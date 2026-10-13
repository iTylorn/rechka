param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$resolvedPython = (Get-Command $PythonExe -ErrorAction Stop).Source
$pythonDirectory = Split-Path -Path $resolvedPython -Parent
$systemDirectory = Join-Path $env:WINDIR "System32"
$env:PATH = @($pythonDirectory, (Join-Path $pythonDirectory "Scripts"), $systemDirectory, $env:WINDIR) -join ";"

& $resolvedPython -B -m unittest discover -s tests -p "test_*.py"
if ($LASTEXITCODE -ne 0) {
    throw "Application tests failed; release build aborted"
}

# PyInstaller creates an EXE in its work directory before COLLECT adds the
# runtime DLLs. Keep that incomplete EXE outside the project to avoid
# mistaking it for the runnable release under dist\WhisperGUI.
$workDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("WhisperGUI-build-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $workDirectory | Out-Null
& $resolvedPython -m PyInstaller --noconfirm --clean --workpath $workDirectory WhisperGUI.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$releaseDirectory = Join-Path $PSScriptRoot "dist\WhisperGUI"
$executable = Join-Path $releaseDirectory "WhisperGUI.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Release executable was not created: $executable"
}
$pythonRuntime = Join-Path $releaseDirectory "_internal\python310.dll"
if (-not (Test-Path -LiteralPath $pythonRuntime -PathType Leaf)) {
    throw "Release Python runtime was not created: $pythonRuntime"
}

$smoke = Start-Process -FilePath $executable -ArgumentList "--smoke-test" -WindowStyle Hidden -Wait -PassThru
if ($smoke.ExitCode -ne 0) {
    throw "Packaged application smoke test failed with exit code $($smoke.ExitCode)"
}

$gpuSmoke = Start-Process -FilePath $executable -ArgumentList "--gpu-smoke-test" -WindowStyle Hidden -Wait -PassThru
if ($gpuSmoke.ExitCode -ne 0) {
    throw "Packaged OpenVINO GPU/FFmpeg test failed with exit code $($gpuSmoke.ExitCode)"
}

$localQwen = Join-Path (Split-Path $PSScriptRoot -Parent) "llm-models\Qwen3-8B-int4-ov"
if (Test-Path -LiteralPath (Join-Path $localQwen "openvino_model.xml")) {
    $quotedModelPath = '"' + $localQwen + '"'
    $llmSmoke = Start-Process -FilePath $executable -ArgumentList @("--llm-smoke-test", $quotedModelPath) -WindowStyle Hidden -Wait -PassThru
    if ($llmSmoke.ExitCode -ne 0) {
        throw "Packaged Qwen GPU test failed with exit code $($llmSmoke.ExitCode)"
    }
}

foreach ($document in @("RELEASE_README.txt", "THIRD_PARTY_NOTICES.txt", "README.md")) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $document) -Destination $releaseDirectory -Force
}

$versionSource = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'app_version.py') -Raw
if ($versionSource -notmatch 'APP_VERSION\s*=\s*"([0-9A-Za-z.\-]+)"') {
    throw 'APP_VERSION not found in app_version.py'
}
$releaseVersion = $Matches[1]
$archive = Join-Path $PSScriptRoot "dist\WhisperGUI-$releaseVersion-win64.zip"
$pendingArchive = "$archive.partial"
if (Test-Path -LiteralPath $pendingArchive) {
    Remove-Item -LiteralPath $pendingArchive -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $releaseDirectory,
    $pendingArchive,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $true
)
Move-Item -LiteralPath $pendingArchive -Destination $archive -Force

$zip = [System.IO.Compression.ZipFile]::OpenRead($archive)
try {
    if ($zip.Entries.FullName -notcontains "WhisperGUI/_internal/python310.dll") {
        throw "Release archive does not contain the Python runtime: $archive"
    }
} finally {
    $zip.Dispose()
}

$resolvedWorkDirectory = [System.IO.Path]::GetFullPath($workDirectory)
$resolvedTempDirectory = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
if (-not $resolvedWorkDirectory.StartsWith($resolvedTempDirectory, [System.StringComparison]::OrdinalIgnoreCase) -or
    (Split-Path -Path $resolvedWorkDirectory -Leaf) -notmatch '^WhisperGUI-build-[0-9a-f]{32}$') {
    throw "Refusing to clean unexpected work directory: $resolvedWorkDirectory"
}
Remove-Item -LiteralPath $resolvedWorkDirectory -Recurse -Force

$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
Write-Output "RUNNABLE EXE (keep _internal beside it): $executable"
Write-Output "DISTRIBUTE THIS ZIP: $archive"
Write-Output "SHA256: $hash"
