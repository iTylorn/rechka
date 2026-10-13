# Разработка и сборка WhisperGUI для Windows

Команды ниже выполняются в PowerShell из `WhisperGUI_Development\app`. Нужны Windows x64, Python 3.10, совместимый Intel GPU и драйвер, видимый OpenVINO. Релизный архив включает Python-библиотеки и FFmpeg; для запуска из исходников зависимости ставятся отдельно. Перед сборкой убедитесь, что рядом с `app/` находятся `diarization/wespeaker_fixed.xml` и `.bin`: `WhisperGUI.spec` берёт их оттуда.

## Подготовка и запуск

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe main.py
```

`start_whisper_gpu.bat` — альтернативный запуск, но он использует строго `%LOCALAPPDATA%\Programs\Python\Python310\python.exe`, а не `.venv`. `requirements-build.txt` включает зависимости приложения из `requirements.txt`, PyInstaller 6.16.0 и `openvino-tokenizers` 2026.3.0.0, которые нужны для сборки.

## Получение моделей

При первом запуске окно предложит установить Whisper large-v3-turbo INT8 (~0,9 ГБ). Кнопка «Скачать модели...» открывает установщик; в окне «Обработать текст ИИ» можно установить Qwen3-8B INT4 (~4,9 ГБ) или Qwen3-14B INT4 (~9,8 ГБ). Скачивание требует доступа к Hugging Face и сохраняет файлы по умолчанию в `%LOCALAPPDATA%\WhisperGUI\models`; после установки модель используется локально. При прерывании загрузки повторный запуск продолжает файлы из `_downloads`.

При ручной раскладке поместите папки моделей в `WhisperGUI_Source_1.1.0\turbo-gpu\whisper-large-v3-turbo-int8-ov` и `WhisperGUI_Source_1.1.0\llm-models\Qwen3-8B-int4-ov` (либо `Qwen3-14B-int4-ov`). Это должны быть полные экспортированные модели OpenVINO, а не одни XML-файлы. В настройках `models_folder` указывает на родительскую папку модели Whisper, `llm_model_path` — на конкретную папку Qwen, `embed_model_path` — на XML WeSpeaker. Установщик GUI обновляет эти настройки автоматически.

## Проверки и сборка

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe main.py --smoke-test
.\.venv\Scripts\python.exe main.py --gpu-smoke-test
$qwenModelPath = Join-Path $env:LOCALAPPDATA 'WhisperGUI\models\llm-models\Qwen3-8B-int4-ov'
.\.venv\Scripts\python.exe main.py --llm-smoke-test $qwenModelPath
.\build_release.ps1 -PythonExe .\.venv\Scripts\python.exe
```

`unittest` проверяет установщик моделей и несколько сценариев настроек/первого запуска. `--smoke-test` создаёт окно без интерактивного цикла. `--gpu-smoke-test` проверяет FFmpeg, видимость GPU, компиляцию WeSpeaker и загрузку Whisper, **если** модель найдена. `--llm-smoke-test` загружает указанную Qwen и её токенизатор; команда предполагает установку Qwen3-8B в папку по умолчанию, а для выбранной вручную папки замените `$qwenModelPath`. Ни один из этих коротких тестов не проверяет качество распознавания на записи.

`build_release.ps1` запускает PyInstaller по `WhisperGUI.spec`, затем оба базовых smoke-теста упакованного EXE. Дополнительный тест Qwen выполняется только при наличии `..\llm-models\Qwen3-8B-int4-ov\openvino_model.xml`. При успехе скрипт создаёт `app\dist\WhisperGUI\WhisperGUI.exe`, полный `app\dist\WhisperGUI-1.1.0-win64.zip` и печатает SHA-256 архива. Для передачи нужен весь ZIP: EXE зависит от соседней папки `_internal`.

Промежуточные файлы PyInstaller создаются во временной папке Windows и удаляются после успешной сборки. Если запустить PyInstaller вручную и увидеть `build\WhisperGUI\WhisperGUI.exe`, не используйте его: он не содержит соседнюю `_internal` и не является готовым релизом.

## Диагностика

- Не запускается GUI: проверьте `%LOCALAPPDATA%\WhisperGUI\startup_error.log`, версию Python 3.10 и установку зависимостей в том же окружении, которым запускается приложение.
- «OpenVINO не видит GPU» или не загружается модель: проверьте драйвер Intel GPU и запуск `--gpu-smoke-test`; CPU-режим для нейросетевого вывода здесь не предусмотрен.
- Не найдена WeSpeaker: проверьте оба файла `..\diarization\wespeaker_fixed.xml` и `.bin` и путь `embed_model_path` в `%APPDATA%\WhisperGUI\settings.json`.
- Не извлекается звук из видео: `MediaLoader` ищет `app\tools\ffmpeg.exe`, затем бинарник из `imageio-ffmpeg`, затем `ffmpeg` в `PATH`; убедитесь, что видео содержит звуковую дорожку.
- Модель скачана, но не видна: проверьте `models_folder` (папка с подпапками Whisper) и `llm_model_path` (конкретная папка Qwen), затем обновите список моделей в GUI. Установщик хранит неполную загрузку в `%LOCALAPPDATA%\WhisperGUI\models\_downloads` или в выбранном каталоге.

## Что ещё не подтверждено этими проверками

Исходный пакет не содержит большие веса Whisper/Qwen. В репозитории нет автоматического сквозного теста распознавания, диаризации, работы на конкретном Intel GPU, качества Qwen3-14B и полного пользовательского сценария ZIP на другой машине. После изменения версий зависимостей или модели выполните ручную проверку на целевом Windows ПК. В `llm_processor.py` метод `close()` вызывает `gc.collect()` без импорта `gc`; этот дефект нужно исправить перед доверием длительной LLM-обработке.
