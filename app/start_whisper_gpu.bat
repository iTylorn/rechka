@echo off
setlocal
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Python 3.10 not found: %PYTHON_EXE%
    pause
    exit /b 1
)
cd /d "%~dp0"
"%PYTHON_EXE%" main.py
if errorlevel 1 pause
