@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo CamerBound virtual environment bulunamadi.
    echo Beklenen: .venv\Scripts\python.exe
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "main.py"
set "exit_code=%errorlevel%"

if not "%exit_code%"=="0" (
    echo.
    echo CamerBound hata koduyla kapandi: %exit_code%
    pause
)

exit /b %exit_code%
