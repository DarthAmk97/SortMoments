@echo off
echo ============================================================
echo Sort Moments - Standalone Executable Builder
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Installing/updating PyInstaller...
pip install pyinstaller --upgrade --quiet

echo.
echo Building standalone executable...
echo This may take 5-10 minutes...
echo.

python build_exe.py

echo.
pause
