@echo off
setlocal enabledelayedexpansion
title AerialSketch Launcher 
color 0B

echo =======================================
echo        AerialSketch Launcher
echo =======================================
echo.

:: 1. Setup Virtual Environment ONLY if it doesn't exist
if not exist "venv" (
    echo [!] Virtual environment not found. 
    echo [!] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [!] Installing dependencies...
    python -m pip install -r requirements.txt
) else (
    :: 2. Activate venv
    call venv\Scripts\activate.bat
)

:: 3. Launch Application
echo [OK] Launching AerialSketch...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [!] AerialSketch exited with an error. 
    echo [!] If dependencies are missing, try deleting the "venv" folder and running this script again.
)

echo.
pause
