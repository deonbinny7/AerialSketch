@echo off
echo =======================================
echo   AerialSketch - Starting...
echo =======================================
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo [OK] Starting AerialSketch...
venv\Scripts\python.exe main.py
pause
