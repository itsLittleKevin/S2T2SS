@echo off
echo 🚀 S2T2SS - Activating Virtual Environment
echo =========================================

REM Check if .venv exists
if exist ".venv\Scripts\activate.bat" (
    echo ✅ Found virtual environment at .venv
    echo 🔄 Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
    echo.
    echo 🚀 Starting S2T2SS Launcher...
    python launcher.py
) else if exist "venv\Scripts\activate.bat" (
    echo ✅ Found virtual environment at venv
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
    echo.
    echo 🚀 Starting S2T2SS Launcher...
    python launcher.py
) else (
    echo ❌ No virtual environment found
    echo.
    echo 💡 To create a virtual environment:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    echo.
    echo 🔄 Starting launcher with system Python...
    python launcher.py
)

pause
