@echo off
setlocal enabledelayedexpansion

:: 🚀 S2T2SS Automated Setup Script for Windows
:: This script will install all dependencies and configure the system

echo.
echo ============================================================
echo 🚀 S2T2SS Real-time Translation System Setup
echo ============================================================
echo.
echo This will install:
echo - Python dependencies
echo - PyTorch with CUDA support
echo - Audio processing libraries
echo - Japanese language support
echo - LM Studio integration
echo.

pause

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.9-3.11 first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Found Python %PYTHON_VERSION%

:: Check if we're already in a virtual environment
python -c "import sys; exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo 🐍 Setting up virtual environment...
    
    :: Create virtual environment if it doesn't exist
    if not exist ".venv" (
        echo 📦 Creating virtual environment at .venv...
        python -m venv .venv
        if errorlevel 1 (
            echo ❌ Failed to create virtual environment
            pause
            exit /b 1
        )
        echo ✅ Virtual environment created
    ) else (
        echo ✅ Virtual environment already exists
    )
    
    :: Activate virtual environment
    echo 🔄 Activating virtual environment...
    call .venv\Scripts\activate.bat
    if errorlevel 1 (
        echo ❌ Failed to activate virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment activated
) else (
    echo ✅ Already in virtual environment
)

echo ✅ Python found
python --version

:: Check if we're in a virtual environment
if not defined VIRTUAL_ENV (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo ✅ Virtual environment created and activated
) else (
    echo ✅ Virtual environment already active
)

:: Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA support
echo.
echo 🔥 Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install main requirements
echo.
echo 📦 Installing main dependencies...
pip install -r requirements.txt

:: Install Japanese language support
echo.
echo 🇯🇵 Installing Japanese language support...
pip install fugashi[unidic-lite] cutlet unidic-lite

:: Install additional audio dependencies
echo.
echo 🔊 Installing audio dependencies...
pip install sounddevice soundfile librosa

:: Create necessary directories
echo.
echo 📁 Creating directories...
if not exist "models" mkdir models
if not exist "config" mkdir config

:: Download sample voice file if not exists
if not exist "sample02.wav" (
    echo.
    echo 🎤 Downloading sample voice file...
    echo Please add your voice sample file as 'sample02.wav'
    echo You can record a 10-30 second audio sample of your voice
)

:: Check for LM Studio
echo.
echo 🤖 Checking LM Studio...
where lm-studio >nul 2>&1
if errorlevel 1 (
    echo ⚠️  LM Studio not found in PATH
    echo Please download and install LM Studio from:
    echo https://lmstudio.ai/
    echo.
    echo After installing:
    echo 1. Download Gemma-3-4B-IT model
    echo 2. Start local server on port 1234
) else (
    echo ✅ LM Studio found
)

:: Setup VB-Audio Virtual Cable
echo.
echo 🔊 Audio Setup...
echo For OBS integration, please install VB-Audio Virtual Cable:
echo https://vb-audio.com/Cable/
echo.
echo Run setup_audio_devices.bat after installation.

:: Create configuration file
echo.
echo ⚙️  Creating configuration file...
(
echo {
echo   "deployment": {
echo     "version": "1.0.0",
echo     "setup_date": "%date% %time%",
echo     "platform": "windows"
echo   },
echo   "language_pairs": {
echo     "default": "chinese_to_english",
echo     "available": ["chinese_to_english", "english_to_chinese", "chinese_to_japanese", "english_to_japanese", "auto_to_english", "auto_to_chinese"]
echo   },
echo   "audio": {
echo     "enable_vb_audio": true,
echo     "sample_rate": 16000
echo   },
echo   "performance": {
echo     "use_gpu": true,
echo     "buffer_size": 1,
echo     "similarity_threshold": 0.80,
echo     "tts_speed": 1.0
echo   }
echo }
) > config.json

:: Test installation
echo.
echo 🧪 Testing installation...
python -c "import torch; print('✅ PyTorch:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available())"
python -c "import sounddevice as sd; print('✅ Audio devices:', len(sd.query_devices()))"
python -c "import funasr; print('✅ FunASR installed')" 2>nul || echo "⚠️  FunASR may need manual installation"

echo.
echo ============================================================
echo ✅ Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Place your voice sample as 'sample02.wav'
echo 2. Install and start LM Studio with Gemma-3-4B-IT model
echo 3. Run: python core/main_pipeline.py
echo 4. Choose your translation mode from the menu
echo.
echo For OBS integration:
echo 1. Install VB-Audio Virtual Cable
echo 2. Run: setup_audio_devices.bat
echo 3. Configure OBS to capture from VB-Audio
echo.
echo 📖 See DEPLOYMENT_GUIDE.md for detailed instructions
echo 🎮 Happy translating!
echo.

pause
