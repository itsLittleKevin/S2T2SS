#!/bin/bash

# 🚀 S2T2SS Automated Setup Script for Linux
# This script will install all dependencies and configure the system

set -e

echo ""
echo "============================================================"
echo "🚀 S2T2SS Real-time Translation System Setup"
echo "============================================================"
echo ""
echo "This will install:"
echo "- Python dependencies"
echo "- PyTorch with CUDA support"
echo "- Audio processing libraries"
echo "- Japanese language support"
echo "- LM Studio integration"
echo ""

read -p "Press Enter to continue..."

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    DISTRO=$ID
else
    OS="Unknown"
    DISTRO="unknown"
fi

echo "✅ Detected OS: $OS"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3.9-3.11 first."
    echo "Ubuntu/Debian: sudo apt install python3.10 python3.10-pip python3.10-venv"
    echo "CentOS/RHEL: sudo yum install python39 python39-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Found Python $PYTHON_VERSION"

# Check if we're already in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
else
    echo ""
    echo "🐍 Setting up virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "📦 Creating virtual environment at .venv..."
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create virtual environment"
            echo "Try: sudo apt install python3-venv"
            exit 1
        fi
        echo "✅ Virtual environment created"
    else
        echo "✅ Virtual environment already exists"
    fi
    
    # Activate virtual environment
    echo "🔄 Activating virtual environment..."
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment activated"
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]]; then
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        ffmpeg \
        portaudio19-dev \
        pulseaudio \
        alsa-utils \
        libasound2-dev \
        libsndfile1-dev \
        libffi-dev \
        libssl-dev
elif [[ "$DISTRO" == "fedora" ]] || [[ "$DISTRO" == "centos" ]] || [[ "$DISTRO" == "rhel" ]]; then
    sudo dnf install -y \
        gcc gcc-c++ \
        cmake \
        git \
        curl \
        wget \
        ffmpeg \
        portaudio-devel \
        pulseaudio \
        alsa-lib-devel \
        libsndfile-devel \
        libffi-devel \
        openssl-devel
elif [[ "$DISTRO" == "arch" ]]; then
    sudo pacman -S \
        base-devel \
        cmake \
        git \
        curl \
        wget \
        ffmpeg \
        portaudio \
        pulseaudio \
        alsa-lib \
        libsndfile
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Check for CUDA
echo ""
echo "🔥 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "⚠️  No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install main requirements
echo ""
echo "📦 Installing main dependencies..."
pip install -r requirements.txt

# Install Japanese language support
echo ""
echo "🇯🇵 Installing Japanese language support..."
pip install fugashi[unidic-lite] cutlet unidic-lite

# Install additional audio dependencies
echo ""
echo "🔊 Installing audio dependencies..."
pip install sounddevice soundfile librosa

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p models
mkdir -p config

# Check for sample voice file
if [ ! -f "sample02.wav" ]; then
    echo ""
    echo "🎤 Voice sample not found..."
    echo "Please add your voice sample file as 'sample02.wav'"
    echo "You can record a 10-30 second audio sample of your voice"
fi

# Check for LM Studio
echo ""
echo "🤖 Checking LM Studio..."
if command -v lm-studio &> /dev/null; then
    echo "✅ LM Studio found"
else
    echo "⚠️  LM Studio not found"
    echo "Please download and install LM Studio from:"
    echo "https://lmstudio.ai/"
    echo ""
    echo "For Linux, you can also use:"
    echo "wget https://releases.lmstudio.ai/linux/x64/0.2.25/LM_Studio-0.2.25.AppImage"
    echo "chmod +x LM_Studio-0.2.25.AppImage"
    echo ""
    echo "After installing:"
    echo "1. Download Gemma-3-4B-IT model"
    echo "2. Start local server on port 1234"
fi

# Setup audio
echo ""
echo "🔊 Audio Setup..."
echo "PulseAudio/ALSA should be configured automatically."
echo "For advanced audio routing, consider installing:"
echo "- JACK Audio Connection Kit"
echo "- PipeWire (modern alternative)"

# Create configuration file
echo ""
echo "⚙️  Creating configuration file..."
cat > config.json << EOF
{
  "deployment": {
    "version": "1.0.0",
    "setup_date": "$(date)",
    "platform": "linux",
    "distribution": "$DISTRO"
  },
  "language_pairs": {
    "default": "chinese_to_english",
    "available": ["chinese_to_english", "english_to_chinese", "chinese_to_japanese", "english_to_japanese", "auto_to_english", "auto_to_chinese"]
  },
  "audio": {
    "enable_vb_audio": false,
    "sample_rate": 16000
  },
  "performance": {
    "use_gpu": $(command -v nvidia-smi &> /dev/null && echo "true" || echo "false"),
    "buffer_size": 1,
    "similarity_threshold": 0.80,
    "tts_speed": 1.0
  }
}
EOF

# Test installation
echo ""
echo "🧪 Testing installation..."
python -c "import torch; print('✅ PyTorch:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available())"
python -c "import sounddevice as sd; print('✅ Audio devices:', len(sd.query_devices()))"
python -c "import funasr; print('✅ FunASR installed')" 2>/dev/null || echo "⚠️  FunASR may need manual installation"

echo ""
echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Place your voice sample as 'sample02.wav'"
echo "2. Install and start LM Studio with Gemma-3-4B-IT model"
echo "3. Run: python core/main_pipeline.py"
echo "4. Choose your translation mode from the menu"
echo ""
echo "To activate the environment in future sessions:"
echo "source .venv/bin/activate"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions"
echo "🎮 Happy translating!"
echo ""
