# 🚀 S2T2SS Complete Setup Guide

## 📋 Overview

S2T2SS (Speech-to-Text-to-Speech-to-Speech) is a real-time multilingual live captioning and voice conversion system featuring:

- **Real-time ASR**: Chinese (FunASR) + Multilingual (Whisper/XTTS)
- **LLM Processing**: Text refinement with 6 narrator modes including translation
- **TTS Synthesis**: XTTS v2 multilingual voice synthesis
- **VB-Audio Integration**: Professional audio routing for OBS
- **Multi-platform Support**: Windows, Linux, Docker

---

## 🎯 Quick Start Options

### Option 1: Easy Start with Virtual Environment (Recommended)

**Windows:**
```cmd
git clone <repository>
cd S2T2SS
start.bat
```

**Linux:**
```bash
git clone <repository>
cd S2T2SS
chmod +x start.sh
./start.sh
```

### Option 2: Manual Virtual Environment Setup

**Create Virtual Environment:**
```bash
# Navigate to project directory
cd S2T2SS

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate.bat

# Linux/macOS:
source .venv/bin/activate

# Verify activation (should show virtual environment path)
which python    # Linux/macOS
where python    # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the system
python launcher.py
```

### Option 3: Automated Setup

**Windows:**
```cmd
git clone <repository>
cd S2T2SS
.\setup.bat
```

**Linux:**
```bash
git clone <repository>
cd S2T2SS
chmod +x setup.sh
./setup.sh
```

**Docker:**
```bash
git clone <repository>
cd S2T2SS
docker-compose up --build
```

### Option 2: Manual Setup

Follow the detailed instructions below for complete control over your installation.

---

## 🔧 System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel i5-8th gen / AMD Ryzen 5 | Intel i7-10th gen / AMD Ryzen 7 |
| **RAM** | 16GB | 32GB |
| **GPU** | GTX 1060 6GB / RTX 2060 | RTX 3060 12GB+ |
| **Storage** | 50GB free space | 100GB+ SSD |
| **Audio** | Standard audio I/O | Professional audio interface |

### Operating System Support

- **Windows 10/11** (Tested on Windows 11)
- **Ubuntu 20.04/22.04** (Recommended Linux distribution)
- **Other Linux** (May require additional configuration)
- **Docker** (All platforms with Docker support)

---

## 📦 System Dependencies

### 1. Python Environment

**Python 3.9-3.11 Required** (Python 3.12+ not yet supported by all dependencies)

**Windows:**
```cmd
# Download from python.org or use Microsoft Store
winget install Python.Python.3.11

# Verify installation
python --version
pip --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv python3.11-dev

# Set as default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

### 2. CUDA Toolkit (For GPU Acceleration)

**Required for optimal performance with NVIDIA GPUs**

**Windows:**
1. Download CUDA 12.1 from [NVIDIA Developer](https://developer.nvidia.com/cuda-12-1-0-download-archive)
2. Install with default settings
3. Verify installation:
```cmd
nvcc --version
nvidia-smi
```

**Linux:**
```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify
nvcc --version
nvidia-smi
```

### 3. FFmpeg (Required for audio processing)

**Windows:**
```cmd
# Using winget
winget install Gyan.FFmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### 4. Git (For cloning repository)

**Windows:**
```cmd
winget install Git.Git
```

**Linux:**
```bash
sudo apt install git
```

### 5. VB-Audio Virtual Cable (For OBS integration)

**Download and install from:** https://vb-audio.com/Cable/

**Windows Installation:**
1. Download VB-CABLE Driver Pack
2. Extract and run `VBCABLE_Setup_x64.exe` as Administrator
3. Restart computer when prompted
4. Verify installation in Windows Sound settings

### 6. LM Studio (For local LLM)

**Download from:** https://lmstudio.ai/

**Installation:**
1. Download LM Studio for your platform
2. Install and launch
3. Download a model (recommended: `bartowski/gemma-2-9b-it-GGUF`)
4. Start local server on `localhost:1234`

---

## 🐍 Python Dependencies Installation

### Create Virtual Environment

**IMPORTANT: Always use a virtual environment to avoid conflicts**

**Windows:**
```cmd
cd S2T2SS
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux:**
```bash
cd S2T2SS
python3 -m venv .venv
source .venv/bin/activate
```

**Verify Virtual Environment:**
```bash
# Check that you're in the virtual environment
# The prompt should show (.venv) prefix
# Python path should point to your virtual environment

# Windows
where python
# Should show: C:\path\to\S2T2SS\.venv\Scripts\python.exe

# Linux
which python
# Should show: /path/to/S2T2SS/.venv/bin/python
```

### Install Core Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
pip install -r requirements.txt
```

### Install Japanese Language Support

```bash
# Japanese text processing
pip install fugashi cutlet unidic-lite

# Additional Japanese dependencies
python -c "import unidic; unidic.download()"
```

### Verify Installation

```bash
python testing_suite.py
```

---


---

## 🎙️ Audio Configuration

### VB-Audio Virtual Cable Setup

1. **Install VB-Audio Virtual Cable** (see system dependencies)
2. **Configure Windows Audio:**
   - Set `CABLE Input` as default playback device
   - Keep your microphone as default recording device
3. **Configure OBS:**
   - Add Audio Input Capture source
   - Select `CABLE Output` as device

### Test Audio Configuration

```bash
python testing_suite.py
# Choose option 1: Audio Devices Test
```

### Microphone Configuration

**Windows:**
- Go to Settings → System → Sound
- Ensure microphone privacy settings allow app access
- Test microphone levels

**Linux:**
```bash
# Install PulseAudio utilities
sudo apt install pulseaudio pulseaudio-utils

# List audio devices
pactl list sources
pactl list sinks

# Test recording
arecord -f cd -t wav -d 5 test.wav && aplay test.wav
```

---

## 🧠 LLM Server Setup

### LM Studio Configuration

1. **Download and install LM Studio**
2. **Download recommended model:**
   - `bartowski/gemma-2-9b-it-GGUF` (4-bit quantized)
   - Alternative: `microsoft/Phi-3-mini-4k-instruct`

3. **Start local server:**
   - Click "Local Server" tab
   - Select your model
   - Start server on `localhost:1234`

4. **Test connection:**
```bash
curl http://localhost:1234/v1/models
```

### Alternative LLM Options

**Ollama (Linux/macOS):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download model
ollama pull gemma:7b

# Start server
ollama serve
```

**Text Generation WebUI:**
```bash
# Clone repository
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui

# Install and run
pip install -r requirements.txt
python server.py --api --listen
```

---

## 📁 Project Configuration

### Create Configuration File

```bash
cp config_template.json config.json
```

### Edit Configuration

```json
{
  "audio": {
    "input_device": "auto",
    "output_device": "CABLE Input (VB-Audio Virtual Cable)",
    "sample_rate": 16000
  },
  "asr": {
    "language": "auto",
    "model": "funasr_paraformer"
  },
  "llm": {
    "server_url": "http://localhost:1234/v1",
    "model": "local-model",
    "narrator_mode": "casual"
  },
  "tts": {
    "model": "xtts_v2",
    "voice_file": "sample02.wav",
    "speed": 1.0
  }
}
```

### Voice Reference Setup

1. **Record or download reference voice:**
   - 10-30 seconds of clean speech
   - Save as `sample02.wav` in project root
   - Alternative names: `sample.wav`, `reference.wav`

2. **Voice quality tips:**
   - Clear, noise-free recording
   - Single speaker
   - Natural speech patterns
   - 22050Hz or 24000Hz sample rate

---

## 🚀 Running the System

### Start All Components

1. **Start LM Studio server** (if not already running)
2. **Activate Python environment:**
```bash
source venv/bin/activate  # Linux
# or
venv\Scripts\activate     # Windows
```

3. **Run the main system:**
```bash
python main.py
```

### Using Translation Mode

```bash
# Interactive mode switcher
python translation_modes.py

# Or direct mode setting
python main.py --narrator-mode translate
```

### Testing Individual Components

```bash
# Comprehensive testing suite
python testing_suite.py

# Quick system verification
python setup_check.py

# One-click launcher with checks
python quick_start.py
```

---

## 🐳 Docker Deployment

### Prerequisites

- Docker Desktop (Windows/macOS)
- Docker + docker-compose (Linux)
- NVIDIA Container Toolkit (for GPU support)

### GPU Support Setup

**Linux:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build and Run

```bash
# Build container
docker-compose build

# Run with GPU support
docker-compose up

# Run without GPU
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up
```

### Container Configuration

**Environment Variables:**
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - LM_STUDIO_URL=http://host.docker.internal:1234
  - NARRATOR_MODE=casual
  - AUDIO_DEVICE=pulse
```

---

## 🔧 Troubleshooting

### Common Issues

#### Audio Issues

**Problem:** No audio devices detected
```bash
# Windows: Check audio drivers
# Linux: Install PulseAudio
sudo apt install pulseaudio pulseaudio-utils alsa-utils

# Test audio system
python testing_suite.py  # Option 1
```

**Problem:** VB-Audio not working
1. Restart computer after VB-Audio installation
2. Check Windows Sound settings
3. Run setup_audio_devices.bat as Administrator

#### GPU/CUDA Issues

**Problem:** CUDA not available
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Problem:** Out of GPU memory
- Reduce batch sizes in configuration
- Close other GPU applications
- Use CPU mode as fallback

#### ASR Issues

**Problem:** FunASR model not loading
```bash
# Clear model cache
rm -rf ~/.cache/modelscope/

# Reinstall FunASR
pip uninstall funasr
pip install funasr

# Test ASR
python testing_suite.py  # Option 2
```

#### TTS Issues

**Problem:** XTTS synthesis fails
```bash
# Check voice reference file
ls -la *.wav

# Test TTS
python testing_suite.py  # Option 3
```

#### LLM Connection Issues

**Problem:** Cannot connect to LM Studio
1. Ensure LM Studio is running
2. Check server is on localhost:1234
3. Test connection:
```bash
curl http://localhost:1234/v1/models
```


### Performance Optimization

#### Reduce Latency

1. **Audio buffer settings:**
```json
{
  "audio": {
    "chunk_duration": 1.0,
    "buffer_size": 1024
  }
}
```

2. **ASR optimization:**
```json
{
  "asr": {
    "chunk_size": 1,
    "beam_size": 1
  }
}
```

3. **GPU memory management:**
```json
{
  "gpu": {
    "memory_fraction": 0.8,
    "allow_growth": true
  }
}
```

#### Improve Quality

1. **Higher quality voice reference**
2. **Increase TTS sampling rate**
3. **Use larger LLM model**
4. **Enable RVC for voice conversion**

### Log Analysis

**Enable debug logging:**
```bash
export LOG_LEVEL=DEBUG
python main.py
```

**Check log files:**
- `llm_logs.txt` - LLM processing logs
- `live_caption.txt` - Real-time captions
- `output.txt` - Final processed text

---

## 📚 Additional Resources

### Documentation

- [Narrator Modes Reference](NARRATOR_MODES_REFERENCE.md)
- [VB-Audio Setup Guide](VB_AUDIO_SETUP.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

### Model Downloads

- **XTTS Models:** Auto-downloaded on first use
- **FunASR Models:** Auto-downloaded from ModelScope
- **RVC Models:** Manual download required

### Community Support

- GitHub Issues: Report bugs and feature requests
- Discord/Forum: Community discussions
- Documentation: Wiki and guides

### Development

**Contributing:**
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

**Development Setup:**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

---

## 🎉 You're Ready!

After completing this setup, you should have a fully functional S2T2SS system with:

- ✅ Real-time speech recognition
- ✅ LLM text processing with narrator modes
- ✅ High-quality TTS synthesis
- ✅ Optional RVC voice conversion
- ✅ Professional audio routing for streaming
- ✅ Multilingual support with translation

**Next Steps:**
1. Run `python testing_suite.py` to verify all components
2. Configure your preferred narrator mode
3. Test with live audio input
4. Integrate with OBS for streaming
5. Customize voice models and settings

**Happy streaming! 🎙️✨**
