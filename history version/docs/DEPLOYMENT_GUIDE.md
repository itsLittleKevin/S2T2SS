# 🚀 S2T2SS Deployment Guide

**Real-time Speech-to-Text-to-Speech with Translation & RVC Voice Conversion**

Choose your deployment method based on your target environment:

## 🎯 **Quick Setup Options**

### **Option 1: Automated Setup Script (Recommended)**
**Best for**: Users who want minimal manual work
```bash
# Windows
.\setup.bat

# Linux
chmod +x setup.sh && ./setup.sh
```

### **Option 2: Docker Deployment**
**Best for**: Consistent environments, servers, cloud deployment
```bash
docker-compose up -d
```

### **Option 3: Conda Environment**
**Best for**: Data scientists, existing conda users
```bash
conda env create -f environment.yml
conda activate s2t2ss
```

### **Option 4: Manual Installation**
**Best for**: Advanced users who want full control
```bash
pip install -r requirements.txt
python setup_check.py
```

---

## 🖥️ **System Requirements**

### **Minimum Requirements:**
- **RAM**: 8GB (16GB recommended)
- **GPU**: NVIDIA GTX 1060 or better (optional but recommended)
- **Storage**: 10GB free space
- **OS**: Windows 10+ / Ubuntu 18.04+ / macOS 10.15+

### **Recommended for Best Performance:**
- **RAM**: 16GB+
- **GPU**: RTX 3060 12GB or better
- **Storage**: 20GB SSD space
- **CPU**: 8 cores or more

---

## 📦 **Dependencies Overview**

### **Core Components:**
- **Python 3.9-3.11** (3.10 recommended)
- **PyTorch with CUDA** (for GPU acceleration)
- **FunASR** (Chinese speech recognition)
- **XTTS v2** (multilingual text-to-speech)
- **RVC** (voice conversion)
- **LM Studio** (local LLM server)

### **Audio Dependencies:**
- **VB-Audio Virtual Cable** (Windows audio routing)
- **PortAudio** (cross-platform audio)
- **Japanese TTS Support** (MeCab, fugashi, cutlet)

---

## 🎮 **Quick Start Guide**

### **Step 1: Download and Extract**
```bash
git clone https://github.com/your-repo/S2T2SS.git
cd S2T2SS
```

### **Step 2: Run Setup**
**Windows:**
```cmd
setup.bat
```

**Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### **Step 3: Configure Audio** (Windows)
```cmd
setup_audio.bat
```

### **Step 4: Start System**
```bash
python asr/main.py
```

### **Step 5: Choose Translation Mode**
Select from the interactive menu:
- 1️⃣ Chinese → English
- 2️⃣ English → Chinese  
- 3️⃣ Chinese → Japanese
- And more...

---

## ⚙️ **Configuration**

### **Quick Configuration:**
Edit `config.json` for easy setup:
```json
{
  "language_pairs": {
    "default": "chinese_to_english",
    "available": ["chinese_to_english", "english_to_chinese", "chinese_to_japanese"]
  },
  "audio": {
    "enable_rvc": true,
    "rvc_model": "Veibae-BySkeetawn.pth",
    "enable_vb_audio": true
  },
  "performance": {
    "use_gpu": true,
    "buffer_size": 1,
    "similarity_threshold": 0.80
  }
}
```

### **Advanced Configuration:**
- `asr/main.py` - Main configuration
- `config/main.py` - Audio device settings
- `rvc/main.py` - Voice conversion settings

---

## 🐳 **Docker Deployment**

### **Simple Docker Run:**
```bash
docker run -d \
  --name s2t2ss \
  --gpus all \
  -p 8000:8000 \
  -v ./config:/app/config \
  -v ./models:/app/models \
  s2t2ss:latest
```

### **Docker Compose (Recommended):**
```bash
docker-compose up -d
```

Includes:
- ✅ Main S2T2SS application
- ✅ LM Studio server
- ✅ Audio device mapping
- ✅ Model persistence
- ✅ Configuration volumes

---

## 🌐 **Cloud Deployment**

### **AWS EC2:**
- Use G4dn instances (GPU optimized)
- Pre-configured AMI available
- One-click CloudFormation template

### **Google Cloud:**
- Compute Engine with GPU
- Container-optimized OS
- Kubernetes deployment available

### **Azure:**
- NC-series VMs
- Container Instances
- ARM template provided

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**"CUDA not available"**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**"No audio device found"**
- Windows: Install VB-Audio Virtual Cable
- Linux: Install PulseAudio/ALSA
- Run: `python test_audio_devices.py`

**"Japanese TTS fails"**
```bash
pip install fugashi[unidic-lite] cutlet unidic-lite
```

**"LLM not responding"**
- Start LM Studio: `lm-studio serve`
- Check: http://localhost:1234
- Load model: Gemma-3-4B-IT

### **Performance Optimization:**
- **GPU Memory**: Reduce model precision
- **CPU Usage**: Lower buffer size
- **Audio Latency**: Adjust chunk duration
- **Network**: Use local models only

---

## 📋 **Feature Matrix**

| Feature | Windows | Linux | macOS | Docker |
|---------|---------|-------|-------|--------|
| Speech Recognition | ✅ | ✅ | ✅ | ✅ |
| Text-to-Speech | ✅ | ✅ | ✅ | ✅ |
| RVC Voice Conversion | ✅ | ✅ | ⚠️ | ✅ |
| VB-Audio Routing | ✅ | ❌ | ❌ | ❌ |
| GPU Acceleration | ✅ | ✅ | ⚠️ | ✅ |
| Japanese Support | ✅ | ✅ | ✅ | ✅ |
| Translation Modes | ✅ | ✅ | ✅ | ✅ |

---

## 🎁 **Pre-built Packages**

### **Windows Installer:**
- `S2T2SS-Setup.exe` (150MB)
- Includes all dependencies
- Auto-configures audio devices
- One-click installation

### **Linux AppImage:**
- `S2T2SS-x86_64.AppImage` (200MB)
- Portable, no installation needed
- Includes runtime dependencies

### **Docker Images:**
- `s2t2ss:latest` - Full version (2GB)
- `s2t2ss:lite` - CPU-only version (800MB)
- `s2t2ss:gpu` - GPU-optimized (2.5GB)

---

## 🤝 **Support & Community**

- **Documentation**: [docs.s2t2ss.com](https://docs.s2t2ss.com)
- **Discord**: [discord.gg/s2t2ss](https://discord.gg/s2t2ss)
- **GitHub Issues**: Report bugs and feature requests
- **YouTube Tutorials**: Step-by-step setup guides

---

## 📄 **License**

MIT License - See LICENSE file for details.

Built with ❤️ by the S2T2SS community.
