# 🚀 S2T2SS - Speech-to-Text-to-Speech-to-Speech

**Real-time Multilingual Translation & Voice Conversion System**

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Docker-lightgrey.svg)

A comprehensive real-time speech translation and voice conversion system that transforms live audio input through multiple AI processing stages, featuring anime character voice conversion and professional streaming integration.

## ✨ Key Features

- 🗣️ **Real-time ASR**: Chinese (FunASR) + Multilingual (XTTS) speech recognition
- 🤖 **LLM Processing**: 6 narrator modes including real-time translation
- 🎭 **Voice Synthesis**: High-quality XTTS v2 multilingual TTS
- � **Speech Recognition**: Real-time audio transcription with FunASR
- 🎙️ **Professional Audio**: VB-Audio Virtual Cable integration for OBS
- 🌍 **Multilingual Support**: Chinese, English, Japanese with automatic detection
- 🐳 **Easy Deployment**: Docker, automated setup scripts, one-click launcher

## 🎯 Quick Start

### Option 1: Easy Start (Recommended)
```bash
git clone https://github.com/yourusername/S2T2SS.git
cd S2T2SS

# Windows
start.bat

# Linux/macOS
chmod +x start.sh
./start.sh
```

### Option 2: Manual Virtual Environment
```bash
git clone https://github.com/yourusername/S2T2SS.git
cd S2T2SS

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate.bat

# Linux/macOS  
source .venv/bin/activate

# Install dependencies and run
pip install -r requirements.txt
python launcher.py
```

### Option 3: Automated Setup
**Windows:**
```cmd
scripts\setup.bat
```

**Linux:**
```bash
chmod +x scripts/setup.sh
scripts/setup.sh
```

**Docker:**
```bash
docker-compose up --build
```

## 📁 Project Structure

```
S2T2SS/
├── 🚀 launcher.py              # Main application launcher
├── 🎯 main.py                  # Core S2T2SS pipeline
├── 🧪 testing_suite.py         # Comprehensive testing tools
├── 📋 requirements.txt         # Python dependencies
├── ⚙️ config_template.json     # Configuration template
├── 🐳 Dockerfile              # Container configuration
├── 📦 docker-compose.yml       # Multi-service deployment
│
├── 📂 core/                    # Modern S2T2SS System (PRIMARY)
│   ├── main_pipeline.py        # Main interactive system
│   ├── asr_module.py           # Speech recognition
│   ├── tts_manager.py          # TTS synthesis
│   └── data/                   # Output files
├── 📂 asr/                     # Legacy system (deprecated)
│
├── 📂 scripts/                 # Setup & Utilities
│   ├── setup.bat              # Windows automated setup
│   ├── setup.sh               # Linux automated setup
│   ├── setup_audio_devices.bat # Audio configuration
│   └── translation_modes.py    # Mode switcher
│
├── 📂 docs/                    # Documentation
│   ├── COMPLETE_SETUP_GUIDE.md # Comprehensive setup
│   ├── NARRATOR_MODES_REFERENCE.md # Mode documentation
│   ├── VB_AUDIO_SETUP.md      # Audio routing guide
│   └── DEPLOYMENT_GUIDE.md     # Production deployment
│
└── 📂 [api|config|text_edit|tts|utils]/  # Additional modules
```

## 🎮 Usage

### Quick Start
```bash
# Easy start with automatic virtual environment
./start.bat     # Windows
./start.sh      # Linux

# Or manual activation
.venv\Scripts\activate.bat    # Windows
source .venv/bin/activate     # Linux
python launcher.py
```

### Direct Launch
```bash
# Start with modern launcher (recommended)
python launcher.py

# Direct access to core system
cd core && python main_pipeline.py

# Legacy system (deprecated)
# python asr/main.py
```

### Testing
```bash
# Comprehensive system testing
python testing_suite.py

# Quick verification
python setup_check.py
```

## 🔧 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9 | 3.10+ |
| **RAM** | 16GB | 32GB |
| **GPU** | GTX 1060 6GB | RTX 3060 12GB+ |
| **Storage** | 50GB | 100GB SSD |

### Required Software
- **CUDA 12.1+** (for GPU acceleration)
- **FFmpeg** (audio processing)
- **VB-Audio Virtual Cable** (for OBS integration)
- **LM Studio** (local LLM server)

## 🎭 Narrator Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `direct` | No processing | Raw transcription |
| `casual` | Friendly tone | Gaming streams |
| `professional` | Formal tone | Business meetings |
| `adaptive` | Context-aware | Mixed content |
| `first_person` | Personal perspective | Vlogs, personal content |
| `translate` | Real-time translation | Multilingual streams |

## 🌍 Language Support

- **Chinese**: Native support with FunASR
- **English**: XTTS multilingual model
- **Japanese**: Full Unicode support with MeCab/fugashi
- **Auto-detection**: Automatic language switching
- **Translation**: Real-time bidirectional translation

## 🎙️ Audio Configuration

### VB-Audio Virtual Cable
For professional streaming with OBS:

1. Install VB-Audio Virtual Cable
2. Set `CABLE Input` as default playback device
3. In OBS: Add Audio Input Capture → `CABLE Output`
4. Configure S2T2SS to output to `CABLE Input`

### Testing Audio
```bash
python testing_suite.py
# Choose: 1. 🔊 Audio Devices Test
```

## 🤖 LLM Integration

### LM Studio (Recommended)
1. Download and install LM Studio
2. Download model: `bartowski/gemma-2-9b-it-GGUF`
3. Start server on `localhost:1234`
4. Test connection via launcher or testing suite

### Alternative LLM Servers
- Ollama
- Text Generation WebUI
- OpenAI API
- Custom implementations

## 📊 Performance Optimization

### Latency Reduction
- Reduce audio chunk size
- Use GPU acceleration
- Optimize model loading

### Quality Improvement
- Higher quality voice references
- Larger LLM models
- Professional audio equipment
- Proper acoustic treatment

## 🐳 Docker Deployment

### Quick Start
```bash
docker-compose up --build
```

### With GPU Support
```bash
# Ensure NVIDIA Container Toolkit is installed
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Custom Configuration
```bash
# Edit docker-compose.yml environment variables
NARRATOR_MODE=translate
AUDIO_DEVICE=pulse
LM_STUDIO_URL=http://host.docker.internal:1234
```

## 🔧 Troubleshooting

### Common Issues

**Audio Problems:**
```bash
# Test audio system
python testing_suite.py  # Option 1

# Windows: Check audio drivers
# Linux: Install PulseAudio/ALSA
```

**GPU Not Detected:**
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**LLM Connection Failed:**
```bash
# Test LM Studio connection
curl http://localhost:1234/v1/models

# Check if server is running
python testing_suite.py  # Option 6
```

**Dependencies Missing:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Check installation
python launcher.py  # Option 4: System Status
```

### Performance Issues

**High Latency:**
- Reduce audio buffer size
- Use faster GPU
- Optimize chunk processing
- Close unnecessary applications

**Poor Quality:**
- Use higher quality voice reference
- Increase TTS sampling rate
- Use larger LLM model
- Check audio device quality

## 📚 Documentation

- **[Complete Setup Guide](docs/COMPLETE_SETUP_GUIDE.md)** - Detailed installation instructions
- **[Narrator Modes Reference](docs/NARRATOR_MODES_REFERENCE.md)** - All narrator modes explained
- **[VB-Audio Setup Guide](docs/VB_AUDIO_SETUP.md)** - Professional audio routing
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment

## 🛠️ Development

### Setting Up Development Environment
```bash
# Clone repository
git clone https://github.com/yourusername/S2T2SS.git
cd S2T2SS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black isort flake8

# Run tests
python testing_suite.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FunASR** - Chinese speech recognition
- **XTTS v2** - Multilingual text-to-speech
- **LM Studio** - Local LLM serving
- **VB-Audio** - Virtual audio cable
- **OpenAI** - API standards and inspiration

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides in `docs/`
- **Testing Suite**: Built-in diagnostic tools

---

**Ready to transform your voice in real-time? Get started with the launcher! 🚀**
