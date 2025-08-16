# S2T2SS System - Preparation & Setup Guide

## 1. Prerequisites

### Hardware
- **NVIDIA GPU** (RTX 3060 12GB VRAM or better recommended)
- **Windows 10/11** or **Linux** (WSL2 supported)

### Software & Drivers
- **CUDA Toolkit** (v11.8+):  
  Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
  Install matching NVIDIA drivers.

- **Python 3.10+** (Recommended: 3.11)
- **Git** (for cloning the repository)

### Required External Software
- **LM Studio** (for local LLM inference):  
  Download from [LM Studio](https://lmstudio.ai/)  
  - Start LM Studio and download a supported model (e.g., Qwen2, Llama-3, etc.)
  - Enable the HTTP server (default: `http://localhost:1234`)

- **Ollama** (optional, for LLM):  
  Download from [Ollama](https://ollama.com/download)

- **VB-Audio Virtual Cable** (for audio routing, Windows only):  
  Download from [VB-Audio](https://vb-audio.com/Cable/)

- **FFmpeg** (for audio processing):  
  Download from [FFmpeg](https://ffmpeg.org/download.html)  
  Add to your system PATH.

---

## 2. Clone & Prepare the S2T2SS Repository

```powershell
git clone https://github.com/itsLittleKevin/S2T2SS
cd S2T2SS
```

---

## 3. Python Environment Setup

### Option 1: Automated Setup (Recommended)
```powershell
.\scripts\setup.bat
```
or on Linux:
```bash
./scripts/setup.sh
```

### Option 2: Manual Setup
1. Create and activate a virtual environment:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
2. Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```

---

## 4. Configuration

### Main Config Files

- **`config_template.json`**: Reference template for deployment.
- **`config.json`**: Minimal launcher config (used by `launcher.py`).
- **`core/data/s2t2ss_config.json`**: Main runtime config (controls all features).

#### How to Adjust Configs

- **Edit `config.json`** for basic settings:
    ```json
    {
      "narrator_mode": "direct",
      "language": "zh",
      "audio_output": "auto"
    }
    ```

- **Edit `core/data/s2t2ss_config.json`** for advanced options:
    - LLM server URL, narrator mode, output language, TTS voice, similarity threshold, cache settings, CUDA error recovery, sequential TTS, etc.
    - Example:
      ```json
      {
        "LLM_SERVER_URL": "http://localhost:1234",
        "NARRATOR_MODE": "direct",
        "OUTPUT_LANGUAGE": "zh",
        "ENABLE_TTS": true,
        "SIMILARITY_THRESHOLD": 0.8,
        "FORCE_SEQUENTIAL_TTS": true,
        "ENABLE_CUDA_ERROR_RECOVERY": true
      }
      ```

- **Reference `config_template.json`** for all possible options and their defaults.

#### Common Adjustments

- **Change narrator mode**: `"NARRATOR_MODE": "direct"` (see template for all modes)
- **Switch output language**: `"OUTPUT_LANGUAGE": "en"` or `"zh"`
- **Enable/disable TTS**: `"ENABLE_TTS": true/false`
- **Set LLM server**: `"LLM_SERVER_URL": "http://localhost:1234"`
- **Tune chunk duration**: `"chunk_duration": 3.0` (in `audio` section, template default is 5.0)
- **Adjust similarity threshold**: `"SIMILARITY_THRESHOLD": 0.8`
- **Enable CUDA error recovery**: `"ENABLE_CUDA_ERROR_RECOVERY": true`
- **Force sequential TTS**: `"FORCE_SEQUENTIAL_TTS": true` (recommended for CUDA stability)

---

## 5. File Structure Overview

```
S2T2SS/
├── config_template.json         # Reference config template
├── config.json                  # Minimal launcher config
├── launcher.py                  # Main launcher script
├── requirements.txt             # Python dependencies
├── core/
│   ├── main_pipeline.py         # Main processing pipeline
│   ├── asr_module.py            # Speech recognition (ASR)
│   ├── llm_worker.py            # LLM text processing
│   ├── tts_manager.py           # Text-to-speech synthesis
│   ├── caption_manager.py       # Caption management
│   ├── config.py                # Config loader/manager
│   ├── gpu_stream_manager.py    # CUDA stream management
│   ├── data/
│   │   ├── s2t2ss_config.json   # Main runtime config
│   │   ├── buffer_caption.txt   # Caption buffer
│   │   ├── input.srt/.txt       # Input transcripts
│   │   ├── output.srt/.txt      # Output transcripts
│   │   ├── live_caption.txt     # Live captions
│   │   ├── voice_samples/       # Reference voices
│   │   └── cache/               # Audio/temp files
├── docs/                        # Setup and troubleshooting guides
├── scripts/                     # Utility scripts (setup, benchmarks, etc.)
```

---

## 6. Running the System

### Start LM Studio (or Ollama) and load your model.

### Launch S2T2SS:
```powershell
python launcher.py
```
or use the provided batch script:
```powershell
start.bat
```

---

## 7. Troubleshooting

- **CUDA errors**: Ensure drivers and CUDA toolkit match, use `"FORCE_SEQUENTIAL_TTS": true` in config for stability.
- **Audio issues**: Check VB-Audio installation and device selection.
- **LLM connection**: Verify LM Studio/Ollama HTTP server is running and accessible.

See `docs/COMPLETE_SETUP_GUIDE.md` and `docs/CUDA_ERROR_FIX.md` for more help.

---

## 8. Updating & Customizing

- To update configs, edit the relevant JSON files and restart the system.
- For new deployments, copy `config_template.json` to `config.json` and adjust as needed.
- Advanced users can tune `core/data/s2t2ss_config.json` for performance and stability.

---
# References & External Projects Used in S2T2SS

## Core Dependencies & Technologies

### Speech Recognition (ASR)
- **FunASR**  
  - [FunASR GitHub](https://github.com/alibaba/FunASR)  
  - Used for real-time speech-to-text on GPU (cuda:0).
  - Model: `speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`

### Text-to-Speech (TTS)
- **XTTS v2**  
  - [XTTS v2 GitHub](https://github.com/coqui-ai/xtts)  
  - GPU-accelerated multi-lingual TTS engine.
  - Supports voice cloning via reference samples.

### Language Model (LLM)
- **LM Studio**  
  - [LM Studio](https://lmstudio.ai/)  
  - Local LLM inference with HTTP API.
  - Supports models like Qwen2, Llama-3, etc.

- **Ollama** (optional)  
  - [Ollama](https://ollama.com/)  
  - Alternative local LLM server, supports many open-source models.

### Audio Routing
- **VB-Audio Virtual Cable**  
  - [VB-Audio](https://vb-audio.com/Cable/)  
  - Virtual audio device for routing output/input (Windows only).

### Audio Processing
- **FFmpeg**  
  - [FFmpeg](https://ffmpeg.org/)  
  - Used for audio format conversion and processing.

---

## Python Libraries

- **PyTorch** (with CUDA support)  
  - [PyTorch](https://pytorch.org/)  
  - Deep learning backend for ASR and TTS.

- **NumPy, SoundFile, librosa**  
  - Audio and signal processing.

- **Requests, Flask**  
  - HTTP communication and API integration.

---

## Project Structure & References

- **`core/`**: Main pipeline, ASR, TTS, LLM worker, config manager.
- **`core/models/`**: Pretrained ASR models.
- **`core/data/voice_samples/`**: Reference voices for TTS.
- **`docs/`**: Setup, deployment, and troubleshooting guides.

---

## Related Projects & Documentation

- **S2T2SS Documentation**  
  - See `docs/COMPLETE_SETUP_GUIDE.md` for full setup and usage.
  - `docs/OLLAMA_SETUP.md` for Ollama integration.
  - `docs/VB_AUDIO_SETUP.md` for audio routing.
  - `docs/CUDA_ERROR_FIX.md` for GPU troubleshooting.

- **Other Open-Source References**  
  - [Coqui TTS](https://github.com/coqui-ai/TTS) (XTTS base)
  - [OpenAI Whisper](https://github.com/openai/whisper) (alternative ASR, not default)
  - [Transformers](https://github.com/huggingface/transformers) (LLM models, used via LM Studio/Ollama)

---

This project contains contents generated by LLMs.

## Citation

If you use S2T2SS or its components, please cite the original projects:
- FunASR, XTTS, LM Studio, Ollama, VB-Audio, FFmpeg, PyTorch, and any LLM models used.

---

**For more details, see the documentation in the `docs/` folder.**