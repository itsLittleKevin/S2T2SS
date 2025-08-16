# 🦙 Ollama Network Setup Guide

## Installation on Server Machine (10.0.0.43)

### 1. Download and Install Ollama
1. Go to https://ollama.ai/download
2. Download Ollama for Windows
3. Install it on the server machine (10.0.0.43)

### 2. Configure Ollama for Network Access
Ollama needs environment variables to allow network connections:

**Method A: System Environment Variables (Recommended)**
1. Right-click "This PC" → Properties → Advanced System Settings
2. Click "Environment Variables"
3. Under "System Variables" click "New"
4. Add these variables:
   - Variable: `OLLAMA_HOST`
   - Value: `0.0.0.0:11434`
   
**Method B: PowerShell (Temporary)**
```powershell
$env:OLLAMA_HOST = "0.0.0.0:11434"
ollama serve
```

### 3. Download a Model
```cmd
# Chinese language models
ollama pull qwen2.5:7b           # Recommended for Chinese
ollama pull qwen2:7b             # Alternative
ollama pull llama3.1:8b          # Good multilingual option

# Lightweight options
ollama pull qwen2.5:3b           # Faster, smaller
ollama pull gemma2:2b            # Very fast
```

### 4. Start Ollama Server
```cmd
ollama serve
```

Should show:
```
2024/08/15 12:34:56 inference server listening on 0.0.0.0:11434
```

## Network Configuration

### Firewall Rule (Run as Administrator on server machine)
```powershell
New-NetFirewallRule -DisplayName "Ollama Server" -Direction Inbound -Port 11434 -Protocol TCP -Action Allow
```

### Test Connection
From client machine (10.0.0.168):
```cmd
curl http://10.0.0.43:11434/api/tags
```

## S2T2SS Configuration

Update S2T2SS to use Ollama instead of LM Studio:
- Server URL: `http://10.0.0.43:11434`
- API Format: OpenAI-compatible

## Model Recommendations

**For Chinese Speech Processing:**
- `qwen2.5:7b` - Best Chinese understanding
- `qwen2.5:3b` - Faster, good quality
- `llama3.1:8b` - Great multilingual

**For Speed:**
- `gemma2:2b` - Very fast
- `qwen2.5:3b` - Good balance

## Advantages over LM Studio

✅ **Easy network configuration** - Just set OLLAMA_HOST
✅ **Better CLI management** - Easy model switching
✅ **Lighter resource usage** - More efficient
✅ **OpenAI API compatible** - Works with existing code
✅ **Better multi-user support** - Designed for network use
