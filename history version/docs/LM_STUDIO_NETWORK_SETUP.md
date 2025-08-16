# 🌐 LM Studio Network Setup Guide

## Overview
This guide shows how to run LM Studio on a separate machine in your local network and configure S2T2SS to connect to it remotely.

## 🖥️ LM Studio Server Setup (Machine B)

### 1. Install and Configure LM Studio
1. Download and install LM Studio on your dedicated machine
2. Download your preferred model (e.g., Qwen2.5-7B-Instruct, Llama 3.1 8B)
3. Start the model in LM Studio

### 2. Enable Network Access
1. In LM Studio, go to **Local Server** tab
2. Click **Start Server**
3. **IMPORTANT**: Change the server address from `localhost` to `0.0.0.0`
   - This allows connections from other machines on your network
   - Default port is `1234`
4. Note the machine's IP address (e.g., `192.168.1.100` or `10.0.0.50`)

### 3. Firewall Configuration
**Windows (Machine B):**
```powershell
# Allow LM Studio through Windows Firewall
New-NetFirewallRule -DisplayName "LM Studio Server" -Direction Inbound -Port 1234 -Protocol TCP -Action Allow
```

**Alternative: Windows Firewall GUI**
1. Open Windows Defender Firewall
2. Click "Allow an app or feature through Windows Defender Firewall"
3. Click "Change Settings" → "Allow another app"
4. Browse to LM Studio executable and add it
5. Ensure both "Private" and "Public" are checked

## 🎯 S2T2SS Client Setup (Machine A)

### 1. Find LM Studio Server IP
On Machine B (LM Studio server), run:
```cmd
ipconfig
```
Look for IPv4 Address (e.g., `192.168.1.100`)

### 2. Configure S2T2SS
Edit `core/config.py` and change:
```python
LLM_SERVER_URL = "http://192.168.1.100:1234"  # Replace with your LM Studio machine IP
```

### 3. Test Connection
Run this test to verify connectivity:
```python
import requests
import json

# Replace with your LM Studio machine IP
llm_url = "http://192.168.1.100:1234"

try:
    # Test if server is reachable
    response = requests.get(f"{llm_url}/v1/models", timeout=5)
    if response.status_code == 200:
        print("✅ LM Studio server is reachable!")
        models = response.json()
        print(f"Available models: {len(models.get('data', []))}")
    else:
        print(f"❌ Server responded with status: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"❌ Connection failed: {e}")
```

## 🔧 Common Network Configurations

### Home Network (192.168.x.x)
```python
# Examples for home networks
LLM_SERVER_URL = "http://192.168.1.100:1234"    # Common router range
LLM_SERVER_URL = "http://192.168.0.50:1234"     # Alternative range
LLM_SERVER_URL = "http://192.168.1.200:1234"    # Static IP example
```

### Office Network (10.x.x.x)
```python
# Examples for office networks
LLM_SERVER_URL = "http://10.0.0.100:1234"       # Class A private
LLM_SERVER_URL = "http://10.1.1.50:1234"        # Subnet example
```

### Different Port
If you change LM Studio's port from 1234:
```python
LLM_SERVER_URL = "http://192.168.1.100:8080"    # Custom port
```

## 🚀 Performance Benefits

**Machine A (S2T2SS):**
- Focus on ASR processing (CPU/Memory intensive)
- Real-time audio capture and TTS generation
- Caption management and file I/O

**Machine B (LM Studio):**
- Dedicated GPU resources for LLM inference
- No interference from audio processing
- Better thermal management for sustained loads

## 📊 Testing Your Setup

### 1. Network Connectivity Test
```cmd
# From Machine A, test if Machine B is reachable
ping 192.168.1.100
telnet 192.168.1.100 1234
```

### 2. LM Studio API Test
```python
# Test LLM processing through network
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'core'))

import llm_worker
import config

# This will now use your network LM Studio server
llm = llm_worker.LLMWorker(base_url=config.LLM_SERVER_URL)
result = llm.process_text("测试网络连接")
print(f"Network LLM result: {result}")
```

### 3. Full Pipeline Test
Run S2T2SS normally - it will automatically use the network LM Studio server:
```cmd
python launcher.py
```

## 🛠️ Troubleshooting

### Connection Refused
- Verify LM Studio is running and server started
- Check firewall settings on both machines
- Ensure LM Studio is bound to `0.0.0.0`, not `localhost`

### Slow Response Times
- Check network bandwidth between machines
- Use wired connection instead of WiFi if possible
- Consider using a faster/smaller model

### Model Loading Issues
- Ensure the model is fully loaded in LM Studio before testing
- Check LM Studio logs for model loading errors
- Verify sufficient RAM/VRAM on Machine B

## 📝 Configuration Backup

Save your working configuration:
```python
# In core/config.py, document your working setup
LLM_SERVER_URL = "http://192.168.1.100:1234"  # Gaming PC with RTX 4080

# Optional: Add fallback for when server is offline
LLM_FALLBACK_URL = "http://localhost:1234"     # Local backup
```

## 🔄 Dynamic Configuration

For advanced users, you can make the server URL configurable at runtime:
```python
# Example: Environment variable support
import os
LLM_SERVER_URL = os.environ.get('S2T2SS_LLM_URL', 'http://localhost:1234')
```

This allows you to switch between local and network modes:
```cmd
# Use network LM Studio
set S2T2SS_LLM_URL=http://192.168.1.100:1234
python launcher.py

# Use local LM Studio
set S2T2SS_LLM_URL=http://localhost:1234
python launcher.py
```
