# VB-Audio Virtual Cable Setup Guide

## 🎵 Setting Up VB-Audio Virtual Cable for TTS Output

### Step 1: Install VB-Audio Virtual Cable
1. Download from: https://vb-audio.com/Cable/
2. Install the software
3. Restart your computer (required for driver installation)

### Step 2: Test Your Audio Setup
Run the audio device test script:
```bash
python test_audio_devices.py
```

**IMPORTANT: Test your microphone input separately:**
```bash
python test_microphone.py
```

This will:
- List all available audio devices
- Automatically detect VB-Audio devices  
- Allow you to test audio playback on different devices
- **Test microphone input to ensure speech recognition works**

### Step 3: Configure Your Applications

#### For OBS Studio:
1. Add an **Audio Input Capture** source
2. Set Device to: **"CABLE Output (VB-Audio Virtual Cable)"**
3. This will capture the TTS audio from the virtual cable

#### For Other Recording Software:
- Look for **"CABLE Output (VB-Audio Virtual Cable)"** in your audio input sources
- This is where the TTS audio will be available for capture

### Step 4: Run the TTS System
```bash
cd asr
python main.py
```

The script will:
- Automatically detect VB-Audio devices
- Route TTS output to the virtual cable
- Show which device is being used in the console

### Manual Device Configuration

If you need to use a specific audio device, edit `asr/main.py`:

```python
# Find this line in main.py:
# AUDIO_OUTPUT_DEVICE = None  # None = default device

# Replace with your device ID (found using test_audio_devices.py):
AUDIO_OUTPUT_DEVICE = 5  # Replace 5 with your device ID
```

### Troubleshooting

#### No VB-Audio Device Detected:
- Make sure VB-Audio Virtual Cable is installed
- Restart your computer after installation
- Run `test_audio_devices.py` to verify installation

#### No Audio in OBS:
- Check that OBS Audio Input Capture source is set to "CABLE Output"
- Test the virtual cable using `test_audio_devices.py`
- Make sure the TTS system shows "→ CABLE Input" in console output

#### No Audio Detection in ASR:
- **Check microphone device**: VB-Audio might be set as default input
- Run `python test_microphone.py` to verify microphone is working
- Make sure your actual microphone (e.g., "USB Audio Device") is selected, not VB-Audio
- In Windows Sound settings, set your microphone as default recording device
- Check microphone levels and ensure it's not muted

### Advanced: Multiple Virtual Cables

If you have VB-Audio Cable A & B:
- **CABLE Input/Output**: Virtual Cable A
- **CABLE-A Input/Output**: Virtual Cable A (alternative names)
- **CABLE-B Input/Output**: Virtual Cable B

Use different cables for:
- TTS output (this script)
- Music/background audio
- Voice chat
- etc.

### Expected Console Output

When working correctly, you should see:
```
🔊 Audio Device Configuration:
✅ Auto-selected VB-Audio device: CABLE Input (VB-Audio Virtual Cable)
🎵 Audio output device: 5
🔊 Playing TTS (2.4s @ 1.0x speed) → CABLE Input (VB-Audio Virtual Cable)
```

### OBS Audio Setup Summary

1. **Add Audio Input Capture source**
2. **Device**: "CABLE Output (VB-Audio Virtual Cable)"
3. **This captures the TTS audio for your stream/recording**

The virtual cable acts as a bridge:
```
TTS Script → CABLE Input → [Virtual Cable] → CABLE Output → OBS
```
