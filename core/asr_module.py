#!/usr/bin/env python3
"""
🎤 S2T2SS ASR Module  
Automatic Speech Recognition with FunASR
"""

import os
import sys
import numpy as np
import soundfile as sf
import sounddevice as sd
import config

# Add parent directory for model imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ASR imports
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print("FunASR available")
except ImportError:
    print("FunASR not available")
    FUNASR_AVAILABLE = False

class FunASREngine:
    def __init__(self, model_dir=None, vad_model_dir=None, use_gpu=None):
        """Initialize FunASR with CPU/GPU support and automatic fallback"""
        if model_dir is None:
            # Look for models in parent asr directory
            model_dir = os.path.abspath(os.path.join(parent_dir, 'asr', 'models', 'speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'))
        if vad_model_dir is None:
            vad_model_dir = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        
        # Determine device to use
        if use_gpu is None:
            use_gpu = config.ENABLE_GPU_ASR
            
        self.device = "cpu"  # Default fallback
        self.model = None
        self.gpu_attempted = False
        self.gpu_available = False
        
        if FUNASR_AVAILABLE:
            # Try GPU first if requested and available
            if use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        print(f"Attempting GPU ASR initialization on {config.GPU_ASR_DEVICE}...")
                        
                        # Check GPU memory availability
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        current_allocated = torch.cuda.memory_allocated(0) / 1024**3
                        available_memory = gpu_memory_gb - current_allocated
                        
                        print(f"   GPU Memory: {available_memory:.1f}GB available of {gpu_memory_gb:.1f}GB total")
                        
                        if available_memory >= 3.0:  # Need at least 3GB for Paraformer-large
                            self.model = AutoModel(
                                model=model_dir, 
                                vad_model=vad_model_dir, 
                                device=config.GPU_ASR_DEVICE,
                                disable_update=True
                            )
                            self.device = config.GPU_ASR_DEVICE
                            self.gpu_available = True
                            self.gpu_attempted = True
                            print(f"GPU ASR initialized successfully on {self.device}")
                        else:
                            print(f"Insufficient GPU memory ({available_memory:.1f}GB), falling back to CPU")
                            self.gpu_attempted = True
                    else:
                        print("CUDA not available, falling back to CPU ASR")
                        
                except Exception as e:
                    print(f"GPU ASR initialization failed: {e}")
                    print("   Falling back to CPU ASR...")
                    self.gpu_attempted = True
            
            # Fallback to CPU if GPU failed or not requested
            if not self.model:
                try:
                    print("Initializing CPU ASR...")
                    self.model = AutoModel(
                        model=model_dir, 
                        vad_model=vad_model_dir, 
                        device="cpu",
                        disable_update=True
                    )
                    self.device = "cpu"
                    print("CPU ASR initialized successfully")
                except Exception as e:
                    print(f"CPU ASR initialization failed: {e}")
                    self.model = None
        else:
            print("FunASR not available")

    def transcribe_chunk(self, audio_chunk):
        """Optimized transcribe for single audio chunk with GPU support"""
        if not self.model:
            return "ASR not available"
        
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_chunk, bytes):
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_data = audio_chunk
            
            # Convert to float32 and normalize (optimized)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0  # Faster than np.iinfo
            
            # Quick silence detection for performance
            if config.SKIP_SILENCE_DETECTION or np.max(np.abs(audio_data)) > 0.001:
                # GPU memory management for coordination with TTS
                if self.gpu_available and config.ENABLE_ASR_TTS_COORDINATION:
                    try:
                        import torch
                        # Clear cache before ASR processing if memory is tight
                        if torch.cuda.memory_allocated(0) / 1024**3 > 8.0:  # If using >8GB
                            torch.cuda.empty_cache()
                    except:
                        pass
                
                # Generate transcription with optimized settings
                result = self.model.generate(
                    input=audio_data, 
                    input_sample_rate=16000, 
                    is_final=True,
                    # Performance optimizations
                    batch_size=config.ASR_BATCH_SIZE,
                    hotword=""  # Skip hotword processing for speed
                )
                
                text = result[0]["text"] if result and "text" in result[0] else ""
                
                # Performance logging for GPU mode
                if self.gpu_available and not config.MINIMAL_LOGGING:
                    try:
                        import torch
                        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                        if gpu_memory > 0.1:  # Only log if significant GPU usage
                            print(f"   GPU ASR: {gpu_memory:.1f}GB VRAM used")
                    except:
                        pass
                
                return text.strip() if text else ""
            else:
                return ""  # Skip processing of silent chunks
            
        except Exception as e:
            if not config.MINIMAL_LOGGING:
                device_info = f"GPU ({self.device})" if self.gpu_available else "CPU"
                print(f"❌ {device_info} transcription error: {e}")
                
                # GPU error recovery
                if self.gpu_available and config.ENABLE_CUDA_ERROR_RECOVERY:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        print("   🔄 GPU cache cleared, retrying on CPU...")
                        # Create temporary CPU model for this chunk
                        cpu_model = AutoModel(
                            model=os.path.abspath(os.path.join(parent_dir, 'asr', 'models', 'speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')),
                            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                            device="cpu",
                            disable_update=True
                        )
                        result = cpu_model.generate(input=audio_data, input_sample_rate=16000, is_final=True)
                        text = result[0]["text"] if result and "text" in result[0] else ""
                        return text.strip() if text else ""
                    except Exception as recovery_error:
                        print(f"   ❌ CPU recovery failed: {recovery_error}")
            return ""

    def transcribe_stream(self, chunk_generator, sample_rate=16000):
        """Transcribe audio chunks from a generator"""
        if not self.model:
            yield "ASR not available"
            return
            
        for chunk in chunk_generator:
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32) / np.iinfo(chunk.dtype).max
            result = self.model.generate(input=chunk, input_sample_rate=sample_rate, is_final=True)
            text = result[0]["text"] if result and "text" in result[0] else ""
            yield text

    def transcribe_file(self, audio_path):
        """Transcribe a single audio file"""
        if not self.model:
            return "ASR not available"
            
        result = self.model.generate(input=audio_path)
        return result[0]["text"] if result and "text" in result[0] else ""

def list_audio_devices():
    """List all available audio devices (filtered for relevance)"""
    devices = sd.query_devices()
    input_devices = []
    output_devices = []
    seen_input_names = set()
    seen_output_names = set()
    
    # Filter devices to avoid duplicates and irrelevant ones
    for i, device in enumerate(devices):
        device_name = device['name']
        
        # Skip low-quality or duplicate devices
        if any(skip in device_name.lower() for skip in [
            'hands-free', 'bt ', 'bluetooth', 'steam streaming', 
            '声音映射器', '声音驱动程序', 'nvidia high definition audio'
        ]):
            continue
            
        if device['max_input_channels'] > 0:  # Input device
            # Only add if we haven't seen this name before (avoid Host API duplicates)
            base_name = device_name.split('(')[0].strip()
            if base_name not in seen_input_names:
                input_devices.append((i, device))
                seen_input_names.add(base_name)
        
        if device['max_output_channels'] > 0:  # Output device
            base_name = device_name.split('(')[0].strip()
            if base_name not in seen_output_names:
                output_devices.append((i, device))
                seen_output_names.add(base_name)
    
    return input_devices, output_devices

def get_vb_audio_device():
    """Find VB-Audio Virtual Cable device automatically"""
    devices = sd.query_devices()
    vb_devices = []
    
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:  # Output device
            device_name = device['name'].lower()
            if 'vb-audio' in device_name or 'cable' in device_name or 'virtual' in device_name:
                vb_devices.append((i, device))
                print(f"🎵 Found VB-Audio device {i}: {device['name']}")
    
    return vb_devices

def record_microphone_chunk(filename="mic_chunk.wav", duration=5, samplerate=16000):
    """Record audio from microphone and save to WAV file"""
    output_path = os.path.join(config.DEFAULT_OUTPUT_DIR, filename)
    print(f"🎤 Recording {duration}s from microphone...")
    
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        sf.write(output_path, audio, samplerate)
        print(f"✅ Audio recorded: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None

def stream_audio_chunks(audio_path, chunk_size=16000*5):
    """Generator that yields audio chunks from a wav file"""
    try:
        data, samplerate = sf.read(audio_path)
        for start in range(0, len(data), chunk_size):
            yield data[start:start+chunk_size], samplerate
    except Exception as e:
        print(f"❌ Error reading audio file: {e}")

if __name__ == "__main__":
    print("🎤 S2T2SS ASR Module Test")
    
    # Test device listing
    input_devices, output_devices = list_audio_devices()
    print(f"Found {len(input_devices)} input devices and {len(output_devices)} output devices")
    
    # Test VB-Audio detection
    vb_devices = get_vb_audio_device()
    print(f"Found {len(vb_devices)} VB-Audio devices")
    
    # Test ASR engine initialization
    asr_engine = FunASREngine()
    if asr_engine.model:
        print("✅ ASR Engine initialized successfully")
    else:
        print("⚠️ ASR Engine initialization failed")
    
    # Test recording
    print("\nTesting microphone recording (3 seconds)...")
    audio_path = record_microphone_chunk(duration=3)
    if audio_path:
        print("✅ Recording test completed")
    else:
        print("❌ Recording test failed")
