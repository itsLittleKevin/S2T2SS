#!/usr/bin/env python3
"""
🎛️ S2T2SS Configuration Module
Core toggle system configuration
"""

import os
import json
import sys

# ================================================================
# ENCODING HELPER
# ================================================================
def safe_print(text):
    """Print text with encoding fallback for Windows compatibility"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows GBK encoding
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

# ================================================================
# GLOBAL TOGGLE CONFIGURATION
# ================================================================

# Core System Toggles - Your 4 requested features
ENABLE_LLM_EDITING = True      # ✅ True: Use LLM to refine text, False: Use raw ASR output directly
LLM_SERVER_URL = "http://10.0.0.43:1234"  # Ollama server (port 11434)
                               #   Local LM Studio:    "http://localhost:1234"
                               #   Network LM Studio:  "http://10.0.0.43:1234"  
                               #   Local Ollama:       "http://localhost:11434"
                               #   Network Ollama:     "http://10.0.0.43:11434" (CURRENT)
LLM_MODEL_NAME = "" # Model name for Ollama/LM Studio
                               # Ollama: Set specific model name (e.g., "L21-21B:latest", "llama3.1:8b")
                               # LM Studio: Set to "" (empty) for auto-detection of loaded model
                               # Auto-detect: Set to "" or None to use whatever model is currently active
ENABLE_OBS_CAPTIONS = True     # ✅ True: Write captions to live_caption.txt for OBS, False: No caption files
ENABLE_TTS = True              # ✅ True: Generate audio output, False: Text-only mode

# Performance Optimization Toggles
FAST_PROCESSING_MODE = True    # ✅ True: Skip detailed LLM processing for speed, False: Full processing
ENABLE_PARALLEL_PROCESSING = False   # ❌ DISABLED: Causes GPU memory conflicts with multiple chunks
ENABLE_BATCH_PROCESSING = True      # ✅ True: Batch multiple operations, False: Individual processing
USE_FAST_SIMILARITY = True         # ✅ True: Use optimized similarity detection, False: Detailed comparison
STREAMLINE_PIPELINE = True          # ✅ True: Skip redundant operations, False: Full pipeline

# GPU ASR Configuration
ENABLE_GPU_ASR = True              # ✅ True: Use GPU for ASR when available, False: Force CPU
GPU_ASR_DEVICE = "cuda:0"          # GPU device for ASR
GPU_MEMORY_FRACTION = 0.4          # Reduced from 0.6 to leave more memory for TTS
ENABLE_ASR_TTS_COORDINATION = True # ✅ True: Smart GPU memory sharing between ASR and TTS
ASR_BATCH_SIZE = 1                 # Process multiple chunks together (1=sequential for stability)
MAX_CONCURRENT_GPU_OPERATIONS = 1  # Maximum concurrent GPU operations to prevent conflicts

# GPU Stream Management - DISABLED: Sequential processing prevents CUDA indexing errors
ENABLE_GPU_STREAMS = False         # ❌ False: Use sequential processing to prevent CUDA errors
MAX_CONCURRENT_STREAMS = 1         # Force single stream operation
STREAM_SYNC_TIMEOUT = 10.0         # Timeout for stream synchronization in seconds

# CUDA Error Prevention Settings
ENABLE_CUDA_ERROR_RECOVERY = True    # ✅ True: Attempt recovery from CUDA errors, False: Fail immediately
XTTS_MAX_TEXT_LENGTH = 200          # Reduced from 400 to prevent large GPU allocations
XTTS_RETRY_ATTEMPTS = 2             # Number of retry attempts for XTTS synthesis
CUDA_MEMORY_CLEANUP = True          # ✅ True: Clean CUDA memory before synthesis, False: No cleanup
SAFE_MODE_ON_CUDA_ERROR = True      # ✅ True: Use fallback on CUDA errors, False: Skip synthesis
FORCE_SEQUENTIAL_GPU_PROCESSING = True  # ✅ True: Force sequential processing to avoid conflicts
FORCE_SEQUENTIAL_TTS = True         # ✅ True: Force sequential TTS processing (CRITICAL for CUDA stability)
ENABLE_MODEL_REINITIALIZATION = True  # ✅ True: Reinitialize XTTS model after errors, False: Keep broken model
MODEL_RESET_COOLDOWN = 30           # Seconds to wait between model resets
PROACTIVE_MODEL_RESET_HOURS = 24     # Hours between proactive model resets (0 = disabled, 24 = daily)

# Performance Optimization
FAST_MODE = True               # ✅ True: Optimize for speed over quality, False: Optimize for quality
SKIP_SILENCE_DETECTION = True  # ✅ True: Process all chunks, False: Skip silence chunks
OPTIMIZED_CHUNK_SIZE = True    # ✅ True: Use performance-optimized chunk sizes, False: Standard sizes
MINIMAL_LOGGING = True         # ✅ True: Reduce logging overhead, False: Detailed logging

# Streaming TTS Optimizations
ENABLE_STREAMING_TTS = True        # ✅ True: Stream audio immediately, False: Wait for completion
NON_BLOCKING_AUDIO = True          # ✅ True: Don't wait for audio to finish, False: Sequential playback
LINEAR_VOICE_OUTPUT = True          # ✅ True: Queue voices with intervals, False: Concurrent voices
VOICE_OUTPUT_INTERVAL = 0.3         # Minimum seconds between voice outputs (prevents overlap)
CAPTION_SYNC_MODE = "voice"         # ✅ "voice": Sync captions with voice timing, "immediate": Show ASAP
AUDIO_QUEUE_SIZE = 5               # Maximum concurrent audio streams
TTS_OVERLAP_THRESHOLD = 1.0        # Seconds - start next TTS when current has 1s left
FAST_TTS_MODE = True               # ✅ True: Prioritize speed over quality, False: High quality

# Additional Configuration
NARRATOR_MODE = "direct"  # Options: "direct", "first_person", "casual_narrator", "professional_narrator", "adaptive", "translate"
OUTPUT_LANGUAGE = "auto"  # Options: "auto", "zh", "en", "ja", "ko", "es", "fr", "de", "ru"
TTS_VOICE = "default"     # Options: "default", "male", "female", "young", "elder", "neutral"
SIMILARITY_THRESHOLD = 0.8
FAST_SIMILARITY_THRESHOLD = 0.75  # Lower threshold for faster processing
MAX_PARALLEL_WORKERS = 3          # Maximum parallel processing threads
BATCH_SIZE = 3                    # Number of chunks to batch process
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")  # Use core/data directory
CACHE_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "cache")
AUDIO_CACHE_DIR = os.path.join(CACHE_DIR, "audio")

# File Management Settings
ENABLE_AUDIO_CLEANUP = True    # ✅ True: Automatically delete old audio files, False: Keep all files
AUDIO_CACHE_MAX_FILES = 50     # Maximum number of audio files to keep in cache
AUDIO_CACHE_MAX_AGE_HOURS = 24 # Delete audio files older than this many hours
AUTO_CLEANUP_ON_EXIT = True    # ✅ True: Cleanup cache when system exits, False: Manual cleanup only

# Ensure directories exist
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

try:
    print("✅ S2T2SS Configuration Module Loaded")
except UnicodeEncodeError:
    print("[OK] S2T2SS Configuration Module Loaded")

def get_config():
    """Get current configuration as dictionary"""
    return {
        'ENABLE_LLM_EDITING': ENABLE_LLM_EDITING,
        'LLM_SERVER_URL': LLM_SERVER_URL,
        'LLM_MODEL_NAME': LLM_MODEL_NAME,
        'ENABLE_OBS_CAPTIONS': ENABLE_OBS_CAPTIONS,
        'ENABLE_TTS': ENABLE_TTS,
        'NARRATOR_MODE': NARRATOR_MODE,
        'OUTPUT_LANGUAGE': OUTPUT_LANGUAGE,
        'TTS_VOICE': TTS_VOICE,
        'SIMILARITY_THRESHOLD': SIMILARITY_THRESHOLD,
        'DEFAULT_OUTPUT_DIR': DEFAULT_OUTPUT_DIR,
        'CACHE_DIR': CACHE_DIR,
        'AUDIO_CACHE_DIR': AUDIO_CACHE_DIR,
        'ENABLE_AUDIO_CLEANUP': ENABLE_AUDIO_CLEANUP,
        'AUDIO_CACHE_MAX_FILES': AUDIO_CACHE_MAX_FILES,
        'AUDIO_CACHE_MAX_AGE_HOURS': AUDIO_CACHE_MAX_AGE_HOURS,
        'AUTO_CLEANUP_ON_EXIT': AUTO_CLEANUP_ON_EXIT,
        'ENABLE_CUDA_ERROR_RECOVERY': ENABLE_CUDA_ERROR_RECOVERY,
        'XTTS_MAX_TEXT_LENGTH': XTTS_MAX_TEXT_LENGTH,
        'XTTS_RETRY_ATTEMPTS': XTTS_RETRY_ATTEMPTS,
        'CUDA_MEMORY_CLEANUP': CUDA_MEMORY_CLEANUP,
        'SAFE_MODE_ON_CUDA_ERROR': SAFE_MODE_ON_CUDA_ERROR,
        'ENABLE_MODEL_REINITIALIZATION': ENABLE_MODEL_REINITIALIZATION,
        'MODEL_RESET_COOLDOWN': MODEL_RESET_COOLDOWN,
        'PROACTIVE_MODEL_RESET_HOURS': PROACTIVE_MODEL_RESET_HOURS
    }

def cleanup_audio_cache(force_all=False):
    """Clean up audio cache based on configuration"""
    import glob
    import time
    
    if not ENABLE_AUDIO_CLEANUP and not force_all:
        return 0
    
    try:
        # Get all audio files in cache
        audio_files = glob.glob(os.path.join(AUDIO_CACHE_DIR, "*.wav"))
        deleted_count = 0
        
        if force_all:
            # Delete all cached audio files
            for audio_file in audio_files:
                try:
                    os.remove(audio_file)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {audio_file}: {e}")
            
            if deleted_count > 0:
                print(f"🗑️ Deleted {deleted_count} cached audio files")
            return deleted_count
        
        # Sort by modification time (oldest first)
        audio_files.sort(key=lambda x: os.path.getmtime(x))
        current_time = time.time()
        
        # Delete files older than max age
        for audio_file in audio_files:
            file_age_hours = (current_time - os.path.getmtime(audio_file)) / 3600
            if file_age_hours > AUDIO_CACHE_MAX_AGE_HOURS:
                try:
                    os.remove(audio_file)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {audio_file}: {e}")
        
        # Delete excess files if over max count
        remaining_files = [f for f in audio_files if os.path.exists(f)]
        if len(remaining_files) > AUDIO_CACHE_MAX_FILES:
            excess_files = remaining_files[:-AUDIO_CACHE_MAX_FILES]
            for audio_file in excess_files:
                try:
                    os.remove(audio_file)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {audio_file}: {e}")
        
        if deleted_count > 0:
            print(f"🗑️ Cleaned up {deleted_count} old audio files from cache")
        
        return deleted_count
        
    except Exception as e:
        print(f"❌ Audio cache cleanup error: {e}")
        return 0

def get_cache_status():
    """Get current cache status"""
    import glob
    
    try:
        audio_files = glob.glob(os.path.join(AUDIO_CACHE_DIR, "*.wav"))
        total_size = sum(os.path.getsize(f) for f in audio_files if os.path.exists(f))
        
        return {
            'audio_files_count': len(audio_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': AUDIO_CACHE_DIR,
            'cleanup_enabled': ENABLE_AUDIO_CLEANUP,
            'max_files': AUDIO_CACHE_MAX_FILES,
            'max_age_hours': AUDIO_CACHE_MAX_AGE_HOURS
        }
    except Exception as e:
        print(f"❌ Cache status error: {e}")
        return {}

def save_config():
    """Save current configuration to file"""
    config = get_config()
    config_path = os.path.join(DEFAULT_OUTPUT_DIR, "s2t2ss_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Configuration saved to {config_path}")
    return config_path

def load_config(config_path=None):
    """Load configuration from file"""
    global ENABLE_LLM_EDITING, LLM_SERVER_URL, LLM_MODEL_NAME, ENABLE_OBS_CAPTIONS, ENABLE_TTS
    global NARRATOR_MODE, OUTPUT_LANGUAGE, TTS_VOICE, SIMILARITY_THRESHOLD, DEFAULT_OUTPUT_DIR
    global ENABLE_CUDA_ERROR_RECOVERY, XTTS_MAX_TEXT_LENGTH, XTTS_RETRY_ATTEMPTS
    global CUDA_MEMORY_CLEANUP, SAFE_MODE_ON_CUDA_ERROR
    global ENABLE_MODEL_REINITIALIZATION, MODEL_RESET_COOLDOWN, PROACTIVE_MODEL_RESET_HOURS
    global ENABLE_MODEL_REINITIALIZATION, MODEL_RESET_COOLDOWN, PROACTIVE_MODEL_RESET_HOURS
    
    if config_path is None:
        config_path = os.path.join(DEFAULT_OUTPUT_DIR, "s2t2ss_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        ENABLE_LLM_EDITING = config.get('ENABLE_LLM_EDITING', True)
        LLM_SERVER_URL = config.get('LLM_SERVER_URL', "http://localhost:1234")
        LLM_MODEL_NAME = config.get('LLM_MODEL_NAME', "L32-21B:latest")
        ENABLE_OBS_CAPTIONS = config.get('ENABLE_OBS_CAPTIONS', True)
        ENABLE_TTS = config.get('ENABLE_TTS', True)
        NARRATOR_MODE = config.get('NARRATOR_MODE', "direct")
        OUTPUT_LANGUAGE = config.get('OUTPUT_LANGUAGE', "auto")
        TTS_VOICE = config.get('TTS_VOICE', "default")
        SIMILARITY_THRESHOLD = config.get('SIMILARITY_THRESHOLD', 0.8)
        DEFAULT_OUTPUT_DIR = config.get('DEFAULT_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), "data"))
        
        # Load CUDA error handling settings
        ENABLE_CUDA_ERROR_RECOVERY = config.get('ENABLE_CUDA_ERROR_RECOVERY', True)
        XTTS_MAX_TEXT_LENGTH = config.get('XTTS_MAX_TEXT_LENGTH', 400)
        XTTS_RETRY_ATTEMPTS = config.get('XTTS_RETRY_ATTEMPTS', 2)
        CUDA_MEMORY_CLEANUP = config.get('CUDA_MEMORY_CLEANUP', True)
        SAFE_MODE_ON_CUDA_ERROR = config.get('SAFE_MODE_ON_CUDA_ERROR', True)
        
        # Load model reinitialization settings
        ENABLE_MODEL_REINITIALIZATION = config.get('ENABLE_MODEL_REINITIALIZATION', True)
        MODEL_RESET_COOLDOWN = config.get('MODEL_RESET_COOLDOWN', 30)
        PROACTIVE_MODEL_RESET_HOURS = config.get('PROACTIVE_MODEL_RESET_HOURS', 1)
        
        # Voice conversion functionality removed
        
        print(f"Configuration loaded from {config_path}")
        return True
    else:
        print(f"⚠️ Config file not found: {config_path}")
        return False

if __name__ == "__main__":
    print("🎛️ S2T2SS Configuration Module Test")
    print("Current config:", get_config())
    save_config()
