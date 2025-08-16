#!/usr/bin/env python3
"""
🔊 S2T2SS TTS Manager Module
Handles text-to-speech synthesis and audio output
"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import pyaudio
import time
import threading
from typing import Optional, Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

class TTSManager:
    """
    TTS Manager for text-to-speech synthesis
    Handles audio generation and playback
    """
    
    def __init__(self, caption_manager=None):
        """Initialize TTS Manager with XTTS v2 (Coqui TTS) support"""
        print("Initializing TTS Manager...")
        
        # Configuration
        self.sample_rate = 24000  # XTTS v2 uses 24kHz
        self.output_dir = config.DEFAULT_OUTPUT_DIR
        
        # Initialize TTS models
        self.tts_model = None
        self.xtts_model = None
        self.model_failed_count = 0
        self.last_model_reset = time.time()  # Initialize to current time to prevent immediate reset
        self.model_reset_cooldown = getattr(config, 'MODEL_RESET_COOLDOWN', 30)  # Use config setting
        
        # Caption manager for clearing timer integration
        self.caption_manager = caption_manager
        
        # Audio output initialization with streaming support
        self.audio_output = None
        self.output_device_id = None
        self.audio_queue = []              # Queue for concurrent audio streams
        self.active_streams = []           # Track active audio streams
        self.stream_lock = threading.Lock()  # Thread safety for audio queue
        
        # CUDA Error Prevention - Force Sequential TTS Processing
        self.tts_processing_lock = threading.Lock()  # CRITICAL: Prevent concurrent XTTS calls
        self.force_cpu_mode = False        # Fallback to CPU when CUDA fails repeatedly
        self.cuda_error_count = 0          # Track consecutive CUDA errors
        self.max_cuda_errors = 3           # Switch to CPU after this many errors
        
        # Voice output sequencing (prevents overlapping voices)
        self.voice_queue = []              # Queue for linear voice output
        self.voice_queue_lock = threading.Lock()
        self.last_voice_time = 0           # Track last voice output time
        self.voice_scheduler_running = False
        
        # VB-Cable detection for output routing will be done when needed
        # (in _initialize_audio_output method)
        
        # Initialize with XTTS v2 (your original implementation)
        try:
            print("Initializing XTTS v2 (Coqui TTS)...")
            from TTS.api import TTS as CoquiTTS
            import torch
            
            # Load XTTS v2 model (multilingual)
            self.xtts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            
            # Apply optimizations from your original code
            if hasattr(self.xtts_model, 'synthesizer') and hasattr(self.xtts_model.synthesizer, 'tts_model'):
                model = self.xtts_model.synthesizer.tts_model
                if hasattr(model, 'gpt'):
                    # Reduce GPT conditioning length for faster processing
                    model.gpt.gpt_cond_len = 12  # Reduced from default 30
                    print("✅ Applied GPT optimization (reduced conditioning length)")
                
                # Clear GPU cache after optimization
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.tts_model = "xtts_v2"
            print("✅ XTTS v2 initialized successfully with optimizations")
            
            # Set up voice reference file
            self._setup_voice_reference()
            
            # Update last reset time to current time after successful initialization
            self.last_model_reset = time.time()
            
        except ImportError as e:
            print(f"⚠️ XTTS (Coqui TTS) not available: {e}")
            print("⚠️ Using placeholder TTS (install TTS>=0.22.0 for real synthesis)")
            self.tts_model = "placeholder"
        except Exception as e:
            print(f"⚠️ XTTS initialization failed: {e}")
            print("⚠️ Using placeholder TTS")
            self.tts_model = "placeholder"
        
        print(f"✅ TTS Manager initialized (model: {self.tts_model})")
        
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device for TTS"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_models(self):
        """Initialize TTS models with XTTS v2 (your original implementation)"""
        try:
            print("🔊 Loading XTTS v2 models...")
            
            # Initialize XTTS v2 from Coqui TTS (your original working implementation)
            from TTS.api import TTS as CoquiTTS
            import re
            
            # Load XTTS v2 model (multilingual support)
            use_gpu = torch.cuda.is_available()
            self.xtts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
            
            # Apply optimizations from your original code for faster synthesis
            if hasattr(self.xtts_model, 'synthesizer') and hasattr(self.xtts_model.synthesizer, 'tts_model'):
                model = self.xtts_model.synthesizer.tts_model
                if hasattr(model, 'gpt'):
                    # Reduce GPT conditioning length for faster processing
                    model.gpt.gpt_cond_len = 12  # Reduced from default 30
                    print("✅ Applied GPT optimization (reduced conditioning length)")
                
                # Clear cache after optimization
                if use_gpu:
                    torch.cuda.empty_cache()
            
            # Set up voice reference file (from your original implementation)
            self._setup_voice_reference()
            
            self.tts_model = "xtts_v2"
            self.sample_rate = 24000  # XTTS v2 sample rate
            
            print(f"✅ XTTS v2 Models loaded successfully on {self.device}")
            
        except ImportError as e:
            print(f"⚠️ XTTS (Coqui TTS) not available: {e}")
            print("📥 Install with: pip install TTS>=0.22.0")
            self._fallback_to_placeholder()
        except Exception as e:
            print(f"⚠️ XTTS Model loading failed: {e}")
            self._fallback_to_placeholder()
    
    def _setup_voice_reference(self):
        """Set up voice reference file for XTTS cloning"""
        try:
            # Look for voice reference files (from your original implementation)
            voice_files = [
                "core/data/voice_samples/sample.wav", 
                "core/data/voice_samples/sample02.wav"
            ]
            
            self.voice_reference = None
            for voice_file in voice_files:
                # Check in current directory and data directory
                for check_path in [voice_file, os.path.join("..", voice_file), os.path.join(config.DEFAULT_OUTPUT_DIR, voice_file)]:
                    if os.path.exists(check_path):
                        self.voice_reference = check_path
                        print(f"✅ Using voice reference: {os.path.basename(check_path)}")
                        return
            
            print("⚠️ No voice reference file found. XTTS will use default voice.")
            print("💡 Add a voice sample file (sample.wav, sample02.wav, etc.) for voice cloning")
            self.voice_reference = None
            
        except Exception as e:
            print(f"⚠️ Voice reference setup failed: {e}")
            self.voice_reference = None
    
    def _fallback_to_placeholder(self):
        """Fallback to placeholder TTS for testing"""
        print("🔄 Falling back to placeholder TTS...")
        self.tts_model = "placeholder"
        self.xtts_model = None
    
    def _initialize_audio_output(self):
        """Initialize audio output system with VB-Cable preference"""
        try:
            if not self.audio_output:
                self.audio_output = pyaudio.PyAudio()
                
                # Try to find VB-Cable output device for TTS
                self.output_device_id = self._find_vb_cable_output()
                if self.output_device_id is not None:
                    print(f"✅ Audio output initialized with VB-Cable device {self.output_device_id}")
                else:
                    print("✅ Audio output initialized with system default")
        except Exception as e:
            print(f"⚠️ Audio output initialization failed: {e}")
    
    def _find_vb_cable_output(self):
        """Find VB-Cable output device for TTS audio"""
        try:
            if not self.audio_output:
                return None
            
            # Look for VB-Cable output devices
            for i in range(self.audio_output.get_device_count()):
                device_info = self.audio_output.get_device_info_by_index(i)
                device_name = device_info['name'].lower()
                
                # Check if it's a VB-Cable output device
                if (('vb-audio' in device_name or 'cable' in device_name) and 
                    device_info['maxOutputChannels'] > 0):
                    print(f"🔊 Found VB-Cable output: {i} - {device_info['name']}")
                    return i
            
            print(f"🔊 No VB-Cable output found, will use system default")
            return None
            
        except Exception as e:
            print(f"⚠️ VB-Cable detection error: {e}")
            return None
    
    def synthesize_text(self, text: str, voice_id: str = "default") -> Optional[np.ndarray]:
        """
        Synthesize text to speech using XTTS v2 with CUDA error prevention
        
        Args:
            text: Text to synthesize
            voice_id: Voice identifier to use
            
        Returns:
            Audio data as numpy array or None if failed
        """
        
        # Check if TTS is enabled
        if not config.ENABLE_TTS:
            print(f"🔊 TTS Synthesis: DISABLED (text-only mode)")
            return None
        
        # Skip empty text
        if not text.strip():
            return None
        
        try:
            # Split long text to prevent CUDA indexing errors
            text_chunks = self._split_long_text(text)
            
            if len(text_chunks) == 1:
                # Single chunk processing
                chunk_text = text_chunks[0]
                print(f"🔊 Synthesizing with XTTS: '{chunk_text[:50]}'")
                
                # Use XTTS v2 if available
                if self.tts_model == "xtts_v2" and self.xtts_model is not None:
                    return self._synthesize_with_xtts(chunk_text)
                else:
                    print(f"⚠️ XTTS not available (model: {self.tts_model}, xtts_obj: {'Loaded' if self.xtts_model else 'None'}), using placeholder")
                    return self._fallback_tts(chunk_text)
            else:
                # Multi-chunk processing for long texts
                print(f"🔊 Synthesizing {len(text_chunks)} chunks with XTTS")
                all_audio = []
                
                for i, chunk_text in enumerate(text_chunks):
                    print(f"🔊 Processing chunk {i+1}/{len(text_chunks)}: '{chunk_text[:30]}...'")
                    
                    if self.tts_model == "xtts_v2" and self.xtts_model is not None:
                        chunk_audio = self._synthesize_with_xtts(chunk_text)
                    else:
                        chunk_audio = self._fallback_tts(chunk_text)
                    
                    if chunk_audio is not None:
                        all_audio.append(chunk_audio)
                        # Add brief pause between chunks
                        pause = np.zeros(int(self.sample_rate * 0.2))  # 200ms pause
                        all_audio.append(pause.astype(np.float32))
                    else:
                        print(f"⚠️ Chunk {i+1} synthesis failed, skipping")
                
                if all_audio:
                    # Concatenate all chunks
                    final_audio = np.concatenate(all_audio)
                    print(f"✅ Multi-chunk synthesis completed: {len(final_audio)/self.sample_rate:.1f}s")
                    return final_audio
                else:
                    print("❌ All chunks failed, using fallback")
                    return self._fallback_tts(text)
                
        except Exception as e:
            print(f"❌ TTS Synthesis Error: {e}")
            return self._fallback_tts(text)
    
    def _split_long_text(self, text: str) -> List[str]:
        """Split long text into smaller chunks to prevent CUDA indexing errors"""
        
        max_length = getattr(config, 'XTTS_MAX_TEXT_LENGTH', 400)
        
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        
        # Try to split at sentence boundaries first
        import re
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add punctuation back
            if sentence == sentences[-1]:
                sentence_with_punct = sentence
            else:
                sentence_with_punct = sentence + "."
            
            # Check if adding this sentence would exceed the limit
            if len(current_chunk + " " + sentence_with_punct) > max_length:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                current_chunk = sentence_with_punct
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence_with_punct
                else:
                    current_chunk = sentence_with_punct
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If sentence splitting still produces chunks that are too long, split by words
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split by words
                words = chunk.split()
                current_word_chunk = ""
                
                for word in words:
                    if len(current_word_chunk + " " + word) > max_length:
                        if current_word_chunk.strip():
                            final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word
                    else:
                        if current_word_chunk:
                            current_word_chunk += " " + word
                        else:
                            current_word_chunk = word
                
                if current_word_chunk.strip():
                    final_chunks.append(current_word_chunk.strip())
        
        print(f"📝 Split text into {len(final_chunks)} chunks (max length: {max_length})")
        return final_chunks
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text (from your original implementation)"""
        # Use config setting if specified and not auto
        if hasattr(config, 'OUTPUT_LANGUAGE') and config.OUTPUT_LANGUAGE != "auto":
            output_lang = config.OUTPUT_LANGUAGE
            
            # Map config language codes to XTTS language codes
            lang_mapping = {
                "zh": "zh-cn",
                "en": "en", 
                "ja": "ja",
                "ko": "ko",
                "es": "es",
                "fr": "fr",
                "de": "de",
                "ru": "ru"
            }
            
            mapped_lang = lang_mapping.get(output_lang, output_lang)
            print(f"🌐 Using configured output language: {mapped_lang}")
            return mapped_lang
        
        # Auto-detect based on character sets only if OUTPUT_LANGUAGE is "auto"
        if any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in text):
            detected_lang = "zh-cn"  # Chinese
        elif any(ord(char) >= 0x3040 and ord(char) <= 0x309f for char in text) or \
             any(ord(char) >= 0x30a0 and ord(char) <= 0x30ff for char in text):
            detected_lang = "ja"     # Japanese
        elif any(ord(char) >= 0xac00 and ord(char) <= 0xd7af for char in text):
            detected_lang = "ko"     # Korean
        else:
            detected_lang = "en"     # Default to English
            
        print(f"🔍 Auto-detected language: {detected_lang}")
        return detected_lang
    
    def _split_by_language(self, text: str):
        """Split text by language segments (simplified version of your original)"""
        import re
        
        # Clean up text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Simple approach: detect predominant language and use that
        lang = self._detect_language(text)
        return [(cleaned_text, lang)]
    
    def _should_reinitialize_model(self) -> bool:
        """Check if model should be proactively reinitialized"""
        
        # Check if model reinitialization is enabled
        if not getattr(config, 'ENABLE_MODEL_REINITIALIZATION', True):
            return False
        
        # Reinitialize after too many failures
        if self.model_failed_count >= 5:
            print(f"🚨 Model failure threshold reached ({self.model_failed_count} failures)")
            return True
        
        # Reinitialize periodically as preventive maintenance
        proactive_hours = getattr(config, 'PROACTIVE_MODEL_RESET_HOURS', 1)
        if proactive_hours > 0:
            current_time = time.time()
            time_since_reset = current_time - self.last_model_reset
            
            # Only trigger if enough time has passed AND model has been initialized
            if time_since_reset > (proactive_hours * 3600) and self.xtts_model is not None:
                print(f"⏰ Proactive reset scheduled ({time_since_reset/3600:.1f} hours since last reset)")
                return True
        
        return False
    
    def _synthesize_with_xtts(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using XTTS v2 with CUDA error handling and input validation"""
        try:
            if self.xtts_model is None:
                print("❌ XTTS model not loaded")
                return None
            
            # Check if model should be proactively reinitialized
            if self._should_reinitialize_model():
                print("🔄 Proactive model reinitialization triggered")
                if not self._reinitialize_xtts_model():
                    print("⚠️ Proactive reinitialization failed, continuing with current model")
            
            # Input validation to prevent CUDA indexing errors
            if not self._validate_xtts_input(text):
                print(f"⚠️ Invalid input for XTTS, using fallback: '{text[:30]}...'")
                return self._fallback_tts(text)
            
            # Clear CUDA cache before synthesis to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Split text by language segments
            segments = self._split_by_language(text)
            if not segments:
                return None
            
            # For single language synthesis (most common case)
            if len(segments) == 1:
                segment_text, lang = segments[0]
                
                # Special handling for Japanese TTS errors (from your original code)
                if lang == 'ja':
                    try:
                        # Test Japanese TTS capability first
                        test_wav = self.xtts_model.tts(
                            text="テスト", 
                            language=lang, 
                            speaker_wav=self.voice_reference,
                            temperature=0.65,
                            length_penalty=1.2,
                            repetition_penalty=5.0,
                            top_k=50,
                            top_p=0.85
                        )
                    except Exception as jp_test_error:
                        print(f"⚠️ Japanese TTS failed, falling back to English: {jp_test_error}")
                        lang = 'en'  # Fall back to English
                
                # CUDA-safe XTTS synthesis with error recovery
                try:
                    wav = self._safe_xtts_synthesis(segment_text, lang)
                    if wav is None:
                        print(f"⚠️ XTTS synthesis failed, using fallback for: '{segment_text[:30]}...'")
                        return self._fallback_tts(text)
                    
                    # Convert to numpy array
                    final_audio = np.array(wav, dtype=np.float32) if isinstance(wav, list) else wav
                    
                    print(f"✅ XTTS synthesized {len(final_audio)/self.sample_rate:.1f}s of speech ({lang})")
                    return final_audio
                
                except RuntimeError as cuda_error:
                    if "CUDA" in str(cuda_error) or "device-side assert" in str(cuda_error):
                        print(f"🛡️ CUDA error detected, recovering: {cuda_error}")
                        return self._recover_from_cuda_error(text)
                    else:
                        raise cuda_error
            
            else:
                # Multi-language synthesis (concatenate segments)
                all_audio = []
                for segment_text, lang in segments:
                    try:
                        wav = self._safe_xtts_synthesis(segment_text, lang)
                        if wav is None:
                            print(f"⚠️ Failed to synthesize segment '{segment_text[:30]}...' ({lang}), skipping")
                            continue
                        
                        audio_segment = np.array(wav, dtype=np.float32) if isinstance(wav, list) else wav
                        all_audio.append(audio_segment)
                        
                    except RuntimeError as cuda_error:
                        if "CUDA" in str(cuda_error) or "device-side assert" in str(cuda_error):
                            print(f"🛡️ CUDA error in segment synthesis, skipping: {cuda_error}")
                            continue
                        else:
                            raise cuda_error
                    except Exception as e:
                        print(f"⚠️ Failed to synthesize segment '{segment_text[:30]}...' ({lang}): {e}")
                        continue
                
                if all_audio:
                    # Concatenate all segments
                    final_audio = np.concatenate(all_audio)
                    print(f"✅ XTTS synthesized {len(final_audio)/self.sample_rate:.1f}s of multi-language speech")
                    return final_audio
                else:
                    print("❌ All synthesis segments failed")
                    return self._fallback_tts(text)
                
        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error) or "device-side assert" in str(cuda_error):
                print(f"🛡️ CUDA error in XTTS synthesis: {cuda_error}")
                return self._recover_from_cuda_error(text)
            else:
                print(f"❌ XTTS runtime error: {cuda_error}")
                return None
        except Exception as e:
            print(f"❌ XTTS synthesis error: {e}")
            return None
    
    def _validate_xtts_input(self, text: str) -> bool:
        """Validate input text to prevent CUDA indexing errors"""
        try:
            # Check if model is in a bad state
            if self.model_failed_count >= 5:
                print(f"⚠️ Model in bad state (failures: {self.model_failed_count}), rejecting input")
                return False
            
            # Check text length against config setting
            max_length = getattr(config, 'XTTS_MAX_TEXT_LENGTH', 400)
            if len(text) > max_length:
                print(f"⚠️ Text too long ({len(text)} chars, max: {max_length}), may cause CUDA errors")
                return False
            
            # Check for empty or whitespace-only text
            if not text.strip():
                return False
            
            # Check for problematic characters that can cause tensor shape issues
            # Remove or replace characters that might cause indexing problems
            import re
            
            # Check for excessive punctuation that might create empty tokens
            if re.search(r'[.!?]{5,}', text):
                print("⚠️ Excessive punctuation detected, may cause tensor issues")
                return False
            
            # Check for very long words that might cause tokenization issues
            words = text.split()
            for word in words:
                if len(word) > 50:  # Very long words can cause problems
                    print(f"⚠️ Very long word detected: '{word[:20]}...', may cause issues")
                    return False
            
            # Check for unusual character patterns
            if len(re.sub(r'[a-zA-Z0-9\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff.,!?，。？！：；""''（）【】《》]', '', text)) > len(text) * 0.3:
                print("⚠️ High ratio of special characters, may cause tensor issues")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Input validation error: {e}")
            return False
    
    def _safe_xtts_synthesis(self, text: str, lang: str, max_retries: int = None):
        """Perform XTTS synthesis with CUDA error handling and retries"""
        
        # CRITICAL: Use lock to prevent concurrent XTTS calls that cause CUDA indexing errors
        with self.tts_processing_lock:
            if max_retries is None:
                max_retries = getattr(config, 'XTTS_RETRY_ATTEMPTS', 2)
            
            # Check if we should force CPU mode due to repeated CUDA failures
            if self.force_cpu_mode:
                print("🔄 Using CPU mode due to repeated CUDA errors")
                return self._cpu_fallback_synthesis(text, lang)
            
            for attempt in range(max_retries + 1):
                try:
                    # Clean up GPU memory before each attempt if enabled
                    if getattr(config, 'CUDA_MEMORY_CLEANUP', True) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for all operations to complete
                    
                    # XTTS synthesis with optimized parameters for fast mode
                    synthesis_params = {
                        "text": text, 
                        "language": lang, 
                        "speaker_wav": self.voice_reference,
                    }
                    
                    # Apply fast mode optimizations if enabled
                    if hasattr(config, 'FAST_MODE') and config.FAST_MODE:
                        synthesis_params.update({
                            "temperature": 0.5,      # Even lower for faster synthesis
                            "length_penalty": 1.0,   # Faster speech rhythm
                            "repetition_penalty": 3.0, # Prevent repetitions efficiently
                            "top_k": 30,            # Reduced for faster sampling
                            "top_p": 0.75           # Reduced for faster sampling
                        })
                    else:
                        synthesis_params.update({
                            "temperature": 0.65,     # Standard optimization
                            "length_penalty": 1.2,   
                            "repetition_penalty": 5.0, 
                            "top_k": 50,            
                            "top_p": 0.85           
                        })
                    
                    # Perform synthesis
                    wav = self.xtts_model.tts(**synthesis_params)
                    
                    # Validate output
                    if wav is None or (isinstance(wav, (list, np.ndarray)) and len(wav) == 0):
                        if attempt < max_retries:
                            print(f"⚠️ Empty output from XTTS, retrying ({attempt + 1}/{max_retries})")
                            continue
                        else:
                            print("❌ XTTS produced empty output after retries")
                            return None
                    
                    # Success - reset error count
                    self.cuda_error_count = 0
                    return wav
                    
                except RuntimeError as e:
                    if "CUDA" in str(e) or "device-side assert" in str(e) or "index" in str(e).lower():
                        print(f"🛡️ CUDA/indexing error in synthesis attempt {attempt + 1}: {e}")
                        
                        # Increment CUDA error count
                        self.cuda_error_count += 1
                        
                        # Check if we should switch to CPU mode
                        if self.cuda_error_count >= self.max_cuda_errors:
                            print(f"⚠️ Too many CUDA errors ({self.cuda_error_count}), switching to CPU mode")
                            self.force_cpu_mode = True
                            return self._cpu_fallback_synthesis(text, lang)
                        
                        if attempt < max_retries and getattr(config, 'ENABLE_CUDA_ERROR_RECOVERY', True):
                            # Try to recover for next attempt
                            self._cleanup_cuda_state()
                            print(f"🔄 Retrying synthesis ({attempt + 2}/{max_retries + 1})")
                            continue
                        else:
                            print("❌ All synthesis attempts failed with CUDA errors")
                            return None
                    else:
                        # Non-CUDA runtime error, re-raise
                        raise e
                except Exception as e:
                    print(f"❌ Unexpected error in synthesis attempt {attempt + 1}: {e}")
                    if attempt >= max_retries:
                        return None
                    continue
            
            return None
    
    def _cpu_fallback_synthesis(self, text: str, lang: str):
        """CPU-only fallback synthesis when CUDA fails"""
        try:
            print("💻 Using CPU fallback for TTS synthesis")
            
            # Move model to CPU if it's on GPU
            if hasattr(self.xtts_model, 'device') and 'cuda' in str(self.xtts_model.device):
                print("🔄 Moving XTTS model to CPU...")
                self.xtts_model = self.xtts_model.cpu()
                torch.cuda.empty_cache()
            
            # Simplified synthesis parameters for CPU
            synthesis_params = {
                "text": text, 
                "language": lang, 
                "speaker_wav": self.voice_reference,
                "temperature": 0.7,
                "length_penalty": 1.0,
                "repetition_penalty": 2.0
            }
            
            # Perform CPU synthesis
            wav = self.xtts_model.tts(**synthesis_params)
            
            if wav is not None and len(wav) > 0:
                print("✅ CPU fallback synthesis successful")
                return wav
            else:
                print("❌ CPU fallback synthesis failed")
                return None
                
        except Exception as e:
            print(f"❌ CPU fallback synthesis error: {e}")
            return None
    
    def _cleanup_cuda_state(self):
        """Clean up CUDA state after errors"""
        try:
            if torch.cuda.is_available():
                # Clear all CUDA caches
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()
                
                print("🧹 CUDA state cleaned up")
                
        except Exception as e:
            print(f"⚠️ CUDA cleanup warning: {e}")
    
    def _reinitialize_xtts_model(self) -> bool:
        """Reinitialize XTTS model after CUDA errors"""
        try:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_model_reset < self.model_reset_cooldown:
                print(f"⏳ Model reset on cooldown ({self.model_reset_cooldown - (current_time - self.last_model_reset):.1f}s remaining)")
                return False
            
            print("🔄 Reinitializing XTTS model after CUDA errors...")
            
            # Clean up current model
            if self.xtts_model is not None:
                try:
                    # Try to explicitly delete the model
                    del self.xtts_model
                    self.xtts_model = None
                except:
                    pass
            
            # Reset TTS model flag to prevent falling back to placeholder during reinitialization
            self.tts_model = None
            
            # Clean up CUDA state thoroughly
            self._cleanup_cuda_state()
            
            # Wait a moment for cleanup
            time.sleep(2)
            
            # Reinitialize XTTS
            print("🔄 Loading fresh XTTS v2 model...")
            from TTS.api import TTS as CoquiTTS
            
            # Load XTTS v2 model again
            self.xtts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            
            # Apply optimizations again
            if hasattr(self.xtts_model, 'synthesizer') and hasattr(self.xtts_model.synthesizer, 'tts_model'):
                model = self.xtts_model.synthesizer.tts_model
                if hasattr(model, 'gpt'):
                    model.gpt.gpt_cond_len = 12
                    print("✅ Applied GPT optimization to reinitialized model")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Set up voice reference again
            self._setup_voice_reference()
            print(f"✅ Voice reference re-established: {self.voice_reference}")
            
            # Reset TTS model flag to indicate XTTS is available
            self.tts_model = "xtts_v2"
            
            # Test the reinitialized model with a simple synthesis
            try:
                print("🧪 Testing reinitialized model...")
                test_result = self._safe_xtts_synthesis("Test", "en", max_retries=1)
                if test_result is not None:
                    print("✅ Reinitialized model test passed")
                else:
                    print("⚠️ Reinitialized model test failed, but continuing")
            except Exception as test_error:
                print(f"⚠️ Reinitialized model test error: {test_error}")
            
            # Reset counters
            self.model_failed_count = 0
            self.last_model_reset = current_time
            
            print("✅ XTTS model reinitialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Model reinitialization failed: {e}")
            import traceback
            print(f"📝 Reinitialization error details: {traceback.format_exc()}")
            
            # Ensure we fall back to placeholder mode
            self.xtts_model = None
            self.tts_model = "placeholder"
            return False
    
    def _recover_from_cuda_error(self, original_text: str) -> Optional[np.ndarray]:
        """Recover from CUDA errors with fallback strategies"""
        try:
            # Check if safe mode is enabled
            if not getattr(config, 'SAFE_MODE_ON_CUDA_ERROR', True):
                print("🛡️ Safe mode disabled, skipping CUDA error recovery")
                return None
            
            print("🛡️ Attempting CUDA error recovery...")
            
            # Increment failure count
            self.model_failed_count += 1
            print(f"📊 Model failure count: {self.model_failed_count}")
            
            # Strategy 1: Clean up CUDA state and try with shorter text
            self._cleanup_cuda_state()
            
            # Shorten text if it's long
            max_length = getattr(config, 'XTTS_MAX_TEXT_LENGTH', 400)
            if len(original_text) > max_length // 2:  # Try with half the max length
                # Don't add ellipses - they cause TTS pronunciation issues
                shortened_text = original_text[:max_length // 2].rstrip('.,!?;:')
                # Add appropriate ending punctuation for the detected language
                is_chinese = any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in shortened_text)
                if is_chinese:
                    shortened_text += '。'  # Chinese period
                else:
                    shortened_text += '.'   # English period
                    
                print(f"🔄 Trying with shortened text: '{shortened_text}'")
                
                try:
                    result = self._safe_xtts_synthesis(shortened_text, "en", max_retries=1)
                    if result is not None:
                        print("✅ Recovery successful with shortened text")
                        return result
                except:
                    pass
            
            # Strategy 2: Try with simple fallback text
            fallback_texts = [
                "Audio synthesis completed.",
                "Text processed.",
                "Audio ready."
            ]
            
            for fallback_text in fallback_texts:
                try:
                    print(f"🔄 Trying fallback text: '{fallback_text}'")
                    result = self._safe_xtts_synthesis(fallback_text, "en", max_retries=1)
                    if result is not None:
                        print("✅ Recovery successful with fallback text")
                        return result
                except:
                    continue
            
            # Strategy 3: Try model reinitialization if multiple failures
            if self.model_failed_count >= 3:
                print("🔄 Multiple CUDA failures detected, attempting model reinitialization...")
                if self._reinitialize_xtts_model():
                    # Try synthesis one more time with reinitialized model
                    try:
                        result = self._safe_xtts_synthesis("Test synthesis after reinitialization", "en", max_retries=1)
                        if result is not None:
                            print("✅ Model reinitialization successful, trying original text...")
                            return self._safe_xtts_synthesis(original_text[:200], "en", max_retries=1)
                    except Exception as reinit_error:
                        print(f"⚠️ Reinitialized model still failing: {reinit_error}")
            
            # Strategy 4: Use non-XTTS fallback
            print("🔄 All XTTS recovery failed, using audio fallback")
            return self._fallback_tts(original_text)
            
        except Exception as e:
            print(f"❌ Error recovery failed: {e}")
            return self._fallback_tts(original_text)
    
    def _select_voice(self, text: str) -> str:
        """Select appropriate voice based on text language and config"""
        try:
            # Detect language or use config
            if hasattr(config, 'OUTPUT_LANGUAGE') and config.OUTPUT_LANGUAGE != "auto":
                lang = config.OUTPUT_LANGUAGE
            else:
                # Simple language detection based on character sets
                if any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in text):
                    lang = "zh"
                elif any(ord(char) >= 0x3040 and ord(char) <= 0x309f for char in text):
                    lang = "ja"
                elif any(ord(char) >= 0xac00 and ord(char) <= 0xd7af for char in text):
                    lang = "ko"
                else:
                    lang = "en"
            
            # Voice selection based on language and TTS_VOICE config
            voice_preference = getattr(config, 'TTS_VOICE', 'default')
            
            if lang == "zh":
                if voice_preference == "female":
                    return "zh-CN-XiaoxiaoNeural"  # Female Chinese
                elif voice_preference == "male":
                    return "zh-CN-YunxiNeural"     # Male Chinese
                else:
                    return "zh-CN-XiaoxiaoNeural"  # Default female Chinese
            elif lang == "ja":
                if voice_preference == "female":
                    return "ja-JP-NanamiNeural"    # Female Japanese
                elif voice_preference == "male":
                    return "ja-JP-KeitaNeural"     # Male Japanese
                else:
                    return "ja-JP-NanamiNeural"    # Default female Japanese
            elif lang == "ko":
                if voice_preference == "female":
                    return "ko-KR-SunHiNeural"     # Female Korean
                elif voice_preference == "male":
                    return "ko-KR-InJoonNeural"    # Male Korean
                else:
                    return "ko-KR-SunHiNeural"     # Default female Korean
            else:  # English and others
                if voice_preference == "female":
                    return "en-US-AriaNeural"      # Female English
                elif voice_preference == "male":
                    return "en-US-DavisNeural"     # Male English
                elif voice_preference == "young":
                    return "en-US-JennyNeural"     # Young female
                elif voice_preference == "elder":
                    return "en-US-GuyNeural"       # Mature male
                else:
                    return "en-US-AriaNeural"      # Default female English
                    
        except Exception as e:
            print(f"⚠️ Voice selection error: {e}, using default")
            return "en-US-AriaNeural"
    
    def _fallback_tts(self, text: str) -> Optional[np.ndarray]:
        """Fallback TTS when XTTS is not available"""
        try:
            print("⚠️ XTTS not available - using notification beep")
            print("📥 Install Coqui TTS for real speech: pip install TTS>=0.22.0")
            
            # Generate a simple notification beep instead of speech-like tone
            duration = 0.5  # Short beep
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            
            # Simple notification beep (2 tones)
            beep1 = 0.3 * np.sin(2 * np.pi * 800 * t[:len(t)//2])  # 800Hz
            beep2 = 0.3 * np.sin(2 * np.pi * 600 * t[len(t)//2:])  # 600Hz
            
            audio_data = np.concatenate([beep1, beep2])
            
            print(f"🔔 Notification beep generated (install XTTS for speech synthesis)")
            return audio_data.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Fallback beep failed: {e}")
            return None
    
    def _queue_voice_output(self, audio_data: np.ndarray, text: str, caption_text: str = None) -> bool:
        """Queue voice for linear output with interval control"""
        
        if not config.LINEAR_VOICE_OUTPUT:
            # Fall back to immediate streaming
            return self._streaming_audio_play(audio_data)
        
        # Add to voice queue with caption support
        voice_item = {
            'audio_data': audio_data,
            'text': text[:30] + '...' if len(text) > 30 else text,
            'caption_text': caption_text,  # Caption to display when voice starts
            'timestamp': time.time(),
            'duration': len(audio_data) / self.sample_rate
        }
        
        with self.voice_queue_lock:
            self.voice_queue.append(voice_item)
            queue_position = len(self.voice_queue)
        
        # Start voice scheduler if not running
        if not self.voice_scheduler_running:
            self._start_voice_scheduler()
        
        caption_info = " + caption" if caption_text else ""
        if not config.MINIMAL_LOGGING:
            print(f"🎵 Voice queued (position {queue_position}): '{voice_item['text']}'{caption_info}")
        
        return True
    
    def _start_voice_scheduler(self):
        """Start the voice output scheduler thread"""
        
        def voice_scheduler():
            """Schedule voice outputs with proper intervals"""
            self.voice_scheduler_running = True
            
            try:
                while True:
                    with self.voice_queue_lock:
                        if not self.voice_queue:
                            # No voices in queue, stop scheduler
                            self.voice_scheduler_running = False
                            break
                        
                        voice_item = self.voice_queue.pop(0)
                    
                    # Check if enough time has passed since last voice
                    current_time = time.time()
                    time_since_last = current_time - self.last_voice_time
                    
                    if time_since_last < config.VOICE_OUTPUT_INTERVAL:
                        # Wait for the interval
                        wait_time = config.VOICE_OUTPUT_INTERVAL - time_since_last
                        if not config.MINIMAL_LOGGING:
                            print(f"⏸️ Voice interval: waiting {wait_time:.2f}s")
                        time.sleep(wait_time)
                    
                    # Play the voice
                    if not config.MINIMAL_LOGGING:
                        print(f"🎧 Playing voice: '{voice_item['text']}' ({voice_item['duration']:.1f}s)")
                    
                    # VOICE-SYNCHRONIZED CAPTION: Display caption exactly when voice starts
                    if voice_item.get('caption_text') and config.ENABLE_OBS_CAPTIONS:
                        try:
                            caption_success = self._display_synchronized_caption(voice_item['caption_text'])
                            if caption_success and not config.MINIMAL_LOGGING:
                                print(f"📺 Caption displayed with voice: '{voice_item['caption_text'][:30]}...'")
                        except Exception as e:
                            print(f"❌ Caption sync error: {e}")
                    
                    # Play audio and start clearing timer when voice finishes
                    self._immediate_audio_play(voice_item['audio_data'], voice_item.get('caption_text'))
                    self.last_voice_time = time.time()
                    
                    # Small processing delay
                    time.sleep(0.05)
                    
            except Exception as e:
                if not config.MINIMAL_LOGGING:
                    print(f"❌ Voice scheduler error: {e}")
                self.voice_scheduler_running = False
        
        # Start scheduler in background thread
        scheduler_thread = threading.Thread(target=voice_scheduler, daemon=True)
        scheduler_thread.start()
    
    def _immediate_audio_play(self, audio_data: np.ndarray, caption_text: str = None) -> bool:
        """Play audio immediately without queueing"""
        
        try:
            self._initialize_audio_output()
            
            if not self.audio_output:
                return False
            
            # Convert to 16-bit PCM
            audio_16bit = (audio_data * 32767).astype(np.int16)
            
            # Open stream and play immediately
            stream = self.audio_output.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_id
            )
            
            # Write audio data
            stream.write(audio_16bit.tobytes())
            
            # Schedule cleanup and caption clearing timer
            def cleanup_stream():
                # Wait for audio to finish playing
                audio_duration = len(audio_data) / self.sample_rate
                time.sleep(audio_duration + 0.1)
                
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
                
                # Start caption clearing timer when voice finishes
                if caption_text and self.caption_manager and config.ENABLE_OBS_CAPTIONS:
                    try:
                        self.caption_manager._start_caption_clear_timer()
                        if not config.MINIMAL_LOGGING:
                            print(f"⏰ Caption clearing timer started after voice finished")
                    except Exception as e:
                        if not config.MINIMAL_LOGGING:
                            print(f"❌ Caption timer start error: {e}")
            
            threading.Thread(target=cleanup_stream, daemon=True).start()
            return True
            
        except Exception as e:
            if not config.MINIMAL_LOGGING:
                print(f"❌ Immediate audio play error: {e}")
            return False
    
    def _streaming_audio_play(self, audio_data: np.ndarray) -> bool:
        """Play audio with streaming support - multiple concurrent streams"""
        
        if not config.NON_BLOCKING_AUDIO:
            # Fall back to standard playback
            return self.play_audio(audio_data, blocking=False)
        
        def stream_audio():
            """Stream audio in separate thread with queue management"""
            try:
                with self.stream_lock:
                    # Clean up finished streams
                    self.active_streams = [stream for stream in self.active_streams if stream.is_active()]
                    
                    # Limit concurrent streams
                    if len(self.active_streams) >= config.AUDIO_QUEUE_SIZE:
                        # Stop oldest stream to make room
                        oldest_stream = self.active_streams.pop(0)
                        try:
                            oldest_stream.stop_stream()
                            oldest_stream.close()
                        except:
                            pass
                
                self._initialize_audio_output()
                
                if not self.audio_output:
                    return
                
                # Convert to 16-bit PCM
                audio_16bit = (audio_data * 32767).astype(np.int16)
                
                # Open new stream
                stream = self.audio_output.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.output_device_id
                )
                
                with self.stream_lock:
                    self.active_streams.append(stream)
                
                # Stream audio data
                stream.write(audio_16bit.tobytes())
                
                # Let it play and clean up
                def cleanup_stream():
                    time.sleep(len(audio_data) / self.sample_rate + 0.1)
                    try:
                        stream.stop_stream()
                        stream.close()
                        with self.stream_lock:
                            if stream in self.active_streams:
                                self.active_streams.remove(stream)
                    except:
                        pass
                
                import threading
                threading.Thread(target=cleanup_stream, daemon=True).start()
                
                if not config.MINIMAL_LOGGING:
                    print(f"🎧 Streaming {len(audio_data)/self.sample_rate:.1f}s audio (concurrent)")
                
            except Exception as e:
                if not config.MINIMAL_LOGGING:
                    print(f"❌ Streaming audio error: {e}")
        
        # Start streaming immediately in background
        import threading
        threading.Thread(target=stream_audio, daemon=True).start()
        return True
    
    def play_audio(self, audio_data: np.ndarray, blocking: bool = False) -> bool:
        """
        Play audio data
        
        Args:
            audio_data: Audio data to play
            blocking: Whether to wait for playback to complete
            
        Returns:
            True if playback started successfully
        """
        
        if not config.ENABLE_TTS:
            print("🔊 Audio Playback: DISABLED")
            return False
        
        if audio_data is None or len(audio_data) == 0:
            return False
        
        try:
            self._initialize_audio_output()
            
            if not self.audio_output:
                print("❌ No audio output available")
                return False
            
            # Convert to 16-bit PCM
            audio_16bit = (audio_data * 32767).astype(np.int16)
            
            # Open audio stream (with VB-Cable if available)
            stream = self.audio_output.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_id  # Use VB-Cable if found
            )
            
            # Play audio
            stream.write(audio_16bit.tobytes())
            
            if blocking:
                # Wait for playback to complete
                stream.stop_stream()
                stream.close()
            else:
                # Non-blocking playback
                def cleanup():
                    time.sleep(len(audio_data) / self.sample_rate)
                    stream.stop_stream()
                    stream.close()
                
                import threading
                threading.Thread(target=cleanup, daemon=True).start()
            
            print(f"🔊 Playing {len(audio_data)/self.sample_rate:.1f}s audio")
            return True
            
        except Exception as e:
            print(f"❌ Audio Playback Error: {e}")
            return False
    
    def save_audio(self, audio_data: np.ndarray, filename: str) -> bool:
        """
        Save audio data to cache directory with automatic cleanup
        
        Args:
            audio_data: Audio data to save
            filename: Output filename
            
        Returns:
            True if saved successfully
        """
        
        if audio_data is None or len(audio_data) == 0:
            return False
        
        try:
            # Ensure cache directory exists
            os.makedirs(config.AUDIO_CACHE_DIR, exist_ok=True)
            
            # Build full path to cache directory
            if not os.path.isabs(filename):
                filepath = os.path.join(config.AUDIO_CACHE_DIR, filename)
            else:
                filepath = filename
            
            # Save audio file to cache
            sf.write(filepath, audio_data, self.sample_rate)
            
            print(f"💾 Audio cached: {os.path.relpath(filepath)}")
            
            # Trigger cache cleanup if enabled
            if config.ENABLE_AUDIO_CLEANUP:
                config.cleanup_audio_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ Audio Save Error: {e}")
            return False
    
    def text_to_speech_pipeline(self, text: str, save_file: bool = True, play_audio: bool = True, 
                               caption_text: str = None, sync_with_audio: bool = False) -> bool:
        """
        Streaming TTS pipeline with immediate audio start and no waiting
        
        Args:
            text: Text to convert to speech
            save_file: Whether to save audio file
            play_audio: Whether to play audio
            caption_text: Text to display as caption (if different from TTS text)
            sync_with_audio: If True, captions appear when audio starts playing
            
        Returns:
            True if pipeline started successfully (not completion)
        """
        
        if not config.ENABLE_TTS:
            if not config.MINIMAL_LOGGING:
                print(f"🔊 TTS Pipeline: DISABLED - Text only: '{text}'")
            
            # In text-only mode, we might still want to save the text
            if save_file:
                self.save_text_output(text)
            
            return False
        
        # STREAMING MODE: Start audio generation in background immediately
        if config.ENABLE_STREAMING_TTS:
            return self._streaming_tts_pipeline(text, save_file, play_audio, caption_text, sync_with_audio)
        
        # STANDARD MODE: Traditional sequential processing
        return self._standard_tts_pipeline(text, save_file, play_audio, caption_text, sync_with_audio)
    
    def _streaming_tts_pipeline(self, text: str, save_file: bool, play_audio: bool, caption_text: str, sync_with_audio: bool) -> bool:
        """Streaming TTS with immediate start and background processing"""
        
        def background_tts_processing():
            """Background thread for TTS processing and playback"""
            try:
                if not config.MINIMAL_LOGGING:
                    print(f"🎧 Streaming TTS for: '{text[:30]}...'")
                
                # Start audio synthesis immediately
                audio_data = self.synthesize_text(text)
                
                if audio_data is None:
                    if not config.MINIMAL_LOGGING:
                        print(f"❌ Streaming TTS failed for: '{text[:30]}...'")
                    return
                
                # STREAMING AUDIO: Queue for linear output with intervals
                if play_audio:
                    if config.LINEAR_VOICE_OUTPUT:
                        # Pass caption text to voice queue for synchronized display
                        queue_caption = caption_text if (sync_with_audio and config.ENABLE_OBS_CAPTIONS) else None
                        self._queue_voice_output(audio_data, text, queue_caption)
                        
                        if queue_caption and not config.MINIMAL_LOGGING:
                            print(f"📺 Caption queued with voice for sync display: '{caption_text[:30]}...'")
                    else:
                        # Immediate streaming - display caption now if needed
                        if sync_with_audio and config.ENABLE_OBS_CAPTIONS and caption_text:
                            self._display_synchronized_caption(caption_text)
                            if not config.MINIMAL_LOGGING:
                                print(f"📺 Caption displayed immediately: '{caption_text[:30]}...'")
                        
                        self._streaming_audio_play(audio_data)
                
                # Save audio file in background
                if save_file:
                    timestamp = int(time.time() * 1000)
                    filename = f"tts_stream_{timestamp}.wav"
                    self.save_audio(audio_data, filename)
                
            except Exception as e:
                if not config.MINIMAL_LOGGING:
                    print(f"❌ Streaming TTS error: {e}")
        
        # Start TTS processing in background thread immediately
        import threading
        tts_thread = threading.Thread(target=background_tts_processing, daemon=True)
        tts_thread.start()
        
        # Return immediately - don't wait for completion
        return True
    
    def _standard_tts_pipeline(self, text: str, save_file: bool, play_audio: bool, caption_text: str, sync_with_audio: bool) -> bool:
        """Standard TTS pipeline for compatibility"""
        
        # Synthesize audio
        if not config.MINIMAL_LOGGING:
            print(f"🔊 Synthesizing TTS for: '{text[:50]}...'")
        audio_data = self.synthesize_text(text)
        
        if audio_data is None:
            if not config.MINIMAL_LOGGING:
                print(f"❌ TTS Pipeline failed for: '{text[:50]}...'")
            return False
        
        success = True
        
        # SYNCHRONIZED CAPTION TIMING: Display captions when audio starts
        if sync_with_audio and config.ENABLE_OBS_CAPTIONS and caption_text:
            if not config.MINIMAL_LOGGING:
                print("📺 Displaying synchronized caption now...")
            
            # Import caption manager to update captions precisely when audio plays
            try:
                caption_success = self._display_synchronized_caption(caption_text)
                if caption_success and not config.MINIMAL_LOGGING:
                    print(f"✅ Caption displayed: '{caption_text[:50]}...'")
                elif not config.MINIMAL_LOGGING:
                    print("⚠️ Caption display failed")
            except Exception as e:
                if not config.MINIMAL_LOGGING:
                    print(f"❌ Caption sync error: {e}")
        
        # Save audio file
        if save_file:
            timestamp = int(time.time() * 1000)
            filename = f"tts_output_{timestamp}.wav"
            success &= self.save_audio(audio_data, filename)
        
        # Play audio (this is when captions should appear if synchronized)
        if play_audio:
            if not config.MINIMAL_LOGGING:
                print("🔊 Playing TTS audio through VB-Cable...")
            success &= self.play_audio(audio_data, blocking=False)
        
        return success
    
    def _display_synchronized_caption(self, caption_text: str) -> bool:
        """Display caption synchronized with voice output"""
        
        try:
            if not config.ENABLE_OBS_CAPTIONS or not self.caption_manager:
                return False
            
            # Use the shared caption manager and cancel any existing clearing timer
            # since new voice is starting
            success = self.caption_manager.update_live_caption(caption_text, "voice_sync")
            
            return success
            
        except Exception as e:
            print(f"❌ Caption file write error: {e}")
            return False
    
    def save_text_output(self, text: str) -> bool:
        """Save text output when TTS is disabled"""
        try:
            os.makedirs(config.DEFAULT_OUTPUT_DIR, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            filename = f"text_output_{timestamp}.txt"
            filepath = os.path.join(config.DEFAULT_OUTPUT_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"📝 Text saved: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Text Save Error: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current TTS manager status including model health"""
        return {
            "enabled": config.ENABLE_TTS,
            "models_loaded": self.tts_model is not None,
            "sample_rate": self.sample_rate,
            "audio_output_ready": self.audio_output is not None,
            "model_failed_count": getattr(self, 'model_failed_count', 0),
            "last_model_reset": getattr(self, 'last_model_reset', 0),
            "model_health": "good" if getattr(self, 'model_failed_count', 0) < 3 else "degraded" if getattr(self, 'model_failed_count', 0) < 5 else "poor",
            "reinitialization_enabled": getattr(config, 'ENABLE_MODEL_REINITIALIZATION', True),
            "cuda_recovery_enabled": getattr(config, 'ENABLE_CUDA_ERROR_RECOVERY', True)
        }
    
    def force_model_reset(self) -> bool:
        """Manually force model reinitialization (emergency reset)"""
        print("🚨 Forcing emergency model reset...")
        
        # Bypass cooldown for emergency reset
        self.last_model_reset = 0
        self.model_failed_count = 0
        
        # Force reinitialization
        return self._reinitialize_xtts_model()
    
    def get_model_health_info(self) -> Dict[str, Any]:
        """Get detailed model health information"""
        current_time = time.time()
        time_since_reset = current_time - getattr(self, 'last_model_reset', 0)
        
        return {
            "failure_count": getattr(self, 'model_failed_count', 0),
            "time_since_last_reset_minutes": time_since_reset / 60,
            "model_loaded": self.xtts_model is not None,
            "health_status": "good" if getattr(self, 'model_failed_count', 0) < 3 else "degraded" if getattr(self, 'model_failed_count', 0) < 5 else "critical",
            "next_proactive_reset_minutes": max(0, (getattr(config, 'PROACTIVE_MODEL_RESET_HOURS', 1) * 3600 - time_since_reset) / 60) if getattr(config, 'PROACTIVE_MODEL_RESET_HOURS', 1) > 0 else -1,
            "cooldown_remaining_seconds": max(0, self.model_reset_cooldown - (current_time - getattr(self, 'last_model_reset', 0)))
        }
    
    def cleanup(self):
        """Clean up TTS manager resources including voice queue"""
        try:
            # Stop voice scheduler
            if hasattr(self, 'voice_queue_lock'):
                with self.voice_queue_lock:
                    if hasattr(self, 'voice_queue'):
                        self.voice_queue.clear()
                    self.voice_scheduler_running = False
            
            # Clean up active streams
            if hasattr(self, 'stream_lock'):
                with self.stream_lock:
                    if hasattr(self, 'active_streams'):
                        for stream in self.active_streams:
                            try:
                                stream.stop_stream()
                                stream.close()
                            except:
                                pass
                        self.active_streams.clear()
            
            # Clean up audio output
            if self.audio_output:
                self.audio_output.terminate()
                self.audio_output = None
                print("🔊 Audio output cleaned up")
            
            # Clean up audio cache if auto-cleanup is enabled
            if config.AUTO_CLEANUP_ON_EXIT:
                deleted_count = config.cleanup_audio_cache()
                if deleted_count > 0:
                    print(f"🗑️ Cleaned up {deleted_count} cached audio files")
                    
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

def test_tts_manager():
    """Test the TTS manager module"""
    print("🧪 Testing TTS Manager Module")
    print("-" * 40)
    
    # Initialize manager
    tts = TTSManager()
    
    # Test text-to-speech
    test_texts = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing one, two, three."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔊 Test {i}: '{text}'")
        
        # Test synthesis
        audio = tts.synthesize_text(text)
        if audio is not None:
            print(f"✅ Synthesized {len(audio)} samples")
            
            # Test save
            filename = f"test_output_{i}.wav"
            saved = tts.save_audio(audio, filename)
            print(f"💾 Save: {'✅ Success' if saved else '❌ Failed'}")
        else:
            print("❌ Synthesis failed")
    
    # Test pipeline
    print(f"\n🔄 Testing complete pipeline...")
    pipeline_success = tts.text_to_speech_pipeline(
        "This is a complete pipeline test.",
        save_file=True,
        play_audio=False  # Don't play during testing
    )
    print(f"Pipeline: {'✅ Success' if pipeline_success else '❌ Failed'}")
    
    # Show status
    status = tts.get_status()
    print(f"\n📊 TTS Manager Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    tts.cleanup()

if __name__ == "__main__":
    print("🔊 S2T2SS TTS Manager Module")
    print("=" * 50)
    
    test_tts_manager()
