#!/usr/bin/env python3
"""
🎯 S2T2SS Main Pipeline Integration
Complete Speech-to-Text-to-Speech-Synthesis system with all toggle features
"""

import sys
import os
import time
import threading
import sounddevice as sd
import numpy as np
from typing import Optional, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all modules
import config
import toggle_control
import asr_module
import llm_worker
import tts_manager
import caption_manager

class S2T2SSPipeline:
    """
    Main S2T2SS Pipeline
    Integrates all modules with complete toggle control
    """
    
    def __init__(self):
        """Initialize the complete pipeline"""
        print("🎯 Initializing S2T2SS Pipeline...")
        print("=" * 60)
        
        # Load configuration
        config.load_config()
        
        # Initialize all modules
        self.asr = asr_module.FunASREngine()
        self.llm = llm_worker.LLMWorker(base_url=config.LLM_SERVER_URL, model_name=config.LLM_MODEL_NAME)
        self.captions = caption_manager.CaptionManager()
        self.tts = tts_manager.TTSManager(caption_manager=self.captions)
        
        # Streaming configuration (optimized for Chinese ASR reliability)
        self.samplerate = 16000
        
        # Performance-optimized chunk sizes
        if config.OPTIMIZED_CHUNK_SIZE:
            self.chunk_duration = 3.0               # Optimized for speed vs accuracy balance
            self.overlap_duration = 0.15            # Reduced overlap for faster processing
        else:
            self.chunk_duration = 3.5               # Standard for maximum accuracy
            self.overlap_duration = 0.25            # Standard overlap
            
        self.chunk_samples = int(self.samplerate * self.chunk_duration)
        self.overlap_samples = int(self.samplerate * self.overlap_duration)
        
        # Parallel processing setup
        if config.ENABLE_PARALLEL_PROCESSING:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_WORKERS)
        else:
            self.executor = None
        
        # Pipeline state
        self.is_running = False
        self.pipeline_thread = None
        self.audio_stream = None
        self.stats = {
            'processed_chunks': 0,
            'total_runtime': 0,
            'start_time': None,
            'duplicates_skipped': 0  # Track skipped duplicates
        }
        
        # Optimized similarity tracking to prevent duplicate processing
        if config.USE_FAST_SIMILARITY:
            self.recent_texts = []          # Store recent processed texts (reduced)
            self.recent_audio_hashes = []   # Store recent audio chunk hashes
            self.max_recent_items = 3       # Reduced for faster comparison
        else:
            self.recent_texts = []          # Store recent processed texts
            self.recent_audio_hashes = []   # Store recent audio chunk hashes
            self.max_recent_items = 5       # Standard comparison window
        
        # Performance tracking
        self.batch_queue = []               # Queue for batch processing
        self.processing_futures = []        # Track parallel operations
        
        try:
            print("✅ S2T2SS Pipeline initialized successfully!")
        except UnicodeEncodeError:
            print("S2T2SS Pipeline initialized successfully!")
        self._show_system_status()
    
    def _show_system_status(self):
        """Show current system configuration"""
        print("\n🎯 Current System Configuration")
        print("=" * 50)
        toggle_control.show_current_configuration()
        
        print(f"\n📊 Module Status:")
        try:
            print(f"   🎤 ASR Engine:    {'✅ Ready' if self.asr.model else '❌ Not Ready'}")
            print(f"   🧠 LLM Worker:    {'✅ Connected' if self.llm.connection_tested else '⚠️ Not Connected'}")
            print(f"   🔊 TTS Manager:   {'✅ Ready' if self.tts.tts_model else '⚠️ Placeholder Models'}")
            print(f"   📺 Caption Mgr:   ✅ Ready")
        except UnicodeEncodeError:
            print("   [OK] ASR Engine:    Ready" if self.asr.model else "   [!] ASR Engine:    Not Ready")
            print("   [OK] LLM Worker:    Connected" if self.llm.connection_tested else "   [!] LLM Worker:    Not Connected")
            print("   [OK] TTS Manager:   Ready" if self.tts.tts_model else "   [!] TTS Manager:   Placeholder Models")
            print("   [OK] Caption Mgr:   Ready")
        
        # Show voice queue status if linear output is enabled
        if config.LINEAR_VOICE_OUTPUT and hasattr(self.tts, 'voice_queue'):
            with self.tts.voice_queue_lock:
                queue_size = len(self.tts.voice_queue)
            try:
                scheduler_status = '✅ Running' if self.tts.voice_scheduler_running else '🔴 Stopped'
                print(f"   🎵 Voice Queue:    {queue_size} items, Scheduler: {scheduler_status}")
            except UnicodeEncodeError:
                scheduler_status = '[Running]' if self.tts.voice_scheduler_running else '[Stopped]'
                print(f"   Voice Queue:    {queue_size} items, Scheduler: {scheduler_status}")
        
        # Show cache status
        cache_status = config.get_cache_status()
        try:
            print(f"\n🗂️ Cache Status:")
            print(f"   🎵 Audio Files:   {cache_status.get('audio_files_count', 0)} files ({cache_status.get('total_size_mb', 0):.1f} MB)")
            print(f"   🗑️ Auto Cleanup:  {'✅ Enabled' if cache_status.get('cleanup_enabled', False) else '❌ Disabled'}")
            print(f"   📁 Cache Dir:     {os.path.relpath(config.AUDIO_CACHE_DIR)}")
        except UnicodeEncodeError:
            print(f"\nCache Status:")
            print(f"   Audio Files:   {cache_status.get('audio_files_count', 0)} files ({cache_status.get('total_size_mb', 0):.1f} MB)")
            print(f"   Auto Cleanup:  {'Enabled' if cache_status.get('cleanup_enabled', False) else 'Disabled'}")
            print(f"   Cache Dir:     {os.path.relpath(config.AUDIO_CACHE_DIR)}")
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using optimized algorithms"""
        if not text1 or not text2:
            return 0.0
        
        # Fast mode: Use simplified similarity for speed
        if config.USE_FAST_SIMILARITY:
            return self._fast_similarity(text1, text2)
        
        # Normalize texts (remove spaces, convert to lowercase)
        norm1 = ''.join(text1.split()).lower()
        norm2 = ''.join(text2.split()).lower()
        
        if not norm1 or not norm2:
            return 0.0
        
        # Early exit for identical texts
        if norm1 == norm2:
            return 1.0
        
        # Early exit for very different lengths
        len_diff = abs(len(norm1) - len(norm2))
        if len_diff > max(len(norm1), len(norm2)) * 0.5:
            return 0.0
        
        # Calculate character-level similarity using Jaccard index
        set1 = set(norm1)
        set2 = set(norm2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Also check sequence similarity (for Chinese character order)
        longer_text = norm1 if len(norm1) > len(norm2) else norm2
        shorter_text = norm2 if len(norm1) > len(norm2) else norm1
        
        # Count matching substrings
        matching_chars = 0
        for i in range(len(shorter_text)):
            if i < len(longer_text) and shorter_text[i] == longer_text[i]:
                matching_chars += 1
        
        sequence_similarity = matching_chars / len(longer_text) if longer_text else 0.0
        
        # Combine both metrics
        final_similarity = (jaccard_similarity + sequence_similarity) / 2
        
        return final_similarity
    
    def _fast_similarity(self, text1: str, text2: str) -> float:
        """Ultra-fast similarity detection for real-time processing"""
        # Normalize texts quickly
        norm1 = text1.replace(' ', '').lower()
        norm2 = text2.replace(' ', '').lower()
        
        if not norm1 or not norm2:
            return 0.0
        
        # Exact match check
        if norm1 == norm2:
            return 1.0
        
        # Length-based early exit
        len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
        if len_ratio < 0.5:
            return 0.0
        
        # Simple character overlap ratio
        common_chars = len(set(norm1) & set(norm2))
        total_unique = len(set(norm1) | set(norm2))
        
        return common_chars / total_unique if total_unique > 0 else 0.0
    
    def _calculate_audio_hash(self, audio_chunk: bytes) -> str:
        """Calculate a simple hash for audio chunk"""
        import hashlib
        return hashlib.md5(audio_chunk[:1000]).hexdigest()[:8]  # Use first 1KB for speed
    
    def _is_duplicate_content(self, raw_text: str, audio_chunk: bytes) -> bool:
        """Optimized duplicate content detection"""
        
        # Skip duplicate detection in streamlined mode for maximum speed
        if config.STREAMLINE_PIPELINE and len(self.recent_texts) == 0:
            return False
        
        # Use appropriate similarity threshold
        threshold = config.FAST_SIMILARITY_THRESHOLD if config.USE_FAST_SIMILARITY else config.SIMILARITY_THRESHOLD
        
        # Early exit for very short texts
        if len(raw_text.strip()) < 3:
            return False
        
        # Check text similarity with recent texts (optimized)
        for recent_text in self.recent_texts:
            # Quick length check before detailed similarity
            if abs(len(raw_text) - len(recent_text)) > max(len(raw_text), len(recent_text)) * 0.7:
                continue
                
            similarity = self._calculate_text_similarity(raw_text, recent_text)
            if similarity > threshold:
                if not config.MINIMAL_LOGGING:
                    print(f"🔍 Duplicate text detected (similarity: {similarity:.2f}): '{raw_text[:30]}...'")
                return True
        
        # Optimized audio hash check (only if text check passed)
        if config.USE_FAST_SIMILARITY:
            # Skip audio hash for speed in fast mode
            return False
        else:
            # Check audio hash (for exact audio duplicates)
            audio_hash = self._calculate_audio_hash(audio_chunk)
            if audio_hash in self.recent_audio_hashes:
                if not config.MINIMAL_LOGGING:
                    print(f"🔍 Duplicate audio chunk detected (hash: {audio_hash})")
                return True
        
        return False
    
    def _update_recent_content(self, raw_text: str, audio_chunk: bytes):
        """Update recent content tracking"""
        
        # Add current text to recent texts
        self.recent_texts.append(raw_text)
        if len(self.recent_texts) > self.max_recent_items:
            self.recent_texts.pop(0)
        
        # Add current audio hash to recent hashes
        audio_hash = self._calculate_audio_hash(audio_chunk)
        self.recent_audio_hashes.append(audio_hash)
        if len(self.recent_audio_hashes) > self.max_recent_items:
            self.recent_audio_hashes.pop(0)
    
    def process_audio_chunk(self, audio_chunk: bytes, start_time: float = None) -> Dict[str, Any]:
        """
        Process a single audio chunk through the complete pipeline
        
        Args:
            audio_chunk: Raw audio data
            start_time: Start time for timing calculations
            
        Returns:
            Processing results dictionary
        """
        
        if start_time is None:
            start_time = time.time()
        
        results = {
            'success': False,
            'raw_text': '',
            'processed_text': '',
            'audio_generated': False,
            'captions_updated': False,
            'processing_time': 0,
            'start_time': start_time
        }
        
        try:
            try:
                print(f"\n🎯 Processing audio chunk ({len(audio_chunk)} bytes)...")
            except UnicodeEncodeError:
                print(f"\nProcessing audio chunk ({len(audio_chunk)} bytes)...")
            
            # Step 1: Speech Recognition (ASR)
            try:
                print("🎤 Step 1: Speech Recognition...")
            except UnicodeEncodeError:
                print("Step 1: Speech Recognition...")
            raw_text = self.asr.transcribe_chunk(audio_chunk)
            
            if not raw_text:
                try:
                    print("⚠️ No speech detected in chunk")
                except UnicodeEncodeError:
                    print("No speech detected in chunk")
                return results
            
            results['raw_text'] = raw_text
            try:
                print(f"🎤 ASR Result: '{raw_text}'")
            except UnicodeEncodeError:
                print(f"ASR Result: '{raw_text}'")
            
            # Step 2: Text Processing (LLM) - Optional
            if config.ENABLE_LLM_EDITING:
                print("🧠 Step 2: LLM Text Processing...")
                processed_text = self.llm.process_text(raw_text, mode="refine")
            else:
                print("🧠 Step 2: LLM Processing DISABLED (using raw text)")
                processed_text = raw_text
            
            results['processed_text'] = processed_text
            print(f"🧠 Processed Text: '{processed_text}'")
            
            # Step 3: Text-to-Speech (TTS) - Optional
            if config.ENABLE_TTS:
                try:
                    print("🔊 Step 3: Text-to-Speech Synthesis...")
                except UnicodeEncodeError:
                    print("Step 3: Text-to-Speech Synthesis...")
                
                audio_generated = self.tts.text_to_speech_pipeline(processed_text, save_file=True, play_audio=True)
                
                results['audio_generated'] = audio_generated
            else:
                try:
                    print("🔊 Step 3: TTS DISABLED (text-only mode)")
                except UnicodeEncodeError:
                    print("Step 3: TTS DISABLED (text-only mode)")
                # Save text output instead
                self.tts.save_text_output(processed_text)
                results['audio_generated'] = False
            
            # Step 4: Caption Management - Conditional based on TTS status  
            # If TTS enabled: Captions will be synchronized with voice output
            # If TTS disabled: Show captions immediately (text-only mode)
            caption_mode = "immediate" if not config.ENABLE_TTS else config.CAPTION_SYNC_MODE
            
            # Step 4: Caption Management - Always save transcription history
            if config.ENABLE_OBS_CAPTIONS and caption_mode == "immediate":
                try:
                    print("📺 Step 4: Immediate Caption Processing (text-only mode)...")
                except UnicodeEncodeError:
                    print("Step 4: Immediate Caption Processing (text-only mode)...")
                
                # Save transcription history
                self.captions.add_input_transcription(raw_text, start_time)
                self.captions.add_output_transcription(processed_text, start_time)
                
                # Update live captions
                captions_updated = self.captions.update_live_caption(processed_text)
                results['captions_updated'] = captions_updated
            elif config.ENABLE_OBS_CAPTIONS and caption_mode == "voice":
                try:
                    print("📺 Step 4: Voice-Synchronized Caption Mode (captions will sync with audio)")
                except UnicodeEncodeError:
                    print("Step 4: Voice-Synchronized Caption Mode (captions will sync with audio)")
                
                # Save transcription history even in voice mode
                self.captions.add_input_transcription(raw_text, start_time)
                self.captions.add_output_transcription(processed_text, start_time)
                results['captions_updated'] = True  # Will be handled by voice queue
            else:
                try:
                    print("📺 Step 4: Caption Processing DISABLED")
                except UnicodeEncodeError:
                    print("Step 4: Caption Processing DISABLED")
                
                # Still save transcription history even if captions disabled
                self.captions.add_input_transcription(raw_text, start_time)
                self.captions.add_output_transcription(processed_text, start_time)
                results['captions_updated'] = False
            
            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            results['success'] = True
            
            # Update stats
            self.stats['processed_chunks'] += 1
            
            print(f"✅ Pipeline completed in {results['processing_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Pipeline Error: {e}")
            results['processing_time'] = time.time() - start_time
        
        return results
    
    def start_live_processing(self, device_id: int = None, duration: float = None):
        """
        Start live audio processing
        
        Args:
            device_id: Audio input device ID (None for default)
            duration: Maximum processing duration in seconds (None for infinite)
        """
        
        if self.is_running:
            try:
                print("⚠️ Pipeline already running!")
            except UnicodeEncodeError:
                print("Pipeline already running!")
            return
        
        try:
            print(f"\n🚀 Starting Live S2T2SS Processing...")
        except UnicodeEncodeError:
            print(f"\nStarting Live S2T2SS Processing...")
        print("=" * 50)
        
        # Set up input device (microphone)
        if device_id is None:
            try:
                input_devices, _ = asr_module.list_audio_devices()
                
                # Use system default input device (microphone)
                # This respects whatever microphone the user has set as default in Windows
                device_id = None  # None means use system default
                
                # Show available input devices for reference
                try:
                    print(f"🎤 Available input devices:")
                except UnicodeEncodeError:
                    print("Available input devices:")
                
                mic_devices = []
                for dev_id, dev_info in input_devices:
                    if dev_info['max_input_channels'] > 0:  # Only input devices
                        device_name = dev_info['name']
                        is_default = "(DEFAULT)" if dev_id == 0 else ""
                        try:
                            print(f"   {dev_id}: {device_name} {is_default}")
                        except UnicodeEncodeError:
                            print(f"   {dev_id}: {device_name.encode('ascii', errors='replace').decode('ascii')} {is_default}")
                        mic_devices.append((dev_id, device_name))
                
                try:
                    print(f"🎤 Using system default microphone (respects Windows default setting)")
                except UnicodeEncodeError:
                    print("Using system default microphone (respects Windows default setting)")
                    
            except Exception as e:
                print(f"⚠️ Device detection error: {e}")
                device_id = None
                print(f"🎤 Using system default microphone")
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        def processing_loop():
            """Main processing loop with gap-free streaming"""
            try:
                print(f"🎤 Starting gap-free audio streaming from device {device_id}...")
                print(f"� Stream config: {self.chunk_duration}s chunks, {self.overlap_duration*1000}ms overlap")
                print(f"�💡 Press Ctrl+C to stop processing")
                
                chunk_count = 0
                start_time = time.time()
                
                # Initialize gap-free audio stream
                audio_stream = self._create_audio_stream(device_id)
                
                # Process continuous audio chunks with overlap
                for audio_chunk in audio_stream:
                    if not self.is_running:
                        break
                        
                    if duration and (time.time() - start_time) > duration:
                        print(f"⏰ Duration limit reached ({duration}s)")
                        break
                    
                    chunk_count += 1
                    print(f"\n📡 Processing overlapping chunk #{chunk_count} ({len(audio_chunk)} samples)...")
                    
                    # Convert numpy array to bytes for processing
                    audio_bytes = audio_chunk.astype(np.int16).tobytes()
                    
                    # Process through pipeline with synchronized timing
                    results = self.process_audio_chunk_with_sync(audio_bytes, chunk_count)
                    
                    if results['success']:
                        print(f"✅ Chunk #{chunk_count} completed successfully")
                    else:
                        print(f"⚠️ Chunk #{chunk_count} had issues")
                
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping live processing...")
            except Exception as e:
                print(f"❌ Processing loop error: {e}")
            finally:
                self.is_running = False
        
        # Start processing in separate thread
        self.pipeline_thread = threading.Thread(target=processing_loop, daemon=True)
        self.pipeline_thread.start()
    
    def _create_audio_stream(self, device_id):
        """Create gap-free audio stream with overlapping chunks (from your original implementation)"""
        import queue
        
        audio_buffer = queue.Queue()
        stop_flag = [False]
        last_audio_level_time = [0]
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"⚠️ Audio input status: {status}")
            
            # Monitor audio levels every 2 seconds
            current_time = time.time()
            if current_time - last_audio_level_time[0] > 2.0:
                audio_level = np.max(np.abs(indata)) if len(indata) > 0 else 0
                audio_db = 20 * np.log10(audio_level + 1e-10)
                
                if audio_level > 0.01:  # Threshold for detecting speech
                    print(f"🎤 Audio detected: Level={audio_level:.3f}, dB={audio_db:.1f}")
                else:
                    print(f"🔇 No audio detected: Level={audio_level:.3f}, dB={audio_db:.1f}")
                
                last_audio_level_time[0] = current_time
            
            audio_buffer.put(indata.copy())
        
        def record_thread():
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype='int16',
                callback=audio_callback,
                device=device_id
            ):
                print(f"🎤 Gap-free recording started (device: {device_id or 'default'})")
                while not stop_flag[0] and self.is_running:
                    sd.sleep(50)
        
        # Start recording thread
        record_thread_obj = threading.Thread(target=record_thread, daemon=True)
        record_thread_obj.start()
        
        # Generator that yields overlapping audio chunks
        collected = np.zeros((0,), dtype='int16')
        
        # Wait for initial chunk
        while len(collected) < self.chunk_samples and self.is_running:
            if not audio_buffer.empty():
                chunk = audio_buffer.get()
                collected = np.concatenate([collected, chunk.flatten()])
            else:
                sd.sleep(10)
        
        if len(collected) >= self.chunk_samples:
            yield collected[:self.chunk_samples]
            collected = collected[-self.overlap_samples:]  # Keep overlap for next chunk
        
        # Continue yielding overlapping chunks
        try:
            while self.is_running:
                while len(collected) < self.chunk_samples and self.is_running:
                    if not audio_buffer.empty():
                        chunk = audio_buffer.get()
                        collected = np.concatenate([collected, chunk.flatten()])
                    else:
                        sd.sleep(10)
                
                if len(collected) >= self.chunk_samples and self.is_running:
                    yield collected[:self.chunk_samples]
                    collected = collected[-self.overlap_samples:]  # Maintain 250ms overlap
        except KeyboardInterrupt:
            stop_flag[0] = True
            return
    
    def process_audio_chunk_with_sync(self, audio_chunk: bytes, chunk_num: int) -> Dict[str, Any]:
        """Optimized audio chunk processing with streamlined pipeline"""
        start_time = time.time()
        
        results = {
            'success': False,
            'raw_text': '',
            'processed_text': '',
            'audio_generated': False,
            'captions_updated': False,
            'processing_time': 0,
            'start_time': start_time
        }
        
        try:
            # Minimal logging in fast mode
            if not config.MINIMAL_LOGGING:
                print(f"🎯 Processing audio chunk #{chunk_num} ({len(audio_chunk)} bytes)...")
            
            # Step 1: Speech Recognition (ASR)
            if not config.MINIMAL_LOGGING:
                print("🎤 Step 1: Speech Recognition...")
            raw_text = self.asr.transcribe_chunk(audio_chunk)
            
            if not raw_text:
                if not config.MINIMAL_LOGGING:
                    print("⚠️ No speech detected in chunk")
                return results
            
            results['raw_text'] = raw_text
            if not config.MINIMAL_LOGGING:
                print(f"🎤 ASR Result: '{raw_text}'")
            
            # Step 1.5: Optimized Duplicate Content Check
            if self._is_duplicate_content(raw_text, audio_chunk):
                if not config.MINIMAL_LOGGING:
                    print("🔍 Skipping duplicate content")
                # Mark as success but don't process further
                self.stats['duplicates_skipped'] += 1
                results['success'] = True
                results['processed_text'] = raw_text
                results['processing_time'] = time.time() - start_time
                return results
            
            # Update recent content tracking (optimized)
            self._update_recent_content(raw_text, audio_chunk)
            
            # Step 2: Text Processing (LLM) - Optional
            if config.ENABLE_LLM_EDITING:
                print("🧠 Step 2: LLM Text Processing...")
                processed_text = self.llm.process_text(raw_text, mode="refine")
            else:
                print("🧠 Step 2: LLM Processing DISABLED (using raw text)")
                processed_text = raw_text
            
            results['processed_text'] = processed_text
            print(f"🧠 Processed Text: '{processed_text}'")
            
            # Step 3: Streaming TTS + Caption Synchronization
            if config.ENABLE_TTS:
                if not config.MINIMAL_LOGGING:
                    print("🎧 Step 3: Streaming Text-to-Speech...")
                
                # STREAMING MODE: Start TTS immediately, don't wait for completion
                if config.ENABLE_STREAMING_TTS:
                    # Start TTS processing in background immediately
                    tts_started = self.tts.text_to_speech_pipeline(
                        processed_text, 
                        save_file=True, 
                        play_audio=True,
                        caption_text=processed_text,  # Caption text
                        sync_with_audio=(config.CAPTION_SYNC_MODE == "voice")  # Sync based on mode
                    )
                    
                    # Mark as successful if TTS started (don't wait for completion)
                    results['audio_generated'] = tts_started
                    
                    if not config.MINIMAL_LOGGING:
                        print("🎧 Streaming TTS: Audio will play when ready")
                    
                    # Save transcription history (streaming mode)
                    self.captions.add_input_transcription(raw_text, start_time)
                    self.captions.add_output_transcription(processed_text, start_time)
                    
                    # Captions are handled by TTS manager with immediate timing
                    if config.ENABLE_OBS_CAPTIONS:
                        if not config.MINIMAL_LOGGING:
                            print("📺 Caption sync: Displaying immediately with streaming audio")
                        results['captions_updated'] = True
                    
                else:
                    # STANDARD MODE: Wait for TTS completion (traditional behavior)
                    if not config.MINIMAL_LOGGING:
                        print("🔊 Step 3: Standard Text-to-Speech with Synchronized Captions...")
                    
                    # SYNCHRONIZED CAPTION TIMING: Captions appear when TTS audio starts playing
                    if not config.MINIMAL_LOGGING:
                        print("🔊 TTS Output: Standard (captions will sync with TTS output)")
                    
                    # Generate TTS audio with synchronized caption display
                    audio_generated = self.tts.text_to_speech_pipeline(
                        processed_text, 
                        save_file=True, 
                        play_audio=True,
                        caption_text=processed_text,  # Caption text
                        sync_with_audio=(config.CAPTION_SYNC_MODE == "voice")  # Sync based on mode
                    )
                    
                    results['audio_generated'] = audio_generated
                    
                    # Save transcription history (standard mode)
                    self.captions.add_input_transcription(raw_text, start_time)
                    self.captions.add_output_transcription(processed_text, start_time)
                    
                    # Captions are handled by TTS manager with perfect timing
                    if config.ENABLE_OBS_CAPTIONS:
                        if not config.MINIMAL_LOGGING:
                            print("📺 Caption sync: Will display when audio starts playing")
                        results['captions_updated'] = True
                    else:
                        if not config.MINIMAL_LOGGING:
                            print("📺 OBS Captions: DISABLED")
                        
            else:
                if not config.MINIMAL_LOGGING:
                    print("🔊 Step 3: TTS DISABLED (text-only mode)")
                
                # In text-only mode, show captions immediately
                if config.ENABLE_OBS_CAPTIONS:
                    if not config.MINIMAL_LOGGING:
                        print("📺 Step 4: Immediate Caption Processing (text-only mode)...")
                    
                    # Save transcription history
                    self.captions.add_input_transcription(raw_text, start_time)
                    self.captions.add_output_transcription(processed_text, start_time)
                    
                    # Update live captions
                    captions_updated = self.captions.update_live_caption(processed_text)
                    results['captions_updated'] = captions_updated
                else:
                    if not config.MINIMAL_LOGGING:
                        print("📺 Caption Processing: DISABLED")
                    
                    # Still save transcription history even if captions disabled
                    self.captions.add_input_transcription(raw_text, start_time)
                    self.captions.add_output_transcription(processed_text, start_time)
                
                # Save text output instead
                self.tts.save_text_output(processed_text)
                results['audio_generated'] = False
            
            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            results['success'] = True
            
            # Update stats
            self.stats['processed_chunks'] += 1
            
            print(f"✅ Pipeline completed in {results['processing_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Pipeline Error: {e}")
            results['processing_time'] = time.time() - start_time
        
        return results
        
        # Wait for thread completion if duration is specified
        if duration:
            self.pipeline_thread.join()
            self.stop_live_processing()
    
    def _handle_completed_chunks(self, processing_queue, completed_chunks, next_output_chunk):
        """Handle completed chunks while maintaining chronological order"""
        # Check for completed futures
        completed_futures = [(i, chunk_num, future) for i, (chunk_num, future) in enumerate(processing_queue) if future.done()]
        
        # Remove completed futures from queue
        for i, chunk_num, future in reversed(completed_futures):
            processing_queue.pop(i)
            try:
                result = future.result()
                completed_chunks[chunk_num] = result
                
                if result['success']:
                    print(f"✅ Chunk #{chunk_num} completed successfully (parallel)")
                else:
                    print(f"⚠️ Chunk #{chunk_num} had issues (parallel)")
                    
            except Exception as e:
                print(f"❌ Chunk #{chunk_num} processing error: {e}")
        
        # Output chunks in chronological order
        while next_output_chunk[0] in completed_chunks:
            chunk_num = next_output_chunk[0]
            result = completed_chunks.pop(chunk_num)
            
            # Here we could add ordered output handling if needed
            # For now, TTS audio is played immediately when generated
            
            next_output_chunk[0] += 1
    
    def _output_remaining_chunks(self, completed_chunks, next_output_chunk, max_chunk):
        """Output any remaining completed chunks in order"""
        while next_output_chunk[0] <= max_chunk:
            chunk_num = next_output_chunk[0]
            if chunk_num in completed_chunks:
                result = completed_chunks.pop(chunk_num)
                # Process any remaining ordered output
            next_output_chunk[0] += 1
    
    def _handle_completed_chunks(self, processing_queue, completed_chunks, next_output_chunk):
        """Handle completed chunks while maintaining chronological order"""
        # Check for completed futures
        completed_futures = [(i, chunk_num, future) for i, (chunk_num, future) in enumerate(processing_queue) if future.done()]
        
        # Remove completed futures from queue
        for i, chunk_num, future in reversed(completed_futures):
            processing_queue.pop(i)
            try:
                result = future.result()
                completed_chunks[chunk_num] = result
                
                if result['success']:
                    print(f"✅ Chunk #{chunk_num} completed successfully (parallel)")
                else:
                    print(f"⚠️ Chunk #{chunk_num} had issues (parallel)")
                    
            except Exception as e:
                print(f"❌ Chunk #{chunk_num} processing error: {e}")
        
        # Output chunks in chronological order
        while next_output_chunk[0] in completed_chunks:
            chunk_num = next_output_chunk[0]
            result = completed_chunks.pop(chunk_num)
            
            # Here we could add ordered output handling if needed
            # For now, TTS audio is played immediately when generated
            
            next_output_chunk[0] += 1
    
    def _output_remaining_chunks(self, completed_chunks, next_output_chunk, max_chunk):
        """Output any remaining completed chunks in order"""
        while next_output_chunk[0] <= max_chunk:
            chunk_num = next_output_chunk[0]
            if chunk_num in completed_chunks:
                result = completed_chunks.pop(chunk_num)
                # Process any remaining ordered output
            next_output_chunk[0] += 1
    
    def process_batch_chunks(self, audio_chunks: list) -> list:
        """Process multiple audio chunks in batch for improved performance"""
        if not config.ENABLE_BATCH_PROCESSING or not audio_chunks:
            # Fall back to individual processing
            return [self.process_audio_chunk(chunk) for chunk in audio_chunks]
        
        results = []
        start_time = time.time()
        
        try:
            print(f"🚀 Batch processing {len(audio_chunks)} chunks...")
            
            # Batch ASR processing
            raw_texts = []
            for chunk in audio_chunks:
                raw_text = self.asr.transcribe_chunk(chunk)
                raw_texts.append(raw_text)
            
            # Batch duplicate detection
            unique_pairs = []
            for i, (chunk, raw_text) in enumerate(zip(audio_chunks, raw_texts)):
                if raw_text and not self._is_duplicate_content(raw_text, chunk):
                    unique_pairs.append((i, chunk, raw_text))
                    self._update_recent_content(raw_text, chunk)
            
            # Batch LLM processing if enabled
            if config.ENABLE_LLM_EDITING and unique_pairs:
                llm_texts = []
                for _, _, raw_text in unique_pairs:
                    processed_text = self.llm.process_text(raw_text, mode="refine")
                    llm_texts.append(processed_text)
            else:
                llm_texts = [raw_text for _, _, raw_text in unique_pairs]
            
            # Batch TTS processing if enabled
            if config.ENABLE_TTS and unique_pairs:
                for (idx, chunk, raw_text), processed_text in zip(unique_pairs, llm_texts):
                    # Generate TTS for each unique chunk
                    audio_generated = self.tts.text_to_speech_pipeline(
                        processed_text, 
                        save_file=True, 
                        play_audio=True
                    )
                    
                    # Build result
                    result = {
                        'success': True,
                        'raw_text': raw_text,
                        'processed_text': processed_text,
                        'audio_generated': audio_generated,
                        'captions_updated': config.ENABLE_OBS_CAPTIONS,
                        'processing_time': time.time() - start_time,
                        'start_time': start_time
                    }
                    results.append(result)
            
            # Fill in skipped/duplicate results
            while len(results) < len(audio_chunks):
                results.append({
                    'success': False,
                    'raw_text': '',
                    'processed_text': '',
                    'audio_generated': False,
                    'captions_updated': False,
                    'processing_time': 0,
                    'start_time': start_time
                })
            
            total_time = time.time() - start_time
            print(f"✅ Batch processing completed in {total_time:.2f}s ({len(unique_pairs)} unique chunks)")
            
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            # Fall back to individual processing
            return [self.process_audio_chunk(chunk) for chunk in audio_chunks]
        
        return results

    def stop_live_processing(self):
        """Stop live audio processing"""
        if not self.is_running:
            print("⚠️ Pipeline not running!")
            return
        
        print(f"\n🛑 Stopping S2T2SS Pipeline...")
        self.is_running = False
        
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=5.0)
        
        # Calculate total runtime
        if self.stats['start_time']:
            self.stats['total_runtime'] = time.time() - self.stats['start_time']
        
        print(f"✅ Pipeline stopped")
        self._show_session_stats()
    
    def _show_session_stats(self):
        """Show session statistics"""
        print(f"\n📊 Session Statistics")
        print("=" * 30)
        print(f"🔢 Processed Chunks: {self.stats['processed_chunks']}")
        print(f"⏱️ Total Runtime: {self.stats['total_runtime']:.1f}s")
        print(f"🔍 Duplicates Skipped: {self.stats['duplicates_skipped']}")
        
        if self.stats['processed_chunks'] > 0:
            avg_time = self.stats['total_runtime'] / self.stats['processed_chunks']
            print(f"📈 Average per Chunk: {avg_time:.2f}s")
            
            efficiency = (self.stats['processed_chunks'] - self.stats['duplicates_skipped']) / self.stats['processed_chunks'] * 100
            print(f"⚡ Processing Efficiency: {efficiency:.1f}%")
        
        # Show voice queue final status
        if config.LINEAR_VOICE_OUTPUT and hasattr(self.tts, 'voice_queue'):
            with self.tts.voice_queue_lock:
                remaining_voices = len(self.tts.voice_queue)
            if remaining_voices > 0:
                print(f"🎵 Voices still queued: {remaining_voices}")
                print(f"📝 Note: Voices will continue playing until queue is empty")
    
    def process_file(self, audio_file: str) -> bool:
        """
        Process an audio file through the pipeline
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            True if processing successful
        """
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return False
        
        print(f"\n🎵 Processing Audio File: {audio_file}")
        print("=" * 50)
        
        try:
            # Read audio file (simplified - would need proper audio loading)
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Process through pipeline
            results = self.process_audio_chunk(audio_data)
            
            if results['success']:
                print(f"✅ File processing completed successfully")
                print(f"📝 Result: '{results['processed_text']}'")
                return True
            else:
                print(f"❌ File processing failed")
                return False
            
        except Exception as e:
            print(f"❌ File processing error: {e}")
            return False
    
    def interactive_mode(self):
        """Run interactive mode with menu system"""
        print(f"\n🎮 S2T2SS Interactive Mode")
        print("=" * 50)
        
        while True:
            print(f"\n🎯 S2T2SS Main Menu")
            print("=" * 30)
            print("1. 🎛️  Configure Toggles")
            print("2. 🚀 Start Live Processing")
            print("3. 🎵 Process Audio File")
            print("4. 🔧 Show System Status")
            print("5. 📊 Show Statistics")
            print("6. 🛑 Exit")
            
            try:
                choice = input("\n👉 Choose an option (1-6): ").strip()
                
                if choice == "1":
                    toggle_control.interactive_configuration_menu()
                    self._show_system_status()
                
                elif choice == "2":
                    try:
                        print(f"\n🎤 Audio Setup for S2T2SS:")
                        print("=" * 40)
                        print("📡 INPUT:  System default microphone (respects Windows setting)")
                        print("🔊 OUTPUT: VB-Cable (for OBS/streaming capture)")
                    except UnicodeEncodeError:
                        print("\nAudio Setup for S2T2SS:")
                        print("=" * 40)
                        print("INPUT:  System default microphone (respects Windows setting)")
                        print("OUTPUT: VB-Cable (for OBS/streaming capture)")
                    print("")
                    try:
                        input_devices, output_devices = asr_module.list_audio_devices()
                        
                        # Show key devices in a compact format
                        try:
                            print(f"🎤 Key Input Devices:")
                        except UnicodeEncodeError:
                            print("Key Input Devices:")
                        
                        # Show most relevant input devices (max 8)
                        for dev_id, dev_info in input_devices[:8]:
                            device_name = dev_info['name'][:40]  # Truncate long names
                            try:
                                print(f"   {dev_id}: {device_name}")
                            except UnicodeEncodeError:
                                print(f"   {dev_id}: {device_name.encode('ascii', errors='replace').decode('ascii')}")
                        
                        if len(input_devices) > 8:
                            try:
                                print(f"   ... and {len(input_devices) - 8} more devices")
                            except UnicodeEncodeError:
                                print(f"   ... and {len(input_devices) - 8} more devices")
                        
                        try:
                            print(f"\n📡 Press Enter to use system default microphone, or specify device ID:")
                        except UnicodeEncodeError:
                            print("\nPress Enter to use system default microphone, or specify device ID:")
                        
                        device_input = input("Device ID (or Enter for default): ").strip()
                        
                        device_id = None
                        if device_input.isdigit():
                            device_id = int(device_input)
                            try:
                                print(f"🎤 Using specified device: {device_id}")
                            except UnicodeEncodeError:
                                print(f"Using specified device: {device_id}")
                        else:
                            print(f"🎤 Using system default microphone")
                        
                        print(f"\n⏰ Duration (seconds, or Enter for unlimited):")
                        duration_input = input("Duration: ").strip()
                        
                        duration = None
                        if duration_input.replace('.', '').isdigit():
                            duration = float(duration_input)
                        
                        self.start_live_processing(device_id, duration)
                        
                    except KeyboardInterrupt:
                        print(f"\n🛑 Live processing cancelled")
                
                elif choice == "3":
                    audio_file = input("\n📁 Enter audio file path: ").strip()
                    if audio_file:
                        self.process_file(audio_file)
                
                elif choice == "4":
                    self._show_system_status()
                
                elif choice == "5":
                    self._show_session_stats()
                
                elif choice == "6":
                    print(f"\n👋 Goodbye!")
                    break
                
                else:
                    print(f"❌ Invalid choice. Please select 1-6.")
            
            except KeyboardInterrupt:
                print(f"\n\n👋 Exiting S2T2SS...")
                break
            except Exception as e:
                print(f"❌ Menu error: {e}")
    
    def cleanup(self):
        """Clean up all modules"""
        print(f"\n🧹 Cleaning up S2T2SS Pipeline...")
        
        try:
            if self.is_running:
                self.stop_live_processing()
            
            self.tts.cleanup()
            self.captions.cleanup()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

def main():
    """Main entry point"""
    print("🎯 S2T2SS - Speech to Text to Speech Synthesis")
    print("=" * 60)
    print("Complete modular system with full toggle control!")
    
    pipeline = None
    
    try:
        # Initialize pipeline
        pipeline = S2T2SSPipeline()
        
        # Run interactive mode
        pipeline.interactive_mode()
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline:
            pipeline.cleanup()

if __name__ == "__main__":
    main()
