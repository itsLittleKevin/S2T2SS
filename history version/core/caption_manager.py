#!/usr/bin/env python3
"""
📺 S2T2SS Caption Manager Module
Handles caption file generation for OBS and streaming
"""

import sys
import os
import time
import re
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

class CaptionManager:
    """
    Caption Manager for OBS integration
    Handles live captions, SRT files, and text overlays
    """
    
    def __init__(self):
        """Initialize caption manager"""
        self.live_caption_file = "live_caption.txt"
        self.buffer_caption_file = "buffer_caption.txt" 
        self.raw_caption_file = "raw_caption.txt"
        
        # Transcription history files
        self.input_srt_file = "input.srt"      # ASR transcription history (SRT format)
        self.input_txt_file = "input.txt"      # ASR transcription history (plain text)
        self.output_srt_file = "output.srt"    # LLM processed history (SRT format)
        self.output_txt_file = "output.txt"    # LLM processed history (plain text)
        
        # Legacy compatibility
        self.srt_file = self.output_srt_file
        self.output_text_file = self.output_txt_file
        
        self.caption_history = []
        self.srt_entries = []          # For output SRT entries (LLM processed)
        self.input_srt_entries = []    # For input SRT entries (ASR raw)
        self.current_caption_id = 1
        self.session_start_time = datetime.now()  # Track session start for relative timestamps
        
        # Caption clearing configuration
        self.caption_clear_delay = 3.0  # Seconds of silence before clearing
        self.last_caption_time = 0
        self.clear_timer = None
        self.clear_timer_lock = threading.Lock()
        
        # Caption wrapping configuration
        self.caption_limits = {
            'cjk': {
                'max_chars': 25,     # CJK characters are wider
                'max_words': None,   # Word concept less relevant for CJK
                'max_lines': 2
            },
            'latin': {
                'max_chars': 45,     # Latin characters are narrower
                'max_words': 8,      # Word-based wrapping for Latin scripts
                'max_lines': 2
            }
        }
        
        print(f"📺 Caption Manager initialized")
        self._initialize_caption_files()
    
    def _initialize_caption_files(self):
        """Initialize caption files"""
        try:
            # Ensure data directory exists
            os.makedirs(config.DEFAULT_OUTPUT_DIR, exist_ok=True)
            
            # Initialize empty caption files
            caption_files = [
                self.live_caption_file,
                self.buffer_caption_file,
                self.raw_caption_file,
                self.input_txt_file,
                self.output_txt_file
            ]
            
            for filename in caption_files:
                filepath = self._get_caption_path(filename)
                if not os.path.exists(filepath):
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write("")
                    print(f"📺 Created caption file: {filename}")
            
            # Initialize SRT files with headers
            srt_files = [self.input_srt_file, self.output_srt_file]
            for srt_filename in srt_files:
                srt_path = self._get_caption_path(srt_filename)
                if not os.path.exists(srt_path):
                    with open(srt_path, 'w', encoding='utf-8') as f:
                        session_time = self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")
                        file_type = "ASR Raw Input" if "input" in srt_filename else "LLM Processed Output"
                        f.write(f"# S2T2SS Transcription History - {file_type}\n")
                        f.write(f"# Session started: {session_time}\n\n")
                    print(f"📺 Created SRT file: {srt_filename}")
            
            # Legacy SRT file initialization (for backward compatibility)
            srt_path = self._get_caption_path(self.srt_file)
            if not os.path.exists(srt_path):
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write("")
                print(f"📺 Created SRT file: {self.srt_file}")
            
        except Exception as e:
            print(f"❌ Caption file initialization error: {e}")
    
    def _detect_text_language(self, text: str) -> str:
        """
        Detect if text is primarily CJK or Latin script
        
        Args:
            text: Text to analyze
            
        Returns:
            'cjk' or 'latin'
        """
        if not text:
            return 'latin'
        
        # Count CJK characters (Chinese, Japanese, Korean)
        cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]'
        cjk_chars = len(re.findall(cjk_pattern, text))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return 'latin'
        
        # If more than 30% CJK characters, treat as CJK
        cjk_ratio = cjk_chars / total_chars
        return 'cjk' if cjk_ratio > 0.3 else 'latin'
    
    def _normalize_punctuation(self, text: str) -> str:
        """
        Normalize Chinese punctuation to ASCII equivalents for cleaner display
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized punctuation
        """
        if not text:
            return text
        
        # Replace Chinese single quotes with ASCII single quote
        text = text.replace('\u2018', "'")  # Left single quotation mark (U+2018)
        text = text.replace('\u2019', "'")  # Right single quotation mark (U+2019)
        
        return text
    
    def _wrap_caption_text(self, text: str) -> str:
        """
        Process caption text with punctuation normalization only
        Line breaking is handled by OBS directly
        
        Args:
            text: Text to process
            
        Returns:
            Processed text without automatic line breaks
        """
        if not text or not text.strip():
            return text
        
        # Clean and normalize the text (punctuation only)
        text = text.strip()
        text = self._normalize_punctuation(text)
        
        # Return as single line - OBS handles line breaking
        return text
    
    def _wrap_cjk_text(self, text: str, limits: Dict) -> str:
        """
        Wrap CJK text (Chinese, Japanese, Korean)
        CJK characters are wider and don't use spaces between words
        """
        max_chars = limits['max_chars']
        max_lines = limits['max_lines']
        
        if len(text) <= max_chars:
            return text
        
        lines = []
        current_line = ""
        
        for char in text:
            if len(current_line) >= max_chars:
                # Try to break at punctuation if possible
                if char in '，。！？；：、' or current_line.endswith(('，', '。', '！', '？', '；', '：', '、')):
                    lines.append(current_line)
                    current_line = char if char not in '，。！？；：、' else ""
                elif len(lines) < max_lines - 1:
                    lines.append(current_line)
                    current_line = char
                else:
                    # Last line, must fit everything
                    current_line += char
            else:
                current_line += char
        
        if current_line:
            lines.append(current_line)
        
        # Ensure we don't exceed max lines
        if len(lines) > max_lines:
            # Truncate and add ellipsis to last line
            lines = lines[:max_lines]
            if len(lines[-1]) > max_chars - 1:
                lines[-1] = lines[-1][:max_chars-1] + "…"
            else:
                lines[-1] += "…"
        
        return "\n".join(lines)
    
    def _wrap_latin_text(self, text: str, limits: Dict) -> str:
        """
        Wrap Latin text (English, etc.)
        Uses word-based wrapping for better readability
        """
        max_chars = limits['max_chars']
        max_words = limits['max_words']
        max_lines = limits['max_lines']
        
        words = text.split()
        
        # Check word count limit first
        if max_words and len(words) <= max_words:
            # Check if it fits in character limit
            if len(text) <= max_chars:
                return text
        
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed limits
            test_line = current_line + (" " if current_line else "") + word
            
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                # Need to start new line
                if current_line and len(lines) < max_lines - 1:
                    lines.append(current_line)
                    current_line = word
                elif len(lines) < max_lines - 1:
                    current_line = word
                else:
                    # Last line, must fit everything remaining
                    remaining_words = words[words.index(word):]
                    remaining_text = " ".join(remaining_words)
                    
                    if len(current_line + " " + remaining_text) <= max_chars:
                        current_line += " " + remaining_text
                    else:
                        # Truncate with ellipsis
                        available_space = max_chars - len(current_line) - 4  # -4 for " ..."
                        if available_space > 0:
                            truncated = remaining_text[:available_space].rsplit(' ', 1)[0]
                            current_line += " " + truncated + "..."
                        else:
                            current_line = current_line[:max_chars-3] + "..."
                    break
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    def _get_caption_path(self, filename: str) -> str:
        """Get full path for caption file"""
        if config.ENABLE_OBS_CAPTIONS:
            return os.path.join(config.DEFAULT_OUTPUT_DIR, filename)
        else:
            # Return a null path when captions are disabled
            return os.path.join(config.DEFAULT_OUTPUT_DIR, f"disabled_{filename}")
    
    def _is_test_content(self, text: str) -> bool:
        """
        Check if text content is from test functions and should be blocked
        
        Args:
            text: Text to check
            
        Returns:
            True if this appears to be test content that should be blocked
        """
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Common test content patterns
        test_patterns = [
            'raw speech recognition text',
            'processed speech recognition text',
            'test caption',
            'test pipeline',
            'test buffer',
            'testing caption',
            'caption test',
            'hello, this is a live caption test',
            'testing buffer caption functionality',
            'raw asr output example',
            'final test for all caption types'
        ]
        
        for pattern in test_patterns:
            if pattern in text_lower:
                return True
        
        # Check for very generic/template-like content
        if text_lower in ['test text', 'example text', 'sample text', 'placeholder']:
            return True
        
        # Check for obvious test markers
        if any(marker in text_lower for marker in ['test_', '_test', '[test]', '(test)']):
            return True
        
        return False
    
    def _clean_chinese_spaces(self, text: str) -> str:
        """
        Remove unnecessary spaces between Chinese characters from ASR output
        
        Args:
            text: Text that may contain spaced Chinese characters
            
        Returns:
            Text with Chinese character spacing cleaned up
        """
        if not text:
            return text
        
        import re
        
        # Pattern to match Chinese characters with spaces between them
        # This matches: Chinese_char + space + Chinese_char (and continues the pattern)
        chinese_char_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]'
        
        # Remove spaces between Chinese characters
        # This regex finds Chinese char + space + Chinese char and removes the space
        cleaned_text = re.sub(
            rf'({chinese_char_pattern})\s+({chinese_char_pattern})',
            r'\1\2',
            text
        )
        
        # Run multiple passes to handle longer sequences
        # Sometimes we need multiple passes for sequences like "我 们 都 很 好"
        for _ in range(10):  # Max 10 passes should handle any reasonable length
            new_text = re.sub(
                rf'({chinese_char_pattern})\s+({chinese_char_pattern})',
                r'\1\2',
                cleaned_text
            )
            if new_text == cleaned_text:
                break  # No more changes
            cleaned_text = new_text
        
        # Also handle Chinese punctuation spacing
        chinese_punct_pattern = r'[\u3000-\u303f\uff00-\uffef]'
        
        # Remove spaces before/after Chinese punctuation
        cleaned_text = re.sub(
            rf'({chinese_char_pattern})\s+({chinese_punct_pattern})',
            r'\1\2',
            cleaned_text
        )
        cleaned_text = re.sub(
            rf'({chinese_punct_pattern})\s+({chinese_char_pattern})',
            r'\1\2',
            cleaned_text
        )
        
        return cleaned_text.strip()
    
    def _start_caption_clear_timer(self):
        """Start/restart the caption clearing timer"""
        with self.clear_timer_lock:
            # Cancel existing timer
            if self.clear_timer and self.clear_timer.is_alive():
                self.clear_timer.cancel()
            
            # Start new timer
            self.clear_timer = threading.Timer(self.caption_clear_delay, self._clear_caption_on_silence)
            self.clear_timer.daemon = True
            self.clear_timer.start()
    
    def _clear_caption_on_silence(self):
        """Clear live caption after silence period"""
        try:
            if not config.ENABLE_OBS_CAPTIONS:
                return
            
            # Clear the live caption file
            filepath = self._get_caption_path(self.live_caption_file)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("")
            
            print(f"📺 Caption cleared after {self.caption_clear_delay}s silence")
            
        except Exception as e:
            print(f"❌ Caption clear error: {e}")
    
    def _cancel_caption_clear_timer(self):
        """Cancel the caption clearing timer"""
        with self.clear_timer_lock:
            if self.clear_timer and self.clear_timer.is_alive():
                self.clear_timer.cancel()
                self.clear_timer = None
    
    def update_live_caption(self, text: str, source: str = "live") -> bool:
        """
        Update live caption file for OBS with language-aware wrapping
        
        Args:
            text: Caption text to display
            source: Source of the caption (live, buffer, etc.)
            
        Returns:
            True if caption updated successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS:
            print(f"📺 Live Caption: DISABLED - '{text[:50]}...'")
            return False
        
        try:
            # Apply language-aware wrapping
            wrapped_text = self._wrap_caption_text(text)
            
            filepath = self._get_caption_path(self.live_caption_file)
            
            # Write wrapped caption
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(wrapped_text)
            
            # Update last caption time and cancel any existing clear timer
            # (Timer will be started when voice finishes playing)
            self.last_caption_time = time.time()
            self._cancel_caption_clear_timer()
            
            # Add to history with language info
            lang_type = self._detect_text_language(text)
            self.caption_history.append({
                'text': text,
                'wrapped_text': wrapped_text,
                'language_type': lang_type,
                'timestamp': time.time(),
                'source': source
            })
            
            # Keep only last 50 captions in history
            self.caption_history = self.caption_history[-50:]
            
            # Show language-aware logging
            lang_emoji = "🇯" if lang_type == 'cjk' else "🅰️"
            print(f"📺 Live Caption [{source}] {lang_emoji}: '{wrapped_text[:50]}...'")
            return True
            
        except Exception as e:
            print(f"❌ Live Caption Error: {e}")
            return False
    
    def update_buffer_caption(self, text: str) -> bool:
        """
        Update buffer caption file with language-aware wrapping
        Used for showing ongoing transcription
        
        Args:
            text: Buffer text to display
            
        Returns:
            True if buffer updated successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS:
            return False
        
        try:
            # Apply language-aware wrapping
            wrapped_text = self._wrap_caption_text(text)
            
            filepath = self._get_caption_path(self.buffer_caption_file)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(wrapped_text)
            
            lang_type = self._detect_text_language(text)
            lang_emoji = "🇯" if lang_type == 'cjk' else "🅰️"
            print(f"📺 Buffer Caption {lang_emoji}: '{wrapped_text[:30]}...'")
            return True
            
        except Exception as e:
            print(f"❌ Buffer Caption Error: {e}")
            return False
    
    def update_raw_caption(self, text: str) -> bool:
        """
        Update raw caption file and add to input transcription history
        Shows unprocessed ASR output
        
        Args:
            text: Raw text from ASR
            
        Returns:
            True if raw caption updated successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS:
            return False
        
        try:
            filepath = self._get_caption_path(self.raw_caption_file)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Also add to input transcription history
            self.add_input_transcription(text)
            
            print(f"📺 Raw Caption: '{text[:30]}...'")
            return True
            
        except Exception as e:
            print(f"❌ Raw Caption Error: {e}")
            return False
    
    def add_input_transcription(self, text: str, timestamp: float = None) -> bool:
        """
        Add ASR transcription to input history files (both SRT and TXT)
        
        Args:
            text: Raw ASR transcription text
            timestamp: Time relative to session start (auto-calculated if None)
            
        Returns:
            True if added successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS or not text.strip():
            return False
        
        # Check for test contamination protection
        if self._is_test_content(text):
            print(f"🛡️ Blocked test content from input transcription: '{text[:30]}...'")
            return False
        
        try:
            # Clean Chinese character spacing for ASR input
            cleaned_text = self._clean_chinese_spaces(text.strip())
            
            # Calculate timestamp relative to session start
            if timestamp is None:
                timestamp = (datetime.now() - self.session_start_time).total_seconds()
            
            # Add to input SRT entries
            entry_id = len(self.input_srt_entries) + 1
            start_time = timestamp
            end_time = timestamp + 3.0  # 3 second duration
            
            srt_entry = {
                'id': entry_id,
                'start_time': start_time,
                'end_time': end_time,
                'text': cleaned_text,  # Use cleaned text
                'srt_text': f"{entry_id}\n{self._seconds_to_srt_time(start_time)} --> {self._seconds_to_srt_time(end_time)}\n{cleaned_text}\n\n"
            }
            
            self.input_srt_entries.append(srt_entry)
            
            # Write to input SRT file
            self._write_input_srt_file()
            
            # Append to input TXT file with timestamp
            self._append_input_txt_file(cleaned_text, timestamp)
            
            print(f"📝 Input transcription added (cleaned): '{cleaned_text[:30]}...'")
            return True
            
        except Exception as e:
            print(f"❌ Input transcription error: {e}")
            return False
    
    def add_output_transcription(self, text: str, timestamp: float = None) -> bool:
        """
        Add LLM processed transcription to output history files (both SRT and TXT)
        
        Args:
            text: LLM processed transcription text
            timestamp: Time relative to session start (auto-calculated if None)
            
        Returns:
            True if added successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS or not text.strip():
            return False
        
        # Check for test contamination protection
        if self._is_test_content(text):
            print(f"🛡️ Blocked test content from output transcription: '{text[:30]}...'")
            return False
        
        try:
            # Calculate timestamp relative to session start
            if timestamp is None:
                timestamp = (datetime.now() - self.session_start_time).total_seconds()
            
            # Add to output SRT entries (existing method compatibility)
            entry_id = len(self.srt_entries) + 1
            start_time = timestamp
            end_time = timestamp + 3.0  # 3 second duration
            
            srt_entry = {
                'id': entry_id,
                'start_time': start_time,
                'end_time': end_time,
                'text': text.strip(),
                'srt_text': f"{entry_id}\n{self._seconds_to_srt_time(start_time)} --> {self._seconds_to_srt_time(end_time)}\n{text.strip()}\n\n"
            }
            
            self.srt_entries.append(srt_entry)
            
            # Write to output SRT file
            self._write_srt_file()
            
            # Append to output TXT file with timestamp
            self._append_output_txt_file(text, timestamp)
            
            print(f"📝 Output transcription added: '{text[:30]}...'")
            return True
            
        except Exception as e:
            print(f"❌ Output transcription error: {e}")
            return False
    
    def add_srt_entry(self, text: str, start_time: float, duration: float = 3.0) -> bool:
        """
        Add entry to SRT subtitle file with language-aware wrapping
        
        Args:
            text: Subtitle text
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if SRT entry added successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS:
            return False
        
        try:
            # Apply language-aware wrapping for SRT
            wrapped_text = self._wrap_caption_text(text)
            
            end_time = start_time + duration
            
            # Create SRT time format (HH:MM:SS,mmm)
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            # Create SRT entry with wrapped text
            srt_entry = {
                'id': self.current_caption_id,
                'start_time': start_time,
                'end_time': end_time,
                'text': text,
                'wrapped_text': wrapped_text,
                'language_type': self._detect_text_language(text),
                'srt_text': f"{self.current_caption_id}\n{start_srt} --> {end_srt}\n{wrapped_text}\n\n"
            }
            
            self.srt_entries.append(srt_entry)
            self.current_caption_id += 1
            
            # Write to SRT file
            self._write_srt_file()
            
            print(f"📺 SRT Entry: #{srt_entry['id']} '{text[:30]}...'")
            return True
            
        except Exception as e:
            print(f"❌ SRT Entry Error: {e}")
            return False
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _write_srt_file(self):
        """Write complete output SRT file (LLM processed)"""
        try:
            filepath = self._get_caption_path(self.output_srt_file)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write header
                session_time = self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"# S2T2SS Transcription History - LLM Processed Output\n")
                f.write(f"# Session started: {session_time}\n\n")
                
                # Write all entries
                for entry in self.srt_entries:
                    f.write(entry['srt_text'])
            
        except Exception as e:
            print(f"❌ Output SRT Write Error: {e}")
    
    def _write_input_srt_file(self):
        """Write complete input SRT file (ASR raw)"""
        try:
            filepath = self._get_caption_path(self.input_srt_file)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write header
                session_time = self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"# S2T2SS Transcription History - ASR Raw Input\n")
                f.write(f"# Session started: {session_time}\n\n")
                
                # Write all entries
                for entry in self.input_srt_entries:
                    f.write(entry['srt_text'])
            
        except Exception as e:
            print(f"❌ Input SRT Write Error: {e}")
    
    def _append_input_txt_file(self, text: str, timestamp: float):
        """Append to input TXT file with timestamp"""
        try:
            # Check for test contamination protection
            if self._is_test_content(text):
                print(f"🛡️ Blocked test content from input history: '{text[:30]}...'")
                return
            
            filepath = self._get_caption_path(self.input_txt_file)
            
            # Format timestamp as MM:SS
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{time_str}] {text.strip()}\n")
            
        except Exception as e:
            print(f"❌ Input TXT Write Error: {e}")
    
    def _append_output_txt_file(self, text: str, timestamp: float):
        """Append to output TXT file with timestamp"""
        try:
            # Check for test contamination protection
            if self._is_test_content(text):
                print(f"🛡️ Blocked test content from output history: '{text[:30]}...'")
                return
            
            filepath = self._get_caption_path(self.output_txt_file)
            
            # Format timestamp as MM:SS
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{time_str}] {text.strip()}\n")
            
        except Exception as e:
            print(f"❌ Output TXT Write Error: {e}")
    
    def update_output_text(self, text: str) -> bool:
        """
        Update main output text file and add to output transcription history
        
        Args:
            text: Final processed text
            
        Returns:
            True if output updated successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS:
            return False
        
        try:
            # Legacy: Append to output file with timestamp for backward compatibility
            filepath = self._get_caption_path(self.output_text_file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {text}\n")
            
            # Also add to output transcription history
            self.add_output_transcription(text)
            
            print(f"📺 Output Text: '{text[:50]}...'")
            return True
            
        except Exception as e:
            print(f"❌ Output Text Error: {e}")
            return False
    
    def clear_live_caption_immediately(self) -> bool:
        """Immediately clear live caption (useful for manual control)"""
        try:
            if not config.ENABLE_OBS_CAPTIONS:
                return False
            
            # Cancel any pending clear timer
            self._cancel_caption_clear_timer()
            
            # Clear the caption file
            filepath = self._get_caption_path(self.live_caption_file)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("")
            
            print(f"📺 Live caption cleared manually")
            return True
            
        except Exception as e:
            print(f"❌ Manual caption clear error: {e}")
            return False
    
    def clear_captions(self) -> bool:
        """Clear all caption files"""
        
        if not config.ENABLE_OBS_CAPTIONS:
            print("📺 Clear Captions: DISABLED")
            return False
        
        try:
            caption_files = [
                self.live_caption_file,
                self.buffer_caption_file,
                self.raw_caption_file
            ]
            
            for filename in caption_files:
                filepath = self._get_caption_path(filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("")
            
            print("📺 Caption files cleared")
            return True
            
        except Exception as e:
            print(f"❌ Clear Captions Error: {e}")
            return False
    
    def process_caption_pipeline(self, raw_text: str, processed_text: str, 
                                start_time: float = None) -> bool:
        """
        Complete caption processing pipeline
        
        Args:
            raw_text: Raw ASR output
            processed_text: LLM-processed text
            start_time: Start time for SRT entry
            
        Returns:
            True if pipeline completed successfully
        """
        
        if not config.ENABLE_OBS_CAPTIONS:
            print(f"📺 Caption Pipeline: DISABLED")
            return False
        
        if start_time is None:
            start_time = time.time()
        
        success = True
        
        # Update raw caption
        success &= self.update_raw_caption(raw_text)
        
        # Update live caption with processed text
        success &= self.update_live_caption(processed_text, "processed")
        
        # Add to SRT file
        success &= self.add_srt_entry(processed_text, start_time)
        
        # Update output text
        success &= self.update_output_text(processed_text)
        
        return success
    
    def get_status(self) -> Dict[str, Any]:
        """Get current caption manager status"""
        with self.clear_timer_lock:
            timer_active = self.clear_timer and self.clear_timer.is_alive()
            time_since_last = time.time() - self.last_caption_time if self.last_caption_time > 0 else 0
        
        return {
            "enabled": config.ENABLE_OBS_CAPTIONS,
            "output_directory": config.DEFAULT_OUTPUT_DIR,
            "caption_history_count": len(self.caption_history),
            "srt_entries_count": len(self.srt_entries),
            "current_caption_id": self.current_caption_id,
            "caption_clear_delay": self.caption_clear_delay,
            "clear_timer_active": timer_active,
            "time_since_last_caption": time_since_last,
            "will_clear_in": max(0, self.caption_clear_delay - time_since_last) if timer_active else 0,
            "language_limits": self.caption_limits,
            "transcription_history": {
                "input_entries": len(self.input_srt_entries),
                "output_entries": len(self.srt_entries),
                "session_duration": (datetime.now() - self.session_start_time).total_seconds()
            },
            "files": {
                "live_caption": self.live_caption_file,
                "buffer_caption": self.buffer_caption_file,
                "raw_caption": self.raw_caption_file,
                "input_srt": self.input_srt_file,
                "input_txt": self.input_txt_file,
                "output_srt": self.output_srt_file,
                "output_txt": self.output_txt_file,
                # Legacy compatibility
                "srt_file": self.srt_file,
                "output_text": self.output_text_file
            }
        }
    
    def get_transcription_summary(self) -> Dict[str, Any]:
        """Get a summary of the current transcription session"""
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            "session_start": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration_minutes": round(session_duration / 60, 1),
            "input_transcriptions": len(self.input_srt_entries),
            "output_transcriptions": len(self.srt_entries),
            "files_created": {
                "input_srt": self._get_caption_path(self.input_srt_file),
                "input_txt": self._get_caption_path(self.input_txt_file),
                "output_srt": self._get_caption_path(self.output_srt_file),
                "output_txt": self._get_caption_path(self.output_txt_file)
            }
        }
    
    def cleanup(self):
        """Clean up caption manager"""
        try:
            # Cancel caption clear timer
            self._cancel_caption_clear_timer()
            
            if config.ENABLE_OBS_CAPTIONS:
                self.clear_captions()
            print("📺 Caption Manager cleaned up")
        except Exception as e:
            print(f"⚠️ Caption cleanup error: {e}")

def test_caption_manager():
    """Test the caption manager module"""
    print("🧪 Testing Caption Manager Module")
    print("-" * 40)
    
    # Initialize manager
    captions = CaptionManager()
    
    # Test individual caption functions
    test_texts = [
        ("Hello, this is a live caption test.", "This is the processed version."),
        ("Testing buffer caption functionality.", "Buffer test processed."),
        ("Raw ASR output example.", "Cleaned ASR output."),
        ("Final test for all caption types.", "Final processed text.")
    ]
    
    start_time = time.time()
    
    for i, (raw_text, processed_text) in enumerate(test_texts, 1):
        print(f"\n📺 Test {i}:")
        print(f"   Raw: '{raw_text}'")
        print(f"   Processed: '{processed_text}'")
        
        # Test individual functions
        success_raw = captions.update_raw_caption(raw_text)
        success_live = captions.update_live_caption(processed_text, f"test_{i}")
        success_srt = captions.add_srt_entry(processed_text, start_time + i, 3.0)
        success_output = captions.update_output_text(processed_text)
        
        print(f"   Raw Caption: {'✅' if success_raw else '❌'}")
        print(f"   Live Caption: {'✅' if success_live else '❌'}")
        print(f"   SRT Entry: {'✅' if success_srt else '❌'}")
        print(f"   Output Text: {'✅' if success_output else '❌'}")
    
    # Test complete pipeline
    print(f"\n🔄 Testing complete pipeline...")
    pipeline_success = captions.process_caption_pipeline(
        "Raw pipeline test text",
        "Processed pipeline test text",
        start_time + 10
    )
    print(f"Pipeline: {'✅ Success' if pipeline_success else '❌ Failed'}")
    
    # Test buffer caption
    print(f"\n📝 Testing buffer caption...")
    buffer_success = captions.update_buffer_caption("This is a buffer test...")
    print(f"Buffer: {'✅ Success' if buffer_success else '❌ Failed'}")
    
    # Show status
    status = captions.get_status()
    print(f"\n📊 Caption Manager Status:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"      {subkey}: {subvalue}")
        else:
            print(f"   {key}: {value}")
    
    # Cleanup
    captions.cleanup()

if __name__ == "__main__":
    print("📺 S2T2SS Caption Manager Module")
    print("=" * 50)
    
    test_caption_manager()
