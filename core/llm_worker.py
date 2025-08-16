#!/usr/bin/env python3
"""
S2T2SS LLM Worker Module
Handles text processing and refinement using language models
"""

import sys
import os
import requests
import json
import time
from typing import Optional, Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

class LLMWorker:
    """
    LLM Worker for text processing and refinement
    Supports local LLM servers (LM Studio, Ollama, etc.)
    """
    
    def __init__(self, base_url: str = "http://localhost:1234", model_name: str = None):
        """Initialize LLM worker"""
        self.base_url = base_url.rstrip('/')
        # Use configured model name or fall back to auto-detection
        configured_model = model_name or config.LLM_MODEL_NAME
        
        # Smart model handling based on server type
        if configured_model:
            # Check if the configured model makes sense for this server
            is_ollama_server = ":11434" in self.base_url
            is_ollama_model = any(x in configured_model.lower() for x in ["llama", "mistral", "qwen", "l32", "l21"])
            
            # If server and model don't match, clear model for auto-detection
            if is_ollama_server != is_ollama_model:
                print(f"[LLM] Model '{configured_model}' doesn't match server type, using auto-detection")
                self.model_name = None
            else:
                self.model_name = configured_model
        else:
            self.model_name = None
            
        self.connection_tested = False
        self.recent_english_texts = []  # For bilingual handling
        
        try:
            print(f"[LLM] Worker initialized with server: {base_url}")
            if self.model_name:
                print(f"[LLM] Using configured model: {self.model_name}")
            else:
                print(f"[LLM] Model auto-detection enabled")
        except UnicodeEncodeError:
            print("[LLM] Worker initialized with server:", base_url)
    
    def test_connection(self) -> bool:
        """Test connection to LLM server"""
        try:
            # If we already have a configured model name, test with that
            if self.model_name:
                # For Ollama, try both the OpenAI-compatible API and native API
                if ":11434" in self.base_url:
                    # Try Ollama's native API first (more reliable for local models)
                    try:
                        test_response = requests.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": "Hello",
                                "stream": False,
                                "options": {"num_predict": 1}
                            },
                            timeout=10
                        )
                        
                        if test_response.status_code == 200:
                            self.connection_tested = True
                            print(f"[LLM] Ollama server connected with model (native API): {self.model_name}")
                            return True
                        else:
                            print(f"⚠️ Ollama native API failed. Status: {test_response.status_code}")
                    except Exception as e:
                        print(f"⚠️ Ollama native API error: {e}")
                
                # Try OpenAI-compatible API (fallback)
                test_response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 1,
                        "temperature": 0.1
                    },
                    timeout=10
                )
                
                if test_response.status_code == 200:
                    self.connection_tested = True
                    try:
                        server_type = "Ollama" if ":11434" in self.base_url else "LM Studio"
                        print(f"[LLM] {server_type} server connected with model (OpenAI API): {self.model_name}")
                    except UnicodeEncodeError:
                        print(f"[LLM] Server connected with model: {self.model_name}")
                    return True
                else:
                    print(f"⚠️ Model '{self.model_name}' not responding on OpenAI API. Status: {test_response.status_code}")
                    # Print response for debugging
                    try:
                        print(f"⚠️ Response: {test_response.text[:200]}...")
                    except:
                        pass
                    # Fall back to auto-detection
                    print(f"[LLM] Falling back to auto-detection...")
            
            # Auto-detection fallback (original logic)
            response = requests.get(
                f"{self.base_url}/v1/models", 
                timeout=5
            )
            
            if response.status_code == 200:
                models_data = response.json()
                # Handle both LM Studio format (data array) and Ollama format (models array)
                if models_data.get('data'):
                    # LM Studio format: {"data": [{"id": "model-name"}]}
                    detected_model = models_data['data'][0].get('id', 'unknown')
                elif models_data.get('models'):
                    # Ollama format: {"models": [{"name": "model-name"}]}
                    detected_model = models_data['models'][0].get('name', 'llama3.2:latest')
                    print(f"[LLM] Available models: {[m.get('name') for m in models_data['models'][:3]]}...")
                else:
                    # Fallback
                    detected_model = "llama3.2:latest" if ":11434" in self.base_url else "local-model"
                
                # Use detected model if we don't have a configured one
                if not self.model_name:
                    self.model_name = detected_model
                
                self.connection_tested = True
                try:
                    # Detect server type for logging
                    server_type = "Ollama" if ":11434" in self.base_url else "LM Studio"
                    print(f"[LLM] {server_type} server connected: {self.model_name}")
                except UnicodeEncodeError:
                    print(f"[LLM] Server connected: {self.model_name}")
                return True
            
            print(f"⚠️ LLM Server response error: {response.status_code}")
            return False
            
        except requests.exceptions.RequestException as e:
            print(f"❌ LLM Server connection failed: {e}")
            return False
    
    def ensure_connection(self) -> bool:
        """Ensure we have a valid connection"""
        if not self.connection_tested:
            return self.test_connection()
        return True
    
    def process_text(self, text: str, mode: str = "refine") -> str:
        """
        Process text using LLM with Chinese text segmentation
        
        Args:
            text: Input text to process
            mode: Processing mode ("refine", "correct", "enhance")
            
        Returns:
            Processed text or original text if LLM disabled/failed
        """
        
        # Check if LLM is enabled
        if not config.ENABLE_LLM_EDITING:
            print(f"🧠 LLM Processing: DISABLED (returning raw text)")
            return text
        
        # Skip processing for very short text
        if len(text.strip()) < 3:
            return text
        
        # Apply Chinese text segmentation first
        segmented_text = self._apply_chinese_segmentation(text)
        
        # Check if translation is needed based on OUTPUT_LANGUAGE setting
        needs_translation = self._needs_translation(segmented_text)
        
        # Check if narrator mode requires LLM processing (override fast mode)
        requires_llm_processing = (
            hasattr(config, 'NARRATOR_MODE') and 
            config.NARRATOR_MODE not in ["direct"] and  # Only "direct" mode can skip LLM
            config.NARRATOR_MODE != "direct"
        )
        
        # Fast processing mode: Use segmentation only for speed (unless translation or narrator mode needed)
        if (hasattr(config, 'FAST_PROCESSING_MODE') and 
            config.FAST_PROCESSING_MODE and 
            not needs_translation and 
            not requires_llm_processing):
            print(f"🧠 Fast Mode: Using segmentation only ('{segmented_text[:50]}...')")
            return segmented_text
        
        # Check connection (needed for translation or full processing)
        if not self.ensure_connection():
            print(f"🧠 LLM Processing: CONNECTION FAILED (returning segmented text)")
            return segmented_text
        
        # Handle translation if needed
        if needs_translation:
            return self._translate_text(segmented_text)
        
        try:
            # Build prompt based on mode (using segmented text)
            raw_prompt = self._build_prompt(segmented_text, mode)
            
            # Sanitize prompt for LM Studio if needed
            if ":1234" in self.base_url:  # LM Studio port
                prompt = self._sanitize_text_for_lm_studio(raw_prompt)
            else:
                prompt = raw_prompt
            
            # Detect language for system message
            is_chinese = any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in segmented_text)
            
            if is_chinese:
                system_msg = "你是一个专业的文本编辑助手，负责改进语音转文字的输出。重要：始终保持中文输出，不要翻译成其他语言。"
            else:
                system_msg = "You are a helpful text editor that improves speech-to-text output. CRITICAL: Always preserve the original language - do not translate to other languages."
            
            # Make API request - try OpenAI-compatible first, then Ollama native
            response = None
            
            # Try OpenAI-compatible API first
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_name or "local-model",
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,  # Lower temperature for faster, more consistent processing
                        "max_tokens": 200,   # Reduced token limit for faster responses
                        "stream": False
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    processed_text = result['choices'][0]['message']['content'].strip()
                    
                    # Clean up common LLM artifacts
                    processed_text = self._clean_llm_output(processed_text)
                    
                    # Clean text for better TTS quality
                    processed_text = self._clean_text_for_tts(processed_text)
                    
                    print(f"🧠 LLM Processed (OpenAI API): '{segmented_text[:50]}' → '{processed_text[:50]}'")
                    return processed_text
                else:
                    print(f"⚠️ OpenAI API Error {response.status_code}")
                    if ":11434" not in self.base_url:
                        print(f"⚠️ Response: {response.text}")
                        return segmented_text
            
            except Exception as e:
                print(f"⚠️ OpenAI API failed: {e}")
                if ":11434" not in self.base_url:
                    return segmented_text
            
            # Try Ollama native API if OpenAI API failed and this is Ollama
            if ":11434" in self.base_url:
                try:
                    # Combine system message and user prompt for Ollama
                    combined_prompt = f"{system_msg}\n\nUser: {prompt}\nAssistant:"
                    
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": self.model_name or "llama3.2:latest",
                            "prompt": combined_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 200
                            }
                        },
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        processed_text = result.get('response', '').strip()
                        
                        if processed_text:
                            # Clean up common LLM artifacts
                            processed_text = self._clean_llm_output(processed_text)
                            
                            # Clean text for better TTS quality
                            processed_text = self._clean_text_for_tts(processed_text)
                            
                            print(f"🧠 LLM Processed (Ollama API): '{segmented_text[:50]}' → '{processed_text[:50]}'")
                            return processed_text
                    
                    print(f"⚠️ Ollama native API Error {response.status_code}: {response.text[:100]}...")
                    
                except Exception as e:
                    print(f"⚠️ Ollama native API failed: {e}")
            
            return segmented_text
                
        except Exception as e:
            print(f"❌ LLM Processing Error: {e}")
            return segmented_text
    
    def _apply_chinese_segmentation(self, text: str) -> str:
        """Apply Chinese text segmentation and basic formatting"""
        import re
        
        # Remove extra spaces between Chinese characters
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        
        # Add punctuation for natural pauses
        # Add commas for "的" "了" "呢" "啊" patterns
        text = re.sub(r'(的|了|呢|啊)(?=[\u4e00-\u9fff])', r'\1，', text)
        
        # Add periods at natural sentence endings
        text = re.sub(r'(吧|好)$', r'\1。', text)
        text = re.sub(r'(什么|怎么|为什么)$', r'\1？', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[，。？！]{2,}', '，', text)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _needs_translation(self, text: str) -> bool:
        """Check if translation is needed based on OUTPUT_LANGUAGE setting"""
        if not hasattr(config, 'OUTPUT_LANGUAGE') or config.OUTPUT_LANGUAGE == "auto":
            return False
        
        # Detect input language
        is_chinese = any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in text)
        is_english = self._is_likely_english(text)
        
        target_lang = config.OUTPUT_LANGUAGE
        
        # Check if translation is needed
        if is_chinese and target_lang in ["en", "english"]:
            print(f"🌐 Translation needed: Chinese → English")
            return True
        elif is_english and target_lang in ["zh", "chinese", "zh-cn"]:
            print(f"🌐 Translation needed: English → Chinese")
            return True
        elif target_lang not in ["auto", "zh", "en", "chinese", "english", "zh-cn"]:
            print(f"🌐 Translation needed: Auto-detected → {target_lang}")
            return True
        
        return False
    
    def _translate_text(self, text: str) -> str:
        """Translate text to target language using LLM"""
        try:
            target_lang = config.OUTPUT_LANGUAGE
            
            # Map language codes to full names
            lang_names = {
                "en": "English",
                "english": "English", 
                "zh": "Chinese",
                "chinese": "Chinese",
                "zh-cn": "Chinese",
                "ja": "Japanese",
                "ko": "Korean",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "ru": "Russian"
            }
            
            target_name = lang_names.get(target_lang, target_lang.title())
            
            # Sanitize text for LM Studio to avoid Jinja template issues
            if ":1234" in self.base_url:  # LM Studio port
                sanitized_text = self._sanitize_text_for_lm_studio(text)
            else:
                sanitized_text = text
            
            # Build translation prompt - much more directive to avoid explanations
            prompt = f"{sanitized_text}"
            
            # Make API request - try OpenAI-compatible first, then Ollama native
            response = None
            
            # Try OpenAI-compatible API first
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_name or "local-model",
                        "messages": [
                            {"role": "system", "content": f"Translate the following text to {target_name}. Output ONLY the {target_name} translation with no explanations, prefixes, or meta-commentary."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 100,
                        "stream": False
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translated_text = result['choices'][0]['message']['content'].strip()
                    
                    # Clean up common LLM artifacts
                    translated_text = self._clean_llm_output(translated_text)
                    
                    # Clean text for better TTS quality
                    translated_text = self._clean_text_for_tts(translated_text)
                    
                    print(f"🌐 Translated (OpenAI API): '{text[:30]}...' → '{translated_text[:30]}...'")
                    return translated_text
                else:
                    print(f"⚠️ Translation OpenAI API Error {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"⚠️ Error details: {error_detail}")
                    except:
                        print(f"⚠️ Response text: {response.text[:200]}")
                    if ":11434" not in self.base_url:
                        return text
            
            except Exception as e:
                print(f"⚠️ Translation OpenAI API failed: {e}")
                if ":11434" not in self.base_url:
                    return text
            
            # Try Ollama native API if OpenAI API failed and this is Ollama
            if ":11434" in self.base_url:
                try:
                    # Combine system message and user prompt for Ollama
                    system_msg = f"Translate the following text to {target_name}. Output ONLY the {target_name} translation with no explanations, prefixes, or meta-commentary."
                    combined_prompt = f"{system_msg}\n\nUser: {prompt}\nAssistant:"
                    
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": self.model_name or "llama3.2:latest",
                            "prompt": combined_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 100
                            }
                        },
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        translated_text = result.get('response', '').strip()
                        
                        if translated_text:
                            # Clean up common LLM artifacts
                            translated_text = self._clean_llm_output(translated_text)
                            
                            # Clean text for better TTS quality
                            translated_text = self._clean_text_for_tts(translated_text)
                            
                            print(f"🌐 Translated (Ollama API): '{text[:30]}...' → '{translated_text[:30]}...'")
                            return translated_text
                    
                    print(f"⚠️ Translation Ollama API Error {response.status_code}")
                    
                except Exception as e:
                    print(f"⚠️ Translation Ollama API failed: {e}")
            
            return text
                
        except Exception as e:
            print(f"❌ Translation Error: {e}")
            return text
    
    def _build_prompt(self, text: str, mode: str) -> str:
        """Build appropriate prompt for the processing mode"""
        
        # Detect the input language to preserve it
        is_chinese = any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in text)
        is_english = self._is_likely_english(text)
        
        # Check if narrator mode is enabled and override the processing mode
        narrator_mode = getattr(config, 'NARRATOR_MODE', 'direct')
        
        # Apply narrator mode transformations
        if narrator_mode == "first_person":
            if is_chinese:
                return f"""请将这段语音转文字的输出转换为第一人称视角，要求：
1. 将第三人称的表述转换为第一人称
2. 保持原意和语调
3. 添加适当的标点符号
4. 保持中文输出，不要翻译成其他语言
5. 让文本读起来像是说话者本人在叙述

原始文本："{text}"

只返回转换后的第一人称中文文本，不要解释。"""
            else:
                return f"""Convert this speech-to-text output to first-person perspective:
1. Change third-person references to first-person
2. Preserve the original meaning and tone
3. Add proper punctuation
4. IMPORTANT: Keep the SAME LANGUAGE as the input (do not translate)
5. Make it sound like the speaker is narrating their own experience

Original text: "{text}"

Return only the first-person converted text in the SAME LANGUAGE, no explanations."""

        elif narrator_mode == "casual_narrator":
            if is_chinese:
                return f"""请将这段语音转文字的输出转换为轻松的叙述风格，要求：
1. 使用更轻松、自然的表达方式
2. 添加适当的语气词和连接词
3. 保持原意但让语言更生动
4. 保持中文输出，不要翻译成其他语言
5. 让文本读起来更有趣和引人入胜

原始文本："{text}"

只返回转换后的轻松叙述风格中文文本，不要解释。"""
            else:
                return f"""Convert this speech-to-text output to casual storytelling style:
1. Use more relaxed, natural expressions
2. Add appropriate transitional words and fillers
3. Preserve meaning but make the language more engaging
4. IMPORTANT: Keep the SAME LANGUAGE as the input (do not translate)
5. Make it sound more interesting and conversational

Original text: "{text}"

Return only the casual storytelling version in the SAME LANGUAGE, no explanations."""

        elif narrator_mode == "professional_narrator":
            if is_chinese:
                return f"""请将这段语音转文字的输出转换为专业的叙述风格，要求：
1. 使用更正式、专业的表达方式
2. 改善语法结构和用词
3. 保持原意但提升语言的专业性
4. 保持中文输出，不要翻译成其他语言
5. 让文本读起来更权威和专业

原始文本："{text}"

只返回转换后的专业叙述风格中文文本，不要解释。"""
            else:
                return f"""Convert this speech-to-text output to professional narration style:
1. Use more formal, professional expressions
2. Improve grammar structure and vocabulary
3. Preserve meaning but elevate the language quality
4. IMPORTANT: Keep the SAME LANGUAGE as the input (do not translate)
5. Make it sound more authoritative and polished

Original text: "{text}"

Return only the professional narration version in the SAME LANGUAGE, no explanations."""

        elif narrator_mode == "adaptive":
            if is_chinese:
                return f"""请智能分析并改进这段语音转文字的输出，要求：
1. 根据内容类型自动选择最佳的叙述风格
2. 如果是对话，保持对话风格；如果是描述，使用叙述风格
3. 纠正转录错误并添加标点符号
4. 保持中文输出，不要翻译成其他语言
5. 让文本自然流畅，符合内容特点

原始文本："{text}"

只返回智能优化后的中文文本，不要解释。"""
            else:
                return f"""Intelligently analyze and improve this speech-to-text output:
1. Automatically choose the best narration style based on content type
2. If dialogue, keep conversational; if description, use narrative style
3. Correct transcription errors and add punctuation
4. IMPORTANT: Keep the SAME LANGUAGE as the input (do not translate)
5. Make the text natural and appropriate for its content

Original text: "{text}"

Return only the intelligently optimized text in the SAME LANGUAGE, no explanations."""

        # Fall back to the original mode-based prompts for "direct" mode or standard processing modes
        if mode == "refine":
            if is_chinese:
                return f"""纠正这段语音转文字的错误，添加标点符号，改善语法：

"{text}"

要求：
- 只返回修正后的文本
- 不要添加任何解释、评论或前缀
- 不要说"这句话可以..."之类的话
- 保持原意不变
- 保持中文

修正后的文本："""
            else:
                return f"""Fix this speech-to-text output:

"{text}"

Requirements:
- Return ONLY the corrected text
- No explanations, comments, or prefixes  
- Do not say "This can be..." or similar
- Keep original meaning
- Same language as input

Corrected text:"""

        elif mode == "correct":
            if is_chinese:
                return f"""纠正这段语音转文字的错误：

"{text}"

要求：
- 只返回纠正后的文本
- 不要添加解释或评论
- 保持原意不变
- 保持中文

纠正后的文本："""
            else:
                return f"""Correct this speech-to-text output:

"{text}"

Requirements:
- Return ONLY the corrected text
- No explanations or comments
- Keep original meaning
- Same language as input

Corrected text:"""

        elif mode == "enhance":
            if is_chinese:
                return f"""请增强这段语音转文字的输出：
- 提高清晰度和可读性
- 添加适当的标点符号和大小写
- 改善语法，保持原意
- 保持说话者的原意和语调
- 保持中文输出

文本："{text}"

只返回增强后的中文文本。"""
            else:
                return f"""Please enhance this speech-to-text output:
- Improve clarity and readability
- Add proper punctuation and capitalization
- Fix grammar while keeping the original meaning
- Maintain the speaker's intended tone
- IMPORTANT: Keep the SAME LANGUAGE as input

Text: "{text}"

Return only the enhanced text in the SAME LANGUAGE."""

        else:
            # Generic fallback
            if is_chinese:
                return f'请改进这段文本，保持中文输出："{text}"'
            else:
                return f'Please improve this text in the SAME LANGUAGE: "{text}"'
    
    def _clean_llm_output(self, text: str) -> str:
        """Clean up common LLM output artifacts and verbose responses"""
        
        # For translations, extract just the translated text if there's explanatory content
        lines = text.split('\n')
        
        # If there are multiple lines, try to find the actual translation
        if len(lines) > 1:
            # Look for lines that start with quotes (likely the translation)
            for line in lines:
                line = line.strip()
                if line.startswith('"') and line.endswith('"') and len(line) > 10:
                    text = line
                    break
            # If no quoted line found, use the first substantial line
            else:
                for line in lines:
                    line = line.strip()
                    if len(line) > 5 and not line.startswith('(') and not line.lower().startswith('note:'):
                        text = line
                        break
        
        # Remove quotes if the entire response is quoted
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Remove common prefixes and explanatory text
        prefixes_to_remove = [
            "Here's the improved text: ",
            "Improved text: ",
            "Corrected text: ",
            "Enhanced text: ",
            "Here is the corrected version: ",
            "Here is the improved version: ",
            "Translation: ",
            "Here's the translation: ",
            "The translation is: "
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        
        # Remove explanatory content in parentheses at the end
        if '(' in text and text.strip().endswith(')'):
            # Find the last opening parenthesis
            last_paren = text.rfind('(')
            if last_paren > 0:
                before_paren = text[:last_paren].strip()
                if len(before_paren) > 5:  # Keep if there's substantial content before
                    text = before_paren
        
        # Truncate if response is too long (likely verbose explanation)
        if len(text) > 200:
            # Try to find a sentence break
            for punct in ['. ', '! ', '? ']:
                idx = text.find(punct)
                if 10 < idx < 150:  # Reasonable sentence length
                    text = text[:idx + 1]
                    break
            else:
                # No sentence break found, just truncate
                text = text[:150].strip()
        
        return text.strip()
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text to improve TTS quality and reduce artifacts"""
        import re
        
        original_text = text
        cleaned = text
        
        # Remove or fix elongated sounds that cause TTS artifacts
        # Fix extended vowels (ummmmm, ahhhhh, etc.)
        cleaned = re.sub(r'([aeiouAEIOU])\1{2,}', r'\1', cleaned)
        
        # Fix extended consonants (nnnnno, sssso, etc.) 
        cleaned = re.sub(r'([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])\1{2,}', r'\1', cleaned)
        
        # Fix stuttering patterns (i...i...it, w...w...what)
        cleaned = re.sub(r'(\w)\.{2,}\1\.{2,}(\w+)', r'\1\2', cleaned)
        cleaned = re.sub(r'(\w)-{2,}\1-{2,}(\w+)', r'\1\2', cleaned)
        
        # Normalize ellipses - this model loves to write 4+ dots, fix them!
        cleaned = re.sub(r'\.{4,}', '...', cleaned)    # 4+ dots -> 3 dots
        cleaned = re.sub(r'\.{3,}', '...', cleaned)    # Ensure max 3 dots
        cleaned = re.sub(r'…{2,}', '...', cleaned)     # Multiple unicode ellipses -> 3 dots
        cleaned = re.sub(r'…', '...', cleaned)         # Single unicode ellipsis -> 3 dots
        
        # Remove excessive dashes that cause pauses  
        cleaned = re.sub(r'-{3,}', '--', cleaned)    # Max 2 dashes
        
        # Remove or replace filler sounds
        filler_patterns = {
            r'\bumm+\b': 'um',           # ummmmm -> um
            r'\bahh+\b': 'ah',           # ahhhhh -> ah  
            r'\buhh+\b': 'uh',           # uhhhh -> uh
            r'\berr+\b': 'er',           # errrr -> er
            r'\bhm+\b': 'hmm',           # hmmmm -> hmm
        }
        
        for pattern, replacement in filler_patterns.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # TTS-specific fixes for better pronunciation
        # Remove or replace punctuation that TTS tries to voice
        print(f"🔧 TTS cleaning input: '{cleaned}'")
        tts_fixes = {
            # Remove ellipses completely (TTS often says "dot dot dot")
            r'\.{3}': '',                # Exactly 3 dots -> (nothing)
            r'\.{4,}': '',               # 4+ dots -> (nothing)
            r'…': '',                    # Unicode ellipsis -> (nothing)
            
            # Remove other problematic punctuation
        #    r'--+': ' ',                 # Dashes -> space
        #    r'—': ' ',                   # Em dash -> space  
        #    r'–': ' ',                   # En dash -> space
        #    r'\*+': '',                  # Asterisks -> (nothing)
        #    r'#+': '',                   # Hash symbols -> (nothing)
        #    r'&': ' and ',               # Ampersand -> "and"
            
            # Remove excessive punctuation that causes weird pauses
        #    r'[!]{2,}': '!',             # Multiple exclamations
        #    r'[?]{2,}': '?',             # Multiple questions
        #    r'[,]{2,}': ',',             # Multiple commas
        #    r'[;]{2,}': ';',             # Multiple semicolons
            
            # Remove quotation marks that TTS often mispronounces
        #    r'["""]': '',                # Curly quotes
        #    r"[''']": '',                # Curly apostrophes
        #    r'"': '',                    # Straight quotes
        #    r"'": '',                    # Straight apostrophes
            
            # Fix spacing around punctuation
        #    r'\s*([.!?;:,])\s*': r'\1 ', # Normalize punctuation spacing
            
            # Remove parenthetical expressions that confuse TTS
        #    r'\([^)]*\)': '',            # Remove anything in parentheses
        #    r'\[[^\]]*\]': '',           # Remove anything in square brackets
        #    r'\{[^}]*\}': '',            # Remove anything in curly braces
            
            # Fix common abbreviations that TTS mispronounces
        #    r'\bvs\.?\b': 'versus',      # vs -> versus
        #    r'\betc\.?\b': 'etcetera',   # etc -> etcetera
        #    r'\bi\.e\.?\b': 'that is',   # i.e. -> that is
        #    r'\be\.g\.?\b': 'for example', # e.g. -> for example
        #    r'\bw\/\b': 'with',          # w/ -> with
        #    r'\b&\b': 'and',             # & -> and
        }
        
        for pattern, replacement in tts_fixes.items():
            before_fix = cleaned
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            if before_fix != cleaned:
                print(f"🔧 TTS Fix applied: '{before_fix}' → '{cleaned}' (pattern: {pattern})")
        
        # Clean up spacing issues
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces -> single space
        
        # Remove leading/trailing spaces
        cleaned = cleaned.strip()
        
        # Chinese-specific TTS fixes for proper sentence ending
        is_chinese = any(ord(char) >= 0x4e00 and ord(char) <= 0x9fff for char in cleaned)
        
        if is_chinese:
            # For Chinese text, don't add period after ellipses - it causes pronunciation issues
            if cleaned and cleaned.endswith('...'):
                # Chinese ellipses should not have additional punctuation
                pass  # Keep as is
            elif cleaned and not cleaned[-1] in '。！？...':
                # Add Chinese period only if no punctuation
                cleaned += '。'
        else:
            # English text handling
            if cleaned and not cleaned[-1] in '.!?':
                cleaned += '.'
        
        # Debug logging to track the cleaning process
        if original_text != cleaned:
            print(f"🔧 Text cleaned: '{original_text}' → '{cleaned}'")
        
        return cleaned
    
    def _sanitize_text_for_lm_studio(self, text: str) -> str:
        """Sanitize text to avoid Jinja template parsing issues in LM Studio"""
        
        # Simple, safe sanitization
        sanitized = text
        
        # Only replace the most problematic characters
        sanitized = sanitized.replace('[', '(')
        sanitized = sanitized.replace(']', ')')
        sanitized = sanitized.replace('{', '(')
        sanitized = sanitized.replace('}', ')')
        
        return sanitized.strip()
    
    def process_bilingual_text(self, text: str, detected_language: str = "auto") -> str:
        """
        Handle bilingual text processing
        Track recent English texts for context
        """
        
        if not config.ENABLE_LLM_EDITING:
            return text
        
        # Detect if this might be English
        is_english = self._is_likely_english(text)
        
        if is_english:
            # Add to recent English texts
            self.recent_english_texts.append(text)
            # Keep only last 5 texts
            self.recent_english_texts = self.recent_english_texts[-5:]
            
            # Process with English-specific refinement
            return self.process_text(text, mode="refine")
        else:
            # Process with general refinement
            return self.process_text(text, mode="refine")
    
    def _is_likely_english(self, text: str) -> bool:
        """Simple heuristic to detect if text is likely English"""
        
        # Common English words
        english_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }
        
        words = text.lower().split()
        if not words:
            return False
        
        english_count = sum(1 for word in words if word.strip('.,!?;:') in english_words)
        return (english_count / len(words)) > 0.3
    
    def get_status(self) -> Dict[str, Any]:
        """Get current LLM worker status"""
        return {
            "enabled": config.ENABLE_LLM_EDITING,
            "server_url": self.base_url,
            "model_name": self.model_name,
            "connected": self.connection_tested,
            "recent_texts_count": len(self.recent_english_texts)
        }

def test_llm_worker():
    """Test the LLM worker module"""
    print("🧪 Testing LLM Worker Module")
    print("-" * 40)
    
    # Initialize worker
    llm = LLMWorker()
    
    # Test connection
    connected = llm.test_connection()
    print(f"Connection test: {'✅ Success' if connected else '❌ Failed'}")
    
    # Test text processing
    test_texts = [
        "hello world this is a test",
        "the quick brown fox jumps over the lazy dog",
        "this text has some errors and missing punctuation",
        "i think this needs some improvement dont you agree"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: '{text}'")
        processed = llm.process_text(text, mode="refine")
        print(f"🧠 Result: '{processed}'")
    
    # Show status
    status = llm.get_status()
    print(f"\n📊 LLM Worker Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    print("🧠 S2T2SS LLM Worker Module")
    print("=" * 50)
    
    test_llm_worker()
