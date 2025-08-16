# TTS Narrator Modes Reference

## 📝 **All Available Modes:**

### **1. "direct"** 
- **Purpose**: Clean text without changing perspective
- **Example**: "嗯我现在很高兴呃" → "我现在很高兴。"
- **Use case**: Keep original first-person perspective, just clean up

### **2. "first_person"**
- **Purpose**: Enhanced first-person cleaning with better flow
- **Example**: "嗯我觉得这个挺好的呃" → "我觉得这个挺好的。"
- **Use case**: Polished first-person speech

### **3. "casual_narrator"**
- **Purpose**: Casual third-person narrator style
- **Example**: "我现在很高兴" → "他说他现在很高兴。"
- **Use case**: Casual storytelling/narration

### **4. "professional_narrator"**
- **Purpose**: Formal third-person narrator style  
- **Example**: "我现在很高兴" → "他表示他现在很高兴。"
- **Use case**: Professional reporting/documentation

### **5. "adaptive"**
- **Purpose**: Smart context-aware narrator with emotion detection
- **Example**: 
  - Questions → "他询问..."
  - Emotions → "他兴奋地说..."
  - Confirmations → "他确认..."
- **Use case**: Dynamic narration based on content type

### **6. "translate"** ⭐ **NEW**
- **Purpose**: Translate to any target language
- **Example**: "我现在很高兴" → "I am very happy now." (English)
- **Configuration**: Set `TRANSLATION_TARGET_LANGUAGE`
- **Use case**: Real-time translation for multilingual audiences

## 🌍 **Translation Target Languages:**

```python
# BIDIRECTIONAL PRESETS FOR CHINESE/ENGLISH SPEAKERS
TRANSLATION_PRESETS = {
    'chinese_to_english': 'Chinese → English (中文 → English)',
    'english_to_chinese': 'English → Chinese (English → 中文)', 
    'chinese_to_japanese': 'Chinese → Japanese (中文 → 日本語)',
    'english_to_japanese': 'English → Japanese (English → 日本語)',
    'auto_to_english': 'Auto → English (任何语言 → English)',
    'auto_to_chinese': 'Auto → Chinese (Any language → 中文)'
}

# Quick preset selection:
CURRENT_PRESET = 'chinese_to_english'    # Most common for Chinese speakers

# Manual language override:
TRANSLATION_TARGET_LANGUAGE = "English"    # Default
TRANSLATION_TARGET_LANGUAGE = "Japanese"   # 日本語に翻訳
TRANSLATION_TARGET_LANGUAGE = "Korean"     # 한국어로 번역  
TRANSLATION_TARGET_LANGUAGE = "Spanish"    # Traducir al español
TRANSLATION_TARGET_LANGUAGE = "French"     # Traduire en français
TRANSLATION_TARGET_LANGUAGE = "German"     # Ins Deutsche übersetzen
TRANSLATION_TARGET_LANGUAGE = "Russian"    # Перевести на русский
TRANSLATION_TARGET_LANGUAGE = "Portuguese" # Traduzir para português
```

## ⚡ **Quick Translation Mode Switching:**

### **Interactive Menu Launch:**
```bash
python translation_modes.py
# Shows visual menu: 1️⃣ Chinese→English, 2️⃣ English→Chinese, etc.
```

### **Direct Mode Launch:**
```bash
python translation_modes.py chinese_to_english    # 中文 → English (most common)
python translation_modes.py english_to_chinese    # English → 中文
python translation_modes.py chinese_to_japanese   # 中文 → 日本語
```

### **Runtime Mode Switching:**
```python
switch_translation_preset('chinese_to_english')   # Switch on the fly
print_translation_presets()                       # Show all available
```

## 🎯 **Optimized Language Detection Priority:**

1. **Chinese** (highest) - Perfect for Chinese speakers
2. **English** (second) - Secondary detection  
3. **Japanese** (third) - Future expansion
4. **Others** (lower) - Additional languages

This ensures Chinese-English mixed speech is properly processed!

## 🚀 **Quick Setup:**

1. **For basic cleaning** (keep your perspective):
   ```python
   NARRATOR_MODE = "direct"
   ```

2. **For third-person narration**:
   ```python
   NARRATOR_MODE = "casual_narrator"  # or "professional_narrator"
   ```

3. **For real-time translation**:
   ```python
   NARRATOR_MODE = "translate"
   TRANSLATION_TARGET_LANGUAGE = "English"  # or any other language
   ```

4. **For smart adaptive narration**:
   ```python
   NARRATOR_MODE = "adaptive"
   ```

## 🎯 **Use Cases:**

- **Live Streaming**: "translate" mode for international audience
- **Documentation**: "professional_narrator" for formal records
- **Casual Content**: "casual_narrator" for storytelling
- **Personal Use**: "direct" to keep your own voice
- **Dynamic Content**: "adaptive" for varied content types

The translation mode is particularly powerful for:
- **Multilingual streams/meetings**
- **Language learning content**
- **International presentations**
- **Cross-language communication**
