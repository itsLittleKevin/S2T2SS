#!/usr/bin/env python3
"""
🧹 S2T2SS Chinese Text Cleanup Tool
Removes unnecessary spaces between Chinese characters in existing history files
"""

import os
import sys
import re
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

try:
    import config
except ImportError:
    # Fallback for import issues
    class Config:
        DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core', 'data')
    config = Config()

def clean_chinese_spaces(text):
    """Remove unnecessary spaces between Chinese characters"""
    if not text:
        return text
    
    # Pattern to match Chinese characters
    chinese_char_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]'
    
    # Remove spaces between Chinese characters
    cleaned_text = text
    
    # Run multiple passes to handle longer sequences
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
    
    return cleaned_text

def backup_files():
    """Create backup of files before cleaning"""
    print("📦 Creating backup of existing files...")
    
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(config.DEFAULT_OUTPUT_DIR, f"backup_before_cleanup_{backup_time}")
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = ["input.txt", "input.srt"]
    backed_up = []
    
    for filename in files_to_backup:
        source_path = os.path.join(config.DEFAULT_OUTPUT_DIR, filename)
        if os.path.exists(source_path):
            backup_path = os.path.join(backup_dir, filename)
            try:
                with open(source_path, 'r', encoding='utf-8') as src:
                    content = src.read()
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(content)
                backed_up.append(filename)
                print(f"  ✅ Backed up: {filename}")
            except Exception as e:
                print(f"  ❌ Failed to backup {filename}: {e}")
    
    print(f"📦 Backup created in: {backup_dir}")
    return backup_dir, backed_up

def clean_input_txt():
    """Clean spaces in input.txt file"""
    filepath = os.path.join(config.DEFAULT_OUTPUT_DIR, "input.txt")
    
    if not os.path.exists(filepath):
        print("❌ input.txt not found")
        return False
    
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        changes_made = 0
        
        for line in lines:
            # Skip comment lines and empty lines
            if line.strip().startswith('#') or not line.strip():
                cleaned_lines.append(line)
                continue
            
            # Clean Chinese spaces in content lines
            original_line = line
            if line.strip().startswith('[') and ']' in line:
                # Extract timestamp and content
                bracket_end = line.find(']')
                if bracket_end > 0:
                    timestamp_part = line[:bracket_end + 1]
                    content_part = line[bracket_end + 1:].strip()
                    
                    # Clean the content part
                    cleaned_content = clean_chinese_spaces(content_part)
                    cleaned_line = f"{timestamp_part} {cleaned_content}\n"
                    
                    if cleaned_line != original_line:
                        changes_made += 1
                        print(f"  📝 Cleaned: '{content_part}' → '{cleaned_content}'")
                    
                    cleaned_lines.append(cleaned_line)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        # Write back cleaned content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        print(f"✅ Cleaned input.txt - {changes_made} entries updated")
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning input.txt: {e}")
        return False

def clean_input_srt():
    """Clean spaces in input.srt file"""
    filepath = os.path.join(config.DEFAULT_OUTPUT_DIR, "input.srt")
    
    if not os.path.exists(filepath):
        print("❌ input.srt not found")
        return False
    
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into entries (separated by double newlines)
        entries = content.split('\n\n')
        cleaned_entries = []
        changes_made = 0
        
        for entry in entries:
            if not entry.strip():
                cleaned_entries.append(entry)
                continue
            
            lines = entry.split('\n')
            if len(lines) >= 3:
                # Entry has: ID, timestamp, text, (optional more text lines)
                entry_id = lines[0]
                timestamp = lines[1] 
                text_lines = lines[2:]
                
                # Clean Chinese spaces in text lines
                cleaned_text_lines = []
                entry_changed = False
                
                for text_line in text_lines:
                    if text_line.strip().startswith('#'):
                        # Skip comment lines
                        cleaned_text_lines.append(text_line)
                        continue
                    
                    original_text = text_line
                    cleaned_text = clean_chinese_spaces(text_line)
                    
                    if cleaned_text != original_text:
                        changes_made += 1
                        entry_changed = True
                        print(f"  🎬 SRT Cleaned: '{original_text}' → '{cleaned_text}'")
                    
                    cleaned_text_lines.append(cleaned_text)
                
                # Reconstruct entry
                cleaned_entry = f"{entry_id}\n{timestamp}\n" + '\n'.join(cleaned_text_lines)
                cleaned_entries.append(cleaned_entry)
            else:
                # Header or malformed entry, keep as-is
                cleaned_entries.append(entry)
        
        # Write back cleaned content
        cleaned_content = '\n\n'.join(cleaned_entries)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"✅ Cleaned input.srt - {changes_made} entries updated")
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning input.srt: {e}")
        return False

def show_before_after_sample():
    """Show examples of what the cleaning does"""
    print("📋 Cleaning Examples:")
    print("-" * 30)
    
    test_cases = [
        "好 我 们 接 下 来 呢 进 去",
        "进 行 一 个 字 幕 的 测 试", 
        "现 在 呢 可 以 写 入 啊 现 在 天",
        "是 这 个 中 文 啊 它 是 带 空 格 的 啊",
        "在 这 个 input 里 边 呢"
    ]
    
    for test_case in test_cases:
        cleaned = clean_chinese_spaces(test_case)
        print(f"  Before: '{test_case}'")
        print(f"  After:  '{cleaned}'")
        print()

def main():
    """Main cleanup function"""
    print("🧹 S2T2SS Chinese Text Cleanup Tool")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(config.DEFAULT_OUTPUT_DIR):
        print(f"❌ Output directory not found: {config.DEFAULT_OUTPUT_DIR}")
        return False
    
    print(f"📁 Working with directory: {config.DEFAULT_OUTPUT_DIR}")
    
    # Show examples
    show_before_after_sample()
    
    print("🔧 Starting cleanup process...")
    
    # Step 1: Backup existing files
    backup_dir, backed_up = backup_files()
    
    if not backed_up:
        print("⚠️ No files to backup")
    
    # Step 2: Clean input.txt
    print(f"\n🧹 Cleaning input.txt...")
    txt_success = clean_input_txt()
    
    # Step 3: Clean input.srt
    print(f"\n🧹 Cleaning input.srt...")
    srt_success = clean_input_srt()
    
    print("\n" + "=" * 50)
    
    if txt_success and srt_success:
        print("✅ Cleanup completed successfully!")
        print(f"📦 Backup saved in: {backup_dir}")
        print("🔄 Future transcriptions will automatically have clean Chinese text")
        print("📝 Existing files have been cleaned of unnecessary spaces")
    else:
        print("⚠️ Cleanup completed with some issues")
    
    return txt_success and srt_success

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Chinese text cleanup completed!")
        else:
            print("\n❌ Cleanup process failed")
    except KeyboardInterrupt:
        print("\n⚠️ Cleanup process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()