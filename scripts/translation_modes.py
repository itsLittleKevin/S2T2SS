#!/usr/bin/env python3
"""
🌍 Translation Mode Quick Switcher
=====================================

Quick launcher for different translation modes optimized for Chinese/English speakers.

Usage:
    python translation_modes.py                    # Interactive menu
    python translation_modes.py chinese_to_english # Direct mode
    python translation_modes.py --list             # Show all modes
"""

import sys
import os
import subprocess

# Translation presets optimized for Chinese/English speakers
TRANSLATION_MODES = {
    'chinese_to_english': {
        'description': 'Chinese → English (中文 → English)',
        'use_case': 'Most common: Speak Chinese, get English output',
        'emoji': '🇨🇳→🇺🇸'
    },
    'english_to_chinese': {
        'description': 'English → Chinese (English → 中文)', 
        'use_case': 'Speak English, get Chinese output',
        'emoji': '🇺🇸→🇨🇳'
    },
    'chinese_to_japanese': {
        'description': 'Chinese → Japanese (中文 → 日本語)',
        'use_case': 'Chinese speakers learning Japanese',
        'emoji': '🇨🇳→🇯🇵'
    },
    'english_to_japanese': {
        'description': 'English → Japanese (English → 日本語)',
        'use_case': 'English speakers learning Japanese', 
        'emoji': '🇺🇸→🇯🇵'
    },
    'auto_to_english': {
        'description': 'Auto → English (任何语言 → English)',
        'use_case': 'Any language input, English output',
        'emoji': '🌍→🇺🇸'
    },
    'auto_to_chinese': {
        'description': 'Auto → Chinese (Any language → 中文)',
        'use_case': 'Any language input, Chinese output',
        'emoji': '🌍→🇨🇳'
    }
}

def print_modes():
    """Print all available translation modes."""
    print("\n🌍 Available Translation Modes:")
    print("=" * 60)
    for i, (mode_id, config) in enumerate(TRANSLATION_MODES.items(), 1):
        print(f"{i}️⃣  {config['emoji']} {config['description']}")
        print(f"    💡 {config['use_case']}")
        print()

def interactive_menu():
    """Show interactive menu for mode selection."""
    print("🎙️ Real-time Translation System")
    print("=" * 40)
    print("Select your translation mode:")
    
    print_modes()
    
    print("Your choice:")
    try:
        choice = input("Enter number (1-6) or mode name: ").strip()
        
        # Handle numeric input
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(TRANSLATION_MODES):
                mode_id = list(TRANSLATION_MODES.keys())[choice_num - 1]
                return mode_id
        
        # Handle mode name input
        if choice in TRANSLATION_MODES:
            return choice
            
        print(f"❌ Invalid choice: {choice}")
        return None
        
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        return None

def launch_translation_mode(mode_id):
    """Launch the main system with specified translation mode."""
    if mode_id not in TRANSLATION_MODES:
        print(f"❌ Unknown mode: {mode_id}")
        return False
    
    config = TRANSLATION_MODES[mode_id]
    print(f"\n🚀 Launching: {config['emoji']} {config['description']}")
    print(f"💡 Use case: {config['use_case']}")
    print("=" * 50)
    
    # Modify the main script to use the selected mode
    main_script = "core/main_pipeline.py"
    
    # Read current script
    try:
        with open(main_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update the CURRENT_PRESET line
        import re
        pattern = r"CURRENT_PRESET = '[^']*'"
        replacement = f"CURRENT_PRESET = '{mode_id}'"
        
        if re.search(pattern, content):
            updated_content = re.sub(pattern, replacement, content)
            
            # Write back
            with open(main_script, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ Updated {main_script} with mode: {mode_id}")
        else:
            print(f"⚠️  Could not find CURRENT_PRESET in {main_script}")
            
    except Exception as e:
        print(f"❌ Error updating script: {e}")
        return False
    
    # Launch the main system
    print("🎙️ Starting translation system...")
    try:
        subprocess.run([sys.executable, main_script], check=True)
        return True
    except KeyboardInterrupt:
        print("\n👋 Translation system stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching system: {e}")
        return False

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg in ['--list', '-l']:
            print_modes()
            return
            
        elif arg in TRANSLATION_MODES:
            # Direct mode launch
            launch_translation_mode(arg)
            return
            
        else:
            print(f"❌ Unknown argument: {arg}")
            print("Available modes:", ', '.join(TRANSLATION_MODES.keys()))
            return
    
    # Interactive menu
    mode_id = interactive_menu()
    if mode_id:
        launch_translation_mode(mode_id)

if __name__ == "__main__":
    main()
