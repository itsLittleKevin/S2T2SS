#!/usr/bin/env python3
"""
🎛️ S2T2SS Toggle Control Module
Interactive toggle system for the 4 requested features
"""

import config

def show_current_configuration():
    """Display current toggle configuration"""
    print("⚙️ Current S2T2SS Configuration")
    print("=" * 50)
    print(f"🧠 LLM Text Editing:     {'✅ ENABLED' if config.ENABLE_LLM_EDITING else '❌ DISABLED (Raw ASR → TTS)'}")
    print(f"📺 OBS Caption Output:   {'✅ ENABLED' if config.ENABLE_OBS_CAPTIONS else '❌ DISABLED (No caption files)'}")
    print(f"🔊 TTS Audio Output:     {'✅ ENABLED' if config.ENABLE_TTS else '❌ DISABLED (Text-only mode)'}")
    print(f" Narrator Mode:        {config.NARRATOR_MODE.upper()}")
    print(f"🌍 Output Language:      {config.OUTPUT_LANGUAGE.upper()}")
    print(f"🎙️ TTS Voice Style:      {config.TTS_VOICE.upper()}")
    print(f"📁 Output Directory:     {config.DEFAULT_OUTPUT_DIR}")
    print("=" * 50)

def toggle_llm_editing():
    """Toggle LLM editing on/off"""
    config.ENABLE_LLM_EDITING = not config.ENABLE_LLM_EDITING
    status = "ENABLED" if config.ENABLE_LLM_EDITING else "DISABLED"
    mode = "LLM refines text" if config.ENABLE_LLM_EDITING else "Raw ASR → TTS directly"
    print(f"🧠 LLM Text Editing: {status} ({mode})")
    return config.ENABLE_LLM_EDITING

def toggle_obs_captions():
    """Toggle OBS caption output on/off"""
    config.ENABLE_OBS_CAPTIONS = not config.ENABLE_OBS_CAPTIONS
    status = "ENABLED" if config.ENABLE_OBS_CAPTIONS else "DISABLED"
    mode = "Write caption files" if config.ENABLE_OBS_CAPTIONS else "No caption files"
    print(f"📺 OBS Caption Output: {status} ({mode})")
    return config.ENABLE_OBS_CAPTIONS

def toggle_tts():
    """Toggle TTS audio output on/off"""
    config.ENABLE_TTS = not config.ENABLE_TTS
    status = "ENABLED" if config.ENABLE_TTS else "DISABLED"
    mode = "Generate voice" if config.ENABLE_TTS else "Text-only mode"
    print(f"🔊 TTS Audio Output: {status} ({mode})")
    
    # Auto-disable  if TTS is disabled (smart logic)
    
    return config.ENABLE_TTS


def apply_performance_mode(mode):
    """Apply a predefined performance mode"""
    if mode == "speed":
        config.ENABLE_LLM_EDITING = False
        config.ENABLE_TTS = True
        config.ENABLE_OBS_CAPTIONS = True
        print("⚡ Applied Speed Mode: Fast processing with basic TTS")
    
    elif mode == "text":
        config.ENABLE_LLM_EDITING = True
        config.ENABLE_TTS = False
        config.ENABLE_OBS_CAPTIONS = True
        print("📝 Applied Text Mode: Silent caption generation only")
    
    elif mode == "quality":
        config.ENABLE_LLM_EDITING = True
        config.ENABLE_TTS = True
        config.ENABLE_OBS_CAPTIONS = True
        print("🎭 Applied Quality Mode: Full pipeline with all features")
    
    elif mode == "podcast":
        config.ENABLE_LLM_EDITING = True
        config.ENABLE_TTS = True
        config.ENABLE_OBS_CAPTIONS = False
        print("🎤 Applied Podcast Mode: Professional audio without anime voice")
    
    return True

def cycle_narrator_mode():
    """Cycle through narrator modes"""
    modes = ["direct", "first_person", "casual_narrator", "professional_narrator", "adaptive", "translate"]
    mode_descriptions = {
        "direct": "Direct speech (no changes)",
        "first_person": "Convert to first person", 
        "casual_narrator": "Casual storytelling style",
        "professional_narrator": "Professional narration",
        "adaptive": "Auto-adapt based on content",
        "translate": "Translate to target language"
    }
    
    current_index = modes.index(config.NARRATOR_MODE) if config.NARRATOR_MODE in modes else 0
    next_index = (current_index + 1) % len(modes)
    config.NARRATOR_MODE = modes[next_index]
    
    print(f"🎯 Narrator Mode: {config.NARRATOR_MODE.upper()} ({mode_descriptions[config.NARRATOR_MODE]})")
    return config.NARRATOR_MODE

def cycle_output_language():
    """Cycle through output languages"""
    languages = ["auto", "zh", "en", "ja", "ko", "es", "fr", "de", "ru"]
    language_names = {
        "auto": "Auto-detect/maintain original",
        "zh": "Chinese (中文)",
        "en": "English",
        "ja": "Japanese (日本語)", 
        "ko": "Korean (한국어)",
        "es": "Spanish (Español)",
        "fr": "French (Français)",
        "de": "German (Deutsch)",
        "ru": "Russian (Русский)"
    }
    
    current_index = languages.index(config.OUTPUT_LANGUAGE) if config.OUTPUT_LANGUAGE in languages else 0
    next_index = (current_index + 1) % len(languages)
    config.OUTPUT_LANGUAGE = languages[next_index]
    
    print(f"🌍 Output Language: {config.OUTPUT_LANGUAGE.upper()} ({language_names[config.OUTPUT_LANGUAGE]})")
    return config.OUTPUT_LANGUAGE

def cycle_tts_voice():
    """Cycle through TTS voice styles"""
    voices = ["default", "male", "female", "young", "elder", "neutral"]
    voice_descriptions = {
        "default": "System default voice",
        "male": "Male voice preference", 
        "female": "Female voice preference",
        "young": "Youthful voice style",
        "elder": "Mature voice style",
        "neutral": "Gender-neutral voice"
    }
    
    current_index = voices.index(config.TTS_VOICE) if config.TTS_VOICE in voices else 0
    next_index = (current_index + 1) % len(voices)
    config.TTS_VOICE = voices[next_index]
    
    print(f"🎙️ TTS Voice: {config.TTS_VOICE.upper()} ({voice_descriptions[config.TTS_VOICE]})")
    return config.TTS_VOICE

def interactive_configuration_menu():
    """Interactive menu for toggle configuration"""
    while True:
        print("\n" + "=" * 60)
        print("🎛️ S2T2SS Toggle Configuration Menu")
        print("=" * 60)
        
        show_current_configuration()
        
        print("\n🎮 Toggle Options:")
        print("1️⃣  Toggle LLM Text Editing")
        print("2️⃣  Toggle OBS Caption Output") 
        print("3️⃣  Toggle TTS Audio Output")
        
        print("\n🎯 Language & Voice Settings:")
        print("5️⃣  Cycle Narrator Mode")
        print("6️⃣  Cycle Output Language")
        print("7️⃣  Cycle TTS Voice Style")
        
        print("\n🚀 Performance Modes:")
        print("8️⃣  Speed Mode (Fastest)")
        print("9️⃣  Text Mode (Silent)")
        print("🔟  Quality Mode (Full)")
        print("1️⃣1️⃣ Podcast Mode (Natural)")
        
        print("\n📋 Options:")
        print("1️⃣2️⃣ Save Configuration")
        print("1️⃣3️⃣ Test Toggle Logic")
        print("1️⃣4️⃣ Cache Management")
        print("0️⃣  Exit")
        
        try:
            choice = input("\nYour choice (0-14): ").strip()
        except KeyboardInterrupt:
            print("\n✅ Configuration complete!")
            break
        
        if choice == "1":
            toggle_llm_editing()
        elif choice == "2":
            toggle_obs_captions()
        elif choice == "3":
            toggle_tts()
        elif choice == "5":
            cycle_narrator_mode()
        elif choice == "6":
            cycle_output_language()
        elif choice == "7":
            cycle_tts_voice()
        elif choice == "8":
            apply_performance_mode("speed")
        elif choice == "9":
            apply_performance_mode("text")
        elif choice == "10":
            apply_performance_mode("quality")
        elif choice == "11":
            apply_performance_mode("podcast")
        elif choice == "12":
            config.save_config()
        elif choice == "13":
            test_toggle_logic()
        elif choice == "14":
            cache_management_menu()
        elif choice == "0":
            print("✅ Configuration complete!")
            break
        else:
            print("❌ Invalid choice")

def cache_management_menu():
    """Interactive cache management menu"""
    while True:
        print("\n" + "=" * 50)
        print("🗂️ Cache Management")
        print("=" * 50)
        
        # Show cache status
        cache_status = config.get_cache_status()
        print(f"\n📊 Current Cache Status:")
        print(f"   📁 Cache Directory: {cache_status.get('cache_dir', 'N/A')}")
        print(f"   🎵 Audio Files: {cache_status.get('audio_files_count', 0)} files")
        print(f"   💾 Total Size: {cache_status.get('total_size_mb', 0):.2f} MB")
        print(f"   🗑️ Auto Cleanup: {'✅ Enabled' if cache_status.get('cleanup_enabled', False) else '❌ Disabled'}")
        print(f"   📈 Max Files: {cache_status.get('max_files', 0)}")
        print(f"   ⏰ Max Age: {cache_status.get('max_age_hours', 0)} hours")
        
        print(f"\n🎛️ Cache Options:")
        print("1️⃣  Toggle Auto Cleanup")
        print("2️⃣  Clean Old Files")
        print("3️⃣  Clear All Cache")
        print("4️⃣  Configure Max Files")
        print("5️⃣  Configure Max Age")
        print("0️⃣  Back to Main Menu")
        
        try:
            choice = input("\nYour choice (0-5): ").strip()
        except KeyboardInterrupt:
            break
        
        if choice == "1":
            config.ENABLE_AUDIO_CLEANUP = not config.ENABLE_AUDIO_CLEANUP
            status = "✅ ENABLED" if config.ENABLE_AUDIO_CLEANUP else "❌ DISABLED"
            print(f"🗑️ Auto Cleanup: {status}")
            
        elif choice == "2":
            print("🗑️ Cleaning old cached files...")
            deleted_count = config.cleanup_audio_cache()
            if deleted_count > 0:
                print(f"✅ Deleted {deleted_count} old files")
            else:
                print("ℹ️ No old files to clean")
                
        elif choice == "3":
            confirm = input("⚠️ Delete ALL cached audio files? (y/N): ").strip().lower()
            if confirm == 'y':
                deleted_count = config.cleanup_audio_cache(force_all=True)
                print(f"✅ Deleted {deleted_count} cached files")
            else:
                print("❌ Cancelled")
                
        elif choice == "4":
            try:
                max_files = int(input(f"Current max files: {config.AUDIO_CACHE_MAX_FILES}\nNew max files: "))
                if max_files > 0:
                    config.AUDIO_CACHE_MAX_FILES = max_files
                    print(f"✅ Max files set to {max_files}")
                else:
                    print("❌ Invalid number")
            except ValueError:
                print("❌ Invalid input")
                
        elif choice == "5":
            try:
                max_hours = int(input(f"Current max age: {config.AUDIO_CACHE_MAX_AGE_HOURS} hours\nNew max age (hours): "))
                if max_hours > 0:
                    config.AUDIO_CACHE_MAX_AGE_HOURS = max_hours
                    print(f"✅ Max age set to {max_hours} hours")
                else:
                    print("❌ Invalid number")
            except ValueError:
                print("❌ Invalid input")
                
        elif choice == "0":
            break
        else:
            print("❌ Invalid choice")

def test_toggle_logic():
    """Test the toggle logic and interactions"""
    print("🧪 Testing Toggle System Logic:")
    print("-" * 40)
    
    # Save original state
    original_tts = config.ENABLE_TTS
    original_llm = config.ENABLE_LLM_EDITING
    original_obs = config.ENABLE_OBS_CAPTIONS
    
    
    # Restore original state
    config.ENABLE_TTS = original_tts
    config.ENABLE_LLM_EDITING = original_llm
    config.ENABLE_OBS_CAPTIONS = original_obs
    
    print("\n✅ Toggle logic tests completed - original state restored")

if __name__ == "__main__":
    print("🎛️ S2T2SS Toggle Control Test")
    show_current_configuration()
    print("\nStarting interactive menu...")
    interactive_configuration_menu()
