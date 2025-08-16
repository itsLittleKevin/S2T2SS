#!/usr/bin/env python3
"""
🔧 XTTS Model Health Monitor and Reset Tool
Monitor XTTS model health and perform manual resets
"""

import sys
import os
import time
import json

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import config
from tts_manager import TTSManager

def print_model_health(tts_manager):
    """Print detailed model health information"""
    print("\n" + "="*60)
    print("🏥 XTTS MODEL HEALTH STATUS")
    print("="*60)
    
    health_info = tts_manager.get_model_health_info()
    status = tts_manager.get_status()
    
    # Health status with emoji
    health_status = health_info['health_status']
    health_emoji = "🟢" if health_status == "good" else "🟡" if health_status == "degraded" else "🔴"
    
    print(f"{health_emoji} Overall Health: {health_status.upper()}")
    print(f"🔢 Failure Count: {health_info['failure_count']}")
    print(f"⏱️  Time Since Last Reset: {health_info['time_since_last_reset_minutes']:.1f} minutes")
    print(f"🧠 Model Loaded: {'✅ Yes' if health_info['model_loaded'] else '❌ No'}")
    print(f"🔊 TTS Enabled: {'✅ Yes' if status['enabled'] else '❌ No'}")
    print(f"🛡️  Error Recovery: {'✅ Enabled' if status['cuda_recovery_enabled'] else '❌ Disabled'}")
    print(f"🔄 Auto-Reset: {'✅ Enabled' if status['reinitialization_enabled'] else '❌ Disabled'}")
    
    if health_info['next_proactive_reset_minutes'] >= 0:
        print(f"⏰ Next Proactive Reset: {health_info['next_proactive_reset_minutes']:.1f} minutes")
    else:
        print("⏰ Proactive Reset: Disabled")
    
    if health_info['cooldown_remaining_seconds'] > 0:
        print(f"❄️  Reset Cooldown: {health_info['cooldown_remaining_seconds']:.1f} seconds remaining")
    else:
        print("❄️  Reset Cooldown: Ready")

def main():
    """Main monitoring and reset tool"""
    print("🔧 XTTS Model Health Monitor")
    print("=" * 40)
    
    # Initialize TTS manager
    print("🔄 Initializing TTS Manager...")
    tts_manager = TTSManager()
    
    while True:
        try:
            # Display current health
            print_model_health(tts_manager)
            
            print("\n📋 Available Commands:")
            print("  [1] Refresh Status")
            print("  [2] Test Synthesis") 
            print("  [3] Force Model Reset")
            print("  [4] Show Configuration")
            print("  [5] Simulate CUDA Error (for testing)")
            print("  [q] Quit")
            
            choice = input("\n🔍 Enter command: ").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                print("\n👋 Goodbye!")
                break
            elif choice == '1':
                print("🔄 Refreshing status...")
                continue
            elif choice == '2':
                print("\n🧪 Testing synthesis...")
                test_text = "This is a test of the XTTS synthesis system."
                result = tts_manager.synthesize_text(test_text)
                if result is not None:
                    print(f"✅ Synthesis successful: {len(result)/tts_manager.sample_rate:.2f}s audio")
                else:
                    print("❌ Synthesis failed")
            elif choice == '3':
                print("\n🚨 Forcing model reset...")
                if tts_manager.force_model_reset():
                    print("✅ Model reset successful")
                else:
                    print("❌ Model reset failed")
            elif choice == '4':
                print("\n⚙️ Current Configuration:")
                config_data = config.get_config()
                cuda_settings = {k: v for k, v in config_data.items() if 'CUDA' in k or 'XTTS' in k or 'MODEL' in k}
                print(json.dumps(cuda_settings, indent=2))
            elif choice == '5':
                print("\n🧪 Simulating CUDA error for testing...")
                # Increment failure count manually for testing
                tts_manager.model_failed_count += 1
                print(f"🔢 Failure count increased to: {tts_manager.model_failed_count}")
            else:
                print("❓ Invalid command, please try again.")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(1)
    
    # Cleanup
    try:
        tts_manager.cleanup()
        print("🧹 Cleanup completed")
    except:
        pass

if __name__ == "__main__":
    main()