#!/usr/bin/env python3
"""
🧪 S2T2SS Installation Verification Script

Checks all dependencies and system requirements for S2T2SS deployment.
Run this after installation to verify everything is working correctly.
"""

import sys
import os
import subprocess
import platform
import json
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def check_python():
    """Check Python version"""
    print_header("Python Environment")
    
    version = sys.version_info
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python version compatible")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in virtual environment (recommended but not required)")
    
    return True

def check_gpu():
    """Check GPU availability"""
    print_header("GPU Support")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available: {gpu_count} GPU(s) found")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({memory:.1f}GB)")
            
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_audio():
    """Check audio system"""
    print_header("Audio System")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        print(f"✅ Audio system available")
        print(f"✅ Input devices: {len(input_devices)}")
        print(f"✅ Output devices: {len(output_devices)}")
        
        # Check for VB-Audio on Windows
        if platform.system() == "Windows":
            vb_devices = [d for d in devices if 'VB-Audio' in d['name']]
            if vb_devices:
                print(f"✅ VB-Audio Virtual Cable detected: {len(vb_devices)} device(s)")
            else:
                print("⚠️  VB-Audio Virtual Cable not found (optional for OBS)")
        
        return True
        
    except ImportError:
        print("❌ sounddevice not installed")
        return False
    except Exception as e:
        print(f"❌ Audio system error: {e}")
        return False

def check_speech_recognition():
    """Check speech recognition components"""
    print_header("Speech Recognition")
    
    components = {
        'funasr': 'FunASR (Chinese)',
        'whisper': 'OpenAI Whisper',
        'modelscope': 'ModelScope'
    }
    
    results = {}
    for module, name in components.items():
        try:
            __import__(module)
            print(f"✅ {name}")
            results[module] = True
        except ImportError:
            print(f"❌ {name} - not installed")
            results[module] = False
    
    return any(results.values())

def check_text_to_speech():
    """Check text-to-speech components"""
    print_header("Text-to-Speech")
    
    try:
        from TTS.api import TTS
        print("✅ XTTS (Coqui TTS)")
        
        # Test model loading
        try:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            print("✅ XTTS model loadable")
            return True
        except Exception as e:
            print(f"⚠️  XTTS model load test failed: {e}")
            return False
            
    except ImportError:
        print("❌ TTS (Coqui) not installed")
        return False

def check_japanese_support():
    """Check Japanese language support"""
    print_header("Japanese Language Support")
    
    components = {
        'fugashi': 'Fugashi (MeCab)',
        'cutlet': 'Cutlet (Romanization)',
        'unidic_lite': 'UniDic Lite'
    }
    
    results = {}
    for module, name in components.items():
        try:
            __import__(module)
            print(f"✅ {name}")
            results[module] = True
        except ImportError:
            print(f"❌ {name} - not installed")
            results[module] = False
    
    return all(results.values())

def check_llm_server():
    """Check LLM server connection"""
    print_header("LLM Server")
    
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio server running")
            print(f"✅ Available models: {len(models.get('data', []))}")
            return True
        else:
            print(f"⚠️  LM Studio server responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ LM Studio server not running")
        print("   Start LM Studio and load Gemma-3-4B-IT model")
        return False
    except ImportError:
        print("❌ requests module not installed")
        return False

def check_voice_files():
    """Check for voice sample files"""
    print_header("Voice Files")
    
    voice_files = ['sample02.wav', 'sample.wav', 'voice_sample.wav']
    found_files = []
    
    for file in voice_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✅ {file} ({size:.1f}KB)")
            found_files.append(file)
        else:
            print(f"❌ {file} - not found")
    
    if found_files:
        print(f"✅ Voice files available: {len(found_files)}")
        return True
    else:
        print("⚠️  No voice sample files found")
        print("   Record a 10-30 second voice sample as 'sample02.wav'")
        return False

def check_removed_rvc():
    """RVC functionality has been removed"""
    print_header("RVC Voice Models (Removed)")
    print("✅ RVC functionality has been removed - using pure TTS")
    print("   Voice conversion is no longer available")
    return True

def generate_report(results):
    """Generate a comprehensive report"""
    print_header("Installation Report")
    
    # Count results
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r)
    
    # Overall status
    if passed_checks == total_checks:
        status = "✅ EXCELLENT"
        emoji = "🎉"
    elif passed_checks >= total_checks * 0.8:
        status = "✅ GOOD"
        emoji = "👍"
    elif passed_checks >= total_checks * 0.6:
        status = "⚠️  PARTIAL"
        emoji = "⚙️"
    else:
        status = "❌ INCOMPLETE"
        emoji = "🔧"
    
    print(f"{emoji} Overall Status: {status}")
    print(f"📊 Checks passed: {passed_checks}/{total_checks}")
    
    # Detailed results
    print("\n📋 Detailed Results:")
    for check, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {check}")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "status": status,
        "results": results
    }
    
    with open("setup_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: setup_report.json")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not results.get("gpu", False):
        print("  - Install CUDA drivers for GPU acceleration")
    if not results.get("llm_server", False):
        print("  - Start LM Studio server with Gemma-3-4B-IT model")
    if not results.get("voice_files", False):
        print("  - Record voice sample as 'sample02.wav'")
    if not results.get("japanese_support", False):
        print("  - Install Japanese support: pip install fugashi[unidic-lite] cutlet")
    
    return passed_checks >= total_checks * 0.6

def main():
    """Main verification routine"""
    print("🧪 S2T2SS Installation Verification")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Run all checks
    results = {
        "python": check_python(),
        "gpu": check_gpu(),
        "audio": check_audio(),
        "speech_recognition": check_speech_recognition(),
        "text_to_speech": check_text_to_speech(),
        "japanese_support": check_japanese_support(),
        "llm_server": check_llm_server(),
        "voice_files": check_voice_files(),
        "rvc_models": check_removed_rvc()
    }
    
    # Generate report
    success = generate_report(results)
    
    if success:
        print("\n🎉 System ready! You can now run: python core/main_pipeline.py")
        return 0
    else:
        print("\n🔧 Please address the issues above before running the system")
        return 1

if __name__ == "__main__":
    sys.exit(main())
