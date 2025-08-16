#!/usr/bin/env python3
"""
🚀 S2T2SS Launcher v2.0

Modern launcher for the S2T2SS real-time translation system.
Clean, fast, and focused on the new mod        test_items = [
            ("core/test_system.py", "Comprehensive System Tests")
        ]r architecture.
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path

class S2T2SSLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.core_system = self.project_root / "core"
        self.venv_path = self.project_root / ".venv"
        self.config_file = self.project_root / "config.json"

        
    def print_header(self):
        """Print modern application header"""
        print("🚀" + "=" * 58 + "🚀")
        print("🎯 S2T2SS v2.0 - Speech Translation & Voice Conversion")
        print("   Real-time Multilingual AI Pipeline")
        print("🚀" + "=" * 58 + "🚀")
    
    def check_python_version(self):
        """Ensure Python 3.11.9 is installed."""
        try:
            version_output = subprocess.check_output([sys.executable, "--version"], text=True).strip()
            if "3.11.9" in version_output:
                print(f"   ✅ {version_output} is being used.")
                return True
            else:
                print(f"   ❌ {version_output} detected. Python 3.11.9 is required.")
                print("   Please install Python 3.11.9 and try again.")
                return False
        except Exception as e:
            print(f"   ❌ Failed to check Python version: {e}")
            return False

    def ensure_venv(self):
        """Ensure the virtual environment is created and activated."""
        if not self.venv_path.exists():
            print("   ⚠️  Virtual environment not found. Creating one...")
            try:
                if platform.system() == "Windows":
                    subprocess.run(["py", "-3.11", "-m", "venv", str(self.venv_path)], check=True)
                else:
                    subprocess.run(["python3.11", "-m", "venv", str(self.venv_path)], check=True)
                print("   ✅ Virtual environment created successfully.")
            except Exception as e:
                print(f"   ❌ Failed to create virtual environment: {e}")
                return False
        return True

    def activate_venv(self):
        """Activate virtual environment if available."""
        if not self.ensure_venv():
            return sys.executable

        if platform.system() == "Windows":
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"

        if python_exe.exists():
            print("   🐍 Virtual environment: Activated")
            sys.executable = str(python_exe)  # Update sys.executable to use venv Python
            return str(python_exe)
        else:
            print("   ⚠️  Virtual environment: Found but incomplete")
            return sys.executable
    
    def check_system(self):
        """Quick system compatibility check"""
        print("\n🔍 System Check:")

        # Check Python version
        if not self.check_python_version():
            return False

        # Virtual environment - activate if available
        self.activate_venv()
        
        # Core system directory
        if self.core_system.exists():
            print("   ✅ Core system architecture found")
        else:
            print("   ❌ Core system directory missing")
            return False
        
        # Core dependencies
        try:
            import torch
            print("   ✅ PyTorch available")
        except ImportError:
            print("   ❌ PyTorch missing")
            return False
        
        return True
    
    def check_gpu(self):
        """Check GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   🎮 GPU: {gpu_name}")
                return True
            else:
                print("   💻 Running in CPU mode")
                return False
        except:
            return False
    
    def check_lm_studio(self):
        """Quick LM Studio connection check"""
        try:
            import requests
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            if response.status_code == 200:
                print("   🧠 LM Studio: Connected")
                return True
            else:
                print("   🧠 LM Studio: Server not ready")
                return False
        except:
            print("   🧠 LM Studio: Not running")
            return False
    
    def load_config(self):
        """Load or create basic configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Create minimal default config
        default_config = {
            "version": "2.0",
            "last_launched": None,
            "preferred_mode": "interactive"
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
        except:
            pass
        
        return default_config
    
    def show_main_menu(self):
        """Show streamlined main menu"""
        print("\n🎮 S2T2SS Launcher Menu")
        print("=" * 30)
        print("1. 🚀 Launch S2T2SS System")
        print("2. 🧪 Run System Tests")
        print("3. 📊 System Status")
        print("4. 🔧 Setup & Installation")
        print("5. 📚 Documentation")
        print("6. ❌ Exit")
        print("=" * 30)
    
    def launch_main_system(self):
        """Launch the main S2T2SS system"""
        print("\n🚀 Launching S2T2SS System...")
        print("   Location: core/main_pipeline.py")
        print("   Features: Full interactive menu with toggles")
        print("   Press Ctrl+C to return to launcher")
        print("-" * 50)
        
        try:
            # Activate virtual environment and change to core directory
            python_exe = self.activate_venv()
            original_cwd = os.getcwd()
            os.chdir(self.core_system)
            
            subprocess.run([python_exe, "main_pipeline.py"], check=True)
            
        except KeyboardInterrupt:
            print("\n✅ Returned to launcher")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ System error: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            os.chdir(original_cwd)
    
    def run_tests(self):
        """Run system tests"""
        print("\n🧪 Running System Tests...")
        
        test_files = [
            ("core/test_system.py", "Comprehensive System Tests")
        ]
        
        for test_file, description in test_files:
            test_path = self.project_root / test_file
            if test_path.exists():
                print(f"\n📋 {description}")
                print(f"   Running: {test_file}")
                try:
                    python_exe = self.activate_venv()
                    original_cwd = os.getcwd()
                    os.chdir(self.core_system)
                    subprocess.run([python_exe, test_path.name], check=True)
                except Exception as e:
                    print(f"   ❌ Test failed: {e}")
                finally:
                    os.chdir(original_cwd)
            else:
                print(f"   ⚠️  {description}: Test file not found")
        
        # Quick functionality check
        print(f"\n🔍 Quick System Check:")
        try:
            python_exe = self.activate_venv()
            original_cwd = os.getcwd()
            os.chdir(self.core_system)
            
            # Test basic imports
            result = subprocess.run([
                python_exe, "-c", 
                "import config, caption_manager, llm_worker, tts_manager, asr_module, toggle_control; print('✅ All core modules import successfully')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("   ✅ Core module imports: PASSED")
            else:
                print(f"   ❌ Import test failed: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ Quick check failed: {e}")
        finally:
            os.chdir(original_cwd)
        
        input("\nPress Enter to continue...")
    
    def show_system_status(self):
        """Show detailed system status"""
        print("\n📊 Detailed System Status")
        print("=" * 40)
        
        # System check
        self.check_system()
        self.activate_venv()
        self.check_gpu()
        self.check_lm_studio()
        
        # File structure check
        print("\n📁 Project Structure:")
        key_paths = [
            ("core/main_pipeline.py", "Main System"),
            ("core/asr_module.py", "ASR Engine"),
            ("core/llm_worker.py", "LLM Processing"),
            ("core/tts_manager.py", "TTS Engine"),
            ("core/caption_manager.py", "Caption System"),
            ("core/toggle_control.py", "Toggle Control"),
            ("core/config.py", "Configuration"),
            ("requirements.txt", "Dependencies"),
            ("README.md", "Documentation")
        ]
        
        for path_str, description in key_paths:
            path = self.project_root / path_str
            if path.exists():
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}: Missing")
        
        # Configuration status
        print(f"\n⚙️  Configuration:")
        config = self.load_config()
        print(f"   📋 Config file: {'✅ Found' if self.config_file.exists() else '❌ Missing'}")
        print(f"   🆚 Version: {config.get('version', 'Unknown')}")
        print(f"   🎯 Preferred mode: {config.get('preferred_mode', 'Not set')}")
        
        input("\nPress Enter to continue...")
    
    def show_setup_menu(self):
        """Show setup and installation options"""
        print("\n🔧 Setup & Installation")
        print("=" * 30)
        print("1. Install/Update Dependencies")
        print("2. Check Dependencies")
        print("3. Audio Device Setup")
        print("4. Create Virtual Environment")
        print("5. Download Required Models")
        print("6. Back to Main Menu")
        
        choice = input("\nYour choice (1-6): ").strip()
        
        if choice == "1":
            print("\n📦 Installing dependencies...")
            python_exe = self.activate_venv()
            subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"])
        
        elif choice == "2":
            print("\n🔍 Checking dependencies...")
            try:
                python_exe = self.activate_venv()
                subprocess.run([python_exe, "-c", "import torch, sounddevice, funasr, TTS; print('✅ All core dependencies available')"])
            except:
                print("❌ Some dependencies missing")
        
        elif choice == "3":
            print("\n🎵 Audio Device Information:")
            try:
                import sounddevice as sd
                print(sd.query_devices())
            except ImportError:
                print("❌ sounddevice not installed")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "4":
            print("\n🐍 Virtual Environment Setup:")
            print("   Run one of these commands:")
            print("   Windows: python -m venv .venv && .venv\\Scripts\\activate")
            print("   Linux:   python -m venv .venv && source .venv/bin/activate")
        
        elif choice == "5":
            print("\n🤖 Model Download:")
            print("   Models are downloaded automatically on first run")
            print("   FunASR and XTTS v2 models will be cached")
        
        elif choice == "6":
            return
        
        if choice != "6":
            input("\nPress Enter to continue...")
    
    def show_documentation(self):
        """Show documentation information"""
        print("\n📚 Documentation")
        print("=" * 25)
        print("📖 Key Documentation Files:")
        
        docs = [
            ("README.md", "Main documentation"),
            ("requirements.txt", "Dependencies list"),
            ("core/data/", "Output directory")
        ]
        
        for doc_path, description in docs:
            path = self.project_root / doc_path
            if path.exists():
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}: Not found")
        
        print(f"\n💡 Quick Start:")
        print(f"   1. Ensure dependencies: pip install -r requirements.txt")
        print(f"   2. Start LM Studio server on localhost:1234")
        print(f"   3. Launch system: Option 1 from main menu")
        print(f"   4. Use interactive toggles to configure features")
        
        print(f"\n🌟 New Features:")
        print(f"   • Voice-synchronized captions")
        print(f"   • Auto-clearing after 3s silence")
        print(f"   • Chinese quote normalization")
        print(f"   • OBS-controlled line breaking")
        print(f"   • Complete toggle system")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main launcher loop"""
        self.print_header()
        
        # Load configuration
        config = self.load_config()
        
        # Quick system check
        if not self.check_system():
            print("\n❌ System check failed. Please fix issues before continuing.")
            input("Press Enter to exit...")
            return
        
        # Main loop
        while True:
            self.show_main_menu()
            choice = input("\nYour choice (1-6): ").strip()
            
            if choice == "1":
                self.launch_main_system()
            
            elif choice == "2":
                self.run_tests()
            
            elif choice == "3":
                self.show_system_status()
            
            elif choice == "4":
                self.show_setup_menu()
            
            elif choice == "5":
                self.show_documentation()
            
            elif choice == "6":
                print("\n👋 Thank you for using S2T2SS!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-6.")
                input("Press Enter to continue...")

def main():
    """Entry point"""
    try:
        # Ensure the virtual environment is activated at the start
        launcher = S2T2SSLauncher()
        python_exe = launcher.activate_venv()

        # If the activated Python is not 3.11.9, notify the user
        if "3.11.9" not in subprocess.check_output([python_exe, "--version"], text=True):
            print("\n❌ Python 3.11.9 is required but not found in the virtual environment.")
            print("   Please ensure the .venv is powered by Python 3.11.9.")
            input("Press Enter to exit...")
            return

        # Update sys.executable to use the activated Python
        sys.executable = python_exe

        launcher.run()
    except KeyboardInterrupt:
        print("\n\n👋 Launcher interrupted by user")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
