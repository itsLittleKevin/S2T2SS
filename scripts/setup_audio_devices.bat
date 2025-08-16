@echo off
echo Windows Audio Device Quick Setup
echo ===================================
echo.
echo This will help you configure your audio devices for the TTS system:
echo.
echo 1. MICROPHONE INPUT (for speech recognition):
echo    - Should be your real microphone (e.g., "USB Audio Device")  
echo    - NOT VB-Audio Cable
echo.
echo 2. AUDIO OUTPUT (for TTS playback):
echo    - Should be VB-Audio Cable Input (for OBS capture)
echo    - OR your speakers/headphones (for direct listening)
echo.
echo Opening Windows Sound Settings...
echo.
pause
start ms-settings:sound
echo.
echo In the Sound Settings:
echo - Set your USB microphone as DEFAULT INPUT device
echo - This ensures the TTS system uses your real microphone
echo - VB-Audio should be used for OUTPUT only
echo.
echo After configuring, test with:
echo   python test_microphone.py
echo.
pause
