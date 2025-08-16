@echo off
echo ===============================================
echo         Ollama Network Setup for S2T2SS
echo ===============================================
echo.
echo This script helps set up Ollama for network use
echo Run this on the Ollama server machine (10.0.0.43)
echo.

echo Step 1: Set environment variable for network access
setx OLLAMA_HOST "0.0.0.0:11434" /M
echo Environment variable set. You may need to restart your terminal.
echo.

echo Step 2: Configure Windows Firewall
echo Adding firewall rule for Ollama port 11434...
netsh advfirewall firewall add rule name="Ollama Server" dir=in action=allow protocol=TCP localport=11434
echo.

echo Step 3: Enable ping for testing
netsh advfirewall firewall add rule name="ICMP Allow incoming V4 echo request" protocol=icmpv4:8,any dir=in action=allow
echo.

echo ===============================================
echo Configuration complete!
echo ===============================================
echo.
echo Next steps:
echo 1. Download Ollama from https://ollama.ai/download
echo 2. Install Ollama
echo 3. Open a new terminal and run: ollama pull qwen2.5:7b
echo 4. Start server with: ollama serve
echo 5. Test from S2T2SS machine: curl http://10.0.0.43:11434/api/tags
echo.
echo Your S2T2SS is already configured to use Ollama!
echo.
pause
