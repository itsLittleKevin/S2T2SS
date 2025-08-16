@echo off
echo ===============================================
echo    LM Studio Network Firewall Configuration
echo ===============================================
echo.
echo This script will configure Windows Firewall to allow LM Studio network access
echo Run this on the LM Studio machine (10.0.0.43)
echo.
pause

echo Adding firewall rule for LM Studio port 1234...
netsh advfirewall firewall add rule name="LM Studio Server" dir=in action=allow protocol=TCP localport=1234

echo Adding firewall rule for ping (ICMP)...
netsh advfirewall firewall add rule name="ICMP Allow incoming V4 echo request" protocol=icmpv4:8,any dir=in action=allow

echo.
echo ===============================================
echo Firewall rules added successfully!
echo ===============================================
echo.
echo Next steps:
echo 1. Start LM Studio on this machine
echo 2. Go to Local Server tab
echo 3. Change server address to 0.0.0.0:1234
echo 4. Start the server
echo 5. Test from S2T2SS machine with: ping 10.0.0.43
echo.
pause
