@echo off
title NexusAI Web Chatbot

echo.
echo  =====================================================
echo    NexusAI - Professional Chatbot Web Interface
echo  =====================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from python.org
    pause & exit /b 1
)

echo  [*] Installing dependencies...
pip install flask requests --quiet

echo  [*] Starting NexusAI Web Server...
echo.
echo  Open your browser at:  http://localhost:5000
echo.
echo  =====================================================
echo.

python server.py

pause
