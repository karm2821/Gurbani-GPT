@echo off
title Ollama Chatbot

echo.
echo  ============================================
echo    Setting up Ollama Chatbot...
echo  ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install dependencies if needed
echo  [*] Checking dependencies...
pip install -r requirements.txt --quiet

echo  [*] Starting chatbot...
echo.

:: Run the chatbot
python chatbot.py

pause
