@echo off
chcp 65001 >nul
title Gurbani GPT — Web Server
echo.
echo  ============================================================
echo      Gurbani GPT — Sri Guru Granth Sahib Ji
echo      Powered by Ollama + RAG + ChromaDB
echo  ============================================================
echo.

set PYTHON=D:\GurbaniGPT_env\Scripts\python.exe

echo  [CHECK] Verifying environment...
%PYTHON% -c "import flask, chromadb; print('  OK: Dependencies found')" 2>nul
if errorlevel 1 (
    echo  [WARN] Installing dependencies...
    %PYTHON% -m pip install -r requirements.txt -q
)

echo.
echo  Starting server...
echo  Open your browser at: http://localhost:5000
echo.
echo  Gurbani RAG status:  http://localhost:5000/api/gurbani-status
echo  Press Ctrl+C to stop
echo.

%PYTHON% server.py
pause
