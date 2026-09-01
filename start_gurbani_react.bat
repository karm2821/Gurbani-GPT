@echo off
title Gurbani GPT — React UI
color 0E

echo.
echo  ============================================================
echo      Gurbani GPT -- Dedicated React UI + Ollama RAG Server
echo  ============================================================
echo.

cd /d "%~dp0gurbani-gpt"
echo  Starting Gurbani GPT React Frontend on http://localhost:3000 ...
echo.
npm run dev
pause
