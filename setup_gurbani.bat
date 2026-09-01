@echo off
title Gurbani GPT — Setup Pipeline
echo.
echo  ============================================================
echo      Gurbani GPT — Data Setup Pipeline
echo      All packages running from D:\GurbaniGPT_env
echo  ============================================================
echo.

set PYTHON=D:\GurbaniGPT_env\Scripts\python.exe
set SCRIPTS=D:\Chatbot\scripts

echo  [STEP 1/3] Downloading Gurbani data (1430 Angs)...
echo             This may take 15-20 minutes the first time.
echo             Already downloaded Angs will be skipped.
echo.
%PYTHON% "%SCRIPTS%\01_download_data.py"
if errorlevel 1 goto error

echo.
echo  [STEP 2/3] Cleaning and chunking into Shabads...
%PYTHON% "%SCRIPTS%\02_clean_chunk.py"
if errorlevel 1 goto error

echo.
echo  [STEP 3/3] Embedding and storing in ChromaDB...
echo             Make sure Ollama is running first!
echo             Pull embedding model: ollama pull nomic-embed-text
echo.
%PYTHON% "%SCRIPTS%\03_embed_store.py"
if errorlevel 1 goto error

echo.
echo  ============================================================
echo      Setup Complete! Gurbani GPT is ready.
echo      Now run: start_gurbani_server.bat
echo  ============================================================
echo.
pause
goto end

:error
echo.
echo  ERROR: Setup failed. Check the output above.
pause

:end
