@echo off
setlocal enabledelayedexpansion
title Pure Intellect
color 0A

echo.
echo  ================================================
echo   Pure Intellect - Starting...
echo  ================================================
echo.

:: ── Step 1: Kill any existing process on port 8085 ─────────────────────────
echo [1/4] Checking port 8085...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8085 " ^| findstr "LISTENING"') do (
    echo  Found old process (PID %%a), stopping it...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 /nobreak >nul
)
echo  OK: Port 8085 is free

:: ── Step 2: Start Ollama if not running ─────────────────────────────────────
echo.
echo [2/4] Checking Ollama...
curl -s http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo  Starting Ollama...
    start /b "" ollama serve
    echo  Waiting for Ollama to start...
    :wait_ollama
    timeout /t 1 /nobreak >nul
    curl -s http://localhost:11434 >nul 2>&1
    if errorlevel 1 goto wait_ollama
    echo  OK: Ollama started
) else (
    echo  OK: Ollama already running
)

:: ── Step 3: Start Pure Intellect server ─────────────────────────────────────
echo.
echo [3/4] Starting Pure Intellect server...
start "Pure Intellect Server" /min python -m pure_intellect serve --port 8085

:: Wait until server actually responds (not just a timer!)
echo  Waiting for server to be ready...
set ATTEMPTS=0
:wait_server
set /a ATTEMPTS+=1
if !ATTEMPTS! GTR 30 (
    echo.
    echo  ERROR: Server failed to start after 30 seconds!
    echo  Check the minimized window for error messages.
    pause
    exit /b 1
)
timeout /t 1 /nobreak >nul
curl -s http://127.0.0.1:8085 >nul 2>&1
if errorlevel 1 goto wait_server
echo  OK: Server is ready!

:: ── Step 4: Open browser ────────────────────────────────────────────────────
echo.
echo [4/4] Opening browser...
start microsoft-edge:http://127.0.0.1:8085 2>nul
if errorlevel 1 (
    start http://127.0.0.1:8085
)

echo.
echo  ================================================
echo   Pure Intellect is running!
echo   URL: http://127.0.0.1:8085
echo.
echo   To stop: close the minimized server window
echo   Or press any key to stop server now
echo  ================================================
echo.
pause >nul

:: Stop server when user presses any key
echo  Stopping server...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8085 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo  Server stopped. Goodbye!
timeout /t 2 /nobreak >nul
