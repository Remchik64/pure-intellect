@echo off
setlocal enabledelayedexpansion
title Pure Intellect
color 0A

echo.
echo  ================================================
echo   Pure Intellect - Starting
echo  ================================================
echo.

:: ── Step 1: Kill old process on port 7860 ───────────────────────────────────
echo [1/4] Freeing port 7860...
call :kill_port
echo  OK: Port 7860 is free

:: ── Step 2: Check Ollama ────────────────────────────────────────────────────
echo.
echo [2/4] Checking Ollama...
call :check_ollama

:: ── Step 3: Start server and wait ──────────────────────────────────────────
echo.
echo [3/4] Starting Pure Intellect...
start "Pure Intellect Server" /min cmd /c "python -m pure_intellect serve --port 7860 & pause"
call :wait_server
if errorlevel 1 (
    echo.
    echo  ERROR: Server did not start in 30 seconds!
    echo  Check the minimized window for errors.
    pause
    exit /b 1
)
echo  OK: Server is ready!

:: ── Step 4: Open browser ────────────────────────────────────────────────────
echo.
echo [4/4] Opening browser...
start microsoft-edge:http://127.0.0.1:7860
if errorlevel 1 start http://127.0.0.1:7860

echo.
echo  ================================================
echo   Pure Intellect is running!
echo   URL: http://127.0.0.1:7860
echo.
echo   Press any key to STOP the server
echo  ================================================
echo.
pause >nul

:: Stop server
call :kill_port
echo  Server stopped. Goodbye!
timeout /t 2 /nobreak >nul
exit /b 0

:: ============================================================================
:: SUBROUTINES
:: ============================================================================

:kill_port
    for /f "tokens=5" %%p in ('netstat -aon 2^>nul ^| findstr ":7860 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%p >nul 2>&1
    )
    timeout /t 1 /nobreak >nul
exit /b 0

:check_ollama
    curl -s --max-time 2 http://localhost:11434 >nul 2>&1
    if errorlevel 1 (
        echo  Starting Ollama...
        start /b "" ollama serve
        call :wait_ollama
    ) else (
        echo  OK: Ollama already running
    )
exit /b 0

:wait_ollama
    set OLL=0
    :ollama_loop
    set /a OLL+=1
    if !OLL! GTR 15 (
        echo  WARNING: Ollama may not be running
        exit /b 0
    )
    timeout /t 1 /nobreak >nul
    curl -s --max-time 1 http://localhost:11434 >nul 2>&1
    if errorlevel 1 goto ollama_loop
    echo  OK: Ollama started
exit /b 0

:wait_server
    set SRV=0
    :server_loop
    set /a SRV+=1
    if !SRV! GTR 30 exit /b 1
    timeout /t 1 /nobreak >nul
    curl -s --max-time 1 http://127.0.0.1:7860 >nul 2>&1
    if errorlevel 1 goto server_loop
exit /b 0
