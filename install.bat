@echo off
setlocal enabledelayedexpansion
title Pure Intellect — Installer v0.1
color 0A
chcp 65001 >nul 2>&1

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║     Pure Intellect — Installation v0.1      ║
echo  ║     Local AI with unlimited memory          ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ── Step 1: Check Python ────────────────────────────────────────────────────
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found!
    echo  Please install Python 3.11+ from https://python.org
    echo  IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

:: Check Python version >= 3.11
python -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python 3.11 or higher is required.
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo  Found: %%i
    echo  Download: https://python.org/downloads
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo  OK: %%i found

:: ── Step 2: Install Ollama ───────────────────────────────────────────────────
echo.
echo [2/4] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo  Ollama not found. Downloading installer (~150 MB)...
    echo  Please wait...
    curl -L --progress-bar "https://ollama.com/download/ollama-windows-amd64.exe" -o "%TEMP%\ollama-setup.exe"
    if errorlevel 1 (
        echo.
        echo  ERROR: Failed to download Ollama.
        echo  Check your internet connection or download manually:
        echo  https://ollama.com/download
        echo.
        pause
        exit /b 1
    )
    echo  Installing Ollama silently...
    start /wait "%TEMP%\ollama-setup.exe" /S
    del "%TEMP%\ollama-setup.exe" >nul 2>&1
    echo  OK: Ollama installed successfully
) else (
    for /f "tokens=*" %%i in ('ollama --version 2^>^&1') do echo  OK: Ollama %%i already installed
)

:: Start Ollama service in background
echo  Starting Ollama service...
start /b "Ollama" ollama serve >nul 2>&1
timeout /t 3 /nobreak >nul

:: ── Step 3: Install Pure Intellect ──────────────────────────────────────────
echo.
echo [3/4] Installing Pure Intellect...
echo  This may take 5-15 minutes (downloading dependencies)...
echo.

:: Try pip install from GitHub
pip install git+https://github.com/Remchik64/pure-intellect.git --quiet
if errorlevel 1 (
    echo.
    echo  ERROR: Installation failed!
    echo  Try running this script as Administrator.
    echo  Or install manually:
    echo    pip install git+https://github.com/Remchik64/pure-intellect.git
    echo.
    pause
    exit /b 1
)
echo  OK: Pure Intellect installed successfully

:: ── Step 4: Create Desktop Shortcut ─────────────────────────────────────────
echo.
echo [4/4] Creating desktop shortcut...

:: Create start script
set START_SCRIPT=%APPDATA%\PureIntellect\start.bat
mkdir "%APPDATA%\PureIntellect" >nul 2>&1
(
    echo @echo off
    echo title Pure Intellect
    echo echo Starting Pure Intellect...
    echo start /b "" ollama serve ^>nul 2^>^&1
    echo timeout /t 2 /nobreak ^>nul
    echo start http://localhost:8085
    echo pure-intellect serve --port 8085
) > "%START_SCRIPT%"

:: Create shortcut via PowerShell
powershell -NoProfile -Command ^^
    "$ws = New-Object -ComObject WScript.Shell;" ^^
    "$s = $ws.CreateShortcut('%USERPROFILE%\Desktop\Pure Intellect.lnk');" ^^
    "$s.TargetPath = '%START_SCRIPT%';" ^^
    "$s.Description = 'Pure Intellect - Local AI with memory';" ^^
    "$s.WorkingDirectory = '%APPDATA%\PureIntellect';" ^^
    "$s.Save()" >nul 2>&1

if exist "%USERPROFILE%\Desktop\Pure Intellect.lnk" (
    echo  OK: Shortcut created on Desktop
) else (
    echo  WARNING: Could not create shortcut (non-critical)
)

:: ── Done! ────────────────────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║           Installation Complete!            ║
echo  ║                                              ║
echo  ║  Start:   pure-intellect serve              ║
echo  ║           or double-click desktop shortcut  ║
echo  ║                                              ║
echo  ║  Open:    http://localhost:8085             ║
echo  ║                                              ║
echo  ║  First run: go to Models section and        ║
echo  ║  download a model (e.g. qwen2.5:3b)         ║
echo  ╚══════════════════════════════════════════════╝
echo.

set /p START_NOW="Launch Pure Intellect now? (Y/N): "
if /i "%START_NOW%"=="Y" (
    echo  Starting server...
    start http://localhost:8085
    pure-intellect serve --port 8085
)

echo.
echo  Goodbye! Run 'pure-intellect serve' to start anytime.
pause
