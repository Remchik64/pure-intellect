@echo off
setlocal enabledelayedexpansion
title Pure Intellect Installer v0.1
color 0A

echo.
echo  ================================================
echo   Pure Intellect -- Installation v0.1
echo   Local AI with unlimited memory
echo  ================================================
echo.

:: ── Step 1: Check Python ────────────────────────────────────────────────────
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found!
    echo  Please install Python 3.13 from:
    echo  https://www.python.org/ftp/python/3.13.13/python-3.13.13-amd64.exe
    echo.
    echo  IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  OK: Python %PY_VER% found

:: ── Step 2: Check/Install Ollama ────────────────────────────────────────────
echo.
echo [2/4] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo  Ollama not found. Downloading installer...
    echo  Please wait (~150 MB)...
    curl -L "https://ollama.com/download/ollama-windows-amd64.exe" -o "%TEMP%\ollama-setup.exe"
    if errorlevel 1 (
        echo.
        echo  ERROR: Failed to download Ollama.
        echo  Please download manually: https://ollama.com/download
        echo.
        pause
        exit /b 1
    )
    echo  Installing Ollama...
    start /wait "%TEMP%\ollama-setup.exe" /S
    del "%TEMP%\ollama-setup.exe" >nul 2>&1
    echo  OK: Ollama installed
) else (
    for /f "tokens=*" %%v in ('ollama --version 2^>^&1') do echo  OK: Ollama %%v
)

:: Start Ollama in background
echo  Starting Ollama service...
start /b "" ollama serve >nul 2>&1
timeout /t 3 /nobreak >nul

:: ── Step 3: Install Pure Intellect ──────────────────────────────────────────
echo.
echo [3/4] Installing Pure Intellect...
echo  This may take 5-15 minutes (downloading PyTorch, ChromaDB...).
echo.

pip install git+https://github.com/Remchik64/pure-intellect.git
if errorlevel 1 (
    echo.
    echo  ERROR: Installation failed!
    echo  Try: Run this script as Administrator
    echo.
    pause
    exit /b 1
)
echo.
echo  OK: Pure Intellect installed!

:: ── Add Python Scripts to PATH (fix for "not recognized") ───────────────────
echo.
echo  Adding Python Scripts to PATH...
for /f "tokens=*" %%p in ('python -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'Scripts'))"') do set SCRIPTS_DIR=%%p
if not "!SCRIPTS_DIR!"=="" (
    set PATH=!SCRIPTS_DIR!;!PATH!
    echo  OK: Added !SCRIPTS_DIR! to PATH
)

:: ── Step 4: Create Launcher ──────────────────────────────────────────────────
echo.
echo [4/4] Creating desktop shortcut...

set APPDIR=%APPDATA%\PureIntellect
mkdir "%APPDIR%" >nul 2>&1

:: Create start script
(
    echo @echo off
    echo title Pure Intellect
    echo echo Starting Pure Intellect...
    echo start /b "" ollama serve ^>nul 2^>^&1
    echo timeout /t 2 /nobreak ^>nul
    echo start http://localhost:8085
    echo python -m pure_intellect serve --port 8085
) > "%APPDIR%\start.bat"

:: Create desktop shortcut
powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\Pure Intellect.lnk'); $s.TargetPath = '%APPDIR%\start.bat'; $s.Description = 'Pure Intellect AI'; $s.Save()" >nul 2>&1

if exist "%USERPROFILE%\Desktop\Pure Intellect.lnk" (
    echo  OK: Shortcut created on Desktop
) else (
    echo  WARNING: Could not create shortcut
)

:: ── Done! ────────────────────────────────────────────────────────────────────
echo.
echo  ================================================
echo   Installation Complete!
echo.
echo   Start: double-click "Pure Intellect" on Desktop
echo   OR run: python -m pure_intellect serve
echo.
echo   Open browser: http://localhost:8085
echo.
echo   First run: go to Models section and
echo   download a model (e.g. qwen2.5:3b)
echo  ================================================
echo.

set /p START_NOW="Launch Pure Intellect now? (Y/N): "
if /i "%START_NOW%"=="Y" (
    echo  Starting...
    start http://localhost:8085
    python -m pure_intellect serve --port 8085
)

echo.
echo  Done! Run 'python -m pure_intellect serve' to start anytime.
pause
