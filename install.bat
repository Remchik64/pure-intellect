@echo off
setlocal enabledelayedexpansion
title Pure Intellect Installer
color 0A

echo.
echo  ================================================
echo   Pure Intellect -- Installation v0.1
echo   Local AI with unlimited memory
echo  ================================================
echo.

:: ── GitHub Token Input ───────────────────────────────────────────────────────
echo  This installer needs your GitHub Personal Access Token
echo  to download Pure Intellect from the private repository.
echo.
echo  How to get a token:
echo  1. Go to: https://github.com/settings/tokens
echo  2. Generate new token (classic)
echo  3. Select scope: repo
echo  4. Copy and paste it below
echo.
set /p GHTOKEN="  Enter GitHub token (paste and press Enter): "
echo.

if "!GHTOKEN!"=="" (
    echo  ERROR: Token cannot be empty!
    pause
    exit /b 1
)
echo  OK: Token received

:: ── Step 1: Check Python ────────────────────────────────────────────────────
echo.
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found!
    echo  Download Python 3.13:
    echo  https://www.python.org/ftp/python/3.13.13/python-3.13.13-amd64.exe
    echo  IMPORTANT: Check "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  OK: Python %PY_VER%

:: ── Step 2: Check Ollama ─────────────────────────────────────────────────────
echo.
echo [2/4] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo  Ollama not found. Downloading... (~150 MB)
    curl -L "https://ollama.com/download/ollama-windows-amd64.exe" -o "%TEMP%\ollama-setup.exe"
    if errorlevel 1 (
        echo  ERROR: Download failed. Get it from: https://ollama.com/download
        pause
        exit /b 1
    )
    start /wait "%TEMP%\ollama-setup.exe" /S
    del "%TEMP%\ollama-setup.exe" >nul 2>&1
    echo  OK: Ollama installed
) else (
    echo  OK: Ollama found
)

:: Start Ollama only if not already running
curl -s http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo  Starting Ollama service...
    start /b "" ollama serve
    timeout /t 3 /nobreak >nul
    echo  OK: Ollama started
) else (
    echo  OK: Ollama already running
)

:: ── Step 3: Install Pure Intellect ──────────────────────────────────────────
echo.
echo [3/4] Installing Pure Intellect...
echo  Please wait 5-15 minutes (downloading PyTorch, ChromaDB...).
echo.

pip install git+https://!GHTOKEN!@github.com/Remchik64/pure-intellect.git
if errorlevel 1 (
    echo.
    echo  ERROR: Installation failed!
    echo  Check that your token has 'repo' access scope.
    echo.
    pause
    exit /b 1
)
echo.
echo  OK: Pure Intellect installed!

:: Clear token from memory
set GHTOKEN=

:: Add Scripts to PATH for this session
for /f "delims=" %%p in ('python -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),'Scripts'))"') do set SCRIPTS=%%p
if exist "!SCRIPTS!" set PATH=!SCRIPTS!;!PATH!

:: ── Step 4: Create Launcher + Shortcut ──────────────────────────────────────
echo.
echo [4/4] Creating launcher...

set APPDIR=%APPDATA%\PureIntellect
mkdir "%APPDIR%" >nul 2>&1

echo  Downloading start.bat...
curl -s -L "https://raw.githubusercontent.com/Remchik64/pure-intellect/main/start.bat" -o "%APPDIR%\start.bat"
if errorlevel 1 (
    echo  WARNING: Could not download start.bat, creating basic launcher...
    (
        echo @echo off
        echo python -m pure_intellect serve --port 8085
        echo pause
    ) > "%APPDIR%\start.bat"
)
echo  OK: Launcher ready

powershell -NoProfile -Command "$ws=New-Object -ComObject WScript.Shell;$s=$ws.CreateShortcut('%USERPROFILE%\Desktop\Pure Intellect.lnk');$s.TargetPath='%APPDIR%\start.bat';$s.Description='Pure Intellect AI';$s.Save()" >nul 2>&1

if exist "%USERPROFILE%\Desktop\Pure Intellect.lnk" (
    echo  OK: Shortcut created on Desktop
) else (
    echo  OK: Launcher saved: %APPDIR%\start.bat
)

:: ── Launch ───────────────────────────────────────────────────────────────────
echo.
echo  ================================================
echo   Done! Pure Intellect installed.
echo.
echo   Next: open Admin Panel - Models section
echo   Download model: qwen2.5:3b (2GB) to start
echo  ================================================
echo.

set /p GO="Launch now? (Y/N): "
if /i "!GO!"=="Y" (
    echo  Starting server...
    start "Pure Intellect" /min python -m pure_intellect serve --port 8085
    echo  Waiting for server to be ready...
    timeout /t 6 /nobreak >nul
    echo  Opening browser...
    start microsoft-edge:http://127.0.0.1:8085
    echo.
    echo  Server is running! Check taskbar for minimized window.
    echo  Close this window when done.
    echo.
)
if /i "!GO!"=="N" (
    echo.
    echo  To start later: double-click Pure Intellect on Desktop
    echo  Or: python -m pure_intellect serve
    echo.
    pause
)
