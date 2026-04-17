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

:: ── Step 1: Check Python ────────────────────────────────────────────────────
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
echo  Please wait 5-15 minutes...
echo.

:: Install from GitHub (public) or with token (private)
pip install git+https://github.com/Remchik64/pure-intellect.git
if errorlevel 1 (
    echo.
    echo  ERROR: pip install failed!
    echo  If repo is private, run manually:
    echo    pip install git+https://TOKEN@github.com/Remchik64/pure-intellect.git
    echo.
    pause
    exit /b 1
)
echo.
echo  OK: Pure Intellect installed!

:: Add Scripts to PATH for this session
for /f "delims=" %%p in ('python -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),'Scripts'))"') do set SCRIPTS=%%p
if exist "!SCRIPTS!" set PATH=!SCRIPTS!;!PATH!

:: ── Step 4: Create Launcher + Shortcut ──────────────────────────────────────
echo.
echo [4/4] Creating launcher...

set APPDIR=%APPDATA%\PureIntellect
mkdir "%APPDIR%" >nul 2>&1

(
    echo @echo off
    echo title Pure Intellect
    echo echo Starting Pure Intellect...
    echo curl -s http://localhost:11434 ^>nul 2^>^&1 ^|^| start /b "" ollama serve
    echo timeout /t 2 /nobreak ^>nul
    echo start http://localhost:8085
    echo python -m pure_intellect serve --port 8085
    echo pause
) > "%APPDIR%\start.bat"

powershell -NoProfile -Command "$ws=New-Object -ComObject WScript.Shell;$s=$ws.CreateShortcut('%USERPROFILE%\Desktop\Pure Intellect.lnk');$s.TargetPath='%APPDIR%\start.bat';$s.Description='Pure Intellect AI';$s.Save()" >nul 2>&1

if exist "%USERPROFILE%\Desktop\Pure Intellect.lnk" (
    echo  OK: Shortcut created on Desktop
) else (
    echo  OK: Launcher saved to %APPDIR%\start.bat
)

:: ── Launch ───────────────────────────────────────────────────────────────────
echo.
echo  ================================================
echo   Done! Pure Intellect installed.
echo.
echo   Next step: download a model in Admin Panel
echo   Models section -> type qwen2.5:3b -> Download
echo  ================================================
echo.

set /p GO="Launch now? (Y/N): "
if /i "!GO!"=="Y" (
    echo  Opening browser and starting server...
    start http://localhost:8085
    python -m pure_intellect serve --port 8085
)

if /i "!GO!"=="N" (
    echo.
    echo  To start later: double-click Pure Intellect on Desktop
    echo  Or run: python -m pure_intellect serve
    echo.
    pause
)
