#!/bin/bash
# Pure Intellect — Installer v0.1
# Local AI with unlimited memory
# Supports: Linux (Ubuntu/Debian/Arch/Fedora) and macOS

set -e

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()  { echo -e "  ${GREEN}OK:${NC} $*"; }
info(){ echo -e "  ${BLUE}--${NC} $*"; }
warn(){ echo -e "  ${YELLOW}WARN:${NC} $*"; }
fail(){ echo -e "  ${RED}ERROR:${NC} $*" >&2; exit 1; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e ""
echo -e "${BOLD}${CYAN}  ╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}  ║     Pure Intellect — Installation v0.1      ║${NC}"
echo -e "${BOLD}${CYAN}  ║     Local AI with unlimited memory          ║${NC}"
echo -e "${BOLD}${CYAN}  ╚══════════════════════════════════════════════╝${NC}"
echo -e ""

OS=$(uname -s)
ARCH=$(uname -m)
info "Detected OS: $OS ($ARCH)"

# ── Step 1: Check Python ──────────────────────────────────────────────────────
echo -e "\n${BOLD}[1/4] Checking Python...${NC}"

PYTHON_BIN=""
for bin in python3.13 python3.12 python3.11 python3; do
    if command -v "$bin" &>/dev/null; then
        VER=$("$bin" -c 'import sys; print(sys.version_info >= (3,11))' 2>/dev/null)
        if [ "$VER" = "True" ]; then
            PYTHON_BIN="$bin"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "  ${RED}ERROR: Python 3.11+ not found!${NC}"
    echo ""
    if [ "$OS" = "Darwin" ]; then
        echo "  Install with Homebrew:"
        echo "    brew install python@3.11"
    else
        echo "  Install on Ubuntu/Debian:"
        echo "    sudo apt update && sudo apt install python3.11 python3.11-pip"
        echo "  Install on Fedora:"
        echo "    sudo dnf install python3.11"
        echo "  Install on Arch:"
        echo "    sudo pacman -S python"
    fi
    echo ""
    exit 1
fi

PY_VER=$("$PYTHON_BIN" --version 2>&1)
ok "$PY_VER found ($PYTHON_BIN)"

# ── Step 2: Install Ollama ────────────────────────────────────────────────────
echo -e "\n${BOLD}[2/4] Checking Ollama...${NC}"

if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>&1 | head -1)
    ok "Ollama already installed: $OLLAMA_VER"
else
    info "Ollama not found. Installing..."

    if [ "$OS" = "Darwin" ]; then
        # macOS — скачиваем .app или используем brew
        if command -v brew &>/dev/null; then
            info "Installing Ollama via Homebrew..."
            brew install ollama
        else
            info "Downloading Ollama for macOS..."
            curl -fsSL "https://ollama.com/download/Ollama-darwin.zip" -o /tmp/Ollama.zip
            unzip -q /tmp/Ollama.zip -d /tmp/
            mv /tmp/Ollama.app /Applications/Ollama.app 2>/dev/null || true
            rm -f /tmp/Ollama.zip
            # Add CLI to PATH
            sudo ln -sf /Applications/Ollama.app/Contents/MacOS/ollama /usr/local/bin/ollama 2>/dev/null || true
        fi
    else
        # Linux — официальный скрипт установки
        info "Installing Ollama via official script..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    if command -v ollama &>/dev/null; then
        ok "Ollama installed successfully"
    else
        fail "Ollama installation failed. Install manually: https://ollama.com"
    fi
fi

# Start Ollama in background
info "Starting Ollama service..."
if [ "$OS" = "Darwin" ]; then
    # macOS — используем launchctl или просто запускаем
    open -a Ollama 2>/dev/null || ollama serve &>/dev/null &
else
    # Linux — systemd или прямой запуск
    if systemctl is-active --quiet ollama 2>/dev/null; then
        ok "Ollama service already running"
    else
        ollama serve &>/dev/null &
        OLLAMA_PID=$!
        info "Ollama started (PID: $OLLAMA_PID)"
    fi
fi
sleep 3

# ── Step 3: Install Pure Intellect ───────────────────────────────────────────
echo -e "\n${BOLD}[3/4] Installing Pure Intellect...${NC}"
info "This may take 5-15 minutes (downloading PyTorch, ChromaDB, etc.)"
echo ""

# Устанавливаем в пользовательский pip (без sudo)
if ! "$PYTHON_BIN" -m pip install \
    git+https://github.com/Remchik64/pure-intellect.git \
    --quiet --user; then
    fail "Installation failed. Check your internet connection and try again."
fi

# Проверяем что команда доступна
if command -v pure-intellect &>/dev/null; then
    ok "Pure Intellect installed: $(pure-intellect --version 2>/dev/null || echo 'v0.1')"
elif "$PYTHON_BIN" -m pure_intellect --version &>/dev/null; then
    ok "Pure Intellect installed (use: python3 -m pure_intellect serve)"
else
    # Пробуем добавить ~/.local/bin в PATH
    LOCAL_BIN="$HOME/.local/bin"
    if [ -d "$LOCAL_BIN" ] && [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
        warn "Adding $LOCAL_BIN to PATH"
        export PATH="$LOCAL_BIN:$PATH"
        # Добавляем в shell config
        SHELL_RC="$HOME/.bashrc"
        [ -n "$ZSH_VERSION" ] && SHELL_RC="$HOME/.zshrc"
        echo "export PATH=\"$LOCAL_BIN:\$PATH\"" >> "$SHELL_RC"
        info "Added to $SHELL_RC — restart terminal or run: source $SHELL_RC"
    fi
    ok "Pure Intellect installed"
fi

# ── Step 4: Create Launcher ───────────────────────────────────────────────────
echo -e "\n${BOLD}[4/4] Creating launcher...${NC}"

LAUNCHER_DIR="$HOME/.pure-intellect"
mkdir -p "$LAUNCHER_DIR"

cat > "$LAUNCHER_DIR/start.sh" << 'LAUNCHER_EOF'
#!/bin/bash
# Pure Intellect Launcher
echo "Starting Pure Intellect..."

# Start Ollama if not running
if ! curl -s http://localhost:11434 &>/dev/null; then
    ollama serve &>/dev/null &
    sleep 2
fi

# Open browser
if command -v xdg-open &>/dev/null; then
    (sleep 4 && xdg-open http://localhost:8085) &
elif command -v open &>/dev/null; then
    (sleep 4 && open http://localhost:8085) &
fi

# Start server
if command -v pure-intellect &>/dev/null; then
    pure-intellect serve --port 8085
else
    python3 -m pure_intellect serve --port 8085
fi
LAUNCHER_EOF

chmod +x "$LAUNCHER_DIR/start.sh"
ok "Launcher created: $LAUNCHER_DIR/start.sh"

# Desktop entry for Linux
if [ "$OS" = "Linux" ] && [ -d "$HOME/.local/share/applications" ]; then
    cat > "$HOME/.local/share/applications/pure-intellect.desktop" << DESKTOP_EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Pure Intellect
Comment=Local AI with unlimited memory
Exec=$LAUNCHER_DIR/start.sh
Icon=utilities-terminal
Terminal=true
Categories=Utility;Science;ArtificialIntelligence;
Keywords=AI;LLM;Memory;Chat;
DESKTOP_EOF
    ok "Desktop shortcut created"
fi

# ── Done! ────────────────────────────────────────────────────────────────────
echo -e ""
echo -e "${BOLD}${GREEN}  ╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}  ║           Installation Complete!  🎉         ║${NC}"
echo -e "${BOLD}${GREEN}  ╚══════════════════════════════════════════════╝${NC}"
echo -e ""
echo -e "  ${BOLD}Start:${NC}  $LAUNCHER_DIR/start.sh"
echo -e "         or:  pure-intellect serve"
echo -e ""
echo -e "  ${BOLD}Open:${NC}   http://localhost:8085"
echo -e ""
echo -e "  ${YELLOW}First run:${NC} go to 🤖 Models section"
echo -e "            and download a model (e.g. qwen2.5:3b)"
echo -e ""

read -r -p "  Launch Pure Intellect now? (y/N): " LAUNCH
if [[ "$LAUNCH" =~ ^[Yy]$ ]]; then
    echo -e "  ${GREEN}Starting...${NC}"
    "$LAUNCHER_DIR/start.sh"
fi
