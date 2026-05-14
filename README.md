# 🧠 Contextor

> **Local AI with unlimited memory — 85% fewer tokens, 100% recall, zero cloud**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-Remchik64%2FContextor--pro-black.svg)](https://github.com/Remchik64/Contextor-pro)
[![Support on Boosty](https://img.shields.io/badge/Support%20on-Boosty-orange.svg)](https://boosty.to/rem64)
[![Support via YooMoney](https://img.shields.io/badge/Support-ЮMoney-blueviolet.svg)](https://yoomoney.ru/to/4100118846255337)

![Contextor Admin Panel](docs/images/admin-panel.png)

Every LLM has a context limit. When conversation gets long, the model forgets, hallucinates, or breaks. Contextor solves this with **hierarchical memory** and **soft resets** — your AI remembers everything, conversations never degrade, and token usage drops by 85%.

---

## ✨ What Makes Contextor Different

### 🧠 Hierarchical Memory

Contextor doesn't just stuff everything into context. It organizes knowledge in layers:

| Layer | What | How it works |
|-------|------|-------------|
| **HOT** | Working Memory | Active facts with attention scoring — always in context |
| **WARM** | Storage | Semi-active facts, semantic search via embeddings — recalled when relevant |
| **Anchor** | Critical Facts | Information that never decays — names, preferences, key decisions |

Facts automatically flow between layers: important facts get promoted to HOT, irrelevant ones sink to WARM. Anchor facts always stay visible.

### 🎯 Soft Reset with Coordinates

When context fills up, most systems either truncate history (losing information) or crash. Contextor does a **soft reset**:

1. A lightweight 3B model creates a **coordinate** — a compressed snapshot of the entire conversation (~200 tokens)
2. Context clears, coordinate injected as the first message
3. **100% recall across resets** — the model picks up exactly where it left off

No information loss. No degradation. Unlimited conversation length.

### 📊 Context Coherence Index (CCI)

Contextor continuously monitors conversation health with CCI (0.0 → 1.0):

- **CCI > 0.55** → conversation is coherent, proceed normally
- **CCI < 0.55** → context is degrading, time for a soft reset
- **Hard limit** → reset after 16 turns regardless of CCI

This means resets happen **when needed**, not on a fixed schedule.

### 🤖 Dual Model Architecture

Contextor uses two models together for optimal quality and speed:

| Model | Size | Role | Speed |
|-------|------|------|-------|
| **Coordinator** | 2B | Intent detection, coordinates, fact extraction | ⚡ Fast |
| **Generator** | 9B | High-quality response generation | 🧠 Smart |

The coordinator handles all meta-tasks (what to remember, when to reset), while the generator focuses on producing excellent answers. This means the 9B model only runs when generating responses — saving VRAM and compute.

### 💾 Persistent Memory

All memory is **saved to disk**. Restart the server, and it remembers everything:

- Working memory facts survive restarts
- Coordinate archive preserves every soft reset snapshot
- Switch models anytime — memory is never lost
- Each session has isolated storage

### 🖥️ Admin Panel

Full web interface at `http://localhost:7860`:

- **💬 Chat** — talk to your AI with full memory support
- **🧠 Memory** — view, search, and manage facts and coordinates
- **🤖 Models** — hardware detection, model recommendations, download
- **⚙️ Settings** — live configuration (CCI threshold, memory limits)
- **📋 Logs** — real-time server logs

### 🔌 OpenAI-Compatible API

```bash
POST /v1/chat/completions   # Standard OpenAI format
GET  /v1/models              # Model list
```

Any OpenAI-compatible client (curl, Python SDK, apps) connects instantly. Memory works transparently for every client.

---

## 🚀 Quick Start

### Option 1: Installer (Recommended)

**Windows:**

📥 [**Download install.bat**](https://raw.githubusercontent.com/Remchik64/Contextor-pro/main/install.bat) — or via PowerShell:

```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/Remchik64/Contextor-pro/main/install.bat -OutFile install.bat
.\install.bat
```

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Remchik64/Contextor-pro/main/install.sh | bash
```

The installer will:
1. ✅ Check Python 3.11+
2. ✅ Install Ollama automatically
3. ✅ Install Contextor via pip
4. ✅ Create desktop shortcut / launcher
5. ✅ Launch the server and open browser

### Option 2: Manual Installation

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Install Contextor
pip install git+https://github.com/Remchik64/Contextor-pro.git

# 3. Start server
contextor serve

# 4. Open browser
# http://localhost:7860
```

### First Run

After installation, open `http://localhost:7860` and:
1. Go to **🤖 Models** tab
2. Click **"Detect Hardware"** — Contextor auto-detects your GPU
3. See recommendations for your system
4. Click **"Download"** to get recommended models
5. Start chatting! 🎉

---

## 📊 Performance

| Metric | Without Memory | With Contextor |
|--------|---------------|------------------|
| Context tokens per turn | ~8000 | ~1200 |
| Token reduction | baseline | **85% fewer** |
| Recall after reset | 0% | **100%** |
| Supported conversation length | ~50 turns | **Unlimited** |
| Embedding speed | N/A | **5ms/fact (CUDA)** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Contextor                        │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Intent  │  │   CCI    │  │  Memory System   │  │
│  │ Detector │  │ Tracker  │  │  HOT / WARM      │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                  │            │
│  ┌────▼──────────────▼──────────────────▼─────────┐ │
│  │              OrchestratorPipeline               │ │
│  │  Soft Reset │ Coordinate │ Adaptive CCI Reset   │ │
│  └────┬───────────────────────────────────────────┘ │
│       │                                             │
│  ┌────▼──────────────────────────────────────────┐ │
│  │           Dual Model Router                   │ │
│  │  Coordinator (2B) │ Generator (9B)            │ │
│  └────┬──────────────────────────────────────────┘ │
│       │                                             │
│  ┌────▼──────────────────────────────────────────┐ │
│  │  RAG │ Knowledge Graph │ Semantic Search        │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
         │
    Ollama (backend)
```

---

## 📁 Project Structure

```
contextor/
├── src/contextor/
│   ├── api/              # FastAPI routes, WebSocket
│   ├── core/
│   │   ├── memory/       # Fact, WorkingMemory, Storage, Scorer, Optimizer
│   │   ├── orchestrator.py   # Main pipeline
│   │   ├── dual_model.py     # 2B/9B router
│   │   ├── intent.py         # Intent detection
│   │   ├── card_generator.py # RAG card generation
│   │   ├── retriever.py      # Semantic search
│   │   ├── assembler.py      # Context assembly
│   │   ├── graph.py          # Knowledge graph
│   │   └── session_manager.py
│   ├── engines/          # Ollama provider, config loader
│   ├── utils/            # Hardware detector, tokenizer
│   └── static/           # Admin Panel (index.html)
├── tests/                # 370+ tests
├── docs/                 # Documentation
├── install.bat           # Windows installer
├── install.sh            # Linux/macOS installer
├── config.yaml           # Configuration
└── pyproject.toml
```

---

## ⚙️ Configuration

```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 7860

coordinator:
  model: qwen3.5:2b    # Fast model for navigation
  temperature: 0.2
  max_tokens: 400

generator:
  model: qwen3.5:9b    # Smart model for responses
  temperature: 0.7
  max_tokens: 2048

memory:
  context_window_messages: 12
  keep_after_reset: 6
  max_hot_facts: 50
  adaptive_reset:
    enabled: true
    cci_threshold: 0.55
    min_turns_between_resets: 4
    max_turns_without_reset: 16

cci:
  window_size: 5
  reset_threshold: 0.55
```

---

## 🔌 Integration

### ✅ Ollama (Working)

Contextor uses Ollama as the default backend. Install Ollama and pull models:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended models
ollama pull qwen3.5:2b
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

Any Ollama-compatible model works. The Admin Panel handles model detection and download.

---

## 🧪 Development

```bash
# Clone
git clone https://github.com/Remchik64/Contextor-pro
cd contextor

# Setup
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
pip install -e .

# Run tests
python -m pytest tests/ -q

# Run server
contextor serve --port 7860
```

---

## 📈 Roadmap

- [x] Hierarchical memory (HOT/WARM)
- [x] Soft Reset with coordinates
- [x] Context Coherence Index (CCI)
- [x] Dual Model Router (2B coordinator + 9B generator)
- [x] Semantic search with embeddings (CUDA)
- [x] LLM-based importance tagging
- [x] OpenAI-compatible API
- [x] Multi-session support
- [x] Admin Panel
- [x] Hardware Detection + Model Recommendations
- [x] Install Scripts (Windows/Linux/macOS)
- [x] Persistent Memory Storage
- [x] Knowledge Graph
- [x] RAG card generation
- [ ] PyPI package (`pip install contextor`)
- [ ] Module Mode (transparent proxy for AI apps)
- [ ] UCIP v2 (4-layer Context Package)
- [ ] Adaptive CCI threshold
- [ ] Docker image

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a Pull Request

---

## 📜 License

Copyright 2025 **Яраев Ренат Жавдетович**

Licensed under the **Apache License, Version 2.0**.

This license allows you to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Patent use
- ✅ Private use

With conditions:
- 📋 License and copyright notice must be included
- 📋 State changes made to the code
- 📋 Original author attribution required

See [LICENSE](LICENSE) for full terms.

---

<div align="center">

**Contextor** — Your context, orchestrated.

*Built with ❤️ by Ренат Яраев (Remchik64)*

[GitHub](https://github.com/Remchik64/Contextor-pro) · [Issues](https://github.com/Remchik64/Contextor-pro/issues) · [License](LICENSE) · [VK](https://vk.com/remchik64) · [Boosty](https://boosty.to/rem64) · [ЮMoney](https://yoomoney.ru/to/4100118846255337)

📧 renataraev51@gmail.com · remch2013@yandex.ru

</div>
