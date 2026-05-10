# 🧠 Contextor

> **Autonomous Local AI Orchestrator — Infinite Context, Triad Architecture, VRAM Juggler, and 100% Privacy**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-Remchik64%2Fpure--intellect-black.svg)](https://github.com/Remchik64/pure-intellect)

Pure Intellect is a self-sufficient local AI orchestration server. It solves VRAM limitations and KV-Cache degradation by utilizing a unique **Triad Architecture** and **Smart VRAM Juggler (Swap Manager)**. Instead of context collapsing under heavy files or web searches, PI dynamically manages memory, routes intents, and performs "Map-Reduce" chunk reading, bringing enterprise-grade LLM capabilities to consumer hardware.

---

## ✨ Key Capabilities

### 🏛️ Triad Architecture (3 Models, 1 GPU)
Pure Intellect splits cognitive tasks across a specialized triad:
- **Coordinator (3B)** — The gatekeeper. Lightning-fast intent classification (`web_search`, `read_document`, RAG) and Soft Reset coordinate generation.
- **Utility Worker (7B/9B)** — The heavy lifter. Performs background web scraping, reading huge documents, and generating Map-Reduce rolling summaries.
- **Generator (7B/9B)** — The thinker. Synthesizes final, polished responses using the deeply compressed context.

### 🤹 Swap Manager (VRAM Juggler)
Run 3 massive models on just 12GB VRAM + 32GB RAM dynamically.
- Automatically displaces resting models from VRAM to system RAM (`keep_alive: 0`).
- Grants exclusive GPU access to the Utility model for heavy tasks (web scraping, big docs).
- Restores the Generator instantly when generation is required.

### 🌐 Native Web Search & Map-Reduce
- **DuckDuckGo Integration**: Live web searches without external API keys.
- **Map-Reduce Summarizer**: Chunks gigabytes of retrieved sites or PDFs into 3000-token blocks, iteratively summarizing them without ever overflowing VRAM.
- **1-Shot Web Search UI**: Built-in chat toggle to explicitly bypass LLM intent routing and force the Utility Worker to fetch real-time internet data.

### 🧠 Hierarchical Memory & CCI Reset
- **Context Coherence Index (CCI)**: Tracks topic drift. If coherence drops < 0.55, it triggers a background optimization process.
- **Soft Reset System**: Instead of crashing from token overload, the Coordinator collapses the history into a dense "Coordinate" snapshot, flushes the heavy KV cache, and seamlessly injects the Coordinate into the new prompt. 100% conversational recall.
- **HOT / WARM / COLD Storage**: Memory moves from active working context to ChromaDB semantic storage and compressed archives dynamically.

---

## 🚀 Quick Start

### Auto-Install (Recommended)
**Windows:**
```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/Remchik64/pure-intellect/main/install.bat -OutFile install.bat
.\install.bat
```

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Remchik64/pure-intellect/main/install.sh | bash
```

### Manual Installation
```bash
curl -fsSL https://ollama.com/install.sh | sh
pip install git+https://github.com/Remchik64/pure-intellect.git
pure-intellect serve --port 3005
```

### Web UI
Open `http://localhost:3005` in your browser.
1. Navigate to **🤖 Models**.
2. Click **Detect Hardware** to automatically balance the Triad based on your VRAM limits.
3. Download the assigned models via the UI.
4. Start chatting! Use the **🌐 Web Search (1-shot)** toggle to pull live internet data.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       Pure Intellect                        │
│   ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│   │ Coordinator  │  │ CCI Tracker │  │   Memory System  │   │
│   │  (Intent)    │  │ (Coherence) │  │  HOT/WARM/COLD   │   │
│   └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘   │
│          │                 │                  │             │
│   ┌──────▼─────────────────▼──────────────────▼─────────┐   │
│   │                Orchestrator Pipeline                │   │
│   │             (Context Assembly & Reset)              │   │
│   └──────────────────────┬──────────────────────────────┘   │
│   ┌──────────────────────▼──────────────────────────────┐   │
│   │                 SWAP MANAGER (VRAM)                 │   │
│   │  [Utility Worker] ◄── VRAM Swap ──► [Generator]     │   │
│   │  (Web/Map-Reduce)                     (Response)    │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                       Ollama (Backend)
```

## 💾 Persistent Storage & Privacy
Pure Intellect is a **100% private, local-first** application. Except for Organic Web Search commands via DuckDuckGo, it requires no internet access. Memories, system stats, Coordinates, and config profiles are stored systematically in `./storage/sessions/default/` and persist flawlessly across reboots.
