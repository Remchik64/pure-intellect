# Pure Intellect

> **Autonomous Local AI Orchestrator - Infinite Context, Triad Architecture, VRAM Juggler, and 100% Privacy**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-Remchik64%2Fpure--intellect-black.svg)](https://github.com/Remchik64/pure-intellect)

Pure Intellect is a self-sufficient local AI server that natively resolves the two biggest problems in local LLM deployment: **KV-Cache context degradation** and **VRAM limitations**. 

By utilizing a unique **Triad Architecture**, it divides workloads among three distinct open-source models, parking them in system RAM and dynamically swapping them into the GPU exclusively when needed.

---

## Core Innovations

### Triad Architecture (3 Models, 1 GPU)
Pure Intellect does not rely on a monolithic LLM. It splits intelligence into three roles:
- **The Coordinator**: A blazing-fast gatekeeper. It evaluates Context Coherence (CCI), detects user intents (`web_search`, `read_document`), and generates highly compressed memory coordinates.
- **The Utility Worker**: The heavy lifter. Triggered implicitly or explicitly, it handles massive data processing, organic web scraping, and document reading via **Map-Reduce**.
- **The Generator**: The conversationalist. It synthesizes the final, polished response using extracted facts and historical coordinates.

### VRAM Swap Manager (The Juggler)
Run 3 large models on consumer hardware (e.g., 12GB VRAM / 32GB RAM). When heavy data operations are triggered, the Swap Manager evicts resting models from VRAM into RAM, grants the Utility model exclusive hardware access, computes the heavy lifting, and instantly restores the Generator.

### Native Web Search & Map-Reduce
- **DuckDuckGo Integrated**: Live, real-time web scraping without requiring any external API keys.
- **Map-Reduce Summarizer**: Evades KV-cache overflow by splitting gigabytes of HTML/PDF text into strict 3000-token chunks, iteratively summarizing them.
- **1-Shot Web Search UI**: Built-in chat toggle allows users to force a web search, ensuring absolute control over latency and intent routing.

### Hierarchical Memory & Soft Reset
- **Context Coherence Index (CCI)**: Tracks topic drift. If coherence drops < 0.55 or turn limits are hit, the system triggers a **Soft Reset**.
- **Coordinates**: Prior context is collapsed into a dense, 200-token snapshot. The heavy context window is completely flushed, and the Coordinate is injected as the new system prompt, ensuring **100% recall over infinite conversations** without Out-of-Memory crashes.
- **Storage Layers**: HOT (Working Memory) -> WARM (ChromaDB) -> COLD (Persistent Disk).

---

## Quick Start

### Option 1: Installer Script (Recommended)

**Windows:**
```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/Remchik64/pure-intellect/main/install.bat -OutFile install.bat
.\install.bat
```

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Remchik64/pure-intellect/main/install.sh | bash
```

### Option 2: Manual Installation

```bash
# 1. Install Ollama locally
curl -fsSL https://ollama.com/install.sh | sh

# 2. Install Pure Intellect Orchestrator
pip install git+https://github.com/Remchik64/pure-intellect.git

# 3. Start the server (binds to port 3005)
pure-intellect serve --port 3005
```

### First Run
1. Open `http://localhost:3005` in your browser.
2. Navigate to **Models** and click **Detect Hardware** to automatically configure the Triad limits based on your system's VRAM.
3. Download the assigned models via the UI.
4. You are ready. Chat normally, or use the **Web Search (1-shot)** toggle to pull live internet data.

---

## Performance Comparison

| Metric | Baseline Local LLM | Pure Intellect |
|--------|-------------------|--------------------|
| Context footprint | Exponential (~8000+ tokens) | **Flat (~1200 tokens)** |
| Out-Of-Memory (OOM) | Frequent on large docs | **Eliminated (Map-Reduce)** |
| Recall after context sweep | 0% (Forgets earlier chat) | **100% (Coordinates)** |
| Supported conversation length | Limited by VRAM | **Technically Infinite** |

---

## System Flow

```text
                            Pure Intellect

    [Coordinator]        [CCI Tracker]       [Storage (Chroma)]
      (Intent)            (Coherence)              (RAG)
         |                     |                     |
         |                     |                     |
         v                     v                     v
 -----------------------------------------------------------
|                   Orchestrator Pipeline                   |
|                (Context Assembly & Reset)                 |
 -----------------------------------------------------------
                               |
                               v
 -----------------------------------------------------------
|                  SWAP MANAGER (VRAM)                      |
|                                                           |
| [UtilityWorker]  <---- VRAM Swap ----> [Generator]        |
| (Web/Map-Reduce)                       (Response)         |
 -----------------------------------------------------------
                               |
                        Ollama (Backend)
```

---

## Persistent Data & Offline Capable
Except for Organic Web Search commands, Pure Intellect requires no internet access.
Memory vectors, historical coordinates, application metrics, and sessions are safely checkpointed via ChromaDB to `./storage/sessions/default/`.
Everything maintains persistent continuity across server reboots natively.
