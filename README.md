# 🧠 Pure Intellect

> **Исследовательский проект**: Самообновляемая иерархическая память для локальных LLM с обнуляемым контекстом

[![Tests](https://img.shields.io/badge/tests-177%20passed-brightgreen)](#тестирование)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:3b-orange)](#модели)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#лицензия)

---

## 💡 Идея и мотивация

LLM обрывают нить разговора через 20–30 сообщений — контекст заполнен, модель «забывает» начало диалога.

Стандартные решения:
- **Бесконечное расширение контекста** → дорого, медленно, не масштабируется
- **Жёсткое обнуление** → модель теряет всю историю

**Pure Intellect** решает это иначе: **rolling window + координата + иерархическая память**.

```
Вместо того чтобы помнить всё или не помнить ничего —
система запоминает суть и отпускает детали.
```

---

## 🏗️ Архитектура

### Трёхуровневая иерархия памяти

```
┌─────────────────────────────────────────────────────┐
│  L1 — WorkingMemory (горячий буфер, ~1500 токенов) │
│  Горячие факты, anchor facts, rolling window        │
├─────────────────────────────────────────────────────┤
│  L2 — MemoryStorage (долгосрочное хранилище)       │
│  Холодные факты, semantic index, JSON persistence  │
├─────────────────────────────────────────────────────┤
│  L3 — Archive (сжатые / устаревшие факты)          │
│  RAW → SUMMARIZED → ENTITY_ONLY → ARCHIVED         │
└─────────────────────────────────────────────────────┘
```

### Полный pipeline (каждый запрос)

```
User query
    ↓
[CCI]       Context Coherence Index — оценка связности
    ↓ при потере нити — восстановление из L2
[Intent]    Определение типа запроса (rule-based + LLM)
[RAG]       Retrieval из карточек кода / документов
[Graph]     Поиск по графу знаний (NetworkX)
[Memory]    WorkingMemory.get_context() → anchor facts + горячие факты
    ↓
[LLM]       Генерация ответа (qwen2.5:3b через Ollama)
    ↓
[Tagger]    ImportanceTagger: классификация важности
            anchors → add_anchor()  (не decay, не evict)
            facts   → add_text()    (обычный lifecycle)
[Cleanup]   Холодные → L2, горячие остаются в L1
[Optimizer] Каждые 5 turns: promote/compress/archive
[CCI]       Фиксация turn в истории связности
```

### Механизм Soft Reset (ключевая идея)

```
Растущий буфер (нормально):
  Turn 1-12: context = [T1..T12]  ← всё видно

При заполнении (Turn 13):
  qwen2.5:3b читает всю историю
  → создаёт координату: «Александр, pure-intellect, Python 3.13, RTX 3060...»
  → координата → add_anchor() ← НИКОГДА не исчезнет
  → история обрезается до последних 3 turns

После soft reset (Turn 14+):
  context = [coordinate_anchor + T11, T12, T13, ...]
            └── вся суть сохранена        └── живой чат
```

---

## 📊 Результаты

### Benchmark (живой тест с qwen2.5:3b на RTX 3060)

```
═══════════════════════════════════════════════════════
  Сценарий               Baseline   Memory    Прирост
  ─────────────────────────────────────────────────
  30 turns (длинная)       0.0%     83.3%    +83% ▲
  Topic switch (10 t.)     0.0%     95.0%    +95% ▲
  Повторные вопросы       76.5%     88.2%    +15% ▲
═══════════════════════════════════════════════════════
```

### Живой тест (15 turns, 3 soft resets)

```
✅ «Как меня зовут?»        → «Ваше имя — Александр.»
✅ «Как называется проект?» → «pure-intellect»
✅ «На чём написан backend?»→ «Python 3.13, FastAPI, ChromaDB»
✅ «Какую проблему решаем?» → «потеря контекста LLM после 20-30 сообщений»
✅ «Какой GPU установлен?»  → «RTX 3060 с 12GB памяти»

Recall: 5/5 (100%)  |  3 soft resets  |  GPU: 750ms–2.7s/turn
```

---

## 🚀 Быстрый старт

### Требования

- Python 3.13+
- [Ollama](https://ollama.ai) с GPU поддержкой
- NVIDIA GPU (опционально, но рекомендуется)

### Установка

```bash
git clone https://github.com/Remchik64/pure-intellect.git
cd pure-intellect

# Создать виртуальное окружение
python3.13 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Установить зависимости
pip install -e .
```

### Запуск Ollama (требуется)

```bash
# Установить Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Скачать модель
ollama pull qwen2.5:3b

# Запустить сервер
ollama serve
```

### Запуск Pure Intellect

```bash
uvicorn pure_intellect.server:app --port 8085 --reload
```

### Тестирование

```bash
# Все тесты (не требуют Ollama)
python -m pytest tests/ --ignore=tests/test_live_memory.py

# Живой тест с LLM (требует запущенный Ollama)
python tests/test_live_memory.py

# Бенчмарк
python -m benchmarks
```

---

## 🔌 API

```bash
# Отправить запрос через Orchestrator
curl -X POST http://localhost:8085/api/v1/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"query": "Привет! Меня зовут Александр.", "model_key": "qwen2.5:3b"}'

# Статистика памяти
curl http://localhost:8085/api/v1/memory/stats

# Статистика CCI
curl http://localhost:8085/api/v1/cci/stats

# Очистить рабочую память
curl -X POST http://localhost:8085/api/v1/memory/clear
```

---

## 📁 Структура проекта

```
pure-intellect/
├── src/pure_intellect/
│   ├── core/
│   │   ├── memory/              # Система памяти (L1/L2/L3)
│   │   │   ├── fact.py          # Атом памяти с lifecycle + is_anchor
│   │   │   ├── working_memory.py # L1 буфер + add_anchor()
│   │   │   ├── storage.py       # L2 + Ollama embeddings + cosine similarity
│   │   │   ├── scorer.py        # AttentionScorer — оценка важности
│   │   │   ├── optimizer.py     # MemoryOptimizer — promote/compress/archive
│   │   │   ├── cci.py           # Context Coherence Index
│   │   │   └── tagger.py        # ImportanceTagger — LLM классификация
│   │   ├── orchestrator.py      # Главный pipeline + soft reset
│   │   ├── intent.py            # Определение намерений
│   │   ├── retriever.py         # RAG поиск
│   │   └── graph_builder.py     # Граф знаний (NetworkX)
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   └── engine/
│       └── model_manager.py     # Управление llama-cpp моделями
├── tests/                       # 177 тестов
├── benchmarks/                  # Сравнительные тесты
└── docs/                        # Документация
```

---

## 📈 История разработки

Проект разрабатывался итерационно, каждый шаг документирован в коммитах.

### Фаза 0 — Стабилизация `87f230d`
> Апрель 2026

Первичный аудит репозитория выявил критические проблемы:
- `Assembler.assemble()` — метод не существовал (dead code)
- `ModelManager` — утечка памяти VRAM, отсутствие `dispose()`
- Race condition в singleton при параллельных запросах
- Regex-парсинг JSON из LLM ответов — нестабильный
- CORS `allow_origins=['*']` — небезопасная конфигурация

**Исправлено**: `threading.Lock` в ModelManager, robust JSON parser (3 стратегии), `dispose()` + `_build_messages()`, CORS ограничен localhost.

---

### Фаза 1 — Страховочная сетка тестов `ceb1de8`
> Апрель 2026

До начала архитектурных изменений — написаны базовые тесты:
- `test_intent.py` — rule-based классификация намерений
- `test_assembler.py` — сборка контекста
- `test_model_manager.py` — lifecycle GPU моделей
- `test_orchestrator.py` — полный pipeline

**41 тест** — теперь любое изменение сразу видно.

---

### Фаза 2 — Ядро памяти `c0bb487`
> Апрель 2026

Построены три уровня иерархической памяти с нуля:
- `Fact` — атом памяти: `attention_weight`, `decay()`, `touch()`, `compression_level`
- `WorkingMemory` — горячий буфер с `token_budget` и автоматическим eviction
- `MemoryStorage` — долгосрочное хранилище с JSON persistence

**+36 тестов** (77 всего)

---

### Фаза 3 — AttentionScorer `732fc2c`
> Апрель 2026

Система понимает что важно из текущего разговора:
- Если факт упоминается в query/response → `touch()` (вес растёт)
- Если нет → `decay()` (вес падает)
- Горячие остаются в L1, холодные уходят в L2

**+20 тестов** (97 всего)

---

### Фаза 4 — MemoryOptimizer `d6485dd`
> Апрель 2026

Фоновое обслуживание памяти каждые N turns:
- **promote**: горячие факты из L2 → L1 (если часто запрашиваются)
- **compress**: холодные факты сжимаются RAW → SUMMARIZED → ENTITY_ONLY
- **archive**: устаревшие переводятся в L3

**+23 теста** (119 всего)

---

### Фаза 5 — Интеграция в Orchestrator `dfd3456`
> Апрель 2026

Вся система памяти подключена к реальному pipeline:
- `WorkingMemory.get_context()` вставляется в system prompt
- После каждого ответа — факты извлекаются и обновляются
- Singleton pipeline — память персистентна между запросами
- Новые endpoints: `GET /memory/stats`, `POST /memory/clear`

---

### Фаза 6 — Context Coherence Index `e429ece`
> Апрель 2026

`CCITracker` — система понимает когда разговор теряет нить:
- BM25 similarity между соседними turns
- При `coherence < threshold` → восстановление контекста из L2
- Новые endpoints: `GET /cci/stats`, `POST /cci/reset`

**+31 тест** (150 всего)

---

### Фаза 7 — Benchmark suite `22491b0`
> Апрель 2026

Доказательство что система работает лучше baseline:
- 3 сценария: длинная сессия, переключение темы, повторные вопросы
- Baseline (без памяти) vs Memory-augmented
- Метрики: context preservation rate, keyword recall

**Результат**: +83% / +95% / +15% против baseline.

---

### P1 — Semantic Retrieval `e08e03e`
> Апрель 2026

Ключевое улучшение точности: замена BM25 на Ollama embeddings:
- `"Как меня зовут?"` → embedding → `cosine_sim("Меня зовут Александр") = 0.89` ✅
- Vs BM25: `"Как меня зовут?"` → keywords → 0 совпадений ❌
- Fallback на BM25 если Ollama недоступен
- Embeddings сохраняются в JSON (persistence v2.0)

Recall до: **20%** → после: **80%**

---

### P2 — Anchor Facts `e08e03e`
> Апрель 2026

Факты которые **нельзя потерять** (имена, названия, конфигурации):
- `Fact.is_anchor = True` → `decay()` пропускает, `cleanup()` не evict
- `WorkingMemory.add_anchor()` — создаёт защищённый факт
- `OrchestratorPipeline._soft_reset()` → координата → `add_anchor()`
- Rolling window + `_context_window_size = 12` сообщений

Recall: **80%** → **100%**

---

### P3 — LLM-based Importance Tagging `b46b989`
> Апрель 2026

`ImportanceTagger` — модель сама решает что важно:
```
qwen2.5:3b анализирует каждый turn:
  anchors:   ["имя: Александр", "проект: pure-intellect"] → add_anchor()
  facts:     ["Python 3.13", "FastAPI", "RTX 3060"]       → add_text()
  transient: ["вопрос про LoRA", "пример кода"]           → skip
```
- Robust JSON parsing: 3 стратегии (direct / boundaries / markdown)
- Fallback на rule-based если Ollama недоступен
- **+27 тестов** (177 всего)

---

## 🧩 Ключевые компоненты

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `Fact` | `memory/fact.py` | Атом памяти: вес, decay, сжатие, anchor |
| `WorkingMemory` | `memory/working_memory.py` | L1 буфер с токен-бюджетом |
| `MemoryStorage` | `memory/storage.py` | L2 с Ollama embeddings |
| `AttentionScorer` | `memory/scorer.py` | Важность по тексту разговора |
| `MemoryOptimizer` | `memory/optimizer.py` | Promote/compress/archive |
| `CCITracker` | `memory/cci.py` | Context Coherence Index |
| `ImportanceTagger` | `memory/tagger.py` | LLM классификация важности |
| `OrchestratorPipeline` | `core/orchestrator.py` | Главный pipeline + soft reset |

---

## 🛠️ Модели

Проект протестирован с:

| Модель | Размер | Роль | VRAM |
|--------|--------|------|------|
| `qwen2.5:3b` | ~2GB | Навигатор + генератор | ~3GB |

Планируется:

| Модель | Размер | Роль |
|--------|--------|------|
| `qwen2.5:3b` | ~2GB | Координатор (создание координат, tagging) |
| `qwen2.5-coder:7b` | ~5GB | Генератор (качественные ответы) |

---

## 🔭 Roadmap

```
✅ P1 — Semantic retrieval (Ollama embeddings)
✅ P2 — Anchor facts (защита критических фактов)
✅ P3 — LLM-based importance tagging

⬜ P4 — sentence-transformers для storage (torch уже установлен)
         all-MiniLM-L6-v2 быстрее Ollama для embeddings

⬜ P5 — Persistence между сессиями
         SQLite для chat_history, embeddings
         Restart = продолжение той же сессии

⬜ P6 — Двойная дистилляция 3B + 7B
         3B — координатор + tagger
         7B — генератор качественных ответов

⬜ P7 — Web UI
         Simple chat interface с визуализацией памяти

⬜ P8 — Evaluation на реальных датасетах
         LongBench, SCROLLS, MemGPT benchmark
```

---

## 📚 Документация

- [Архитектура](docs/architecture.md)
- [Установка](docs/installation.md)
- [API Reference](docs/api_reference.md)

---

## 🤝 Вклад в проект

Проект находится в активной исследовательской разработке. Issues и PR приветствуются.

---

## 📄 Лицензия

MIT License — свободное использование с сохранением атрибуции.

---

<div align="center">

**Pure Intellect** — потому что интеллект должен помнить главное, а не всё подряд.

*177 тестов · 5/5 recall · GPU inference · Local-first*

</div>
