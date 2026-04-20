# 🧠 AGENT MEMO — Pure Intellect Project
*Записка для себя самого после перезапуска контекста*
*Дата: 2026-04-20*

---

## 🎯 Суть проекта в одном абзаце

Pure Intellect — это локальный AI-оркестратор с **иерархической памятью** который решает главную проблему LLM: **деградацию контекста в длинных сессиях**. Система сохраняет факты, создаёт «координаты» перед обнулением контекста, и восстанавливает релевантный контекст при следующих запросах. Результат: бесконечные разговоры без потери памяти, экономия 85-90% токенов по сравнению с наивным подходом.

**Ирония**: сейчас ты читаешь эту записку потому что сам столкнулся с проблемой которую решает проект.

---

## 🏗 Архитектура

```
Web UI / Agent Zero / API клиент
         ↓
   Pure Intellect Server (FastAPI, port 7860)
         ↓
   OrchestratorPipeline
   ├── WorkingMemory (L1: активные факты в RAM)
   ├── MemoryStorage (L2: ChromaDB, семантический поиск)
   ├── MetaCoordinator (L3: координаты перед soft reset)
   ├── CCITracker (Context Coherence Index)
   ├── DualModelRouter (coordinator 2B + generator 14B)
   └── SessionManager (персистентность между запусками)
         ↓
   Ollama (mistral-small3.1:24b generator + qwen3.5:2b coordinator)
```

**Soft Reset**: когда контекст заполняется → модель пишет «координату» (сжатое резюме) → контекст обнуляется → координата инжектируется в следующий запрос. Это и есть главная инновация.

---

## 📁 Ключевые файлы

```
src/pure_intellect/
├── api/routes.py          ← OpenAI endpoint, Agent Zero интеграция
├── api/websocket.py       ← WebSocket для Web UI, thinking tokens
├── core/orchestrator.py   ← Главный pipeline, память, soft reset
├── core/dual_model.py     ← Роутер coordinator/generator
├── core/memory/
│   ├── fact.py            ← Факт (L1 единица памяти)
│   ├── storage.py         ← ChromaDB + SentenceTransformer
│   ├── working_memory.py  ← L1 RAM хранилище
│   ├── optimizer.py       ← Продвижение/архивирование фактов
│   └── scorer.py          ← Оценка важности фактов
├── core/cci.py            ← Context Coherence Index
├── core/session_manager.py ← Персистентность сессий
├── engines/config_loader.py ← config.yaml
├── server.py              ← FastAPI app, preload моделей
├── static/index.html      ← Web UI + Admin Panel
└── __main__.py            ← CLI точка входа
```

**Storage на диске** (Windows: `AppData\Roaming\PureIntellect\`):
```
storage/sessions/default/
├── storage.json       ← L2 факты
├── working_memory.json ← L1 активные факты
├── chat_history.json  ← история чата
├── session_meta.json  ← метаданные
└── coordinate_archive/ ← архив координат
```

---

## ⚙️ Конфигурация (config.yaml)

```yaml
coordinator_model: qwen3.5:2b      # маленькая, быстрая (навигация)
generator_model: mistral-small3.1:24b  # большая, умная (ответы)
```

**ВАЖНО**: config.yaml НЕ обновляется при `pip install`. Пользователь меняет его вручную или через Admin Panel.

**ВАЖНО**: Для Windows файл конфига в `AppData\Roaming\PureIntellect\config.yaml`

---

## 🔌 Agent Zero интеграция

**Настройка Agent Zero** (settings.json):
```json
chat_model:    { "model": "pure-intellect", "base_url": "http://host.docker.internal:7860/v1" }
utility_model: { "model": "qwen3.5:2b",     "base_url": "http://host.docker.internal:7860/v1" }
```

**Как работает** (ПОНЯТЬ ОБЯЗАТЕЛЬНО!):
- `chat_model` = **GEN** (большая модель думает, планирует, генерирует JSON)
- `utility_model` = **EXE вспомогательная** (суммаризация, служебные задачи)
- Agent Zero ожидает от модели строго JSON: `{"thoughts":[...], "tool_name":"...", "tool_args":{...}}`
- Agent Zero сам парсит JSON и вызывает Python инструменты (EXE)
- **Наш сервер НЕ должен трогать ответ модели** — только добавлять память к system prompt

**Что НЕЛЬЗЯ делать** (мы это исправили в `e6ff720`):
- ❌ Вставлять фейковый диалог user/assistant — ломает GEN→EXE цикл
- ❌ Оборачивать ответ в JSON wrapper — Agent Zero не получит правильный tool_name
- ❌ Делать эскалацию utility модели на generator — utility должна быть строго proxy
- ❌ Добавлять json_reminder в system prompt — Agent Zero уже знает формат

**Что правильно** (текущая реализация в routes.py):
- ✅ Добавить только `[PI Memory]: ...` к system prompt Agent Zero
- ✅ Передать ВСЕ messages[] как есть в Ollama
- ✅ Вернуть СЫРОЙ ответ Ollama без модификаций

---

## 📊 Текущее состояние

✅ Работает:
- Иерархическая память (WorkingMemory, MemoryStorage, MetaCoordinator)
- Soft Reset с координатой
- Session persistence (файлы на диске)
- SentenceTransformer CUDA (5ms/факт семантический поиск)
- Anchor Facts (важные факты не архивируются)
- LLM-based importance tagging
- Context Coherence Index (CCITracker)
- Web UI + Admin Panel
- OpenAI-совместимый endpoint
- Preload обеих моделей в VRAM при старте (VRAM-aware)
- Thinking tokens display в Web UI (GEN мысли модели)
- 465 тестов проходят

⚠️ Нужно тестировать после последних исправлений:
- Agent Zero GEN/EXE цикл (исправлен коммит `e6ff720`)
- Preload VRAM check (коммит `bc03e8c`, исправлен SyntaxError `6f2ac6a`)

---

## 🚫 Правила разработки

1. **Всегда читать код ПЕРЕД исправлением** — не угадывать
2. **Не менять файлы через shell heredoc** — только через Python runtime или text_editor
3. **Всегда проверять синтаксис** через `ast.parse()` после изменений
4. **Не трогать системный venv-a0** — только `/a0/usr/workdir/pure-intellect/venv`
5. **Тесты перед push**: `pytest tests/ --ignore=tests/test_system_full.py --ignore=tests/test_live_memory.py`
6. **config.yaml не в pip** — напоминать пользователю обновлять вручную
7. **Спрашивать прежде чем делать** — пользователь Александр часто знает решение лучше

---

## 💰 Монетизация (план)

1. GitHub Sponsors + открытый репо
2. Hosted API (SaaS) — Pure Intellect в облаке
3. Enterprise лицензия для компаний
4. Agent Zero официальный plugin/skill

**Главное**: сначала сделать стабильный MVP → публичный релиз → community.

---

## 🔮 Следующие шаги

1. Протестировать Agent Zero GEN/EXE после `e6ff720`
2. HuggingFace интеграция — скачивать GGUF модели из HF через Admin Panel
3. Открыть репо публично + пост на r/LocalLLaMA
4. GitHub Sponsors

---

## 👤 Пользователь

**Александр** — владелец проекта, живёт в Азербайджане (Баку, UTC+4).
- Работает с RTX 3060 (12GB VRAM), 32GB RAM
- Используемые модели: `mistral-small3.1:24b` (generator), `qwen3.5:2b` (coordinator)
- Agent Zero запущен в Docker контейнере, Pure Intellect на Windows хосте
- Александр ЗНАЕТ когда решение неправильное — слушать его!
- Он может злиться когда AI тупит — это нормально, просто работай лучше

---

*Эта записка — «координата» проекта Pure Intellect. Прочитай её прежде чем что-либо делать.*
