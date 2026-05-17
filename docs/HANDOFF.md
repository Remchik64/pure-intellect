# 📋 Записка для следующего чата — Contextor Development Context

> Дата: 2026-05-16
> Создал: Agent Zero (developer profile)
> Проект: https://github.com/Remchik64/Contextor-pro (активная разработка)
> Архив: https://github.com/Remchik64/pure-intellect (только чтение)
> Локальная копия: /a0/usr/workdir/pure-intellect/ (пакет переименован в contextor)

---

## 🎯 Суть проекта

**Contextor (CTX)** — локальный AI-оркестратор с иерархической памятью и управлением контекстом.

Ключевая проблема: у всех LLM конечное контекстное окно. При длинных разговорах контекст переполняется, модель деградирует.
CTX решает это через CCI (Context Coherence Index), Soft Reset (сжатие истории в координату), и иерархическую память (HOT/WARM/COLD).

Проект open source (Apache 2.0), ~12K строк Python, ~6K строк тестов.

---

## 🧠 Ключевые архитектурные решения (обсуждены и утверждены)

### 1. CTX Module Mode — контекст-гипервизор

CTX может работать в двух режимах:
- **Standalone** — полноценный чат-бот с Triad Architecture (Coordinator + Generator + Utility)
- **Module** — прозрачный прокси между любым AI-приложением и Ollama, управляющий контекстом

В Module Mode CTX = swap-раздел для AI. Приложения (OpenCode, Agent Zero, LM Studio) проходят через CTX, который управляет контекстным окном.

### 2. Smart Proxy — как CTX встраивается в поток

```
Приложение → CTX Proxy → Ollama
                ↑
          Анализирует контекст (токены, CCI)
          При переполнении → Soft Reset (координата вместо истории)
          Фоном → извлекает факты, обновляет память
```

CTX НЕ модифицирует содержимое запросов. CTX управляет РАЗМЕРОМ контекстного окна:
- PASSTHROUGH: контекст в норме, пропускаем как есть
- ENRICH: добавляем релевантные факты из памяти
- COMPRESS: soft reset — заменяем старую историю на координату
- PREFETCH: инжектируем координату прошлой сессии в новый чат

### 3. Минимальный набор моделей в Module Mode

- Coordinator (3B) — intent detection, coordinate generation, fact extraction (~2GB VRAM)
- Embedding — CCI, memory search, code RAG (~0.4GB VRAM)
- Generator (7B) — НЕ НУЖЕН (агент генерирует ответы своей моделью)
- Utility (7B) — НЕ НУЖЕН (агент делает свою работу)
- Итого: ~2.4GB VRAM вместо ~7.4GB

### 4. UCIP v2 — Universal Context Injection Protocol

**Ключевой инсайт: координата — это НАПРАВЛЕНИЕ, а не СОДЕРЖАНИЕ.**

CTX инжектирует не только координату, а **4-слойный Context Package**:

1. **Coordinate** — куда идём (естественный язык, английский, ~200-400 токенов)
2. **Memory** — какие решения приняли (факты, ~100-200 токенов)
3. **Code RAG** — точные сигнатуры функций и классов (~300-500 токенов)
4. **Graph** — что с чем связано (~50-100 токенов)

Формат: каждое слой — отдельное system-сообщение:
- `[Previous conversation context: ...]`
- `[Relevant context from memory: ...]`
- `[Relevant code context: ...]`
- `[Project structure: ...]`

Координата на АНГЛИЙСКОМ языке (lingua franca LLM, все модели понимают).

### 5. Монетизация

- Community (Apache 2.0): все основные фичи бесплатно
- Pro (Commercial): Dashboard Pro, multi-user, team, GitHub integration, SaaS, API

---

## 📂 Сохранённые документы

| Файл | Содержание |
|------|------------|
| `docs/CTX_MODULE_ARCHITECTURE.md` | Полная архитектура Module Mode, UCIP v2, 4-слойный Context Package, конфиги |
| `docs/ROADMAP.md` | Дорожная карта на 26 недель (MVP → Module Mode → Pro) |
| `docs/HANDOFF.md` | Этот файл — контекст для следующего чата |

---

## 📋 Дорожная карта (кратко)

### Фаза 1: Железобетонное MVP (8 недель)
- P0: Стабильность ядра, багфиксы, error handling
- P0: UCIP v2 (английский, 4-слойный Context Package)
- P0: CCI improvements, adaptive threshold
- P0: `pip install contextor` (PyPI)
- P1: UI очистка (убрать лишнее, добавить CCI визуализацию)
- P1: Memory визуализация в UI
- P2: Рефакторинг routes.py (1976 строк → модули)

### Фаза 2: Module Mode (6 недель)
- P0: Прозрачный HTTP прокси (FastAPI + httpx)
- P0: Decision Engine (PASSTHROUGH / ENRICH / COMPRESS / PREFETCH)
- P0: Context Surgery (замена старой истории на Context Package)
- P0: Background fact extraction
- P1: Интеграция с OpenCode, LM Studio, Agent Zero
- P1: MCP Server

### Фаза 3: CTX Pro (12 недель)
- Закрытый проект (Commercial лицензия)
- Dashboard Pro, Multi-user, Team collaboration
- GitHub integration, One-click deploy
- CTX Cloud (SaaS), CTX Module API

---

## 🔑 Ключевые файлы проекта

| Файл | Строки | Описание |
|------|--------|----------|
| `src/contextor/core/orchestrator.py` | 908 | Главный пайплайн, Soft Reset, координаты |
| `src/contextor/api/routes.py` | 17 | Роутер-хаб (делегирует в модули) |
| `src/contextor/api/chat.py` | 42 | Chat API эндпоинты |
| `src/contextor/api/session.py` | 31 | Session API эндпоинты |
| `src/contextor/api/memory_api.py` | 43 | Memory API эндпоинты |
| `src/contextor/api/models_api.py` | 310 | Models API (status, switch, download) |
| `src/contextor/api/system.py` | 353 | System API (health, config, hardware, OpenAI) |
| `src/contextor/api/websocket.py` | 358 | WebSocket стриминг |
| `src/contextor/api/schemas.py` | 37 | Pydantic модели для API |
| `src/contextor/api/state.py` | 56 | Глобальное состояние пайплайна |
| `src/contextor/core/dual_model.py` | 325 | Маршрутизация Coordinator/Generator |
| `src/contextor/utils/swap_manager.py` | 200 | VRAM Swap Manager |
| `src/contextor/core/memory/working_memory.py` | 396 | Рабочая память (HOT) |
| `src/contextor/core/memory/meta_coordinator.py` | 254 | Мета-координаты (сжатие координат) |
| `src/contextor/config.py` | 76 | Конфигурация (Pydantic Settings, ollama_url) |
| `src/contextor/server.py` | 243 | FastAPI сервер, startup/shutdown |
| `src/contextor/static/index.html` | 341 | Web UI HTML |
| `src/contextor/static/css/style.css` | 1040 | Web UI стили |
| `src/contextor/static/js/app.js` | 1075 | Web UI JavaScript |

---

## ⚠️ Известные проблемы

1. **Coordinator prompt на русском** — для Module Mode нужен английский
2. **Нет proxy pipeline** — основная задача Фазы 2
3. **Нет UCIP v2** — координаты генерируются в русском шаблонном формате
4. **CCI threshold фиксированный (0.55)** — нужен adaptive
5. **Задержка ответа ~25 сек** — pipeline.run() синхронный, ответ появляется целиком
6. **После перезапуска сервера нужен F5** — WebSocket не переподключается автоматически
7. **Тесты модели** — test_config/test_tagger/test_hardware_detector ждут старые имена qwen2.5, код использует qwen3.5

---

## 🚀 Следующий шаг

**Продолжить реализацию Фазы 1 — Стабильность ядра:**
1. Обновить тесты под текущие имена моделей (qwen3.5 вместо qwen2.5)
2. Переписать _create_coordinate() на английский (UCIP format)
3. Добавить 4-слойный Context Package в _build_system_prompt()
4. Улучшить CCI алгоритм (adaptive threshold)
5. Асинхронный pipeline.run() для стриминга ответов
6. WebSocket auto-reconnect на клиенте

Для начала работы прочитать:
- `docs/CTX_MODULE_ARCHITECTURE.md` — полная архитектура Module Mode
- `docs/ROADMAP.md` — дорожная карта с задачами
- `docs/architecture.md` — оригинальная архитектура проекта
- `src/contextor/core/orchestrator.py` — ядро (908 строк)
 
 ---
 
 ## 📝 Дневник разработки
 
### 18 мая 2026 — Книга-рассказ о Contextor

**Выполнено:**
- ✅ Глубоко изучено ядро Contextor: orchestrator.py (896), memory/* (1700+), dual_model.py (325), config.py (103), config_loader.py (285), session.py (223), session_manager.py (400), intent.py (297), assembler.py (162)
- ✅ Написана книга-рассказ о Contextor: `docs/CTX_BOOK.md` (487 строк, 8 глав)
- ✅ Книга описывает архитектуру в живом стиле: контекст как буфер, иерархическая память, CCI, мета-координаты, dual model, поток данных
- ✅ Коммит `a3b2759` — docs: add Contextor book - narrative architecture guide for Agent Zero

**Ключевое понимание:**
- Контекстное окно = обнуляемый буфер для извлечения фактов, а не ограничение
- Мягкий сброс = сжатие и сохранение памяти, а не потеря данных
- Мета-координаты: 4 координаты → 1 мета (~300 токенов), стабильно в prompt
- `num_ctx: 8192` hardcoded в `dual_model.py:180` — нужно вынести в `config.yaml`

**Открытые вопросы:**
- Обсудить контекстное окно по умолчанию (8K vs 16K vs 32K)
- Исправить хардкод `num_ctx: 8192` → вынести в `config.yaml`

### 17 мая 2026 — Auto-detect Ollama URL и редизайн Memory tab

**Выполнено:**
- ✅ fix: auto-detect Docker vs native for Ollama URL (коммит `0b23f2f`)
  - `config.py`: добавлено автоопределение окружения (Docker vs native) для `ollama_url`
  - Docker → `http://host.docker.internal:11434`, Native (Windows/Linux) → `http://localhost:11434`
  - `config.py`: добавлено поле `ollama_timeout` (default: 120s)
  - `state.py`: root logger level = DEBUG для заполнения LOG_BUFFER (вкладка Logs)
- ✅ feat: redesign Memory tab (коммит `3f350dc`)
  - Заменён 3-колоночный layout (📍 Координаты, ⚓ Якоря, 🔥 Hot Facts) на единый блок «Данные чата»
  - Заголовок показывает имя активного чата: «Данные чата: <имя>»
  - Каждый факт имеет бейдж типа (📍/⚓/🔥) и важность
  - `allFacts` изменён с объекта на плоский массив
  - CSS: `memory-cols`/`memory-col` → `memory-single`
- ✅ feat: redesign chat bubbles and improve markdown rendering (коммит `407cdb0`)
  - CSS: скруглённые пузыри с тенями, анимации появления (`slideIn`/`fadeIn`)
  - CSS: тёмные код-блоки с кнопкой копирования (`.copy-btn`, `.code-header`)
  - JS: улучшен `renderMarkdown()` — цитаты, ссылки, код-блоки, таблицы
- ✅ feat: adaptive chat input (коммит `412ffff`)
  - CSS: textarea `max-height 200px`, `border-radius 12px`, focus glow (`box-shadow`)
  - CSS: styled scrollbar, send button hover/active animations
  - JS: auto-resize textarea до 200px с плавным переходом (`scrollHeight`)
- ✅ Push в оба репозитория: contextor и contextor-pro (22+ коммита)

**Что в процессе:**
- 🔄 Обновление оставшихся ссылок `qwen2.5` → `qwen3.5` в тестах
- 🔄 UCIP v2 — координаты на английском

**Блокеры:**
- Нет

**Следующие шаги:**
- Обновить тесты под текущие имена моделей (qwen3.5)
- Переписать `_create_coordinate()` на английский (UCIP format)
- Асинхронный `pipeline.run()` для стриминга ответов
- WebSocket auto-reconnect на клиенте

**Решения и обоснования:**
- Автоопределение Docker/native — убирает необходимость ручной конфигурации при переключении окружений
- Единый блок Memory — упрощает восприятие, убирает визуальный шум от трёх колонок
- Редизайн chat bubbles — улучшает читаемость, современный вид, анимации дают обратную связь
- Adaptive input — textarea растёт с контентом, не занимает лишнее место, плавный UX
- Push в оба репо — поддерживает синхронность между contextor и contextor-pro

 ### 12 мая 2026 — Переименование и новый репозиторий
 
 **Выполнено:**
 - ✅ Переименование Pure Intellect → Contextor (CTX)
   - 92 файла изменено, 357+ вхождений обновлено
   - Пакет: `pure_intellect` → `contextor`
   - PyPI: `pure-intellect` → `contextor`
   - Слоган: «Your context, orchestrated.»
 - ✅ Восстановлен полный README.md (370 строк)
   - Донаты (Boosty, ЮMoney), автор, лицензия, контакты
 - ✅ Создан новый репозиторий Contextor-pro на GitHub
   - Remote: `contextor-pro` → https://github.com/Remchik64/Contextor-pro
   - Вся история проекта запушена
   - Старый репо `origin` (pure-intellect) — архивный, НЕ трогать
 - ✅ Строгое правило: не пушить в неправильный репозиторий
 
 **Коммиты:**
 - `d5280cb` feat: rename project from Pure Intellect to Contextor (CTX)
 - `6ad3a39` fix: restore full README with all sections - donations, author, license, contacts
 - `0e89f10` docs: update HANDOFF.md with Contextor-pro repo info
 
 **Два remote:**
 | Remote | URL | Назначение |
 |--------|-----|------------|
 | `origin` | github.com/Remchik64/pure-intellect | Архив (только чтение) |
 | `contextor-pro` | github.com/Remchik64/Contextor-pro | Активная разработка |
 
 **Важно:** При git push ВСЕГДА показывать remote и branch, получать подтверждение пользователя. Никогда не путать репозитории.
 
 **Следующий шаг:** Начать реализацию Фазы 1 — UCIP v2, стабильность ядра, CCI improvements.

### 14 мая 2026 — Очистка Admin Panel от мёртвого кода

**Выполнено:**

1. ✅ **Удаление API Gateway + Agent Zero Integration** (из предыдущего чата)
   - `index.html` (3392→3009 строк): удалены nav-item Connect, CSS `.conn-card`/`.conn-field`/`.conn-value`, HTML `section-connections`, Utility Model card, JS-функции (`loadConnections`, `checkEndpoints`, `loadOpenAIModels`, `loadAZPluginConfig`, `saveAZPluginConfig`, `setUtilityModel`, `warmUtilityModel`), AZ references в таблице моделей
   - `routes.py` (1976→1616 строк): удалены `_extract_first_json`, `_inject_pi_notifications`, `_create_az_coordinate`, `_AZ_COORDINATE_MSG_THRESHOLD`, весь блок `is_agent_zero`, секция AZ Plugin Config (`_AZ_PLUGIN_CONFIG_FILE`, `_DEFAULT_AZ_PLUGIN_CONFIG`, `AZPluginConfigModel`, `_load/_save_az_plugin_config`, `GET/POST /az-plugin/config`). `source="agent_zero"` → `source="user"`
   - `server.py` (310→224 строк): удалены `_load_az_plugin_utility_model`, `_load_az_plugin_embedding_model`. Упрощён приоритет моделей: `config.yaml` → Ollama (убран fallback `az_plugin_config.yaml`)
   - `az_plugin_config.yaml`: удалён

2. ✅ **Удаление Projects Tab** (из предыдущего чата)
   - `index.html` (3090→2816 строк): удалены nav-item Projects, CSS `.watcher-bar`/`.watcher-status-text`, HTML `section-projects`, JS-функции (`loadProjects`, `updateWatcherStatus`, `startWatcher`, `stopWatcher`, `loadWatcherChanges`, `indexProject`, `openNewProject`, `searchCode`), убран `'projects'` из массива sections
   - `routes.py` (1616→1455 строк): удалены `IndexProjectRequest`, `CodeSearchRequest`, эндпоинты `/code/` (index, search, stats, graph, watcher/status|start|stop|changes|scan)
   - Python-модули (`code_module.py`, `watcher.py`, `code_memory.py`, `watcher_integration.py`) и старые `/watcher/` эндпоинты — сохранены

3. ✅ **Исправление Model Size Detection Warning**
   - Проблема: `/api/show` возвращает `size=0` для незагруженных моделей → «safe mode: generator only»
   - Фикс в `server.py`: добавлен fallback на `/api/tags` когда `/api/show` возвращает `size=0`
   - Результат: корректные размеры (`qwen3.5:2b: 2.6 GB`, `qwen3.5:9b: 6.1 GB`), правильное VRAM-планирование

4. ✅ **Исправление Admin Panel Starts Empty**
   - Проблема: страница загружалась пустой, `showSection('dashboard')` в DOMContentLoaded
   - Фикс: изменено на `showSection('chat')`

5. ✅ **Полная чистка мёртвого dashboard-кода** (этот чат)
   - `index.html` (2816→2448 строк, −368 строк):
     - Удалён закомментированный dashboard nav item
     - Удалена закомментированная dashboard HTML-секция
     - Удалены JS-функции: `loadDashboard()`, `loadSessions()`, `switchSession()`, `deleteSessionById()`, `renameSession()`, `createNewSession()`, `deleteCoordinate()`, `loadCoordinates()`, `startDashboardRefresh()`
     - Удалены переменные: `dashRefreshTimer`, `case 'dashboard'` в switch
     - Удалены вызовы: `loadDashboard()` из `switchSession()`/`createNewSession()`, `startDashboardRefresh()` из DOMContentLoaded
     - Удалены осиротевшие CSS: `.session-item*`, `.coord-item*`, `.cci-bars`, `.cci-bar`, `.grid-4`
     - Удалён `'dashboard'` из sections array
   - Восстановлена `updateDualModelUI()` для Models tab
   - 0 оставшихся ссылок на удалённые функции

**Итого за сессию:** index.html сокращён с 3392→2448 строк (−944 строки, −28%)

**Следующие шаги:**
- Начать реализацию Фазы 1 — UCIP v2

### 14 мая 2026 (продолжение) — Полная чистка мёртвого кода

**Выполнено:**

1. ✅ **Удаление 30 мёртвых API-эндпоинтов из routes.py** (1455→813 строк, −642)
   - `/models` (GET), `/model/load`, `/intent`, `/index`, `/cards/search`
   - `/retrieve`, `/assemble`, `/graph/*`, `/watcher/*`, `/orchestrate`
   - `/coordinates/*`, `/memory/fact`, `/memory/search`, `/cci/reset`
   - `/session/save`, `/sessions/*`, `/dual-model/refresh`, `/models/warm`
   - `/memory/fact/{id}` (DELETE), + 5 Pydantic моделей

2. ✅ **Удаление мёртвых экспортов из core/__init__.py** (23→19 строк)
   - Убраны `FileWatcher`, `WatcherIntegration`

3. ✅ **Удаление мёртвого кода из orchestrator.py** (940→908 строк, −32)
   - Убраны `CodeAwareMemoryIntegration` импорт, `_code_module`, `_code_aware`
   - Удалены блоки C3: Code-Aware Memory (code_context + process_code_turn)
   - Убраны `SESSION_TYPE_CHAT`, `SESSION_TYPE_PROJECT` импорты

4. ✅ **Удаление 5 мёртвых Python-модулей** (~1100 строк)
   - `watcher.py` (154 строк), `watcher_integration.py` (103 строк)
   - `code_module.py` (467 строк), `code_memory.py` (278 строк)
   - `archive.py` (122 строк)

5. ✅ **Удаление 6 тестов мёртвых модулей** (~900 строк)
   - `test_code_module.py`, `test_code_memory.py`, `test_watcher_c2.py`
   - `test_graph.py`, `test_parser.py`, `test_retriever.py`

6. ✅ **Очистка неиспользуемых импортов**
   - `card_generator.py`: удалён `Any`
   - `summarizer.py`: удалены `json`, `Path`
   - `orchestrator.py`: удалены `SESSION_TYPE_CHAT`, `SESSION_TYPE_PROJECT`, `CodeAwareMemoryIntegration`

7. ✅ **Очистка config.py** (81→70 строк, −11)
   - Удалён `archive_dir` (мёртвый — archive.py удалён)
   - Удалена секция File Watcher settings (`supported_extensions`, `ignore_dirs`)

**Итого за сессию чистки:**
- Python: удалено ~2670+ строк кода + ~900 строк тестов
- Frontend: index.html сокращён с 3392→2448 строк (−944 строки, −28%)
- routes.py: 1976→813 строк (−59%)
- Удалено 11 файлов (5 модулей + 6 тестов)
- Все импорты проходят проверку ✅
- Сервер перезапущен, все живые эндпоинты работают, удалённые возвращают 404

**Аудит зависимостей (отложено — чистка pyproject.toml):**

| Зависимость | Используется? | Кем | Действие |
|-------------|--------------|-----|----------|
| `watchdog` | ❌ 0 импортов | Только watcher.py (удалён) | Убрать из pyproject.toml |
| `tree-sitter-javascript` | ❌ 0 импортов | JS parser не существует | Убрать из pyproject.toml |
| `tree-sitter-typescript` | ❌ 0 импортов | TS parser не существует | Убрать из pyproject.toml |
| `aiofiles` | ❌ 0 импортов | Не используется | Убрать из pyproject.toml |
| `python-dotenv` | ❌ 0 импортов | Не используется | Убрать из pyproject.toml |
| `rich` | ❌ 0 импортов | Не используется | Убрать из pyproject.toml |
| `tree-sitter` | ✅ 1 импорт | python_parser.py | Оставить |
| `tree-sitter-python` | ✅ 1 импорт | python_parser.py | Оставить |
| `networkx` | ✅ 1 импорт | graph.py | Оставить |
| `chromadb` | ✅ 4 импорта | retriever.py, card_generator.py | Оставить |
| `tiktoken` | ✅ 1 импорт | tokenizer.py | Оставить |
| `ddgs` | ✅ 1 импорт | utility_worker.py | Оставить |
| `huggingface_hub` | ✅ 1 импорт | model_manager.py | Оставить |
| `click` | ✅ 1 импорт | __main__.py | Оставить |

Также проверить: `llama-cpp-python[cuda]` в optional — всё ещё нужен?

**Известная особенность:** После перезапуска сервера нужно обновить страницу в браузере (F5) для восстановления WebSocket.

**Следующие шаги:**
- ✅ Почистить pyproject.toml (6 неиспользуемых зависимостей) — ВЫПОЛНЕНО
- Начать реализацию Фазы 1 — UCIP v2

### 16 мая 2026 — Рефакторинг UI, API-модули, Code Review

**Выполнено:**

1. ✅ **Рефакторинг index.html: 2448→341 строка (−86%)**
   - CSS вынесен в `static/css/style.css` (1040 строк)
   - JS вынесен в `static/js/app.js` (1065 строк)
   - index.html — чистая HTML-разметка, без инлайн-стилей и скриптов

2. ✅ **Рефакторинг routes.py: 813→17 строк (−98%)**
   - Разделён на модули: `api/state.py` (56), `api/chat.py` (42), `api/session.py` (31), `api/memory_api.py` (43), `api/models_api.py` (310), `api/system.py` (353)
   - `routes.py` — роутер-хаб, делегирует в подмодули

3. ✅ **Code Review — 3 критические проблемы исправлены**
   - `style.css`: незакрытый `@keyframes pulse-think` — исправлено
   - `websocket.py`: `get_pipeline()` → `_get_pipeline()` — исправлено (ленивый импорт)
   - `style.css`: лишняя `}` в конце файла — удалена

4. ✅ **Code Review — 4 важных замечания исправлено**
   - **Централизация Ollama URL**: 21 захардкоженный `http://host.docker.internal:11434` → `settings.ollama_url` в 10 файлах (server.py, swap_manager.py, hardware_detector.py, tagger.py, storage.py, utility_worker.py, dual_model.py, models_api.py, system.py, websocket.py)
   - **Pydantic модели**: `SwitchModelRequest` и `DownloadModelRequest` добавлены в `schemas.py`, `models_api.py` обновлён
   - **Динамическая версия**: `<span>v0.2</span>` → `<span id="version-badge">` + JS fetch `/api/v1/version`
   - Удалены локальные константы `_OLLAMA_BASE`, `OLLAMA_BASE_URL` из swap_manager, tagger, storage, dual_model

5. ✅ **Исправлен баг WebSocket 'Ollama не запущена!'**
   - Удалена blocking `urllib` предпроверка из async handler (коммит 543ec02)
   - Причина: `urllib.request.urlopen()` блокировал event loop → WebSocket timeout

6. ✅ **6 неиспользуемых зависимостей убрано из pyproject.toml**
   - Удалены: watchdog, tree-sitter-javascript, tree-sitter-typescript, aiofiles, python-dotenv, rich

7. ✅ **Push в оба репозитория** (Contextor + Contextor-pro)

**Известные особенности:**
- После перезапуска сервера нужен F5 для WebSocket reconnect
- Задержка ответа ~25 сек (pipeline.run() синхронный) — ответ появляется целиком
- Тесты: 19 fail'ов (старые имена qwen2.5 vs qwen3.5) — pre-existing, не от текущих изменений

**Следующие шаги:**
- Обновить тесты под текущие имена моделей (qwen3.5)
- Начать реализацию Фазы 1 — UCIP v2
- Асинхронный pipeline.run() для стриминга ответов
- WebSocket auto-reconnect на клиенте

---

## 📋 Планы MVP v0.3 — Мульти-чат и рефакторинг UI

### 1. Система мульти-чатов и изоляция данных

**Проблема:** Сейчас все данные валятся в одну кучу — одна сессия, одни факты, одни координаты. После обновления браузера переписка теряется.

**Решение:**
- Чаты не теряются при обновлении браузера — история сохраняется на сервере
- Каждый чат = своя изолированная сессия со своими данными:
  - Факты (HOT/WARM)
  - Координаты (coordinate archive)
  - Якоря (Anchor facts)
  - История переписки (chat_history.json)
- Менеджер памяти показывает данные только активного чата
- Возможность создавать новые чаты

**Ключевые файлы:** `session_manager.py`, `storage/sessions/`, `index.html`

### 2. Рефакторинг UI — вкладки в шапке, фиксированный header

**Проблема:** Боковая навигация занимает место и не нужна для единственной вкладки чата.

**Решение:**
- Перенести все вкладки КРОМЕ чата из боковой панели в шапку админки
- Шапка зафиксирована при скроле (sticky header)
- Боковая панель остаётся только для списка чатов (см. пункт 3)
- Навигация в шапке: Memory | Models | Settings | Logs

### 3. Боковая панель — список чатов и переключение

**Проблема:** Нет возможности создавать новые чаты и переключаться между ними.

**Решение:**
- Боковая панель слева = список чатов (как в ChatGPT/Claude)
- Каждый чат отображается как элемент списка с названием и превью
- Кнопка «+ Новый чат» вверху боковой панели
- Переключение между чатами — клик по элементу списка
- Активный чат всегда открывается в менеджере памяти и содержит все данные по активному чату
- При создании нового чата создаётся новая изолированная сессия

**Порядок реализации:**
1. Сначала — серверная часть: мульти-сессии, изоляция данных (session_manager.py, routes.py)
2. Затем — UI: шапка с вкладками + боковая панель с чатами (index.html)
3. В конце — менеджер памяти, привязанный к активному чату

### 18 мая 2026 — Фиксы кодировки, web search, num_ctx из конфига, Settings API, книга Contextor

**Выполнено:**
- ✅ fix: add encoding='utf-8' to all read_text() calls (коммит 032f0ff) — исправлена ошибка Windows charmap codec в session.py, working_memory.py, storage.py, session_manager.py
- ✅ fix: improve web search query extraction and add web_search prompt instructions (коммит 6438b46) — utility_worker.py: keywords приоритетнее полного текста + regex очистка командных фраз; orchestrator.py: добавлены инструкции для web_search intent (НЕ говорить 'я не могу')
- ✅ feat: improve web search activation — expanded intent rules, query extraction, prompt instructions (коммит 4931cff) — расширены триггеры WEB_SEARCH в intent.py (актуальн, курсов, сегодня, сейчас, google, look up и др.); force_web_search в orchestrator.py извлекает поисковый запрос из сообщения
- ✅ feat: move num_ctx from hardcoded to config.yaml, add settings API endpoints, add num_ctx selector in Settings tab, add no-cache headers
- ✅ fix: pass utility_context to LLM system prompt
- ✅ fix: web search ddgs ImportError handling
- ✅ docs: книга-рассказ о Contextor: docs/CTX_BOOK.md (487 строк, 8 глав)
- ✅ Версия проекта обновлена до 0.3.0

**Что в процессе:**
- 🔄 Settings tab UI (v0.3.0, базовый функционал работает)
- 🔄 Web search intent detection — нуждается в тестировании на разных запросах

**Блокеры:**
- Нет

**Следующие шаги:**
- Обновить тесты под текущие имена моделей (qwen3.5 вместо qwen2.5)
- Переписать _create_coordinate() на английский (UCIP format)
- Добавить 4-слойный Context Package в _build_system_prompt()
- Улучшить CCI алгоритм (adaptive threshold)
- Асинхронный pipeline.run() для стриминга ответов
- WebSocket auto-reconnect на клиенте

**Решения и обоснования:**
- encoding='utf-8' добавлен ко всем read_text() — Windows по умолчанию использует cp1251, что ломает кириллицу в JSON файлах
- Web search triggers расширены — старые триггеры были слишком узкие, запросы типа 'актуальн' или 'сейчас' не попадали в WEB_SEARCH
- force_web_search извлекает запрос из сообщения — раньше LLM получала необработанное сообщение, теперь regex чистит командные фразы
- num_ctx вынесен в config.yaml — был захардкожен в dual_model.py:180, теперь можно менять через Settings API без перезапуска
