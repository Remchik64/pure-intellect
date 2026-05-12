# 🧠 CTX Module Architecture — Окончательное решение

> Архитектурный документ: CTX как модуль памяти для любых локальных AI-приложений.
> Дата: 2026-05-10
> Статус: Утверждённая архитектура

---

## 🎯 Главная идея

**CTX = Контекст-гипервизор.** Прозрачный прокси-слой между любым AI-приложением и LLM, который управляет контекстным окном так, как MMU управляет виртуальной памятью.

Приложения не знают о CTX. Модели не знают о CTX. CTX незримо управляет контекстом, сжимает историю, сохраняет факты и обеспечивает бесконечную работу.

---

## ✅ Выбранная архитектура: Smart Proxy (Module Mode)

### Почему именно Smart Proxy

| Критерий | Smart Proxy | MCP Server | Shared FS | Prompt Inject |
|----------|-------------|-------------|-----------|---------------|
| **Прозрачность для приложений** | ✅ Полная | ❌ Нужна поддержка | 🟡 Нужен prompt | ❌ Ручная |
| **Автоматический soft reset** | ✅ Да | ❌ Нет | ❌ Нет | ❌ Нет |
| **Бесконечный контекст** | ✅ Да | ❌ Нет | 🟡 Частично | ❌ Нет |
| **Работает с ЛЮБЫМ приложением** | ✅ Да | ❌ Только MCP | ❌ Нет | ❌ Нет |
| **Не ломает GEN/EXE** | ✅ Да | ✅ Да | ✅ Да | ✅ Да |
| **Не ломает tool calling** | ✅ Да | ✅ Да | ✅ Да | 🟡 Риск |
| **Сложность реализации** | Средняя | Низкая | Низкая | Минимальная |

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        CTX Module Proxy                          │
│                     (Слушает :11434 или :11435)                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Request Pipeline                         ││
│  │                                                             ││
│  │  1. IDENTIFY — определить сессию приложения                 ││
│  │  2. ANALYZE — токены, CCI, fill percent                     ││
│  │  3. DECIDE — passthrough / enrich / compress / prefetch    ││
│  │  4. EXECUTE — собрать Context Package если нужно             ││
│  │  5. FORWARD — отправить в Ollama                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Response Pipeline                          ││
│  │                                                             ││
│  │  1. PASSTHROUGH — переслать ответ как есть (streaming!)     ││
│  │  2. EXTRACT — извлечь факты из ответа (фоном)                ││
│  │  3. UPDATE — обновить CCI, токен-каунтер, память            ││
│  │  4. SAVE — сохранить факты в HOT/WARM, обновить координату  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    CTX Memory Core                            ││
│  │                                                             ││
│  │  Coordinator (3B) ─── Intent, Coordinates, Facts             ││
│  │  Embedding ───────── CCI, Search, RAG                       ││
│  │  ChromaDB ─────────── WARM/COLD semantic storage             ││
│  │  Sessions ─────────── Per-app isolated contexts              ││
│  │  Coordinates ──────── Compressed session snapshots           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         │                                    │
    ┌────▼────┐                        ┌──────▼──────┐
    │  Ollama  │                        │  Приложения │
    │ :11436   │                        │ (OpenCode,  │
    │(реальный) │                        │  Agent0,    │
    └─────────┘                        │  LM Studio) │
                                       └─────────────┘
```

---

## 📐 Минимальный набор моделей CTX (Module Mode)

```
┌──────────────────────────────────────┐
│         CTX Module (VRAM ~2.4GB)     │
│                                      │
│  Coordinator (3B)    ~2.0GB  ← ALWAYS│
│  Embedding           ~0.4GB  ← ALWAYS│
│                                      │
│  Generator (7B)      НЕ НУЖЕН        │
│  Utility (7B)         НЕ НУЖЕН        │
│  Swap Manager        НЕ НУЖЕН         │
│                                      │
│  Итого: 2.4GB                        │
│  Свободно для агента: ~9.6GB         │
└──────────────────────────────────────┘
```

---

## 🔄 Режимы управления контекстом

### 1. PASSTHROUGH — пропустить как есть
```
Условие: fill < 50% И cci > 0.7
Действие: переслать запрос без изменений
Фон: извлечь факты, обновить CCI
```

### 2. ENRICH — обогатить контекст
```
Условие: fill 50-80% И cci > 0.55
Действие: добавить Context Package system-сообщения
Фон: извлечь факты, обновить CCI
```

### 3. COMPRESS — мягкий сброс (soft reset)
```
Условие: fill > 80% ИЛИ cci < 0.55
Действие: заменить старую историю на Context Package
```

### 4. PREFETCH — предзагрузка для новой сессии
```
Условие: fill < 20% (новый чат)
Действие: инжектировать Context Package прошлой сессии
```

---

## 🧠 UCIP v2 — Universal Context Injection Protocol

### Ключевой инсайт: координата — это НАПРАВЛЕНИЕ, не содержание

Координата ОДНА не работает. Модель получает «ты был в городе, удачи» —
но не знает точных имён, сигнатур, решений, связей.

CTX решает это **4-слойным Context Package**:

| Слой | Что | Зачем | Без него модель... |
|------|-----|-------|-------------------|
| **1. Coordinate** | Направление — что делали, задача | Не знает ЧТО делали |
| **2. Memory** | Решения — что выбрано и почему | Предлагает отвергнутое |
| **3. Code RAG** | Точные сигнатуры функций и классов | Угадывает имена |
| **4. Graph** | Структура — что с чем связано | Не знает зависимости |

### Формат Context Package

Инжектируется как отдельные system-сообщения:

```json
[
  {"role": "system", "content": "You are a coding assistant..."},
  {"role": "system", "content": "[Previous conversation context: <coordinate>]"},
  {"role": "system", "content": "[Relevant context from memory: <facts>]"},
  {"role": "system", "content": "[Relevant code context: <code_snippets>]"},
  {"role": "system", "content": "[Project structure: <graph>]"},
  {"role": "user", "content": "<actual request>"}
]
```

### Layer 1: Coordinate (естественный язык, английский, ~200-400 токенов)
```
[Previous conversation context: We have been working on a todo-app using FastAPI with SQLAlchemy and SQLite. JWT authentication was implemented with HS256. The project has 5 files: main.py, models.py, routes.py, schemas.py, auth.py. Current task: adding pytest tests. Key decisions: SQLite for simplicity, REST conventions, JWT for auth.]
```

### Layer 2: Memory (факты с решениями, ~100-200 токенов)
```
[Relevant context from memory:
- JWT authentication uses HS256 algorithm with 30-minute expiration
- pytest-asyncio is used for async endpoint testing
- SQLite chosen for development simplicity, planning PostgreSQL for production
- bcrypt is used for password hashing
- API routes are mounted at /api/v1 in main.py]
```

### Layer 3: Code RAG (точные сигнатуры, ~300-500 токенов)
```
[Relevant code context:
auth.py:
  def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str
  async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User
models.py:
  class User(Base): id: int, username: str, email: str, hashed_password: str
  class Todo(Base): id: int, title: str, description: str, completed: bool, owner_id: int]
```

### Layer 4: Graph (связи, ~50-100 токенов)
```
[Project structure:
auth.py imports from models.py (User)
routes.py depends on auth.py (get_current_user)
User model linked to Todo model (owner_id)]
```

### Адаптивная длина

| Заполненность контекста | Слои | Размер |
|------------------------|------|--------|
| < 30% (новый чат) | Все 4 слоя, detailed | ~1200 токенов |
| 30-60% | Все 4 слоя, compressed | ~800 токенов |
| 60-80% | Coordinate + Memory + Code, compressed | ~600 токенов |
| > 80% (soft reset) | Coordinate + Memory, ultra_compressed | ~300 токенов |

### CTX уже имеет все компоненты

| Компонент | В CTX | Слой в Context Package |
|-----------|------|----------------------|
| MetaCoordinator | ✅ | Layer 1: Coordinate |
| WorkingMemory | ✅ | Layer 2: Memory |
| CardGenerator + Retriever | ✅ | Layer 3: Code RAG |
| GraphBuilder | ✅ | Layer 4: Graph |
| IntentDetector | ✅ | Выбор WHAT инжектировать |
| CCITracker | ✅ | Выбор КОГДА инжектировать |

---

## 🛡️ Строгие правила модификации контекста

### ЗАПРЕЩЕНО модифицировать:
- System prompt приложения (первое system-сообщение)
- Tool definitions в запросе
- Tool calls в ответе модели
- Tool call IDs
- Последние N сообщений (актуальный контекст)
- Параметры запроса (temperature, top_p, stop, etc)
- Формат ответа (JSON mode, etc)
- Streaming chunks (пересылаются как есть)

### РАЗРЕШЕНО модифицировать:
- Заменять старые сообщения (history > N) на Context Package
- Добавлять [CTX Context] system-сообщение (после system prompt)
- Добавлять [CTX Memory] system-сообщение (с фактами)
- Добавлять [CTX Code] system-сообщение (с сигнатурами)
- Добавлять [CTX Graph] system-сообщение (со связями)

### ПРАВИЛО:
**CTX добавляет, но не удаляет актуальное. CTX сжимает старое, но не трогает свежее.**

---

## 🔌 Как приложения подключаются к CTX

### Способ 1: Смена URL (рекомендуемый)
```bash
opencode --model qwen3:7b --ollama-host http://localhost:11435
```

### Способ 2: Порт-перехват (нулевой конфиг)
```bash
OLLAMA_HOST=0.0.0.0:11436 ollama serve &
PI_PROXY_UPSTREAM=http://localhost:11436 PI_PROXY_LISTEN=0.0.0.0:11434 contextor serve --mode module
```

### Способ 3: MCP Server (дополнительный путь)
```json
{"mcpServers": {"pi-memory": {"command": "pi-mcp-server", "args": ["--port", "3005"]}}}
```

---

## 📊 Сессии: изоляция + общая память

```
┌─────────────────────────────────────────────────┐
│                  CTX Sessions                     │
│                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │ Session #1   │  │ Session #2   │  │Session #3││
│  │ OpenCode     │  │ Agent Zero   │  │LM Studio ││
│  │              │  │              │  │          ││
│  │ Own context  │  │ Own context  │  │Own ctx   ││
│  │ Own CCI      │  │ Own CCI      │  │Own CCI   ││
│  │ Own coords   │  │ Own coords   │  │Own coord ││
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘│
│         └────────┬────────┘───────────────┘      │
│                  │                               │
│         ┌────────▼────────┐                     │
│         │  SHARED MEMORY   │                     │
│         │  HOT/WARM/COLD   │                     │
│         │  + Code RAG      │                     │
│         │  + Graph         │                     │
│         └─────────────────┘                     │
└─────────────────────────────────────────────────┘
```

---

## 🔧 Streaming и Tool Calling

### Streaming: SSE proxy
- CTX пересылает chunks как есть, без буферизации
- После завершения потока — асинхронно извлекает факты
- Задержка для пользователя: 0ms

### Tool Calling: полный passthrough
- CTX НЕ трогает tools[], tool_calls, tool_call_id
- CTX НЕ модифицирует tool results
- CTX извлекает факты из tool call аргументов (фоном)
- При soft reset — tool call история включается в координату

---

## 📐 Конфигурация Module Mode

```yaml
mode: module

coordinator:
  model: "qwen2.5:3b"
  gpu_layers: -1
  temperature: 0.1
  max_tokens: 512

embedding:
  model: "nomic-embed-text"
  gpu_layers: -1

proxy:
  enabled: true
  listen: "0.0.0.0:11435"
  upstream: "http://localhost:11434"
  strategy: smart
  
  compress:
    cci_threshold: 0.55
    fill_threshold: 0.80
    min_messages: 8
    keep_recent: 6
  
  enrich:
    fill_min: 0.20
    fill_max: 0.80
    max_facts: 5
    max_tokens: 1000
  
  prefetch:
    enabled: true
    max_tokens: 500
  
  passthrough_paths:
    - "/api/embed"
    - "/api/show"
    - "/api/tags"
  
  passthrough_models:
    - "nomic*"
  
  memory:
    extract_facts: true
    background: true
    hot_facts_max: 50
    warm_search_top_k: 5

sessions:
  auto_detect: true
  isolation: "per_app"
```

---

## 🚀 План реализации

### Фаза 1: Прозрачный прокси (2 недели)
- HTTP прокси-сервер (FastAPI + httpx)
- Идентификация сессий
- Token counting
- Streaming passthrough (SSE proxy)
- Passthrough для не-чат запросов
- Tool calling passthrough

### Фаза 2: Анализ и решение (2 недели)
- CCI calculation для внешних запросов
- Fill percent calculation
- Decision engine (PASSTHROUGH / ENRICH / COMPRESS / PREFETCH)
- Session management (per-app изоляция)
- Background fact extraction pipeline

### Фаза 3: Context Package (3 недели)
- Coordinator: coordinate generation для внешних запросов (UCIP v2, английский)
- Context surgery: замена старой истории на 4-слойный Context Package
- Enrichment: инжекция релевантных фактов из памяти
- Prefetch: инжекция координаты прошлой сессии
- Code RAG: инжекция релевантных сигнатур кода
- Graph: инжекция связей между сущностями

### Фаза 4: Memory integration (2 недели)
- Per-session CCI tracking
- Per-session token counting
- Cross-session fact sharing (HOT/WARM/COLD)
- Coordinate archive per session
- CTX Dashboard: просмотр сессий, CCI, координат

### Фаза 5: MCP Server (1 неделя)
- MCP server для CTX Memory API
- Инструменты: pi_memory_search, pi_memory_save, pi_context, pi_coordinate
- Документация для интеграции с OpenCode, Claude, etc.

---

## 🔑 Ключевые принципы

1. **CTX НЕ генерирует ответы.** CTX управляет контекстом. Агент генерирует.
2. **CTX НЕ ломает GEN/EXE.** CTX не модифицирует структуру запросов, только управляет размером окна.
3. **CTX добавляет, не удаляет.** Context Package — это ДОБАВЛЕНИЕ, не замена.
4. **CTX прозрачен.** Приложение не знает о CTX, просто работает лучше.
5. **CTX минимален.** 2.4GB VRAM. Coordinator + Embedding. Больше не нужно.
6. **CTX вечен.** Контекст никогда не переполняется. Координаты хранятся вечно.
7. **CTX общий.** Факты разделяются между сессиями. Опыт одного агента доступен другому.
8. **Координата = направление.** За координатой стоит полный Context Package: память + код + граф.
