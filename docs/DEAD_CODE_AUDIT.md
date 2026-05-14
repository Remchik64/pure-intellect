# 🔍 Аудит мёртвого кода — Contextor

**Дата**: 2025-05-14
**Область**: `/a0/usr/workdir/pure-intellect/src/contextor/`

---

## 📋 Сводка

| Категория | Количество | Критичность |
|-----------|-----------|------------|
| Мёртвые модули | 5 | 🔴 Высокая |
| Мёртвые API-эндпоинты | ~25 | 🟡 Средняя |
| Неиспользуемые импорты | 7 | 🟢 Низкая |
| Мёртвые функции/методы | ~15 (не route handlers) | 🟡 Средняя |
| Пустые тестовые файлы | 3 | 🟢 Низкая |
| Тесты для мёртвых модулей | 3 | 🟡 Средняя |

---

## 1. 🗑️ Мёртвые Python-модули

### 1.1 `core/watcher.py` — Файловый наблюдатель
- **Строка**: весь файл (~120 строк)
- **Ссылки**: Импортируется в `watcher_integration.py` и `__init__.py`
- **Доказательство**: Frontend НЕ вызывает `/watcher/*` эндпоинты. Watcher-функционал никогда не активируется в production.
- **Рекомендация**: 🗑️ **Удалить**

### 1.2 `core/watcher_integration.py` — Интеграция watcher
- **Строка**: весь файл (~100 строк)
- **Ссылки**: Импортируется в `code_module.py` (lazy import) и `__init__.py`
- **Доказательство**: Зависит от watcher.py, который мёртвый. Frontend не использует.
- **Рекомендация**: 🗑️ **Удалить**

### 1.3 `core/code_module.py` — Индексация кода (CodeModule)
- **Строка**: весь файл (~450 строк)
- **Ссылки**: `orchestrator.py:117` — `self._code_module = None` (никогда не создаётся)
- **Доказательство**: `_code_module` инициализируется как `None` и нигде не присваивается реальный экземпляр. Условный код в `orchestrator.py:443-482` никогда не выполняется (`if self._code_module is not None` — всегда False).
- **Рекомендация**: 🗑️ **Удалить** (вместе с watcher integration)

### 1.4 `core/code_memory.py` — Контекстная память кода
- **Строка**: весь файл (~260 строк)
- **Ссылки**: `orchestrator.py:16,118,480` — `CodeAwareMemoryIntegration` создаётся, но вызов `process_code_turn()` зависит от `_code_module` (всегда None)
- **Доказательство**: Строка 480: `self._code_aware.process_code_turn(...)` вызывается только если `code_context` получен от `_code_module` (строка 479: `if self._code_module is not None and code_context`), что никогда не true.
- **Рекомендация**: 🗑️ **Удалить** (код никогда не выполняется)

### 1.5 `core/archive.py` — Архив сессий
- **Строка**: весь файл (~100 строк)
- **Ссылки**: Нигде не импортируется. `config.py:47` упоминает `archive_dir` (конфигурация), `optimizer.py` использует слово "archive" в комментариях, но не импортирует модуль.
- **Доказательство**: `grep -rn 'archive' src/` показывает, что ни один файл не делает `from .archive import` или `import archive`.
- **Рекомендация**: 🗑️ **Удалить** (если не планируется использовать в будущем)

---

## 2. 🔌 Мёртвые API-эндпоинты

Frontend вызывает только эти эндпоинты:
```
/api/v1/cci/stats, /api/v1/session/info, /api/v1/dual-model/stats,
/api/v1/session (DELETE), /api/v1/memory/stats, /api/v1/memory/facts,
/api/v1/memory/clear, /api/v1/models/status, /api/v1/ollama/models,
/api/v1/models/switch, /api/v1/models/{name} (DELETE),
/api/v1/hardware/detect, /api/v1/models/download,
/api/v1/models/download/check/{name}, /api/v1/config,
/api/v1/config/reload, /api/v1/logs (GET, DELETE)
```

### Мёртвые эндпоинты (определены в routes.py, не вызываются frontend):

| Эндпоинт | Строка | Метод | Причина смерти | Рекомендация |
|----------|--------|-------|---------------|-------------|
| `/health` | L72 | GET | Docker probe? | ⚠️ Проверить Docker/monitoring |
| `/models` | L84 | GET | Заменён на `/models/status` | 🗑️ Удалить |
| `/model/load` | L101 | POST | Заменён на `/models/switch` | 🗑️ Удалить |
| `/chat` | L121 | POST | WebSocket используется | ⚠️ Проверить клиентов |
| `/intent` | L154 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/index` | L185 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/cards/search` | L205 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/retrieve` | L232 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/assemble` | L275 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/graph/build` | L303 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/graph/stats` | L321 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/graph/search` | L334 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/graph/file` | L351 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/watcher/start` | L364 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/watcher/stop` | L381 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/watcher/scan` | L395 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/watcher/status` | L414 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/orchestrate` | L426 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/memory/fact` | L495 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/memory/fact/{fact_id}` | L1031 | DELETE | Frontend не вызывает | 🗑️ Удалить |
| `/memory/search` | L518 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/cci/reset` | L572 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/coordinates` | L583 | GET | Frontend не вызывает | 🗑️ Удалить |
| `/coordinates/{turn}` | L456 | DELETE | Frontend не вызывает | 🗑️ Удалить |
| `/session/save` | L618 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/sessions` | L764 | GET | Frontend не вызывает | ⚠️ Проверить |
| `/sessions/new` | L775 | POST | Frontend не вызывает | ⚠️ Проверить |
| `/sessions/{id}/switch` | L791 | POST | Frontend не вызывает | ⚠️ Проверить |
| `/sessions/{id}/rename` | L807 | PATCH | Frontend не вызывает | ⚠️ Проверить |
| `/sessions/{id}` | L819 | DELETE | Frontend не вызывает | ⚠️ Проверить |
| `/dual-model/refresh` | L651 | POST | Frontend не вызывает | 🗑️ Удалить |
| `/models/warm` | L984 | POST | Frontend не вызывает | 🗑️ Удалить |

**Итого**: ~25 мёртвых эндпоинтов. Из них ~5 стоит проверить (health, chat, sessions) — возможно используются внешними клиентами или мониторингом.

---

## 3. 📦 Неиспользуемые импорты

| Файл | Импорт | Доказательство | Рекомендация |
|------|--------|----------------|-------------|
| `core/orchestrator.py` | `SESSION_TYPE_CHAT` | Не используется в коде | 🗑️ Удалить |
| `core/orchestrator.py` | `SESSION_TYPE_PROJECT` | Не используется в коде | 🗑️ Удалить |
| `core/code_module.py` | `annotations` | Не используется (только `from __future__ import annotations`) | ⚠️ Проверить, может быть для type hints |
| `core/code_memory.py` | `annotations` | Не используется | ⚠️ Проверить |
| `core/watcher_integration.py` | `Callable` | Не используется | 🗑️ Удалить |
| `core/watcher_integration.py` | `settings` | Не используется | 🗑️ Удалить |
| `core/graph_builder.py` | `CodeCard` | Не используется | 🗑️ Удалить |
| `core/card_generator.py` | `Any` | Не используется | 🗑️ Удалить |
| `core/summarizer.py` | `json` | Не используется | 🗑️ Удалить |
| `core/summarizer.py` | `Path` | Не используется | 🗑️ Удалить |

---

## 4. ☠️ Мёртвые функции/методы

Функции, которые определены, но никогда не вызываются (исключая FastAPI route handlers и __dunder__):

### archive.py (мёртвый модуль — все функции)
| Функция | Строка | Рекомендация |
|---------|--------|-------------|
| `add_pair` | L32 | 🗑️ Удалить с модулем |
| `get_conversation_summary` | L52 | 🗑️ Удалить с модулем |
| `save_session` | L72 | 🗑️ Удалить с модулем |
| `load_session` | L89 | 🗑️ Удалить с модулем |

### code_module.py (мёртвый модуль — все функции)
| Функция | Строка | Рекомендация |
|---------|--------|-------------|
| `is_indexed` (property) | L146 | 🗑️ Удалить с модулем |
| `indexed_files` (property) | L150 | 🗑️ Удалить с модулем |
| `index_project` | L153 | 🗑️ Удалить с модулем |
| `build_graph` | L322 | 🗑️ Удалить с модулем |
| `start_watcher` | L366 | 🗑️ Удалить с модулем |
| `stop_watcher` | L415 | 🗑️ Удалить с модулем |
| `watcher_status` | L428 | 🗑️ Удалить с модулем |
| `scan_changes_now` | L440 | 🗑️ Удалить с модулем |

### code_memory.py (мёртвый модуль)
| Функция | Строка | Рекомендация |
|---------|--------|-------------|
| `format_for_working_memory` | L106 | 🗑️ Удалить с модулем |

### card_generator.py
| Функция | Строка | Рекомендация |
|---------|--------|-------------|
| `get_card_by_id` | L193 | ⚠️ Проверить, может используется через GraphBuilder |

### watcher.py / watcher_integration.py (мёртвые модули)
Все методы — 🗑️ удалить вместе с модулями.

---

## 5. 🧪 Мёртвые и пустые тесты

### Пустые тестовые файлы (0 байт):
| Файл | Рекомендация |
|------|-------------|
| `tests/test_graph.py` | 🗑️ Удалить |
| `tests/test_parser.py` | 🗑️ Удалить |
| `tests/test_retriever.py` | 🗑️ Удалить |

### Тесты для мёртвых модулей:
| Файл | Что тестирует | Рекомендация |
|------|--------------|-------------|
| `tests/test_code_module.py` (8.6 KB) | `code_module.py` | 🗑️ Удалить с модулем |
| `tests/test_code_memory.py` (11 KB) | `code_memory.py` | 🗑️ Удалить с модулем |
| `tests/test_watcher_c2.py` (14.3 KB) | `watcher_integration.py` | 🗑️ Удалить с модулем |

---

## 6. 📊 Неиспользуемые переменные/константы

### config.py
- `model_config` (L69) — класс `ModelConfig`, используется только внутри самого класса конфигурации. Проверить, ссылаются ли на неё внешние модули.
- `archive_dir` (L47) — привязана к мёртвому `archive.py`. Если модуль удалить, поле тоже не нужно.

### __init__.py (core)
Экспортирует мёртвые модули:
```python
from .watcher import FileWatcher          # 🗑️ мёртвый
from .watcher_integration import WatcherIntegration  # 🗑️ мёртвый
```

### routes.py
Импортирует мёртвые модули для watcher-эндпоинтов:
```python
# Строки, связанные с watcher/code_module — можно удалить вместе с эндпоинтами
```

---

## 7. 🏗️ Цепочка зависимостей мёртвого кода

```
watcher.py ← watcher_integration.py ← code_module.py ← orchestrator.py (_code_module=None)
                                           ↑
                                     code_memory.py ← orchestrator.py (_code_aware создан, но process_code_turn никогда не выполняется)

archive.py (никто не импортирует)
```

**Безопасный порядок удаления:**
1. `routes.py`: удалить watcher-эндпоинты (L364-436), /intent, /index, /cards/search, /retrieve, /assemble, /graph/*, /orchestrate, /coordinates/*, /memory/fact, /memory/search, /cci/reset, /sessions/*, /dual-model/refresh, /model/load, /models (старый), /models/warm, /session/save, /chat (если проверено)
2. `core/__init__.py`: убрать экспорт `FileWatcher`, `WatcherIntegration`
3. `orchestrator.py`: убрать `_code_module`, `_code_aware`, `CodeAwareMemoryIntegration` импорт, код в `orchestrate()` (L443-482)
4. Удалить файлы: `watcher.py`, `watcher_integration.py`, `code_module.py`, `code_memory.py`, `archive.py`
5. Удалить тесты: `test_code_module.py`, `test_code_memory.py`, `test_watcher_c2.py`, `test_graph.py`, `test_parser.py`, `test_retriever.py`
6. Почистить импорты в каждом файле
7. Убрать `archive_dir` из `config.py`

---

## 8. 📈 Оценка экономии

| Что удаляем | Строк кода | Размер |
|------------|-----------|--------|
| watcher.py | ~120 | 4 KB |
| watcher_integration.py | ~100 | 3 KB |
| code_module.py | ~450 | 15 KB |
| code_memory.py | ~260 | 9 KB |
| archive.py | ~100 | 3 KB |
| Мёртвые эндпоинты в routes.py | ~400 | 15 KB |
| Мёртвые импорты | ~10 | 0.5 KB |
| Тесты мёртвых модулей | ~34 KB | 34 KB |
| Пустые тесты | 0 | 0 |
| **Итого** | **~1340+ строк** | **~83 KB** |

---

*Аудит проведён без изменения файлов. Все рекомендации требуют ручного подтверждения перед удалением.*
