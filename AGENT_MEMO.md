# AGENT_MEMO — Pure Intellect

> Читай это первым. Игнорируй старые воспоминания если они противоречат этому файлу.

---

## Проект

**Pure Intellect** — иерархическая система памяти для LLM.
Решает проблему деградации длинного контекста: обнуляет контекст (soft reset), сохраняет «координату» (сжатый снимок), восстанавливает память через семантический поиск.

- **Пользователь**: Александр
- **Железо**: RTX 3060 12GB, Windows 11
- **Модели**: `mistral-small3.1:24b` (generator), `qwen3.5:2b` (coordinator)
- **Репо**: `github.com/Remchik64/pure-intellect`
- **PI сервер**: порт `7860`
- **Тесты**: 465 проходят

---

## Что работает в PI

- `WorkingMemory` — факты в RAM
- `MemoryStorage` — ChromaDB + SentenceTransformer CUDA embeddings
- `Anchor Facts` — критические факты не удаляются
- `Soft Reset + Coordinate` — обнуление контекста без потери памяти
- `Context Coherence Index (CCI)` — отслеживает связность диалога
- REST API на порту 7860
- Web UI + Admin Panel
- Сохранение сессий в `storage/sessions/`
- Preload моделей при старте (VRAM check)

---

## Стратегия интеграции с Agent Zero

### НЕПРАВИЛЬНО (старый путь — выбросить из головы)

```
Agent Zero → Pure Intellect (/v1/chat/completions proxy) → Ollama
```

Почему плохо:
- Ломает GEN/EXE цикл Agent Zero
- 1000 строк кода ради связи
- JSON wrapper портит ответы модели
- Fake диалоги сбивают модель с толку
- `reset_turn` ошибки

### ПРАВИЛЬНО (новый путь)

```
Agent Zero → его модель напрямую (GEN/EXE НЕ ТРОНУТ)
     ↕  только memory операции
Pure Intellect REST API (порт 7860)
```

**Конкретно**: клонировать Agent Zero, переписать только файлы памяти (`memory_save`, `memory_load`) чтобы они вызывали PI API вместо встроенной ChromaDB.

Agent Zero думает что использует свою память — но на самом деле использует PI с иерархической памятью, Anchor Facts и Coordinate system.

---

## Следующий шаг (активная задача)

1. Изучить файлы памяти в `/a0/python/tools/` или `/a0/helpers/memory.py`
2. Клонировать Agent Zero: `git clone https://github.com/frdel/agent-zero /a0/usr/workdir/agent-zero-clone`
3. Переписать memory backend → PI API эндпоинты
4. Запустить оба сервера (PI порт 7860, AZ клон другой порт)
5. Тестировать GEN → EXE → memory_save → PI → memory_load
6. Пушить в репо

---

## Что НЕ делать

- НЕ перехватывать LLM запросы AZ через `/v1/chat/completions`
- НЕ добавлять fake диалоги в промпт
- НЕ заворачивать ответы модели в JSON wrapper
- НЕ трогать GEN/EXE цикл Agent Zero
- НЕ писать файлы через shell heredoc (SyntaxError с `\n`)
- НЕ бросаться исправлять без понимания архитектуры

---

## Правила разработки

- Думать на **русском языке**
- Библиотеки устанавливать только в локальный venv: `/a0/usr/workdir/pure-intellect/venv`
- НЕ трогать системный `/opt/venv-a0`
- Файлы писать через **Python runtime**, не через shell heredoc
- Сначала **ПОНЯТЬ**, потом **ДЕЛАТЬ**
- Перед изменением кода — прочитать файл, понять архитектуру

---

## Ключевые файлы PI

```
src/pure_intellect/
  api/routes.py          — REST API эндпоинты
  core/orchestrator.py   — главный pipeline
  core/memory/           — WorkingMemory, MemoryStorage, Scorer, Optimizer
  core/session_manager.py
  core/dual_model.py     — routing coordinator/generator
  server.py              — FastAPI app + preload моделей
  config.py              — конфигурация
config.yaml              — модели, параметры
storage/sessions/        — персистентные данные
```

## PI API эндпоинты (для AZ интеграции)

```
GET  /api/v1/memory/search?q=<query>&limit=5  — семантический поиск фактов
POST /api/v1/memory/fact                       — сохранить факт
GET  /api/v1/memory/stats                      — статистика памяти
DEL  /api/v1/memory/clear                      — очистить память
GET  /api/v1/health                            — статус сервера
```
