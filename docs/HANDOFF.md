# 📋 Записка для следующего чата — Contextor Development Context

> Дата: 2026-05-12
> Создал: Agent Zero (hacker profile)
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
| `src/contextor/core/orchestrator.py` | 940 | Главный пайплайн, Soft Reset, координаты |
| `src/contextor/api/routes.py` | 1976 | API эндпоинты (нуждается в рефакторинге) |
| `src/contextor/core/memory/` | — | Иерархическая память (HOT/WARM/COLD) |
| `src/contextor/core/dual_model.py` | 325 | Маршрутизация Coordinator/Generator |
| `src/contextor/utils/swap_manager.py` | 200 | VRAM Swap Manager |
| `src/contextor/core/memory/meta_coordinator.py` | 254 | Мета-координаты (сжатие координат) |
| `src/contextor/core/memory/working_memory.py` | 396 | Рабочая память (HOT) |
| `src/contextor/config.py` | — | Конфигурация (Pydantic Settings) |

---

## ⚠️ Известные проблемы

1. **routes.py — 1976 строк** — монолит, нужен рефакторинг на модули
2. **Coordinator prompt на русском** — для Module Mode нужен английский
3. **Нет proxy pipeline** — основная задача Фазы 2
4. **Нет UCIP v2** — координаты генерируются в русском шаблонном формате
5. **CCI threshold фиксированный (0.55)** — нужен adaptive

---

## 🚀 Следующий шаг

**Начать реализацию Фазы 1 — Стабильность ядра:**
1. Исправить известные баги в orchestrator.py
2. Переписать _create_coordinate() на английский (UCIP format)
3. Добавить 4-слойный Context Package в _build_system_prompt()
4. Улучшить CCI алгоритм
5. Упростить UI (убрать лишнее)

Для начала работы прочитать:
- `docs/CTX_MODULE_ARCHITECTURE.md` — полная архитектура
- `docs/ROADMAP.md` — дорожная карта с задачами
- `docs/architecture.md` — оригинальная архитектура проекта
- `src/contextor/core/orchestrator.py` — ядро (940 строк)
