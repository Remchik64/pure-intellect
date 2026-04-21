# 🔌 Pure Intellect — API Reference

> Все рабочие эндпоинты с примерами. Эндпоинты помеченные 🚧 экспериментальные.

---

## Базовый URL

```
http://localhost:7860
```

---

## OpenAI-Compatible API

Стандартный интерфейс совместимый с OpenAI SDK. Любой клиент поддерживающий OpenAI API работает из коробки.

### POST /v1/chat/completions

Основной эндпоинт. Принимает сообщения, добавляет контекст памяти, возвращает ответ модели.

**Request:**
```json
{
  "model": "pure-intellect",
  "messages": [
    {"role": "user", "content": "Привет! Как дела?"}
  ],
  "stream": true
}
```

**Response (stream=false):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "pure-intellect",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Привет! Всё отлично."
    },
    "finish_reason": "stop"
  }]
}
```

**cURL пример:**
```bash
curl http://localhost:7860/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "pure-intellect",
    "messages": [{"role": "user", "content": "Что ты помнишь?"}],
    "stream": false
  }'
```

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7860/v1",
    api_key="pure-intellect"  # любая строка
)

response = client.chat.completions.create(
    model="pure-intellect",
    messages=[{"role": "user", "content": "Расскажи что ты помнишь"}]
)
print(response.choices[0].message.content)
```

---

### GET /v1/models

Список доступных моделей.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "pure-intellect",
      "object": "model",
      "created": 1714000000
    }
  ]
}
```

---

## Memory API

Прямой доступ к системе памяти. Полезно для интеграции внешних инструментов.

### GET /api/v1/health

Статус сервера.

```bash
curl http://localhost:7860/api/v1/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "models": {
    "coordinator": "qwen2.5:3b",
    "generator": "qwen2.5:7b"
  }
}
```

---

### GET /api/v1/memory/search

Семантический поиск по базе фактов.

**Parameters:**
- `q` (string) — поисковый запрос
- `limit` (int, default=5) — максимум результатов
- `session_id` (string, default="default") — сессия

```bash
curl "http://localhost:7860/api/v1/memory/search?q=имя+пользователя&limit=3"
```

**Response:**
```json
{
  "results": [
    {
      "text": "Пользователя зовут Александр",
      "importance": 0.9,
      "is_anchor": true,
      "tier": "hot"
    }
  ],
  "total": 1
}
```

---

### POST /api/v1/memory/fact

Сохранить факт в память напрямую.

**Request:**
```json
{
  "text": "Пользователь предпочитает краткие ответы",
  "importance": 0.8,
  "is_anchor": false,
  "session_id": "default"
}
```

```bash
curl -X POST http://localhost:7860/api/v1/memory/fact   -H "Content-Type: application/json"   -d '{
    "text": "Пользователь работает в сфере AI",
    "importance": 0.85,
    "is_anchor": false
  }'
```

---

### GET /api/v1/memory/stats

Статистика системы памяти.

```bash
curl http://localhost:7860/api/v1/memory/stats
```

**Response:**
```json
{
  "hot_facts": 12,
  "warm_facts": 89,
  "cold_facts": 234,
  "anchor_facts": 5,
  "cci_current": 0.73,
  "session_turns": 8
}
```

---

### DELETE /api/v1/memory/clear

Очистить всю память сессии.

⚠️ **Необратимо.** Все факты и координаты будут удалены.

```bash
curl -X DELETE "http://localhost:7860/api/v1/memory/clear?session_id=default"
```

---

## WebSocket API

### WS /ws/chat

Стриминг ответов через WebSocket. Используется Admin Panel.

```javascript
const ws = new WebSocket("ws://localhost:7860/ws/chat");

ws.send(JSON.stringify({
  message: "Привет!",
  session_id: "default"
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.type: "token" | "thinking_start" | "thinking" | "thinking_end" | "done"
  if (data.type === "token") process.stdout.write(data.content);
};
```

**Типы сообщений:**
| type | Описание |
|------|----------|
| `token` | Токен ответа (стриминг) |
| `thinking_start` | Модель начала "думать" (chain-of-thought) |
| `thinking` | Токен мысли (виден в UI как свёрнутый блок) |
| `thinking_end` | Конец блока мышления |
| `done` | Ответ завершён |
| `error` | Ошибка |

---

## Аутентификация

API не требует аутентификации по умолчанию. Для OpenAI-совместимых клиентов передавайте любую строку как `api_key` — она принимается без проверки.

---

## Коды ошибок

| Код | Описание |
|-----|----------|
| 200 | OK |
| 422 | Validation Error — неверный формат запроса |
| 500 | Internal Server Error — проблема с моделью или Ollama |
| 503 | Service Unavailable — Ollama недоступен |
