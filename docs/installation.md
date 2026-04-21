# 📦 Pure Intellect — Installation Guide

> Детальное руководство по установке и настройке.

---

## Требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| Python | 3.11+ | 3.12 |
| RAM | 8 GB | 16 GB |
| VRAM | 0 GB (CPU режим) | 8+ GB (GPU режим) |
| Место на диске | 5 GB | 20+ GB (под модели) |
| ОС | Windows 10 / Ubuntu 20.04 / macOS 12+ | — |

**Обязательно:** [Ollama](https://ollama.com) — локальный runtime для LLM моделей.

---

## Способ 1 — Installer Script (Рекомендуется)

### Windows

```powershell
# PowerShell:
Invoke-WebRequest -Uri https://raw.githubusercontent.com/Remchik64/pure-intellect/main/install.bat -OutFile install.bat
.\install.bat
```

Или скачайте `install.bat` и запустите от имени администратора.

### Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/Remchik64/pure-intellect/main/install.sh | bash
```

Что делает скрипт:
1. Проверяет Python 3.11+
2. Устанавливает Ollama (если нет)
3. Устанавливает Pure Intellect через pip
4. Создаёт ярлык/лаунчер
5. Запускает сервер

---

## Способ 2 — Ручная установка

```bash
# 1. Установить Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Убедиться что Ollama запущен
ollama serve &

# 3. Установить Pure Intellect
pip install git+https://github.com/Remchik64/pure-intellect.git

# 4. Запустить
pure-intellect serve

# Admin Panel: http://localhost:7860
```

---

## Способ 3 — Development Install (для контрибьюторов)

```bash
# 1. Клонировать
git clone https://github.com/Remchik64/pure-intellect
cd pure-intellect

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate      # Linux/macOS
.env\Scriptsctivate        # Windows

# 3. Установить в режиме разработки
pip install -e .

# 4. Запустить
pure-intellect serve --port 7860
```

---

## Первый запуск

1. Открыть `http://localhost:7860`
2. Перейти в раздел **🤖 Models**
3. Нажать **"Определить железо"** — система покажет что ваш GPU поддерживает
4. Нажать **"Скачать"** рядом с рекомендованной моделью
5. Дождаться загрузки (зависит от интернета и размера модели)
6. Перейти в раздел чата и начать разговор

---

## Конфигурация

Файл `config.yaml` в корне проекта (или там где установлен):

```bash
# Найти config.yaml
pure-intellect config --show-path
```

Основные параметры:

```yaml
server:
  port: 7860          # Порт сервера
  host: 0.0.0.0       # 0.0.0.0 = доступен по сети, 127.0.0.1 = только локально

coordinator:
  model: qwen2.5:3b   # Быстрая модель для навигации

generator:
  model: qwen2.5:7b   # Основная модель для ответов
  num_gpu: -1         # -1=авто, 0=CPU, N=N слоёв на GPU

memory:
  hot_facts_max: 50   # Максимум фактов в RAM
```

---

## Рекомендованные модели

| VRAM | Coordinator | Generator |
|------|-------------|----------|
| 12+ GB | qwen2.5:3b | mistral-small3.1:24b |
| 8-12 GB | qwen2.5:3b | qwen2.5:7b |
| 4-8 GB | qwen2.5:3b | qwen2.5:3b |
| CPU | qwen2.5:3b | qwen2.5:3b |

```bash
# Скачать модели через Ollama
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

---

## Troubleshooting

### Сервер не запускается
```bash
# Проверить Ollama
curl http://localhost:11434/api/tags

# Запустить Ollama вручную
ollama serve
```

### Модель не отвечает
```bash
# Проверить что модель скачана
ollama list

# Скачать если нет
ollama pull qwen2.5:7b
```

### Порт 7860 занят
```bash
# Запустить на другом порту
pure-intellect serve --port 8080
```

### Мало VRAM — модель не загружается
В `config.yaml` установите `generator.num_gpu: 0` для CPU-режима или уменьшите модель.

---

## Обновление

```bash
pip install --upgrade git+https://github.com/Remchik64/pure-intellect.git
```
