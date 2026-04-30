# Pure Intellect + Agent Zero — Research Notes

## 🏆 РАБОЧАЯ ТОЧКА: 21 апреля 2026, 17:49

### Что работает

```
AZ v1.7 (порт 5006)
  → GEN/EXE цикл работает нормально ✅
  → memory_save → PI API :7860 ✅
  → memory_load → PI API :7860 ✅
  → chat_model: qwen2.5:7b → Ollama напрямую ✅
  → utility_model: qwen2.5:3b → Ollama напрямую ✅

PI сервер (порт 7860)
  → coordinator: qwen2.5:1.5b ✅
  → SentenceTransformer CUDA ✅
  → Anchor Facts, Soft Reset ✅
  → ChromaDB + JSON storage ✅
```

### Что исправляли (по порядку)

1. **Python 3.13 vs requirements.txt** — установили faiss-cpu отдельно:
   ```bash
   /a0/usr/workdir/agent-zero-pi/venv/bin/pip install faiss-cpu
   ```

2. **ModuleNotFoundError: faiss** — faiss-cpu нужен для инициализации VectorDB в AZ

3. **ImportError: DEFAULT_THRESHOLD** — добавили константу в memory_load.py:
   ```python
   DEFAULT_THRESHOLD = 0.6  # Required by _50_recall_memories.py
   ```

4. **No RFC password** — удалили плагин _promptinclude:
   ```bash
   rm -rf /a0/usr/workdir/agent-zero-pi/plugins/_promptinclude/
   ```
   Этот плагин специфичен для /a0/ контейнера и требует RFC password

5. **Множественные процессы на порту 5006** — убиваем все старые процессы перед запуском:
   ```bash
   for pid in $(ps aux | grep 'run_ui' | grep -v grep | grep -v 'venv-a0' | awk '{print $2}'); do
     kill -9 $pid
   done
   ```

### Команда запуска (рабочая)

```bash
# Терминал 1: PI сервер
cd /a0/usr/workdir/pure-intellect
source venv/bin/activate
python -m pure_intellect serve --port 7860 &

# Терминал 2: AZ клон
cd /a0/usr/workdir/agent-zero-pi
export PI_SERVER=http://localhost:7860
nohup ./venv/bin/python run_ui.py > /tmp/az.log 2>&1 &
```

### Файлы которые мы изменили в agent-zero-pi

```
plugins/_memory/tools/memory_save.py   ← POST localhost:7860/api/v1/memory/fact
plugins/_memory/tools/memory_load.py   ← GET localhost:7860/api/v1/memory/search
                                           + DEFAULT_THRESHOLD = 0.6
plugins/_promptinclude/                ← УДАЛЁН (RFC password required)
```

### Модели (Ollama)

| Роль | Модель | Размер |
|---|---|---|
| AZ chat | qwen2.5:7b | ~4.5GB |
| AZ utility | qwen2.5:3b | ~2GB |
| PI coordinator | qwen2.5:1.5b | ~1GB |

### Тест который подтверждает работу

```
Пользователь → AZ: "Меня зовут Александр. Запомни это."
AZ → GEN: решает использовать memory_save
AZ → EXE: memory_save tool вызывается
memory_save → PI API: POST /api/v1/memory/fact
PI → сохраняет факт "Пользователя зовут Александр."
AZ → отвечает: "Запомнил, Александр! 👋"

Проверка: curl -G :7860/api/v1/memory/search?query=Александр
Результат: 5 фактов найдено ✅
```

### Следующие шаги

- [ ] Проверить персистентность памяти: перезапустить AZ → спросить "Как меня зовут?"
- [ ] Длинный диалог → Soft Reset → проверить что PI помнит
- [ ] Настроить автозапуск скриптом
- [ ] Протестировать с более сложными задачами (код, файлы)
