"""Живой тест памяти Pure Intellect — правильная логика буфера.

Концепция:
- Rolling window: держим последние N токенов как обычный чат
- Когда буфер заполняется → 3B создаёт координату (GPS-маркер)
- Soft reset: убираем старое, оставляем координату + последние 3 turns
- WorkingMemory хранит anchor facts которые никогда не теряются

Запуск: python tests/test_live_memory.py
Требует: ollama serve + qwen2.5:3b
"""

import sys
import asyncio
import httpx
import time
sys.path.insert(0, 'src')

from pure_intellect.core.memory import (
    WorkingMemory, MemoryStorage, AttentionScorer,
    MemoryOptimizer, CCITracker
)
from pure_intellect.core.memory.fact import Fact, CompressionLevel

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"
ROLLING_WINDOW_TURNS = 8   # Сколько turns держим в живом контексте
SOFT_RESET_AT = 6          # После скольких turns делаем soft reset
SEP = "═" * 65
sep = "─" * 65


# ─── Ollama helpers ──────────────────────────────────────────────────────────

async def ollama_chat(
    client: httpx.AsyncClient,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 512,
) -> str:
    """Запрос к Ollama."""
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)
    
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": all_messages,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens}
            },
            timeout=90.0
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"


async def create_coordinate(
    client: httpx.AsyncClient,
    conversation_history: list[dict],
) -> str:
    """Попросить 3B создать GPS-координату перед soft reset.
    
    Координата = компактный маркер позиции в разговоре:
    - Кто участвует, что обсуждаем
    - Ключевые факты которые нельзя потерять
    - На чём остановились, что открыто
    """
    # Берём последние N turns для создания координаты
    recent = conversation_history[-ROLLING_WINDOW_TURNS:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in recent
    )
    
    coordinate_prompt = [
        {
            "role": "user",
            "content": f"""Прочитай этот фрагмент разговора и создай краткую 'навигационную координату' — компактный маркер который позволит продолжить разговор после сброса контекста.

Фрагмент разговора:
{history_text}

Создай координату в формате:
УЧАСТНИКИ: [кто участвует, имена]
ТЕМА: [о чём разговор, 1-2 предложения]
КЛЮЧЕВЫЕ ФАКТЫ: [список важных фактов которые нельзя забыть]
ПОСЛЕДНЯЯ ТЕМА: [на чём остановились]
ОТКРЫТЫЕ ВОПРОСЫ: [что ещё не решено]

Отвечай только координатой, без лишних слов."""
        }
    ]
    
    coordinate = await ollama_chat(client, coordinate_prompt, max_tokens=300)
    return coordinate


# ─── Основной тест ────────────────────────────────────────────────────────────

async def run_live_test():
    print(f"\n{SEP}")
    print("  🧠 PURE INTELLECT — ЖИВОЙ ТЕСТ (правильный буфер)")
    print(f"  Модель: {MODEL} | Rolling window: {ROLLING_WINDOW_TURNS} turns")
    print(SEP)
    
    # Инициализируем систему памяти
    storage = MemoryStorage()
    memory = WorkingMemory(token_budget=2000, storage=storage)
    scorer = AttentionScorer()
    optimizer = MemoryOptimizer(run_every_n_turns=5)
    cci = CCITracker(threshold=0.15)
    
    # Живая история чата (rolling window)
    chat_history = []       # Полная история для rolling window
    coordinate = None       # Текущая координата (после soft reset)
    soft_reset_count = 0    # Сколько раз делали soft reset
    recall_results = []
    
    # ── Сценарий разговора ──────────────────────────────────────────
    conversation = [
        # Фаза 1: Представление и контекст (turns 1-4)
        {"q": "Привет! Меня зовут Александр, я разрабатываю проект pure-intellect — систему памяти для LLM. Приятно познакомиться!",
         "phase": "intro", "recall_test": False},
        {"q": "Проект написан на Python 3.13, используем FastAPI и ChromaDB. GPU у нас RTX 3060 с 12GB памяти.",
         "phase": "intro", "recall_test": False},
        {"q": "Главная проблема которую решаем: LLM теряет контекст на 20-30 сообщении. Хотим решить это через иерархическую память.",
         "phase": "intro", "recall_test": False},
        {"q": "Наш подход: rolling window буфер + координата перед обнулением + semantic retrieval из памяти.",
         "phase": "intro", "recall_test": False},
        
        # Фаза 2: Техническая беседа (turns 5-10) — «отвлечение»
        {"q": "Расскажи про attention mechanism в трансформерах — как он работает математически?",
         "phase": "technical", "recall_test": False},
        {"q": "Чем отличается self-attention от cross-attention?",
         "phase": "technical", "recall_test": False},
        {"q": "Что такое LoRA и как он помогает при fine-tuning больших моделей?",
         "phase": "technical", "recall_test": False},
        {"q": "Объясни принцип работы RLHF — reinforcement learning from human feedback.",
         "phase": "technical", "recall_test": False},
        {"q": "Как работает quantization моделей — в чём разница между GGUF Q4 и Q8?",
         "phase": "technical", "recall_test": False},
        {"q": "Что такое speculative decoding и как он ускоряет inference?",
         "phase": "technical", "recall_test": False},
        
        # Фаза 3: Recall тесты (turns 11-15)
        {"q": "Кстати, как меня зовут?",
         "phase": "recall", "recall_test": True, "expected": "Александр"},
        {"q": "Напомни как называется наш проект?",
         "phase": "recall", "recall_test": True, "expected": "pure-intellect"},
        {"q": "На каком языке и фреймворке написан backend?",
         "phase": "recall", "recall_test": True, "expected": "Python"},
        {"q": "Какую главную проблему мы решаем?",
         "phase": "recall", "recall_test": True, "expected": "контекст"},
        {"q": "Какой GPU у нас установлен?",
         "phase": "recall", "recall_test": True, "expected": "RTX"},
    ]
    
    async with httpx.AsyncClient() as client:
        # Проверяем Ollama
        try:
            await client.get(f"{OLLAMA_URL}/", timeout=5.0)
            print(f"\n✅ Ollama доступен | Модель: {MODEL}")
        except Exception as e:
            print(f"\n❌ Ollama недоступен: {e}")
            return
        
        print(f"\n{'T':<3} {'Фаза':<10} {'Запрос':<42} {'Time':>7} {'Buffer'} {'Status'}")
        print(sep)
        
        for turn_id, turn_data in enumerate(conversation, 1):
            query = turn_data["q"]
            phase = turn_data["phase"]
            is_recall = turn_data["recall_test"]
            expected = turn_data.get("expected", "")
            
            start = time.perf_counter()
            
            # ── Soft reset если буфер заполнен ─────────────────────
            if len(chat_history) >= SOFT_RESET_AT * 2:  # *2 т.к. user+assistant
                print(f"\n  🔄 Soft reset #{soft_reset_count+1} — создаём координату...")
                
                # 3B создаёт координату
                coordinate = await create_coordinate(client, chat_history)
                soft_reset_count += 1
                
                # Сохраняем координату как anchor fact в памяти
                anchor = memory.add_text(
                    f"[КООРДИНАТА #{soft_reset_count}] {coordinate}",
                    source="coordinate"
                )
                anchor.attention_weight = 1.0  # Anchor — не decay
                anchor.stability = 1.0
                
                # Soft reset: оставляем только последние 3 turns
                chat_history = chat_history[-6:]  # последние 3 пары
                
                coord_short = coordinate[:80].replace('\n', ' ')
                print(f"  📍 Координата: {coord_short}...")
                print(f"  📚 Buffer: {len(chat_history)//2} turns | Memory: {memory.size()} facts")
                print()
            
            # ── CCI evaluate ────────────────────────────────────────
            coherence = cci.evaluate(query)
            
            # ── Строим system prompt ─────────────────────────────────
            system_parts = ["Ты полезный ассистент. Отвечай кратко и конкретно."]
            
            # Добавляем координату если есть
            if coordinate:
                system_parts.append(f"\n=== Контекст сессии ===\n{coordinate}")
            
            # Добавляем рабочую память
            memory_ctx = memory.get_context()
            if memory_ctx:
                system_parts.append(f"\n=== Важные факты ===\n{memory_ctx}")
            
            # При low coherence — пробуем восстановить из storage
            if coherence.needs_context_restore():
                recent_kw = cci.get_recent_keywords(n_turns=3)
                if recent_kw:
                    kw_q = " ".join(list(recent_kw)[:8])
                    restored = storage.retrieve(kw_q, top_k=3)
                    if restored:
                        restored_text = "\n".join(f.content for f in restored)
                        system_parts.append(f"\n=== Восстановленный контекст ===\n{restored_text}")
            
            system_prompt = "\n".join(system_parts)
            
            # ── Запрос к Ollama с rolling window ────────────────────
            chat_history.append({"role": "user", "content": query})
            response = await ollama_chat(
                client,
                chat_history,  # ПОЛНАЯ rolling window история
                system_prompt,
                max_tokens=256
            )
            chat_history.append({"role": "assistant", "content": response})
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            # ── Обновляем память ─────────────────────────────────────
            new_facts = scorer.extract_facts_from_response(response, source=f"t{turn_id}")
            for fc in new_facts:
                memory.add_text(fc, source=f"t{turn_id}")
            
            # Query тоже добавляем — вдруг там важное
            if phase == "intro":
                # Intro факты — высокий начальный вес
                f = memory.add_text(query, source=f"intro_{turn_id}")
                if f:
                    f.attention_weight = 0.9
            
            memory.cleanup(turn=turn_id, query=query, response=response)
            optimizer.run_if_needed(memory, storage, current_turn=turn_id)
            cci.add_turn(query=query, response=response, coherence_score=coherence.score)
            
            # ── Проверяем recall ─────────────────────────────────────
            recall_ok = None
            status_str = ""
            if is_recall:
                recall_ok = expected.lower() in response.lower()
                recall_results.append({
                    "turn": turn_id, "question": query,
                    "expected": expected, "response": response,
                    "found": recall_ok
                })
                status_str = "✅" if recall_ok else "❌"
            
            phase_icon = {"intro": "📝", "technical": "🔬", "recall": "🔍"}.get(phase, "")
            q_short = query[:40] + ".." if len(query) > 42 else query
            buf_info = f"{len(chat_history)//2}t/{memory.size()}f"
            
            print(f"{turn_id:<3} {phase_icon+phase:<10} {q_short:<42} {latency_ms:>6.0f}ms {buf_info:<8} {status_str}")
    
    # ── Итоговый отчёт ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  📊 РЕЗУЛЬТАТЫ RECALL ТЕСТОВ")
    print(sep)
    
    passed = sum(1 for r in recall_results if r["found"])
    total = len(recall_results)
    
    for r in recall_results:
        icon = "✅" if r["found"] else "❌"
        print(f"  {icon} {r['question']}")
        print(f"     Ожидали: '{r['expected']}'")
        resp_short = r['response'][:120].replace('\n', ' ')
        print(f"     Получили: '{resp_short}'")
        print()
    
    print(sep)
    print(f"  Recall:              {passed}/{total} ({passed/max(total,1)*100:.0f}%)")
    print(f"  Soft resets:         {soft_reset_count}")
    print(f"  WorkingMemory facts: {memory.size()}")
    print(f"  Storage facts:       {storage.size()}")
    print(f"  CCI avg score:       {cci.stats()['avg_coherence']:.3f}")
    
    if coordinate:
        print(f"\n  📍 Последняя координата:")
        for line in coordinate.split('\n')[:6]:
            print(f"     {line}")
    
    print()
    if passed == total:
        verdict = "✅ ИДЕАЛЬНО — система помнит всё"
    elif passed >= total * 0.8:
        verdict = "✅ ХОРОШО — система помнит большинство"
    elif passed >= total * 0.6:
        verdict = "⚠️  УДОВЛЕТВОРИТЕЛЬНО — часть теряется"
    else:
        verdict = "❌ ПЛОХО — нужны улучшения"
    
    print(f"  {verdict}")
    print(SEP)
    return passed, total


if __name__ == "__main__":
    asyncio.run(run_live_test())
