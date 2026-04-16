#!/usr/bin/env python3
"""
Full System Test — диалог + нагрузка + проверка всех компонентов.
Запуск: python tests/test_system_full.py
"""

import time
import json
import requests
from datetime import datetime
from typing import Optional

BASE = "http://localhost:8085"
API  = f"{BASE}/api/v1"

# ── Цвета ──────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

results = []

def ok(msg):  print(f"  {GREEN}✅ {msg}{RESET}");  results.append((msg, True))
def fail(msg): print(f"  {RED}❌ {msg}{RESET}");  results.append((msg, False))
def info(msg): print(f"  {CYAN}ℹ  {msg}{RESET}")
def head(msg): print(f"\n{BOLD}{BLUE}{'─'*55}{RESET}\n{BOLD}{BLUE}  {msg}{RESET}\n{DIM}{'─'*55}{RESET}")
def sep():     print(f"{DIM}  {'·'*50}{RESET}")

def chat(query: str, timeout: int = 120) -> Optional[dict]:
    """Отправить сообщение в чат."""
    try:
        r = requests.post(
            f"{API}/orchestrate",
            json={"query": query, "temperature": 0.7},
            timeout=timeout
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except requests.exceptions.Timeout:
        return {"error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}

def get(path: str) -> Optional[dict]:
    try:
        r = requests.get(f"{API}{path}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def post(path: str, data: dict = None) -> Optional[dict]:
    try:
        r = requests.post(f"{API}{path}", json=data or {}, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None


print(f"\n{BOLD}{'═'*55}{RESET}")
print(f"{BOLD}  🧠 Pure Intellect — Full System Test{RESET}")
print(f"{BOLD}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
print(f"{BOLD}{'═'*55}{RESET}")


# Очистка сессии перед тестом
try:
    requests.delete(f"{API}/session", timeout=5)
    import time; time.sleep(1)
except:
    pass


# ══════════════════════════════════════════════════════════
# 1. HEALTH CHECK
# ══════════════════════════════════════════════════════════
head("1. Health Check — все эндпоинты")

try:
    r = requests.get(BASE + "/", timeout=5)
    if r.status_code == 200 and "Pure Intellect" in r.text:
        ok(f"Web UI: HTTP {r.status_code} — index.html отдаётся")
    else:
        fail(f"Web UI: HTTP {r.status_code}")
except Exception as e:
    fail(f"Web UI недоступен: {e}")

for path, name in [
    ("/health",        "Health endpoint"),
    ("/memory/stats",  "Memory stats"),
    ("/cci/stats",     "CCI stats"),
    ("/dual-model/stats", "Dual-model stats"),
    ("/session/info",  "Session info"),
]:
    data = get(path)
    if data is not None:
        ok(f"{name}: OK")
    else:
        fail(f"{name}: недоступен")


# ══════════════════════════════════════════════════════════
# 2. DUAL MODEL STATUS
# ══════════════════════════════════════════════════════════
head("2. Dual Model Router — статус моделей")

dm = get("/dual-model/stats")
if dm:
    coord = dm.get("coordinator_model", "?")
    gen   = dm.get("generator_model", "?")
    gen_ok = dm.get("generator_available", False)
    info(f"Coordinator: {coord}")
    info(f"Generator:   {gen} — {'✅ Online' if gen_ok else '⚠️  Fallback'}")
    ok("Dual Model Router отвечает")
    if gen_ok:
        ok("qwen2.5:7b активен как generator")
    else:
        info("Generator использует fallback (3B)")
else:
    fail("Dual Model Router недоступен")


# ══════════════════════════════════════════════════════════
# 3. ДИАЛОГ С НАГРУЗКОЙ
# ══════════════════════════════════════════════════════════
head("3. Диалог — 12 turns с накоплением фактов")

dialogue = [
    # Представление
    "Привет! Меня зовут Александр. Я разрабатываю проект pure-intellect.",
    "Проект написан на Python 3.13 с использованием FastAPI и ChromaDB.",
    "Главная цель проекта — решить проблему деградации контекста LLM.",
    "У меня видеокарта NVIDIA RTX 3060 с 12GB VRAM.",
    # Техническое обсуждение
    "Объясни мне как работает механизм attention в трансформерах?",
    "Чем отличается self-attention от cross-attention?",
    "Что такое LoRA и зачем нужна fine-tuning?",
    "Как работает квантизация моделей — Q4 vs Q8?",
    # Нагрузка на систему
    "Расскажи о принципах RAG архитектуры.",
    "Какие основные проблемы у длинного контекста в LLM?",
    # Recall тесты
    "Как меня зовут?",
    "Назови проект который я разрабатываю и его главную цель.",
]

turn_times = []
recall_results = []
chat_responses = []

for i, msg in enumerate(dialogue, 1):
    turn_num = i
    is_recall = i >= 11
    prefix = f"Turn {i:2d}/12"

    start = time.time()
    resp = chat(msg, timeout=90)
    elapsed = time.time() - start
    turn_times.append(elapsed)

    if resp and not resp.get("error"):
        response_text = resp.get("response", "")
        model_used = resp.get("model_used", "?")
        coherence = resp.get("coherence", {})
        cci_score = coherence.get("score", 0) if coherence else 0
        chat_responses.append(response_text)

        short = response_text[:80].replace("\n", " ")
        print(f"  {DIM}{prefix}{RESET} [{elapsed:.1f}s] {DIM}{short}...{RESET}")

        if is_recall:
            # Проверяем recall
            resp_lower = response_text.lower()
            if i == 11:
                found = any(name in resp_lower for name in ["александр", "alexander", "алекс"])
                if found:
                    ok(f"Recall имени: НАЙДЕНО 'Александр'")
                    recall_results.append(True)
                else:
                    fail(f"Recall имени: не найдено. Ответ: {short}")
                    recall_results.append(False)
            elif i == 12:
                found_project = "pure-intellect" in resp_lower or "pure intellect" in resp_lower
                found_goal = any(w in resp_lower for w in ["контекст", "деградац", "память", "context"])
                if found_project and found_goal:
                    ok(f"Recall проекта и цели: НАЙДЕНО")
                    recall_results.append(True)
                elif found_project:
                    ok(f"Recall названия: НАЙДЕНО (цель не распознана)")
                    recall_results.append(True)
                else:
                    fail(f"Recall проекта: не найдено")
                    recall_results.append(False)
    else:
        err = resp.get("error", "нет ответа") if resp else "timeout"
        fail(f"{prefix}: {err}")
        chat_responses.append("")
        if is_recall:
            recall_results.append(False)

sep()
avg_time = sum(turn_times) / len(turn_times) if turn_times else 0
max_time = max(turn_times) if turn_times else 0
min_time = min(turn_times) if turn_times else 0
info(f"Среднее время ответа: {avg_time:.1f}s")
info(f"Min: {min_time:.1f}s  Max: {max_time:.1f}s")


# ══════════════════════════════════════════════════════════
# 4. RECALL ИТОГ
# ══════════════════════════════════════════════════════════
head("4. Recall — результаты проверки памяти")

recall_score = sum(recall_results)
recall_total = len(recall_results)
recall_pct   = int(recall_score / recall_total * 100) if recall_total else 0

if recall_pct == 100:
    print(f"  {GREEN}{BOLD}  Recall: {recall_score}/{recall_total} (100%) 🏆{RESET}")
    ok("Идеальный recall — система помнит все факты")
elif recall_pct >= 50:
    print(f"  {YELLOW}{BOLD}  Recall: {recall_score}/{recall_total} ({recall_pct}%) ⚠️{RESET}")
    info("Частичный recall — есть потери")
else:
    print(f"  {RED}{BOLD}  Recall: {recall_score}/{recall_total} ({recall_pct}%) ❌{RESET}")
    fail("Низкий recall — память не работает")


# ══════════════════════════════════════════════════════════
# 5. ПАМЯТЬ ПОСЛЕ ДИАЛОГА
# ══════════════════════════════════════════════════════════
head("5. Состояние памяти после диалога")

mem = get("/memory/stats")
if mem:
    wm = mem.get("working_memory", {})
    st = mem.get("storage", {})
    turn = mem.get("turn", 0)

    info(f"Turns пройдено:     {turn}")
    info(f"Горячих фактов:     {wm.get('facts_count', 0)}")
    info(f"Anchor фактов:      {wm.get('anchor_count', 0)}")
    info(f"Long-term storage:  {st.get('total_facts', 0)}")
    info(f"Токен бюджет:       {wm.get('budget_used_pct', 0)}%")

    if wm.get("facts_count", 0) > 0:
        ok("WorkingMemory содержит факты")
    else:
        fail("WorkingMemory пуста")

    if wm.get("anchor_count", 0) > 0:
        ok(f"Anchor facts: {wm.get('anchor_count')} — координаты сохранены")
    else:
        info("Anchor facts: 0 (soft reset ещё не произошёл)")
else:
    fail("Memory stats недоступны")

cci = get("/cci/stats")
if cci:
    score = cci.get("avg_coherence_score", 0)
    low   = cci.get("low_coherence_count", 0)
    color = GREEN if score > 0.5 else YELLOW if score > 0.2 else RED
    info(f"CCI avg score:      {color}{score:.3f}{RESET}")
    info(f"Low coherence:      {low} events")
    ok("CCI tracker работает")


# ══════════════════════════════════════════════════════════
# 6. SESSION PERSISTENCE
# ══════════════════════════════════════════════════════════
head("6. Session Persistence — сохранение сессии")

saved = post("/session/save")
if saved:
    ok(f"Сессия сохранена: turn={saved.get('turn')}, "
       f"wm={saved.get('working_memory_facts')} фактов")
else:
    fail("Не удалось сохранить сессию")

info_data = get("/session/info")
if info_data and info_data.get("exists"):
    ok(f"Сессия существует на диске: {info_data.get('session_id')}")
    info(f"Файлы: {info_data.get('files_present', [])}")
else:
    fail("Сессия не найдена на диске")


# ══════════════════════════════════════════════════════════
# 7. НАГРУЗОЧНЫЙ ТЕСТ
# ══════════════════════════════════════════════════════════
head("7. Нагрузочный тест — 5 быстрых запросов")

load_queries = [
    "Что такое температура в LLM?",
    "Объясни top-p sampling.",
    "Что такое токенизация?",
    "Чем отличается GPT от BERT?",
    "Что такое embedding?",
]

load_times = []
load_errors = 0

for i, q in enumerate(load_queries, 1):
    start = time.time()
    r = chat(q, timeout=60)
    elapsed = time.time() - start
    load_times.append(elapsed)

    if r and not r.get("error"):
        print(f"  {DIM}Query {i}: {elapsed:.1f}s — OK{RESET}")
    else:
        load_errors += 1
        print(f"  {RED}Query {i}: ОШИБКА{RESET}")

sep()
avg_load = sum(load_times) / len(load_times) if load_times else 0
if load_errors == 0:
    ok(f"Нагрузочный тест: 5/5 успешно (avg {avg_load:.1f}s)")
else:
    fail(f"Нагрузочный тест: {load_errors} ошибок из 5")


# ══════════════════════════════════════════════════════════
# ИТОГОВЫЙ ОТЧЁТ
# ══════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*55}{RESET}")
print(f"{BOLD}  📊 ИТОГОВЫЙ ОТЧЁТ{RESET}")
print(f"{BOLD}{'═'*55}{RESET}")

passed = sum(1 for _, ok_ in results if ok_)
failed = sum(1 for _, ok_ in results if not ok_)
total  = len(results)

print(f"\n  Checks:  {GREEN}{passed}{RESET} passed / {RED}{failed}{RESET} failed / {total} total")
print(f"  Recall:  {GREEN if recall_pct==100 else YELLOW}{recall_score}/{recall_total} ({recall_pct}%){RESET}")
print(f"  Avg resp: {avg_time:.1f}s  |  Load avg: {avg_load:.1f}s")

if failed == 0 and recall_pct == 100:
    print(f"\n  {GREEN}{BOLD}🏆 СИСТЕМА РАБОТАЕТ ИДЕАЛЬНО!{RESET}")
elif failed <= 2 and recall_pct >= 50:
    print(f"\n  {YELLOW}{BOLD}⚠️  СИСТЕМА РАБОТАЕТ УДОВЛЕТВОРИТЕЛЬНО{RESET}")
else:
    print(f"\n  {RED}{BOLD}❌ СИСТЕМА ТРЕБУЕТ ВНИМАНИЯ{RESET}")

print(f"{BOLD}{'═'*55}{RESET}\n")
