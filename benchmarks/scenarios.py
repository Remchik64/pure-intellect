"""Тестовые сценарии для benchmark Pure Intellect memory system.

Каждый сценарий — это набор turns симулирующий реальный разговор.
Сценарии проверяют конкретные гипотезы о работе памяти.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """Один turn разговора."""
    query: str
    expected_keywords: list[str] = field(default_factory=list)  # Что должно быть в памяти
    mock_response: str = ""  # Заглушка ответа для тестирования без LLM
    turn_id: int = 0


@dataclass
class Scenario:
    """Сценарий тестирования — набор связанных turns."""
    name: str
    description: str
    turns: list[Turn]
    tags: list[str] = field(default_factory=list)
    
    def total_turns(self) -> int:
        return len(self.turns)


# ─── Сценарий 1: Длинная техническая сессия ──────────────────────────────────
# Проверяет: сохраняются ли факты из начала сессии в конце

LONG_SESSION = Scenario(
    name="long_technical_session",
    description="30 turns технической сессии о разработке Python проекта. "
                "Проверяет что факты из turn 1-5 доступны в turn 25-30.",
    tags=["long_session", "fact_recall", "python"],
    turns=[
        Turn(
            turn_id=1,
            query="Мы разрабатываем проект pure-intellect — систему памяти для LLM",
            expected_keywords=["pure-intellect", "память", "LLM"],
            mock_response="Понял. pure-intellect — это система иерархической памяти для LLM с уровнями L1/L2/L3."
        ),
        Turn(
            turn_id=2,
            query="Основная архитектура: WorkingMemory → MemoryStorage → CCITracker",
            expected_keywords=["WorkingMemory", "MemoryStorage", "CCITracker"],
            mock_response="Архитектура: WorkingMemory (горячий буфер), MemoryStorage (долгосрочное), CCITracker (связность)."
        ),
        Turn(
            turn_id=3,
            query="Главная проблема: LLM теряет контекст в длинных сессиях",
            expected_keywords=["LLM", "контекст", "сессия"],
            mock_response="Да, деградация контекста — это ключевая проблема. Решение — обнуление + внешняя память."
        ),
        Turn(
            turn_id=4,
            query="Мы используем Python 3.13 и FastAPI для backend",
            expected_keywords=["Python", "FastAPI", "backend"],
            mock_response="Python 3.13 + FastAPI — хороший выбор для async backend с type hints."
        ),
        Turn(
            turn_id=5,
            query="ChromaDB используется для векторного хранилища",
            expected_keywords=["ChromaDB", "векторное"],
            mock_response="ChromaDB обеспечивает векторный поиск для RAG retrieval."
        ),
        Turn(
            turn_id=6,
            query="Как реализовать автоматическую компрессию фактов?",
            expected_keywords=["компрессия", "факты"],
            mock_response="Компрессия: RAW → SUMMARIZED (первые 2 предложения) → ENTITY_ONLY → ARCHIVED."
        ),
        Turn(
            turn_id=7,
            query="Какой алгоритм scoring используем для attention_weight?",
            expected_keywords=["scoring", "attention_weight"],
            mock_response="AttentionScorer использует keyword matching: touch() при упоминании, decay() иначе."
        ),
        Turn(
            turn_id=8,
            query="Объясни как работает BM25 в CCITracker",
            expected_keywords=["BM25", "CCITracker"],
            mock_response="BM25 считает TF-нормализованный score между ключевыми словами запроса и истории turns."
        ),
        Turn(
            turn_id=9,
            query="Какой порог coherence считается низким?",
            expected_keywords=["coherence", "порог"],
            mock_response="Порог 0.15 по умолчанию. Ниже — система восстанавливает контекст из storage."
        ),
        Turn(
            turn_id=10,
            query="Как MemoryOptimizer решает что компрессировать?",
            expected_keywords=["MemoryOptimizer", "компрессия"],
            mock_response="Optimizer: hot_retrieval_threshold=3 → promote, cold_weight_threshold=0.1 → compress."
        ),
        Turn(
            turn_id=11,
            query="Расскажи про AttentionScorer.extract_facts_from_response",
            expected_keywords=["AttentionScorer", "extract"],
            mock_response="Метод разбивает ответ LLM на предложения, фильтрует по длине >= 20 символов и >= 4 слов."
        ),
        Turn(
            turn_id=12,
            query="Как тестируем memory subsystem?",
            expected_keywords=["тестирование", "memory"],
            mock_response="pytest тесты: test_memory.py, test_scorer.py, test_optimizer.py, test_cci.py — всего 150 тестов."
        ),
        Turn(
            turn_id=13,
            query="Что такое CompressionLevel в системе?",
            expected_keywords=["CompressionLevel"],
            mock_response="CompressionLevel: RAW=0, SUMMARIZED=1, ENTITY_ONLY=2, ARCHIVED=3. Enum в fact.py."
        ),
        Turn(
            turn_id=14,
            query="Как устроен singleton в ModelManager?",
            expected_keywords=["singleton", "ModelManager"],
            mock_response="ModelManager.get_instance() использует threading.Lock() для thread-safe singleton."
        ),
        Turn(
            turn_id=15,
            query="Какие порты проброшены в Docker контейнере?",
            expected_keywords=["Docker", "порты"],
            mock_response="Порты: 32752:80, 3005:3005, 5006:5006, 8085:8085. GPU доступен через --gpus all."
        ),
        Turn(
            turn_id=16,
            query="Какой GPU используется?",
            expected_keywords=["GPU", "RTX"],
            mock_response="RTX 3060 с 12GB VRAM. CUDA доступен для llama-cpp-python."
        ),
        Turn(
            turn_id=17,
            query="Как работает endpoint /orchestrate?",
            expected_keywords=["orchestrate", "endpoint"],
            mock_response="POST /api/v1/orchestrate: query → intent → RAG → graph → memory → LLM → response."
        ),
        Turn(
            turn_id=18,
            query="Что возвращает /memory/stats?",
            expected_keywords=["memory", "stats"],
            mock_response="JSON: turn, working_memory stats, storage stats, optimizer config."
        ),
        Turn(
            turn_id=19,
            query="Опиши структуру Fact dataclass",
            expected_keywords=["Fact", "dataclass"],
            mock_response="Fact: fact_id, content, source, attention_weight, stability, reference_count, last_referenced, compression_level."
        ),
        Turn(
            turn_id=20,
            query="Как cleanup() в WorkingMemory работает с threshold?",
            expected_keywords=["cleanup", "WorkingMemory", "threshold"],
            mock_response="cleanup(): is_hot() если weight >= 0.7, is_cold() если weight <= 0.05. Холодные → storage."
        ),
        Turn(
            turn_id=21,
            query="Какой token_budget установлен для WorkingMemory?",
            expected_keywords=["token_budget", "WorkingMemory"],
            mock_response="token_budget=1500 токенов. При превышении — принудительный вывод в storage."
        ),
        Turn(
            turn_id=22,
            query="Как работает decay() в Fact?",
            expected_keywords=["decay", "Fact"],
            mock_response="decay(current_turn, decay_rate=0.1): weight *= (1 - rate * turns_since_referenced). Min=0.0."
        ),
        Turn(
            turn_id=23,
            query="Расскажи про run_every_n_turns в MemoryOptimizer",
            expected_keywords=["MemoryOptimizer", "run_every_n_turns"],
            mock_response="run_every_n_turns=5: оптимизация каждые 5 turns. Проверяется в run_if_needed()."
        ),
        Turn(
            turn_id=24,
            query="Какой движок моделей используется — llama-cpp или ollama?",
            expected_keywords=["llama-cpp", "ollama"],
            mock_response="Оба движка присутствуют: engine/model_manager.py (llama-cpp) и engines/ollama.py. Основной — llama-cpp."
        ),
        Turn(
            turn_id=25,
            query="Напомни, как называется наш проект?",  # Тест recall с turn 1
            expected_keywords=["pure-intellect"],
            mock_response="Проект называется pure-intellect."
        ),
        Turn(
            turn_id=26,
            query="Какая основная проблема которую решает проект?",  # Тест recall с turn 3
            expected_keywords=["контекст", "LLM"],
            mock_response="Деградация контекста LLM в длинных сессиях."
        ),
        Turn(
            turn_id=27,
            query="На каком языке и фреймворке написан backend?",  # Тест recall с turn 4
            expected_keywords=["Python", "FastAPI"],
            mock_response="Python 3.13 + FastAPI."
        ),
        Turn(
            turn_id=28,
            query="Какое векторное хранилище используем?",  # Тест recall с turn 5
            expected_keywords=["ChromaDB"],
            mock_response="ChromaDB."
        ),
        Turn(
            turn_id=29,
            query="Сколько всего тестов в проекте?",  # Тест recall с turn 12
            expected_keywords=["тестов", "150"],
            mock_response="150 тестов (pytest)."
        ),
        Turn(
            turn_id=30,
            query="Какие три основных компонента архитектуры памяти?",  # Тест recall с turn 2
            expected_keywords=["WorkingMemory", "MemoryStorage", "CCITracker"],
            mock_response="WorkingMemory, MemoryStorage, CCITracker."
        ),
    ]
)


# ─── Сценарий 2: Смена темы (Topic Switch) ────────────────────────────────────
# Проверяет: восстанавливает ли CCI контекст после резкой смены темы

TOPIC_SWITCH = Scenario(
    name="topic_switch_recovery",
    description="10 turns: 5 о Python, резкий переход на кулинарию, возврат к Python. "
                "Проверяет CCI topic_switch detection и context restore.",
    tags=["topic_switch", "cci", "recovery"],
    turns=[
        Turn(turn_id=1, query="Как работают декораторы в Python?",
             expected_keywords=["декоратор", "python"],
             mock_response="Декоратор — это функция принимающая функцию и возвращающая функцию."),
        Turn(turn_id=2, query="Покажи пример functools.wraps",
             expected_keywords=["functools", "wraps"],
             mock_response="@functools.wraps(func) сохраняет __name__ и __doc__ оборачиваемой функции."),
        Turn(turn_id=3, query="Что такое closure в Python?",
             expected_keywords=["closure", "python"],
             mock_response="Closure — функция запоминающая переменные из внешней области видимости."),
        Turn(turn_id=4, query="Как работает __slots__?",
             expected_keywords=["slots", "python"],
             mock_response="__slots__ ограничивает атрибуты объекта, экономя память."),
        Turn(turn_id=5, query="Объясни метакласс в Python",
             expected_keywords=["метакласс", "python"],
             mock_response="Метакласс — это класс для классов. type является базовым метаклассом."),
        # Резкая смена темы
        Turn(turn_id=6, query="Как приготовить борщ?",
             expected_keywords=["борщ", "приготовить"],
             mock_response="Борщ: свёкла, капуста, картофель, мясо. Варить 1.5 часа."),
        Turn(turn_id=7, query="Какие специи нужны для украинской кухни?",
             expected_keywords=["специи", "кухня"],
             mock_response="Укроп, петрушка, чеснок, лавровый лист."),
        # Возврат к Python
        Turn(turn_id=8, query="Вернёмся к Python — что такое descriptor protocol?",
             expected_keywords=["python", "descriptor"],
             mock_response="Descriptor: объект с __get__, __set__, __delete__. Используется в property."),
        Turn(turn_id=9, query="Как мы обсуждали декораторы?",  # Recall после switch
             expected_keywords=["декоратор"],
             mock_response="Декоратор принимает функцию и возвращает функцию."),
        Turn(turn_id=10, query="Что мы говорили про functools?",  # Recall
             expected_keywords=["functools"],
             mock_response="functools.wraps сохраняет метаданные функции."),
    ]
)


# ─── Сценарий 3: Повторные вопросы ────────────────────────────────────────────
# Проверяет: распознаёт ли система повторные вопросы через промежуток

REPEAT_QUESTIONS = Scenario(
    name="repeat_questions",
    description="Вопрос задаётся в turn 1, тот же вопрос через 15 turns. "
                "Проверяет fact recall через длинный промежуток.",
    tags=["repeat", "fact_recall"],
    turns=[
        Turn(turn_id=1, query="Наш проект называется pure-intellect",
             expected_keywords=["pure-intellect"],
             mock_response="pure-intellect — система памяти для LLM."),
        Turn(turn_id=2, query="Версия Python: 3.13",
             expected_keywords=["python", "3.13"],
             mock_response="Python 3.13 — последняя стабильная версия."),
        *[
            Turn(turn_id=i, query=f"Промежуточный вопрос {i} о разработке",
                 expected_keywords=[],
                 mock_response=f"Ответ на вопрос {i}.")
            for i in range(3, 16)
        ],
        Turn(turn_id=16, query="Как называется наш проект?",
             expected_keywords=["pure-intellect"],
             mock_response="pure-intellect."),
        Turn(turn_id=17, query="Какую версию Python используем?",
             expected_keywords=["3.13"],
             mock_response="Python 3.13."),
    ]
)


# ─── Все сценарии ─────────────────────────────────────────────────────────────

ALL_SCENARIOS = [
    LONG_SESSION,
    TOPIC_SWITCH,
    REPEAT_QUESTIONS,
]


def get_scenario(name: str) -> Optional[Scenario]:
    """Получить сценарий по имени."""
    return next((s for s in ALL_SCENARIOS if s.name == name), None)
