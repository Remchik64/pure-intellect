"""Benchmark runner для Pure Intellect memory system.

Запускает сценарии в двух режимах:
1. baseline — без памяти (каждый turn изолирован)
2. with_memory — с полной системой памяти (WorkingMemory + CCI + Optimizer)

Измеряет и сравнивает метрики для доказательства эффективности памяти.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from pure_intellect.core.memory import (
    WorkingMemory, MemoryStorage, AttentionScorer,
    MemoryOptimizer, CCITracker
)
from pure_intellect.core.memory.fact import Fact
from .scenarios import Scenario, Turn

logger = logging.getLogger(__name__)


# ─── Результаты одного turn ───────────────────────────────────────────────────

@dataclass
class TurnResult:
    """Результат одного turn в benchmark."""
    turn_id: int
    query: str
    mode: str                    # 'baseline' или 'with_memory'
    latency_ms: float            # Время обработки
    memory_facts_count: int      # Фактов в рабочей памяти
    storage_facts_count: int     # Фактов в долгосрочном хранилище
    coherence_score: float       # CCI score (0.0-1.0)
    coherence_signal: str        # coherent / low_coherence / topic_switch
    keyword_recall: float        # % ожидаемых ключевых слов найденных в памяти
    context_tokens: int          # Токенов в контексте


# ─── Результаты сценария ──────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Результат прогона одного сценария."""
    scenario_name: str
    mode: str
    total_turns: int
    turns: list[TurnResult] = field(default_factory=list)
    
    # Итоговые метрики
    avg_coherence_score: float = 0.0
    avg_keyword_recall: float = 0.0
    avg_latency_ms: float = 0.0
    topic_switches_detected: int = 0
    context_restorations: int = 0
    total_facts_stored: int = 0
    peak_memory_facts: int = 0
    
    def compute_metrics(self) -> None:
        """Вычислить итоговые метрики из results."""
        if not self.turns:
            return
        
        self.avg_coherence_score = sum(t.coherence_score for t in self.turns) / len(self.turns)
        self.avg_keyword_recall = sum(t.keyword_recall for t in self.turns) / len(self.turns)
        self.avg_latency_ms = sum(t.latency_ms for t in self.turns) / len(self.turns)
        self.topic_switches_detected = sum(
            1 for t in self.turns if t.coherence_signal == "topic_switch"
        )
        self.total_facts_stored = max(
            (t.storage_facts_count for t in self.turns), default=0
        )
        self.peak_memory_facts = max(
            (t.memory_facts_count for t in self.turns), default=0
        )
    
    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "mode": self.mode,
            "total_turns": self.total_turns,
            "metrics": {
                "avg_coherence_score": round(self.avg_coherence_score, 3),
                "avg_keyword_recall": round(self.avg_keyword_recall, 3),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "topic_switches_detected": self.topic_switches_detected,
                "context_restorations": self.context_restorations,
                "total_facts_stored": self.total_facts_stored,
                "peak_memory_facts": self.peak_memory_facts,
            }
        }


# ─── Baseline Runner (без памяти) ─────────────────────────────────────────────

class BaselineRunner:
    """Прогоняет сценарий БЕЗ памяти.
    
    Каждый turn изолирован — никакого контекста из предыдущих turns.
    Это эмулирует простой LLM без memory augmentation.
    """
    
    def run(self, scenario: Scenario) -> ScenarioResult:
        result = ScenarioResult(
            scenario_name=scenario.name,
            mode="baseline",
            total_turns=scenario.total_turns(),
        )
        
        for turn in scenario.turns:
            start = time.perf_counter()
            
            # В baseline нет памяти — просто считаем что контекст пустой
            # keyword_recall = 0 если ожидаются ключевые слова (их нет в контексте)
            keyword_recall = 0.0
            if not turn.expected_keywords:
                keyword_recall = 1.0  # Нет ожиданий — считается выполненным
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            turn_result = TurnResult(
                turn_id=turn.turn_id,
                query=turn.query,
                mode="baseline",
                latency_ms=latency_ms,
                memory_facts_count=0,        # Нет памяти
                storage_facts_count=0,       # Нет памяти
                coherence_score=0.0,         # Нет CCI
                coherence_signal="none",
                keyword_recall=keyword_recall,
                context_tokens=0,            # Пустой контекст
            )
            result.turns.append(turn_result)
        
        result.compute_metrics()
        return result


# ─── Memory Runner (с памятью) ────────────────────────────────────────────────

class MemoryRunner:
    """Прогоняет сценарий С полной системой памяти.
    
    Эмулирует реальную работу Pure Intellect:
    - WorkingMemory хранит горячие факты
    - CCITracker отслеживает coherence
    - MemoryOptimizer управляет lifecycle фактов
    - При смене темы — restore из storage
    """
    
    def __init__(
        self,
        token_budget: int = 1500,
        cci_threshold: float = 0.15,
        optimizer_interval: int = 5,
    ):
        self.token_budget = token_budget
        self.cci_threshold = cci_threshold
        self.optimizer_interval = optimizer_interval
    
    def _build_memory_system(self):
        """Создать свежую систему памяти для каждого сценария."""
        storage = MemoryStorage()
        memory = WorkingMemory(token_budget=self.token_budget, storage=storage)
        scorer = AttentionScorer()
        optimizer = MemoryOptimizer(run_every_n_turns=self.optimizer_interval)
        cci = CCITracker(threshold=self.cci_threshold)
        return storage, memory, scorer, optimizer, cci
    
    def _check_keyword_recall(
        self,
        keywords: list[str],
        memory: WorkingMemory,
        storage: MemoryStorage,
    ) -> float:
        """Проверить % ожидаемых ключевых слов доступных в памяти."""
        if not keywords:
            return 1.0
        
        # Собираем весь доступный контекст
        working_context = memory.get_context().lower()
        
        # Ищем в storage
        storage_facts = []
        for kw in keywords[:3]:  # Берём первые 3 keyword для поиска
            results = storage.retrieve(kw, top_k=5)
            storage_facts.extend(results)
        
        storage_context = " ".join(f.content for f in storage_facts).lower()
        full_context = working_context + " " + storage_context
        
        found = sum(
            1 for kw in keywords
            if kw.lower() in full_context
        )
        return found / len(keywords)
    
    def run(self, scenario: Scenario) -> ScenarioResult:
        result = ScenarioResult(
            scenario_name=scenario.name,
            mode="with_memory",
            total_turns=scenario.total_turns(),
        )
        
        # Инициализируем систему памяти
        storage, memory, scorer, optimizer, cci = self._build_memory_system()
        current_turn = 0
        
        for turn in scenario.turns:
            start = time.perf_counter()
            current_turn += 1
            
            # 1. Оцениваем coherence ПЕРЕД обработкой
            coherence_result = cci.evaluate(turn.query)
            
            # 2. При low coherence — восстанавливаем контекст из storage
            if coherence_result.needs_context_restore():
                recent_keywords = cci.get_recent_keywords(n_turns=3)
                if recent_keywords:
                    keyword_query = " ".join(list(recent_keywords)[:10])
                    restored = storage.retrieve(keyword_query, top_k=5)
                    for fact in restored:
                        memory.add(fact)
                    result.context_restorations += 1
                    logger.debug(
                        f"Turn {current_turn}: restored {len(restored)} facts "
                        f"(coherence={coherence_result.score:.3f})"
                    )
            
            # 3. Эмулируем response (используем mock_response)
            response = turn.mock_response or f"Ответ на: {turn.query}"
            
            # 4. Извлекаем новые факты из response и добавляем в память
            new_facts = scorer.extract_facts_from_response(
                response, source=f"turn_{current_turn}"
            )
            for fact_content in new_facts:
                memory.add_text(fact_content, source=f"turn_{current_turn}")
            
            # Также добавляем query как факт
            memory.add_text(turn.query, source=f"query_{current_turn}")
            
            # 5. Cleanup с attention scoring
            memory.cleanup(
                turn=current_turn,
                query=turn.query,
                response=response,
            )
            
            # 6. Периодическая оптимизация
            optimizer.run_if_needed(memory, storage, current_turn=current_turn)
            
            # 7. Фиксируем turn в CCI истории
            cci.add_turn(
                query=turn.query,
                response=response,
                coherence_score=coherence_result.score,
            )
            
            # 8. Измеряем keyword recall
            keyword_recall = self._check_keyword_recall(
                turn.expected_keywords, memory, storage
            )
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            turn_result = TurnResult(
                turn_id=turn.turn_id,
                query=turn.query,
                mode="with_memory",
                latency_ms=latency_ms,
                memory_facts_count=memory.size(),
                storage_facts_count=storage.size(),
                coherence_score=coherence_result.score,
                coherence_signal=coherence_result.signal,
                keyword_recall=keyword_recall,
                context_tokens=memory.stats()["total_tokens"],
            )
            result.turns.append(turn_result)
            
            logger.debug(
                f"Turn {current_turn}/{scenario.total_turns()}: "
                f"coherence={coherence_result.score:.3f} ({coherence_result.signal}), "
                f"recall={keyword_recall:.2f}, "
                f"memory={memory.size()}, storage={storage.size()}"
            )
        
        result.compute_metrics()
        return result
