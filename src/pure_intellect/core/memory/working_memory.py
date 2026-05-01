"""WorkingMemory — рабочий буфер памяти Pure Intellect.

Всегда маленький, всегда чистый, всегда focused.
После каждого turn автоматически чистится:
- Горячие факты остаются
- Холодные перемещаются в MemoryStorage
- Бюджет токенов никогда не превышается

P5: save_state()/load_state() для persistence между сессиями.
"""


import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from .fact import Fact, CompressionLevel
from .scorer import AttentionScorer

if TYPE_CHECKING:
    from .storage import MemoryStorage

logger = logging.getLogger(__name__)

# Настройки по умолчанию
DEFAULT_TOKEN_BUDGET = 2000    # Максимум токенов в рабочей памяти
HOT_THRESHOLD = 0.6             # Выше — факт горячий, остаётся
COLD_THRESHOLD = 0.15           # Ниже — факт холодный, уходит в storage
DECAY_RATE = 0.05               # Скорость затухания веса за turn


class WorkingMemory:
    """Рабочий буфер — чистая, маленькая, focused память.
    
    Принцип работы:
    1. Факты добавляются через add()
    2. После каждого ответа LLM вызывается cleanup(turn)
    3. cleanup() применяет decay, перемещает холодные в storage
    4. get_context() возвращает строку с фактами для LLM
    """
    
    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        storage: Optional['MemoryStorage'] = None,
    ):
        self.token_budget = token_budget
        self.storage = storage
        self._facts: list[Fact] = []
        self.current_turn: int = 0
        self._evicted_count: int = 0  # статистика
        self._scorer = AttentionScorer()  # для анализа разговора
    
    def add(self, fact: Fact) -> None:
        """Добавить факт в рабочую память.
        
        Если добавление превышает бюджет — вытесняем самые холодные.
        """
        # Проверяем дубликаты по fact_id
        existing_ids = {f.fact_id for f in self._facts}
        if fact.fact_id in existing_ids:
            return
        
        self._facts.append(fact)
        logger.debug(f"Added fact: {fact.fact_id[:8]}... (weight={fact.attention_weight:.2f})")
        
        # Если превысили бюджет — чистим
        if self._total_tokens() > self.token_budget:
            self._evict_to_budget()
    
    def add_text(self, content: str, source: str = "", **metadata) -> Fact:
        """Создать факт из текста и добавить в рабочую память."""
        fact = Fact(
            content=content,
            source=source,
            metadata=metadata,
        )
        self.add(fact)
        return fact

    def add_anchor(self, content: str, source: str = "coordinate") -> Fact:
        """Создать якорный факт и добавить в рабочую память.
        
        Anchor facts: не decay, не evict — всегда остаются в WorkingMemory.
        Используется для координат сессии, имён пользователей, ключевых параметров.
        """
        fact = Fact(
            content=content,
            source=source,
            attention_weight=1.0,  # Максимальный вес
            stability=1.0,          # Максимальная стабильность
            is_anchor=True,
        )
        # Anchor facts не дублируются по контенту
        for existing in self._facts:
            if existing.content == content and existing.is_anchor:
                return existing
        self.add(fact)
        logger.info(f"Anchor fact added: '{content[:60]}...' " if len(content) > 60 else f"Anchor fact added: '{content}'")
        return fact

    
    def touch(self, fact_id: str) -> bool:
        """Отметить факт как использованный в текущем turn."""
        for fact in self._facts:
            if fact.fact_id == fact_id:
                fact.touch(self.current_turn)
                return True
        return False
    
    def cleanup(
        self,
        turn: Optional[int] = None,
        query: str = "",
        response: str = "",
    ) -> dict:
        """Очистить рабочую память после turn.
        
        Применяет scoring если переданы query/response.
        Применяет decay ко всем фактам.
        Перемещает холодные факты в storage (если задан).
        Оставляет горячие в рабочей памяти.
        
        Args:
            turn: Номер текущего turn (если None — автоинкремент)
            query: Запрос пользователя для scoring
            response: Ответ LLM для scoring
        
        Returns:
            Статистика очистки: {kept, evicted, total_tokens, scored}
        """
        if turn is not None:
            self.current_turn = turn
        else:
            self.current_turn += 1
        
        # Шаг 1: если есть разговор — обновляем веса через scorer
        scored_count = 0
        if query or response:
            results = self._scorer.score_facts(
                self._facts, query, response, self.current_turn
            )
            scored_count = sum(1 for r in results if r.matched)
        
        # Шаг 2: decay + classify
        kept = []
        evicted = []
        
        for fact in self._facts:
            # Anchor facts: decay пропускаем (защищены в fact.decay())
            fact.decay(self.current_turn, DECAY_RATE)
            fact.update_stability()
            
            # Anchor facts НИКОГДА не evict
            if fact.is_anchor:
                kept.append(fact)
                continue
            
            # Решаем: оставить или убрать
            if fact.is_hot(HOT_THRESHOLD):
                kept.append(fact)
            elif fact.is_cold(COLD_THRESHOLD):
                evicted.append(fact)
            else:
                # В промежуточной зоне — оставляем пока
                kept.append(fact)
        
        # Перемещаем холодные в storage
        for fact in evicted:
            if self.storage is not None:
                self.storage.store(fact)
                logger.debug(f"Evicted to storage: {fact.fact_id[:8]}... (weight={fact.attention_weight:.2f})")
        
        self._facts = kept
        self._evicted_count += len(evicted)
        
        stats = {
            "turn": self.current_turn,
            "kept": len(kept),
            "evicted": len(evicted),
            "scored": scored_count,
            "total_tokens": self._total_tokens(),
        }
        
        if evicted:
            logger.info(f"Memory cleanup turn {self.current_turn}: kept={len(kept)}, evicted={len(evicted)}, tokens={stats['total_tokens']}")
        
        return stats
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """Вернуть контент рабочей памяти для вставки в LLM промпт.
        
        Факты сортируются по attention_weight (важные первыми).
        Обрезается по max_tokens если задан.
        """
        if not self._facts:
            return ""
        
        # Сортируем по весу — важные первыми
        sorted_facts = sorted(self._facts, key=lambda f: f.attention_weight, reverse=True)
        
        budget = max_tokens or self.token_budget
        parts = []
        tokens_used = 0
        
        for fact in sorted_facts:
            fact_tokens = fact.token_size()
            if tokens_used + fact_tokens > budget:
                break
            parts.append(fact.content)
            tokens_used += fact_tokens
        
        if not parts:
            return ""
        
        return "\n---\n".join(parts)
    
    def get_facts(self) -> list[Fact]:
        """Вернуть все факты рабочей памяти."""
        return list(self._facts)
    
    def size(self) -> int:
        """Количество фактов в рабочей памяти."""
        return len(self._facts)
    
    def clear(self) -> None:
        """Полностью очистить рабочую память (все факты уходят в storage)."""
        if self.storage:
            for fact in self._facts:
                self.storage.store(fact)
        self._facts = []

    def save_state(self, path) -> None:
        """Сохранить состояние WorkingMemory в JSON файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "current_turn": self.current_turn,
            "evicted_count": self._evicted_count,
            "token_budget": self.token_budget,
            "facts": [f.to_dict() for f in self._facts],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.debug(f"WorkingMemory saved: {len(self._facts)} facts to {path}")

    def load_state(self, path) -> bool:
        """Загрузить состояние WorkingMemory из JSON файла.
        
        Returns: True если загрузка успешна, False если файл не найден.
        """
        path = Path(path)
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            logger.warning(f"WorkingMemory file empty, starting fresh: {path}")
            return False
        try:
            data = json.loads(path.read_text())
            self.current_turn = data.get("current_turn", 0)
            self._evicted_count = data.get("evicted_count", 0)
            loaded_facts = [
                Fact.from_dict(fd) for fd in data.get("facts", [])
            ]
            self._facts = loaded_facts
            logger.info(
                f"WorkingMemory loaded: {len(self._facts)} facts "
                f"(turn={self.current_turn}) from {path}"
            )
            return True
        except Exception as e:
            logger.error(f"WorkingMemory load failed: {e}")
            return False

    
    def stats(self, max_hot_facts: int = 50) -> dict:
        """Статистика рабочей памяти."""
        non_anchor = [f for f in self._facts if not f.is_anchor]
        return {
            "facts_count": len(self._facts),
            "anchor_count": len([f for f in self._facts if f.is_anchor]),
            "hot_facts": len(non_anchor),
            "max_hot_facts": max_hot_facts,
            "memory_pressure": round(self.get_memory_pressure(max_hot_facts), 2),
            "total_tokens": self._total_tokens(),
            "token_budget": self.token_budget,
            "budget_used_pct": round(self._total_tokens() / self.token_budget * 100, 1),
            "current_turn": self.current_turn,
            "evicted_total": self._evicted_count,
            "avg_attention": round(
                sum(f.attention_weight for f in self._facts) / len(self._facts), 2
            ) if self._facts else 0.0,
        }
    
    def _total_tokens(self) -> int:
        """Суммарный размер всех фактов в токенах."""
        return sum(f.token_size() for f in self._facts)
    

    def get_memory_pressure(self, max_hot_facts: int = 50) -> float:
        """Давление на RAM: отношение текущего числа фактов к максимальному.

        R3 roadmap: метрика для принятия решений об evict.

        Returns:
            0.0 — память пуста
            0.5 — половина заполнена
            1.0 — полностью заполнена
            > 1.0 — переполнена (нужен evict!)
        """
        non_anchor = [f for f in self._facts if not f.is_anchor]
        return len(non_anchor) / max(max_hot_facts, 1)

    def evict_below_threshold(
        self,
        threshold: float = 0.2,
        max_facts: int = 50,
    ) -> int:
        """Агрессивный evict маловажных фактов из RAM (R3 roadmap).

        Вызывается когда memory pressure > 0.8 (более 80% от max_facts).
        Выгружает факты с attention_weight < threshold в cold storage.
        Anchor facts НИКОГДА не выгружаются.

        Args:
            threshold: порог внимания для evict (по умолчанию 0.2)
            max_facts: максимальное число горячих фактов в RAM

        Returns:
            Число выгруженных фактов
        """
        non_anchor = [f for f in self._facts if not f.is_anchor]

        # Выгружаем только если давление > 0.8
        if len(non_anchor) <= max_facts * 0.8:
            return 0

        # Сортируем по весу — самые слабые первыми
        candidates = sorted(
            [f for f in non_anchor if f.attention_weight < threshold],
            key=lambda f: f.attention_weight,
        )

        if not candidates:
            return 0

        evicted_count = 0
        evict_ids = {f.fact_id for f in candidates}

        kept = []
        for fact in self._facts:
            if fact.fact_id in evict_ids:
                # Перемещаем в cold storage
                if self.storage is not None:
                    self.storage.store(fact)
                self._evicted_count += 1
                evicted_count += 1
                logger.info(
                    f"[R3 evict] Fact evicted to cold: {fact.fact_id[:8]}... "
                    f"(weight={fact.attention_weight:.3f} < threshold={threshold})"
                )
            else:
                kept.append(fact)

        self._facts = kept

        if evicted_count > 0:
            logger.info(
                f"[R3 evict] Evicted {evicted_count} low-attention facts "
                f"(RAM: {len(self._facts)} facts remaining)"
            )

        return evicted_count

    def _evict_to_budget(self) -> None:
        """Вытеснить самые холодные факты пока не уложимся в бюджет."""
        while self._total_tokens() > self.token_budget and self._facts:
            # Вытесняем факт с наименьшим весом
            coldest = min(self._facts, key=lambda f: f.attention_weight)
            self._facts.remove(coldest)
            if self.storage:
                self.storage.store(coldest)
            self._evicted_count += 1
            logger.debug(f"Budget eviction: {coldest.fact_id[:8]}...")
    
    def __repr__(self) -> str:
        return (
            f"WorkingMemory(facts={len(self._facts)}, "
            f"tokens={self._total_tokens()}/{self.token_budget}, "
            f"turn={self.current_turn})"
        )
