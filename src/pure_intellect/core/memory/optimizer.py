"""MemoryOptimizer — фоновая оптимизация памяти Pure Intellect.

Периодически анализирует MemoryStorage и принимает решения:
- Горячие факты (часто запрашиваемые) → promote обратно в WorkingMemory
- Холодные факты (давно не используемые) → compress (сжать уровень)
- Очень старые факты → archive (мягкое удаление)

Цикл: каждые N turns вызывать MemoryOptimizer.run()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from .fact import Fact, CompressionLevel
from .working_memory import WorkingMemory
from .storage import MemoryStorage

logger = logging.getLogger(__name__)

# Пороги по умолчанию
DEFAULT_HOT_RETRIEVAL_THRESHOLD = 3    # Факт запрошен N+ раз → promote
DEFAULT_COLD_WEIGHT_THRESHOLD = 0.1    # Вес ниже N → кандидат на компрессию
DEFAULT_ARCHIVE_AGE_THRESHOLD = 50     # Не использовался N+ turns → archive
DEFAULT_RUN_EVERY_N_TURNS = 5          # Запускать оптимизацию каждые N turns


@dataclass
class OptimizationStats:
    """Статистика одного прогона оптимизатора."""
    turn: int
    promoted: int = 0          # Фактов повышено в WorkingMemory
    compressed: int = 0        # Фактов сжато
    archived: int = 0          # Фактов заархивировано
    storage_size: int = 0      # Итоговый размер storage
    working_size: int = 0      # Итоговый размер WorkingMemory
    skipped: int = 0           # Пропущено (уже оптимальны)
    
    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "promoted": self.promoted,
            "compressed": self.compressed,
            "archived": self.archived,
            "storage_size": self.storage_size,
            "working_size": self.working_size,
            "skipped": self.skipped,
        }
    
    def __repr__(self) -> str:
        return (
            f"OptimizationStats(turn={self.turn}, "
            f"promoted={self.promoted}, compressed={self.compressed}, "
            f"archived={self.archived})"
        )


class MemoryOptimizer:
    """Фоновый оптимизатор памяти.
    
    Принцип работы:
    1. Promote: горячие факты из Storage → WorkingMemory
    2. Compress: холодные факты → повышаем compression_level
    3. Archive: очень старые факты → мягко удаляем
    
    Запускать через run() каждые N turns.
    """
    
    def __init__(
        self,
        hot_retrieval_threshold: int = DEFAULT_HOT_RETRIEVAL_THRESHOLD,
        cold_weight_threshold: float = DEFAULT_COLD_WEIGHT_THRESHOLD,
        archive_age_threshold: int = DEFAULT_ARCHIVE_AGE_THRESHOLD,
        run_every_n_turns: int = DEFAULT_RUN_EVERY_N_TURNS,
    ):
        self.hot_retrieval_threshold = hot_retrieval_threshold
        self.cold_weight_threshold = cold_weight_threshold
        self.archive_age_threshold = archive_age_threshold
        self.run_every_n_turns = run_every_n_turns
        self._last_run_turn: int = -run_every_n_turns  # чтобы запускался с turn=0
        self._total_runs: int = 0
    
    def should_run(self, current_turn: int) -> bool:
        """Проверить нужно ли запускать оптимизатор в этот turn."""
        return (current_turn - self._last_run_turn) >= self.run_every_n_turns
    
    def run(
        self,
        working_memory: WorkingMemory,
        storage: MemoryStorage,
        current_turn: Optional[int] = None,
    ) -> OptimizationStats:
        """Запустить оптимизацию памяти.
        
        Args:
            working_memory: Рабочая память для promote операций
            storage: Долгосрочное хранилище для анализа
            current_turn: Номер текущего turn
        
        Returns:
            Статистика оптимизации
        """
        turn = current_turn or working_memory.current_turn
        stats = OptimizationStats(turn=turn)
        
        logger.info(f"MemoryOptimizer.run() at turn={turn}, storage={storage.size()} facts")
        
        # Шаг 1: Promote горячих фактов из storage в WorkingMemory
        stats.promoted = self._promote_hot_facts(working_memory, storage)
        
        # Шаг 2: Compress холодных фактов
        stats.compressed = self._compress_cold_facts(storage, turn)
        
        # Шаг 3: Archive очень старых фактов
        stats.archived = self._archive_old_facts(storage, turn)
        
        # Финальная статистика
        stats.storage_size = storage.size()
        stats.working_size = working_memory.size()
        
        self._last_run_turn = turn
        self._total_runs += 1
        
        if stats.promoted or stats.compressed or stats.archived:
            logger.info(
                f"Optimization done: promoted={stats.promoted}, "
                f"compressed={stats.compressed}, archived={stats.archived}"
            )
        else:
            logger.debug("Optimization: nothing to do")
        
        return stats
    
    def run_if_needed(
        self,
        working_memory: WorkingMemory,
        storage: MemoryStorage,
        current_turn: Optional[int] = None,
    ) -> Optional[OptimizationStats]:
        """Запустить оптимизацию только если пришло время.
        
        Returns:
            OptimizationStats если запустился, None если ещё рано.
        """
        turn = current_turn or working_memory.current_turn
        if not self.should_run(turn):
            return None
        return self.run(working_memory, storage, turn)
    
    def _promote_hot_facts(
        self,
        working_memory: WorkingMemory,
        storage: MemoryStorage,
    ) -> int:
        """Перенести горячие факты из Storage в WorkingMemory.
        
        Горячий факт = часто запрашивался (retrieval_count >= threshold).
        Такие факты лучше держать прямо в рабочей памяти.
        """
        promoted = 0
        hot_facts = storage.get_hot_facts(self.hot_retrieval_threshold)
        
        for fact in hot_facts:
            # Сбрасываем retrieval_count чтобы не промоутить бесконечно
            fact.reference_count = 0
            # Добавляем в рабочую память (если не переполнена)
            working_memory.add(fact)
            promoted += 1
            logger.debug(f"Promoted hot fact {fact.fact_id[:8]}... to WorkingMemory")
        
        return promoted
    
    def _compress_cold_facts(
        self,
        storage: MemoryStorage,
        current_turn: int,
    ) -> int:
        """Сжать холодные факты в Storage.
        
        Холодный факт = низкий attention_weight.
        Сжимаем: RAW → SUMMARIZED → ENTITY_ONLY.
        """
        compressed = 0
        cold_facts = storage.get_cold_facts(self.cold_weight_threshold)
        
        for fact in cold_facts:
            if fact.compression_level == CompressionLevel.ARCHIVED:
                continue
            if fact.compression_level == CompressionLevel.ENTITY_ONLY:
                continue  # Уже максимально сжат (до ARCHIVED сжимаем отдельно)
            
            result = storage.compress(fact.fact_id)
            if result:
                compressed += 1
                logger.debug(
                    f"Compressed fact {fact.fact_id[:8]}... "
                    f"to {storage.get(fact.fact_id).compression_level.name}"
                )
        
        return compressed
    
    def _archive_old_facts(
        self,
        storage: MemoryStorage,
        current_turn: int,
    ) -> int:
        """Архивировать очень старые неиспользуемые факты.
        
        Кандидаты: не referenced давно + очень низкий weight.
        Мягкое удаление — ARCHIVED, не физическое.
        """
        archived = 0
        
        for fact in list(storage._facts.values()):
            if fact.compression_level == CompressionLevel.ARCHIVED:
                continue
            
            age = current_turn - fact.last_referenced
            is_very_old = age >= self.archive_age_threshold
            is_very_cold = fact.attention_weight < (self.cold_weight_threshold / 2)
            is_entity_only = fact.compression_level == CompressionLevel.ENTITY_ONLY
            
            if is_very_old and is_very_cold and is_entity_only:
                storage.delete(fact.fact_id)  # → ARCHIVED
                archived += 1
                logger.debug(f"Archived old fact {fact.fact_id[:8]}...")
        
        return archived
    
    def optimizer_stats(self) -> dict:
        """Статистика работы оптимизатора."""
        return {
            "total_runs": self._total_runs,
            "last_run_turn": self._last_run_turn,
            "run_every_n_turns": self.run_every_n_turns,
        }
