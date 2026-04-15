"""Тесты для MemoryOptimizer."""

import pytest
from pure_intellect.core.memory.fact import Fact, CompressionLevel
from pure_intellect.core.memory.working_memory import WorkingMemory
from pure_intellect.core.memory.storage import MemoryStorage
from pure_intellect.core.memory.optimizer import MemoryOptimizer, OptimizationStats


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def storage():
    return MemoryStorage()


@pytest.fixture
def memory(storage):
    return WorkingMemory(token_budget=2000, storage=storage)


@pytest.fixture
def optimizer():
    return MemoryOptimizer(
        hot_retrieval_threshold=3,
        cold_weight_threshold=0.1,
        archive_age_threshold=50,
        run_every_n_turns=5,
    )


# ─── OptimizationStats Tests ──────────────────────────────────────────────────

class TestOptimizationStats:
    def test_to_dict_structure(self):
        stats = OptimizationStats(turn=10)
        d = stats.to_dict()
        assert "turn" in d
        assert "promoted" in d
        assert "compressed" in d
        assert "archived" in d
        assert "storage_size" in d
        assert "working_size" in d

    def test_defaults_are_zero(self):
        stats = OptimizationStats(turn=1)
        assert stats.promoted == 0
        assert stats.compressed == 0
        assert stats.archived == 0

    def test_repr(self):
        stats = OptimizationStats(turn=5, promoted=2, compressed=3)
        r = repr(stats)
        assert "promoted=2" in r
        assert "compressed=3" in r


# ─── MemoryOptimizer.should_run() Tests ──────────────────────────────────────

class TestShouldRun:
    def test_should_run_at_start(self, optimizer):
        """Оптимизатор должен запускаться с самого начала."""
        assert optimizer.should_run(0) is True

    def test_should_run_every_n_turns(self, optimizer):
        optimizer._last_run_turn = 0
        assert optimizer.should_run(5) is True   # ровно N turns
        assert optimizer.should_run(4) is False  # ещё рано
        assert optimizer.should_run(6) is True   # прошло больше N

    def test_should_not_run_before_n_turns(self, optimizer):
        optimizer._last_run_turn = 10
        assert optimizer.should_run(13) is False  # прошло только 3 из 5
        assert optimizer.should_run(15) is True   # прошло 5


# ─── MemoryOptimizer.run() Tests ─────────────────────────────────────────────

class TestOptimizerRun:
    def test_run_returns_stats(self, optimizer, memory, storage):
        stats = optimizer.run(memory, storage, current_turn=5)
        assert isinstance(stats, OptimizationStats)
        assert stats.turn == 5

    def test_run_empty_storage_no_ops(self, optimizer, memory, storage):
        """Пустой storage — ничего не делаем."""
        stats = optimizer.run(memory, storage, current_turn=5)
        assert stats.promoted == 0
        assert stats.compressed == 0
        assert stats.archived == 0

    def test_run_updates_last_run_turn(self, optimizer, memory, storage):
        optimizer.run(memory, storage, current_turn=10)
        assert optimizer._last_run_turn == 10

    def test_run_increments_total_runs(self, optimizer, memory, storage):
        optimizer.run(memory, storage, current_turn=5)
        optimizer.run(memory, storage, current_turn=10)
        assert optimizer._total_runs == 2


# ─── Promote Tests ────────────────────────────────────────────────────────────

class TestPromoteHotFacts:
    def test_promotes_frequently_retrieved_fact(self, optimizer, memory, storage):
        """Факт запрошенный 3+ раз должен вернуться в WorkingMemory."""
        fact = Fact(content="Frequently accessed fact about parsing")
        fact.reference_count = 5  # Превышает threshold=3
        storage.store(fact)
        
        stats = optimizer.run(memory, storage, current_turn=5)
        assert stats.promoted >= 1
        # Факт теперь в working memory
        assert memory.size() > 0

    def test_does_not_promote_cold_fact(self, optimizer, memory, storage):
        """Факт с малым retrieval_count не должен промоутиться."""
        fact = Fact(content="Rarely accessed fact")
        fact.reference_count = 1  # Меньше threshold=3
        storage.store(fact)
        
        stats = optimizer.run(memory, storage, current_turn=5)
        assert stats.promoted == 0

    def test_promoted_fact_reference_count_reset(self, optimizer, memory, storage):
        """После promote reference_count сбрасывается чтобы не промоутить снова."""
        fact = Fact(content="Hot fact to promote")
        fact.reference_count = 5
        storage.store(fact)
        
        optimizer.run(memory, storage, current_turn=5)
        # reference_count должен быть сброшен
        assert fact.reference_count == 0


# ─── Compress Tests ───────────────────────────────────────────────────────────

class TestCompressColdFacts:
    def test_compresses_low_weight_fact(self, optimizer, memory, storage):
        """Факт с низким весом должен быть сжат."""
        fact = Fact(
            content="This is a cold fact that should be compressed. Some extra text here.",
            attention_weight=0.05,  # Ниже cold_weight_threshold=0.1
        )
        storage.store(fact)
        
        stats = optimizer.run(memory, storage, current_turn=5)
        assert stats.compressed >= 1
        
        compressed = storage.get(fact.fact_id)
        assert compressed.compression_level != CompressionLevel.RAW

    def test_does_not_compress_hot_fact(self, optimizer, memory, storage):
        """Горячий факт не должен сжиматься."""
        fact = Fact(
            content="Hot fact that should not be compressed at all",
            attention_weight=0.8,  # Выше cold_weight_threshold
        )
        storage.store(fact)
        
        stats = optimizer.run(memory, storage, current_turn=5)
        assert stats.compressed == 0
        
        stored = storage.get(fact.fact_id)
        assert stored.compression_level == CompressionLevel.RAW

    def test_does_not_compress_archived(self, optimizer, memory, storage):
        """Архивированный факт не сжимается повторно."""
        fact = Fact(content="Archived fact", attention_weight=0.01)
        fact.compression_level = CompressionLevel.ARCHIVED
        storage.store(fact)
        
        stats = optimizer.run(memory, storage, current_turn=5)
        assert stats.compressed == 0


# ─── Archive Tests ────────────────────────────────────────────────────────────

class TestArchiveOldFacts:
    def test_archives_very_old_cold_fact(self, optimizer, memory, storage):
        """Старый холодный ENTITY_ONLY факт должен быть заархивирован."""
        fact = Fact(
            content="Very old fact",
            attention_weight=0.01,  # Очень холодный
        )
        fact.last_referenced = 0
        fact.compression_level = CompressionLevel.ENTITY_ONLY
        storage.store(fact)
        
        # Запускаем при turn=60 (age=60 >= archive_age_threshold=50)
        stats = optimizer.run(memory, storage, current_turn=60)
        assert stats.archived >= 1
        
        archived = storage.get(fact.fact_id)
        assert archived.compression_level == CompressionLevel.ARCHIVED

    def test_does_not_archive_recent_fact(self, optimizer, memory, storage):
        """Недавно использованный факт не архивируется."""
        fact = Fact(content="Recent fact", attention_weight=0.01)
        fact.last_referenced = 55  # Недавно (turn=60, age=5)
        fact.compression_level = CompressionLevel.ENTITY_ONLY
        storage.store(fact)
        
        stats = optimizer.run(memory, storage, current_turn=60)
        assert stats.archived == 0


# ─── run_if_needed() Tests ───────────────────────────────────────────────────

class TestRunIfNeeded:
    def test_runs_when_due(self, optimizer, memory, storage):
        """Запускается если прошло N turns."""
        result = optimizer.run_if_needed(memory, storage, current_turn=5)
        assert result is not None
        assert isinstance(result, OptimizationStats)

    def test_skips_when_not_due(self, optimizer, memory, storage):
        """Не запускается если рано."""
        optimizer._last_run_turn = 3  # Последний раз на turn 3
        result = optimizer.run_if_needed(memory, storage, current_turn=6)  # Прошло 3 из 5
        assert result is None

    def test_runs_after_skip(self, optimizer, memory, storage):
        """После паузы запускается снова."""
        optimizer._last_run_turn = 3
        result = optimizer.run_if_needed(memory, storage, current_turn=8)  # Прошло 5
        assert result is not None


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestOptimizerIntegration:
    def test_full_lifecycle(self):
        """E2E тест: факты создаются, уходят в storage, оптимизируются."""
        storage = MemoryStorage()
        memory = WorkingMemory(token_budget=2000, storage=storage)
        optimizer = MemoryOptimizer(
            hot_retrieval_threshold=2,
            cold_weight_threshold=0.15,
            run_every_n_turns=3,
        )
        
        # Добавляем факты
        f_hot = memory.add_text("parse_file processes python code")
        f_cold = memory.add_text("unrelated information rarely used")
        
        # Симулируем 12 turns — упоминаем только f_hot
        for turn in range(1, 13):
            memory.cleanup(
                turn=turn,
                query="parse_file python",
                response="parse_file processes code",
            )
            # Запускаем оптимизатор каждые 3 turns
            optimizer.run_if_needed(memory, storage, current_turn=turn)
        
        # Итоговое состояние
        opt_stats = optimizer.optimizer_stats()
        assert opt_stats["total_runs"] > 0
        
        # Что-то было сжато или промоутировано
        # (конкретное значение зависит от параметров decay)
        total_facts = memory.size() + storage.size()
        assert total_facts == 2  # Все факты сохранились (хотя бы где-то)
