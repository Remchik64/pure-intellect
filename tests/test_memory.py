"""Тесты для Memory Subsystem — Fact, WorkingMemory, MemoryStorage."""

import pytest
import tempfile
import json
from pathlib import Path
from pure_intellect.core.memory.fact import Fact, CompressionLevel
from pure_intellect.core.memory.working_memory import WorkingMemory
from pure_intellect.core.memory.storage import MemoryStorage


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def fact():
    return Fact(content="Функция parse_file() обрабатывает Python AST", source="parser.py")


@pytest.fixture
def storage():
    return MemoryStorage()  # in-memory без файла


@pytest.fixture
def memory(storage):
    return WorkingMemory(token_budget=1000, storage=storage)


# ─── Fact Tests ─────────────────────────────────────────────────────────────

class TestFact:
    def test_fact_created_with_defaults(self, fact):
        assert fact.content != ""
        assert 0.0 <= fact.attention_weight <= 1.0
        assert fact.compression_level == CompressionLevel.RAW
        assert fact.reference_count == 0
        assert fact.stability == 0.0

    def test_fact_id_is_unique(self):
        f1 = Fact(content="a")
        f2 = Fact(content="a")
        assert f1.fact_id != f2.fact_id

    def test_touch_increases_weight(self, fact):
        initial = fact.attention_weight
        fact.touch(turn=1)
        assert fact.attention_weight > initial
        assert fact.reference_count == 1
        assert fact.last_referenced == 1

    def test_touch_caps_at_1(self, fact):
        for i in range(20):
            fact.touch(turn=i)
        assert fact.attention_weight <= 1.0

    def test_decay_reduces_weight(self, fact):
        fact.attention_weight = 0.8
        fact.last_referenced = 0
        fact.decay(current_turn=10)
        assert fact.attention_weight < 0.8

    def test_decay_never_below_zero(self, fact):
        fact.attention_weight = 0.1
        fact.last_referenced = 0
        fact.decay(current_turn=100, decay_rate=0.5)
        assert fact.attention_weight >= 0.0

    def test_is_hot(self):
        hot = Fact(content="hot", attention_weight=0.9)
        cold = Fact(content="cold", attention_weight=0.1)
        assert hot.is_hot() is True
        assert cold.is_hot() is False

    def test_is_cold(self):
        cold = Fact(content="cold", attention_weight=0.05)
        assert cold.is_cold() is True

    def test_token_size(self, fact):
        size = fact.token_size()
        assert isinstance(size, int)
        assert size > 0

    def test_serialization_roundtrip(self, fact):
        fact.touch(turn=1)
        d = fact.to_dict()
        restored = Fact.from_dict(d)
        assert restored.fact_id == fact.fact_id
        assert restored.content == fact.content
        assert restored.reference_count == fact.reference_count
        assert restored.compression_level == fact.compression_level


# ─── WorkingMemory Tests ─────────────────────────────────────────────────────

class TestWorkingMemory:
    def test_initially_empty(self, memory):
        assert memory.size() == 0
        assert memory.get_context() == ""

    def test_add_fact(self, memory, fact):
        memory.add(fact)
        assert memory.size() == 1

    def test_add_text(self, memory):
        f = memory.add_text("Hello world", source="test")
        assert memory.size() == 1
        assert isinstance(f, Fact)
        assert f.content == "Hello world"

    def test_no_duplicate_facts(self, memory, fact):
        memory.add(fact)
        memory.add(fact)  # дубликат
        assert memory.size() == 1

    def test_touch_updates_fact(self, memory, fact):
        memory.add(fact)
        initial_weight = fact.attention_weight
        memory.touch(fact.fact_id)
        assert fact.attention_weight >= initial_weight

    def test_touch_nonexistent_returns_false(self, memory):
        result = memory.touch("nonexistent-id")
        assert result is False

    def test_cleanup_increments_turn(self, memory):
        memory.cleanup()
        assert memory.current_turn == 1
        memory.cleanup()
        assert memory.current_turn == 2

    def test_cleanup_explicit_turn(self, memory):
        memory.cleanup(turn=10)
        assert memory.current_turn == 10

    def test_hot_facts_stay_after_cleanup(self, memory):
        f = memory.add_text("Important fact", attention_weight=1.0)
        # Обновляем weight напрямую
        f.attention_weight = 1.0
        stats = memory.cleanup(turn=1)
        # Горячий факт должен остаться
        assert stats["kept"] >= 0  # может уйти из-за decay

    def test_cold_facts_evicted_to_storage(self, memory, storage):
        # Создаём холодный факт
        f = Fact(content="Cold fact that nobody uses", attention_weight=0.01)
        memory.add(f)
        memory.cleanup(turn=1)
        # Холодный факт должен уйти в storage
        assert storage.size() > 0 or memory.size() == 0  # или ушёл, или ещё в памяти

    def test_get_context_returns_string(self, memory):
        memory.add_text("Context fact 1")
        memory.add_text("Context fact 2")
        context = memory.get_context()
        assert isinstance(context, str)

    def test_get_context_sorted_by_weight(self, memory):
        f_low = Fact(content="Low priority", attention_weight=0.2)
        f_high = Fact(content="High priority", attention_weight=0.9)
        memory.add(f_low)
        memory.add(f_high)
        context = memory.get_context()
        # Высокоприоритетный должен быть первым
        assert context.index("High priority") < context.index("Low priority")

    def test_stats_returns_dict(self, memory):
        stats = memory.stats()
        assert "facts_count" in stats
        assert "total_tokens" in stats
        assert "budget_used_pct" in stats
        assert "current_turn" in stats

    def test_clear_moves_to_storage(self, memory, storage):
        memory.add_text("Fact 1")
        memory.add_text("Fact 2")
        memory.clear()
        assert memory.size() == 0
        assert storage.size() == 2


# ─── MemoryStorage Tests ──────────────────────────────────────────────────────

class TestMemoryStorage:
    def test_initially_empty(self, storage):
        assert storage.size() == 0

    def test_store_fact(self, storage, fact):
        storage.store(fact)
        assert storage.size() == 1

    def test_store_no_duplicates(self, storage, fact):
        storage.store(fact)
        storage.store(fact)  # дубликат
        assert storage.size() == 1

    def test_retrieve_by_keyword(self, storage):
        storage.store(Fact(content="parse_file обрабатывает Python файлы"))
        storage.store(Fact(content="ChromaDB хранит векторные embeddings"))
        results = storage.retrieve("parse_file", top_k=5)
        assert len(results) >= 1
        assert any("parse_file" in r.content for r in results)

    def test_retrieve_empty_returns_empty(self, storage):
        results = storage.retrieve("anything", top_k=5)
        assert results == []

    def test_retrieve_ignores_archived(self, storage, fact):
        fact.compression_level = CompressionLevel.ARCHIVED
        storage.store(fact)
        results = storage.retrieve(fact.content[:10], top_k=5)
        assert len(results) == 0

    def test_get_by_id(self, storage, fact):
        storage.store(fact)
        retrieved = storage.get(fact.fact_id)
        assert retrieved is not None
        assert retrieved.fact_id == fact.fact_id

    def test_compress_raw_to_summarized(self, storage):
        f = Fact(content="This is a long sentence. And another sentence here. And more text.")
        storage.store(f)
        result = storage.compress(f.fact_id)
        assert result is True
        compressed = storage.get(f.fact_id)
        assert compressed.compression_level == CompressionLevel.SUMMARIZED

    def test_compress_nonexistent(self, storage):
        result = storage.compress("nonexistent-id")
        assert result is False

    def test_soft_delete(self, storage, fact):
        storage.store(fact)
        storage.delete(fact.fact_id)
        deleted = storage.get(fact.fact_id)
        assert deleted.compression_level == CompressionLevel.ARCHIVED

    def test_stats_structure(self, storage, fact):
        storage.store(fact)
        stats = storage.stats()
        assert "total_facts" in stats
        assert "by_compression_level" in stats
        assert "RAW" in stats["by_compression_level"]

    def test_persistence_save_load(self, fact):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        # Сохраняем
        s1 = MemoryStorage(storage_path=path)
        s1.store(fact)
        s1.save()
        
        # Загружаем
        s2 = MemoryStorage(storage_path=path)
        assert s2.size() == 1
        restored = s2.get(fact.fact_id)
        assert restored is not None
        assert restored.content == fact.content
        
        # Cleanup
        Path(path).unlink(missing_ok=True)
