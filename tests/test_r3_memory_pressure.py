"""Тесты для R3: Умная фильтрация WorkingMemory.

Проверяет:
- get_memory_pressure() — давление на RAM
- evict_below_threshold() — агрессивный evict маловажных фактов
- stats() — обновлённая статистика с memory_pressure
"""

from unittest.mock import MagicMock

import pytest

from pure_intellect.core.memory.fact import Fact
from pure_intellect.core.memory.working_memory import WorkingMemory


@pytest.fixture
def wm():
    """WorkingMemory без storage."""
    return WorkingMemory(token_budget=10000)


@pytest.fixture
def wm_with_storage():
    """WorkingMemory с mock storage."""
    storage = MagicMock()
    return WorkingMemory(token_budget=10000, storage=storage), storage


def make_fact(content: str, weight: float, is_anchor: bool = False) -> Fact:
    """Создать тестовый факт с заданным весом."""
    fact = Fact(content=content, source="test")
    fact.attention_weight = weight
    fact.is_anchor = is_anchor
    return fact


# ── get_memory_pressure() tests ───────────────────────────

class TestGetMemoryPressure:
    """Тесты get_memory_pressure()."""

    def test_empty_memory_zero_pressure(self, wm):
        assert wm.get_memory_pressure(max_hot_facts=50) == 0.0

    def test_half_full_returns_05(self, wm):
        for i in range(25):
            wm.add(make_fact(f"Факт {i}", weight=0.5))
        pressure = wm.get_memory_pressure(max_hot_facts=50)
        assert pressure == pytest.approx(0.5, abs=0.01)

    def test_full_returns_1(self, wm):
        for i in range(50):
            wm.add(make_fact(f"Факт {i}", weight=0.5))
        pressure = wm.get_memory_pressure(max_hot_facts=50)
        assert pressure == pytest.approx(1.0, abs=0.01)

    def test_overfull_returns_over_1(self, wm):
        for i in range(60):
            wm.add(make_fact(f"Факт {i}", weight=0.5))
        pressure = wm.get_memory_pressure(max_hot_facts=50)
        assert pressure > 1.0

    def test_anchor_facts_excluded_from_pressure(self, wm):
        """Anchor facts не учитываются в давлении на RAM."""
        # Добавляем 10 anchor facts
        for i in range(10):
            wm.add_anchor(f"Anchor {i}", source="test")
        # Добавляем 25 обычных фактов
        for i in range(25):
            wm.add(make_fact(f"Обычный факт {i}", weight=0.5))

        pressure = wm.get_memory_pressure(max_hot_facts=50)
        # 25 обычных / 50 max = 0.5 (anchor не учитываются)
        assert pressure == pytest.approx(0.5, abs=0.01)

    def test_custom_max_hot_facts(self, wm):
        for i in range(10):
            wm.add(make_fact(f"Факт {i}", weight=0.5))
        # При max=10 давление = 10/10 = 1.0
        assert wm.get_memory_pressure(max_hot_facts=10) == pytest.approx(1.0, abs=0.01)
        # При max=20 давление = 10/20 = 0.5
        assert wm.get_memory_pressure(max_hot_facts=20) == pytest.approx(0.5, abs=0.01)

    def test_zero_max_does_not_raise(self, wm):
        """max_hot_facts=0 не вызывает деление на ноль."""
        # Должен использовать max(max_hot_facts, 1)
        try:
            pressure = wm.get_memory_pressure(max_hot_facts=0)
            assert pressure >= 0.0
        except ZeroDivisionError:
            pytest.fail("get_memory_pressure() вызвал ZeroDivisionError при max_hot_facts=0")


# ── evict_below_threshold() tests ────────────────────────

class TestEvictBelowThreshold:
    """Тесты evict_below_threshold()."""

    def test_no_evict_when_pressure_low(self, wm_with_storage):
        """Evict не происходит при давлении < 0.8."""
        wm, storage = wm_with_storage
        # Добавляем 30 фактов с низким весом (60% от max=50)
        for i in range(30):
            wm.add(make_fact(f"Слабый факт {i}", weight=0.1))

        evicted = wm.evict_below_threshold(threshold=0.2, max_facts=50)
        assert evicted == 0
        storage.store.assert_not_called()

    def test_evicts_when_pressure_high(self, wm_with_storage):
        """Evict происходит при давлении > 0.8."""
        wm, storage = wm_with_storage
        # Добавляем 45 фактов — давление 45/50 = 0.9 > 0.8
        for i in range(35):
            wm.add(make_fact(f"Сильный факт {i}", weight=0.8))  # выше threshold
        for i in range(10):
            wm.add(make_fact(f"Слабый факт {i}", weight=0.1))  # ниже threshold

        evicted = wm.evict_below_threshold(threshold=0.2, max_facts=50)
        assert evicted == 10  # только слабые выгружены
        assert storage.store.call_count == 10

    def test_anchor_facts_never_evicted(self, wm_with_storage):
        """Anchor facts НИКОГДА не выгружаются."""
        wm, storage = wm_with_storage
        # Добавляем 10 anchor с низким весом
        for i in range(10):
            fact = make_fact(f"Anchor {i}", weight=0.05, is_anchor=True)
            wm._facts.append(fact)
        # Добавляем 35 обычных слабых фактов → давление (35+0)/50 = 0.7
        # Нет! anchor не считается — только 35 non-anchor. 35/50 = 0.7 < 0.8
        # Нужно 41+: добавим ещё слабые
        for i in range(5):
            wm.add(make_fact(f"Слабый {i}", weight=0.05))
        # Итого non-anchor = 5, давление 5/50 = 0.1 < 0.8
        # Добавляем ещё до 42 non-anchor чтобы давление > 0.8
        for i in range(37):
            wm.add(make_fact(f"Доп {i}", weight=0.05))

        total_before = wm.size()
        anchor_count_before = len([f for f in wm._facts if f.is_anchor])

        wm.evict_below_threshold(threshold=0.2, max_facts=50)

        anchor_count_after = len([f for f in wm._facts if f.is_anchor])
        assert anchor_count_after == anchor_count_before  # anchor не тронуты

    def test_only_low_weight_evicted(self, wm_with_storage):
        """Только факты ниже threshold выгружаются."""
        wm, storage = wm_with_storage
        # 42 факта чтобы давление > 0.8
        for i in range(20):
            wm.add(make_fact(f"Высокий {i}", weight=0.9))  # выше threshold
        for i in range(22):
            wm.add(make_fact(f"Низкий {i}", weight=0.1))  # ниже threshold

        evicted = wm.evict_below_threshold(threshold=0.2, max_facts=50)
        assert evicted == 22
        # Проверяем что все оставшиеся факты выше threshold
        remaining_non_anchor = [f for f in wm._facts if not f.is_anchor]
        for fact in remaining_non_anchor:
            assert fact.attention_weight >= 0.2 or fact.is_anchor

    def test_evict_updates_counter(self, wm_with_storage):
        """Evicted counter обновляется."""
        wm, storage = wm_with_storage
        for i in range(42):
            wm.add(make_fact(f"Слабый {i}", weight=0.05))

        initial_count = wm.stats()["evicted_total"]
        evicted = wm.evict_below_threshold(threshold=0.2, max_facts=50)
        assert wm.stats()["evicted_total"] == initial_count + evicted

    def test_returns_zero_when_nothing_to_evict(self, wm_with_storage):
        """Возвращает 0 если нет кандидатов ниже threshold."""
        wm, storage = wm_with_storage
        # 42 факта с высоким весом → давление > 0.8 но нечего выгружать
        for i in range(42):
            wm.add(make_fact(f"Сильный {i}", weight=0.9))

        evicted = wm.evict_below_threshold(threshold=0.2, max_facts=50)
        assert evicted == 0

    def test_no_storage_evict_still_removes_from_ram(self, wm):
        """Без storage факты удаляются из RAM но не сохраняются."""
        # wm без storage
        for i in range(42):
            wm.add(make_fact(f"Слабый {i}", weight=0.05))

        size_before = wm.size()
        evicted = wm.evict_below_threshold(threshold=0.2, max_facts=50)
        size_after = wm.size()
        assert evicted > 0
        assert size_after < size_before


# ── stats() with memory_pressure tests ───────────────────

class TestStatsWithPressure:
    """Тесты обновлённого stats() с memory_pressure."""

    def test_stats_has_memory_pressure(self, wm):
        stats = wm.stats()
        assert "memory_pressure" in stats
        assert "hot_facts" in stats
        assert "max_hot_facts" in stats
        assert "anchor_count" in stats

    def test_stats_pressure_zero_when_empty(self, wm):
        stats = wm.stats()
        assert stats["memory_pressure"] == 0.0
        assert stats["hot_facts"] == 0
        assert stats["anchor_count"] == 0

    def test_stats_pressure_correct(self, wm):
        for i in range(25):
            wm.add(make_fact(f"Факт {i}", weight=0.5))
        stats = wm.stats(max_hot_facts=50)
        assert stats["hot_facts"] == 25
        assert stats["max_hot_facts"] == 50
        assert stats["memory_pressure"] == pytest.approx(0.5, abs=0.01)

    def test_stats_anchor_counted_separately(self, wm):
        for i in range(3):
            wm.add_anchor(f"Anchor {i}", source="test")
        for i in range(10):
            wm.add(make_fact(f"Факт {i}", weight=0.5))

        stats = wm.stats(max_hot_facts=50)
        assert stats["anchor_count"] == 3
        assert stats["hot_facts"] == 10
        # Pressure считается только по non-anchor
        assert stats["memory_pressure"] == pytest.approx(0.2, abs=0.01)
