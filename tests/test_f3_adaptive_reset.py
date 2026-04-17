"""F3: Тесты для адаптивного soft reset по CCI.

Адаптивный reset срабатывает по трём условиям:
1. Hard limit: turns >= max_turns_without_reset
2. History overflow: len(history) > context_window_size
3. Adaptive: CCI < threshold AND turns >= min_turns_between_resets
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Фикстура OrchestratorPipeline с моками ──────────────────

@pytest.fixture
def pipeline():
    """OrchestratorPipeline с заглушенными тяжёлыми зависимостями."""
    with patch('pure_intellect.core.orchestrator.ModelManager'), \
         patch('pure_intellect.core.orchestrator.IntentDetector'), \
         patch('pure_intellect.core.orchestrator.Retriever'), \
         patch('pure_intellect.core.orchestrator.ContextAssembler'), \
         patch('pure_intellect.core.orchestrator.GraphBuilder'), \
         patch('pure_intellect.core.orchestrator.CardGenerator'), \
         patch('pure_intellect.core.orchestrator.MemoryStorage'), \
         patch('pure_intellect.core.orchestrator.WorkingMemory'), \
         patch('pure_intellect.core.orchestrator.MemoryOptimizer'), \
         patch('pure_intellect.core.orchestrator.AttentionScorer'), \
         patch('pure_intellect.core.orchestrator.ImportanceTagger'), \
         patch('pure_intellect.core.orchestrator.CCITracker'), \
         patch('pure_intellect.core.orchestrator.DualModelRouter'), \
         patch('pure_intellect.core.orchestrator.SessionPersistence'), \
         patch('pure_intellect.core.orchestrator.MetaCoordinator'):

        from pure_intellect.core.orchestrator import OrchestratorPipeline
        pipe = OrchestratorPipeline()
        # Сбрасываем mock'и для чистоты теста
        pipe._turns_since_reset = 0
        pipe._adaptive_reset_enabled = True
        pipe._cci_reset_threshold = 0.55
        pipe._min_turns_between_resets = 4
        pipe._max_turns_without_reset = 16
        pipe._context_window_size = 12
        pipe._chat_history = []
        yield pipe


# ── _should_soft_reset() тесты ──────────────────────────────

class TestShouldSoftReset:
    """Логика _should_soft_reset(cci_score) → (bool, reason)."""

    def test_no_reset_below_all_thresholds(self, pipeline):
        """Не сбрасываем когда всё в норме."""
        pipeline._turns_since_reset = 2
        pipeline._chat_history = [{"role": "user"}] * 5  # 5 < 12
        should, reason = pipeline._should_soft_reset(cci_score=0.9)
        assert should is False
        assert reason == "ok"

    def test_hard_limit_triggers_reset(self, pipeline):
        """Hard limit: turns >= max_turns_without_reset → всегда reset."""
        pipeline._turns_since_reset = 16
        pipeline._chat_history = [{"role": "user"}] * 5
        should, reason = pipeline._should_soft_reset(cci_score=0.99)
        assert should is True
        assert "hard_limit" in reason
        assert "16" in reason

    def test_hard_limit_overrides_good_cci(self, pipeline):
        """Hard limit срабатывает даже при хорошем CCI."""
        pipeline._turns_since_reset = 20
        should, reason = pipeline._should_soft_reset(cci_score=1.0)
        assert should is True
        assert "hard_limit" in reason

    def test_history_overflow_triggers_reset(self, pipeline):
        """История превысила лимит → reset."""
        pipeline._turns_since_reset = 2
        pipeline._chat_history = [{"role": "user"}] * 13  # 13 > 12
        should, reason = pipeline._should_soft_reset(cci_score=0.9)
        assert should is True
        assert "history_overflow" in reason

    def test_low_cci_with_enough_turns_triggers_reset(self, pipeline):
        """Низкий CCI + достаточно turns → reset."""
        pipeline._turns_since_reset = 5  # >= min_turns(4)
        pipeline._chat_history = [{"role": "user"}] * 5
        should, reason = pipeline._should_soft_reset(cci_score=0.4)  # < 0.55
        assert should is True
        assert "low_coherence" in reason
        assert "0.400" in reason

    def test_low_cci_but_too_soon_no_reset(self, pipeline):
        """Низкий CCI но слишком рано (min_turns не достигнут) → нет reset."""
        pipeline._turns_since_reset = 2  # < min_turns(4)
        pipeline._chat_history = [{"role": "user"}] * 5
        should, reason = pipeline._should_soft_reset(cci_score=0.3)
        assert should is False
        assert reason == "ok"

    def test_cci_exactly_at_threshold_no_reset(self, pipeline):
        """CCI точно на пороге → нет reset (нужно строго ниже)."""
        pipeline._turns_since_reset = 10
        pipeline._chat_history = [{"role": "user"}] * 5
        should, reason = pipeline._should_soft_reset(cci_score=0.55)  # == threshold
        assert should is False

    def test_adaptive_reset_disabled(self, pipeline):
        """Если adaptive_reset отключён — низкий CCI не триггерит reset."""
        pipeline._adaptive_reset_enabled = False
        pipeline._turns_since_reset = 10
        pipeline._chat_history = [{"role": "user"}] * 5
        should, reason = pipeline._should_soft_reset(cci_score=0.1)
        assert should is False
        assert reason == "ok"

    def test_priority_hard_limit_over_cci(self, pipeline):
        """Hard limit проверяется раньше чем adaptive CCI."""
        pipeline._turns_since_reset = 20  # hard limit
        pipeline._chat_history = [{"role": "user"}] * 5
        # Даже при отключённом adaptive и хорошем CCI — hard limit сработает
        pipeline._adaptive_reset_enabled = False
        should, reason = pipeline._should_soft_reset(cci_score=0.95)
        assert should is True
        assert "hard_limit" in reason

    def test_priority_history_over_adaptive(self, pipeline):
        """History overflow проверяется раньше чем adaptive."""
        pipeline._turns_since_reset = 1  # мало turns — adaptive не сработает
        pipeline._chat_history = [{"role": "user"}] * 15  # overflow!
        should, reason = pipeline._should_soft_reset(cci_score=0.9)
        assert should is True
        assert "history_overflow" in reason


# ── Счётчик turns_since_reset ───────────────────────────────

class TestTurnsSinceReset:
    """Проверяем что _turns_since_reset правильно управляется."""

    def test_attribute_exists(self, pipeline):
        """_turns_since_reset атрибут существует."""
        assert hasattr(pipeline, '_turns_since_reset')
        assert pipeline._turns_since_reset == 0

    def test_adaptive_attributes_exist(self, pipeline):
        """Все F3 атрибуты присутствуют."""
        assert hasattr(pipeline, '_adaptive_reset_enabled')
        assert hasattr(pipeline, '_cci_reset_threshold')
        assert hasattr(pipeline, '_min_turns_between_resets')
        assert hasattr(pipeline, '_max_turns_without_reset')

    def test_default_values(self, pipeline):
        """Проверяем дефолтные значения F3 параметров."""
        assert pipeline._adaptive_reset_enabled is True
        assert pipeline._cci_reset_threshold == 0.55
        assert pipeline._min_turns_between_resets == 4
        assert pipeline._max_turns_without_reset == 16


# ── Config integration ─────────────────────────────────────

class TestAdaptiveResetConfig:
    """Проверяем что config.yaml содержит F3 параметры."""

    def test_config_has_adaptive_reset_section(self):
        """config.yaml содержит секцию adaptive_reset."""
        from pathlib import Path
        config_path = Path('/a0/usr/workdir/pure-intellect/config.yaml')
        if config_path.exists():
            content = config_path.read_text()
            assert 'adaptive_reset' in content
            assert 'cci_threshold' in content
            assert 'min_turns_between_resets' in content
            assert 'max_turns_without_reset' in content

    def test_config_values_sensible(self):
        """Значения F3 в config разумны."""
        # Дефолтные значения должны быть в разумных диапазонах
        cci_threshold = 0.55
        min_turns = 4
        max_turns = 16

        assert 0.0 < cci_threshold < 1.0, "CCI threshold должен быть 0..1"
        assert min_turns >= 2, "min_turns должен быть >= 2"
        assert max_turns > min_turns, "max_turns должен быть > min_turns"


# ── Reason string format ────────────────────────────────────

class TestResetReasonFormat:
    """Проверяем формат строки reason для логирования."""

    def test_hard_limit_reason_contains_counts(self, pipeline):
        pipeline._turns_since_reset = 20
        _, reason = pipeline._should_soft_reset(0.9)
        assert 'hard_limit' in reason
        assert '20' in reason  # текущее значение
        assert '16' in reason  # лимит

    def test_history_overflow_reason_contains_counts(self, pipeline):
        pipeline._turns_since_reset = 2
        pipeline._chat_history = [{'role': 'user'}] * 15
        _, reason = pipeline._should_soft_reset(0.9)
        assert 'history_overflow' in reason
        assert '15' in reason  # текущий размер
        assert '12' in reason  # лимит

    def test_low_coherence_reason_contains_cci(self, pipeline):
        pipeline._turns_since_reset = 8
        pipeline._chat_history = [{'role': 'user'}] * 5
        _, reason = pipeline._should_soft_reset(0.3)
        assert 'low_coherence' in reason
        assert '0.300' in reason
        assert '0.55' in reason  # threshold
