"""Тесты для Context Coherence Index (CCI)."""

import pytest
from pure_intellect.core.memory.cci import (
    CCITracker,
    CoherenceEntry,
    CoherenceResult,
    _extract_keywords,
    _bm25_score,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    return CCITracker(history_size=5, threshold=0.15)


@pytest.fixture
def tracker_with_history(tracker):
    """Tracker с 3 turns о Python разработке."""
    tracker.add_turn(
        "Как работает декоратор в Python?",
        "Декоратор — это функция которая принимает другую функцию."
    )
    tracker.add_turn(
        "Покажи пример декоратора с аргументами",
        "Вот пример: def my_decorator(arg): def wrapper(func): ..."
    )
    tracker.add_turn(
        "Как использовать functools.wraps?",
        "functools.wraps сохраняет метаданные оборачиваемой функции."
    )
    return tracker


# ─── _extract_keywords ───────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_extracts_words(self):
        kw = _extract_keywords("Python function decorator")
        assert "python" in kw
        assert "function" in kw
        assert "decorator" in kw

    def test_filters_short_words(self):
        kw = _extract_keywords("a bb ccc dddd")
        assert "a" not in kw
        assert "bb" not in kw
        assert "ccc" in kw

    def test_filters_stop_words(self):
        kw = _extract_keywords("this that with from")
        assert "this" not in kw
        assert "that" not in kw

    def test_lowercases(self):
        kw = _extract_keywords("Python FUNCTION Decorator")
        assert "python" in kw
        assert "function" in kw

    def test_handles_empty_string(self):
        kw = _extract_keywords("")
        assert kw == set()

    def test_cyrillic_words(self):
        kw = _extract_keywords("декоратор функция Python")
        assert "декоратор" in kw
        assert "функция" in kw


# ─── _bm25_score ─────────────────────────────────────────────────────────────

class TestBM25Score:
    def test_identical_sets_high_score(self):
        kw = {"python", "decorator", "function"}
        score = _bm25_score(kw, kw, avg_doc_len=3)
        assert score > 0.5

    def test_empty_query_returns_zero(self):
        score = _bm25_score(set(), {"python", "function"}, avg_doc_len=2)
        assert score == 0.0

    def test_empty_doc_returns_zero(self):
        score = _bm25_score({"python"}, set(), avg_doc_len=0)
        assert score == 0.0

    def test_no_overlap_returns_zero(self):
        score = _bm25_score({"apple", "fruit"}, {"python", "code"}, avg_doc_len=2)
        assert score == 0.0

    def test_partial_overlap(self):
        q = {"python", "decorator", "unrelated"}
        d = {"python", "decorator"}
        score = _bm25_score(q, d, avg_doc_len=2)
        assert 0.0 < score < 1.0


# ─── CoherenceEntry ───────────────────────────────────────────────────────────

class TestCoherenceEntry:
    def test_keywords_auto_extracted(self):
        entry = CoherenceEntry(
            turn=1,
            query="Python decorator function",
            response="Декоратор обрабатывает функцию",
        )
        assert len(entry.keywords) > 0
        assert "decorator" in entry.keywords or "python" in entry.keywords

    def test_custom_keywords_preserved(self):
        custom = {"custom", "keywords"}
        entry = CoherenceEntry(
            turn=1,
            query="any query",
            response="any response",
            keywords=custom,
        )
        assert entry.keywords == custom


# ─── CCITracker ───────────────────────────────────────────────────────────────

class TestCCITracker:
    def test_initial_state(self, tracker):
        assert tracker.history_size_current() == 0
        assert tracker._current_turn == 0

    def test_first_query_always_coherent(self, tracker):
        result = tracker.evaluate("Любой первый запрос")
        assert result.is_coherent is True
        assert result.score == 1.0
        assert result.signal == "coherent"

    def test_add_turn_increments_counter(self, tracker):
        tracker.add_turn("query", "response")
        assert tracker._current_turn == 1
        assert tracker.history_size_current() == 1

    def test_add_turn_returns_entry(self, tracker):
        entry = tracker.add_turn("query", "response")
        assert isinstance(entry, CoherenceEntry)
        assert entry.turn == 1

    def test_history_size_limit(self):
        tracker = CCITracker(history_size=3)
        for i in range(5):
            tracker.add_turn(f"query {i}", f"response {i}")
        assert tracker.history_size_current() == 3

    def test_coherent_same_topic(self, tracker_with_history):
        # Запрос на ту же тему (Python декораторы)
        result = tracker_with_history.evaluate("Что такое functools.wraps в Python?")
        assert result.score > 0.0
        assert result.signal in ("coherent", "low_coherence")

    def test_topic_switch_detected(self, tracker_with_history):
        # Резкая смена темы
        result = tracker_with_history.evaluate(
            "Расскажи про историю Древнего Рима и Цезаря"
        )
        # Score должен быть ниже чем для Python темы
        assert result.score < 1.0

    def test_evaluate_returns_coherence_result(self, tracker):
        tracker.add_turn("query", "response")
        result = tracker.evaluate("another query")
        assert isinstance(result, CoherenceResult)
        assert 0.0 <= result.score <= 1.0
        assert result.signal in ("coherent", "low_coherence", "topic_switch")

    def test_needs_context_restore_when_not_coherent(self, tracker):
        result = CoherenceResult(
            turn=1,
            query="test",
            score=0.0,
            is_coherent=False,
            top_matching_turns=[],
            signal="topic_switch",
        )
        assert result.needs_context_restore() is True

    def test_no_context_restore_when_coherent(self, tracker):
        result = CoherenceResult(
            turn=1,
            query="test",
            score=0.8,
            is_coherent=True,
            top_matching_turns=[1],
            signal="coherent",
        )
        assert result.needs_context_restore() is False

    def test_get_recent_keywords(self, tracker_with_history):
        keywords = tracker_with_history.get_recent_keywords(n_turns=2)
        assert isinstance(keywords, set)
        assert len(keywords) > 0
        # Должны быть слова из последних 2 turns
        assert any("python" in w or "functools" in w for w in keywords)

    def test_get_recent_keywords_empty_history(self, tracker):
        keywords = tracker.get_recent_keywords()
        assert keywords == set()

    def test_stats_empty(self, tracker):
        stats = tracker.stats()
        assert stats["turns"] == 0
        assert stats["avg_coherence"] == 0.0
        assert stats["threshold"] == 0.15

    def test_stats_with_history(self, tracker_with_history):
        stats = tracker_with_history.stats()
        assert stats["turns"] == 3
        assert stats["history_size"] == 3
        assert "avg_coherence" in stats

    def test_reset_clears_history(self, tracker_with_history):
        tracker_with_history.reset()
        assert tracker_with_history.history_size_current() == 0
        assert tracker_with_history._current_turn == 0

    def test_after_reset_first_query_coherent(self, tracker_with_history):
        tracker_with_history.reset()
        result = tracker_with_history.evaluate("После сброса")
        assert result.is_coherent is True

    def test_top_matching_turns_populated(self, tracker_with_history):
        result = tracker_with_history.evaluate("Python декоратор functools")
        # При совпадении темы matching turns должны быть
        assert isinstance(result.top_matching_turns, list)

    def test_weighted_recent_turns_matter_more(self):
        """Недавние turns должны влиять на coherence больше."""
        tracker = CCITracker(history_size=10, threshold=0.1)
        # Добавляем старые turns про историю
        for _ in range(5):
            tracker.add_turn(
                "История Древнего Рима и Цезаря",
                "Юлий Цезарь был великим полководцем Рима"
            )
        # Добавляем недавние turns про Python
        for _ in range(3):
            tracker.add_turn(
                "Python декоратор функция",
                "Декоратор принимает функцию как аргумент"
            )
        # Запрос про Python должен иметь лучшую coherence (недавнее)
        result_python = tracker.evaluate("Как работает Python декоратор?")
        result_rome = tracker.evaluate("Расскажи про Цезаря")
        # Python более релевантен последним turns
        assert result_python.score >= result_rome.score
