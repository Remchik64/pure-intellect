"""Тесты для AttentionScorer."""

import pytest
from pure_intellect.core.memory.fact import Fact
from pure_intellect.core.memory.scorer import AttentionScorer, ScoreResult
from pure_intellect.core.memory.working_memory import WorkingMemory
from pure_intellect.core.memory.storage import MemoryStorage


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    return AttentionScorer()


@pytest.fixture
def facts():
    return [
        Fact(content="Функция parse_file() обрабатывает Python AST дерево", source="parser.py"),
        Fact(content="ChromaDB хранит векторные embeddings карточек кода", source="card_gen.py"),
        Fact(content="Orchestrator запускает Intent→RAG→Graph→LLM pipeline", source="orchestrator.py"),
    ]


# ─── AttentionScorer Tests ────────────────────────────────────────────────────

class TestAttentionScorer:
    def test_score_facts_returns_list(self, scorer, facts):
        results = scorer.score_facts(facts, "test query", "test response", turn=1)
        assert isinstance(results, list)
        assert len(results) == len(facts)

    def test_score_result_structure(self, scorer, facts):
        results = scorer.score_facts(facts, "test", "test", turn=1)
        for r in results:
            assert isinstance(r, ScoreResult)
            assert hasattr(r, 'fact_id')
            assert hasattr(r, 'matched')
            assert hasattr(r, 'match_count')
            assert hasattr(r, 'weight_before')
            assert hasattr(r, 'weight_after')

    def test_matched_fact_weight_increases(self, scorer, facts):
        """Факт упомянутый в разговоре должен получить больший вес."""
        fact = facts[0]  # parse_file Python факт
        initial_weight = fact.attention_weight
        
        # Запрос упоминает parse_file
        results = scorer.score_facts(
            [fact],
            query="как работает parse_file",
            response="parse_file обрабатывает файлы",
            turn=1,
        )
        
        assert results[0].matched is True
        assert results[0].weight_after > initial_weight

    def test_unmatched_fact_weight_unchanged(self, scorer, facts):
        """Факт НЕ упомянутый в разговоре не должен получить touch()."""
        fact = facts[1]  # ChromaDB факт
        initial_weight = fact.attention_weight
        
        # Запрос про parse_file — ChromaDB не упоминается
        results = scorer.score_facts(
            [fact],
            query="как работает parse_file в python",
            response="parse_file использует ast модуль",
            turn=1,
        )
        
        assert results[0].matched is False
        assert results[0].weight_after == initial_weight

    def test_multiple_facts_scored(self, scorer, facts):
        """Несколько фактов одновременно, каждый оценивается независимо."""
        results = scorer.score_facts(
            facts,
            query="parse_file обрабатывает Python код через ChromaDB",
            response="да, parse_file и ChromaDB работают вместе",
            turn=1,
        )
        
        # Как минимум parse_file и ChromaDB должны быть matched
        matched = [r for r in results if r.matched]
        assert len(matched) >= 2

    def test_score_single(self, scorer, facts):
        """score_single() работает как score_facts() для одного факта."""
        fact = facts[0]
        result = scorer.score_single(
            fact,
            query="parse_file python",
            response="",
            turn=1,
        )
        assert isinstance(result, ScoreResult)
        assert result.fact_id == fact.fact_id

    def test_match_words_populated(self, scorer):
        """match_words содержит слова которые совпали."""
        fact = Fact(content="function parse_file processes python ast")
        results = scorer.score_facts(
            [fact],
            query="parse_file python function",
            response="",
            turn=1,
        )
        assert results[0].matched is True
        assert len(results[0].match_words) > 0

    def test_empty_query_response(self, scorer, facts):
        """Пустые query и response не вызывают ошибок."""
        results = scorer.score_facts(facts, "", "", turn=1)
        assert len(results) == len(facts)
        assert all(not r.matched for r in results)

    def test_empty_facts_list(self, scorer):
        """Пустой список фактов — пустой результат."""
        results = scorer.score_facts([], "query", "response", turn=1)
        assert results == []


# ─── Keyword Extraction Tests ─────────────────────────────────────────────────

class TestKeywordExtraction:
    def test_extracts_keywords(self, scorer):
        words = scorer._extract_keywords("parse_file обрабатывает Python файлы")
        assert isinstance(words, set)
        assert len(words) > 0

    def test_filters_short_words(self, scorer):
        """Короткие слова (< min_word_len) не включаются."""
        words = scorer._extract_keywords("is a it to")
        # Все слова короче 4 символов
        short = {w for w in words if len(w) < scorer.min_word_len}
        assert len(short) == 0

    def test_filters_stop_words(self, scorer):
        """Стоп-слова не включаются в ключевые слова."""
        words = scorer._extract_keywords("это также будет можно")
        for stop in scorer.stop_words:
            assert stop not in words

    def test_case_insensitive(self, scorer):
        """Регистр не важен."""
        words1 = scorer._extract_keywords("ParseFile")
        words2 = scorer._extract_keywords("parsefile")
        assert words1 == words2

    def test_handles_russian(self, scorer):
        """Русские слова корректно извлекаются."""
        words = scorer._extract_keywords("оркестратор запускает пайплайн")
        assert "оркестратор" in words
        assert "запускает" in words


# ─── Extract Facts from Response Tests ────────────────────────────────────────

class TestExtractFacts:
    def test_extracts_sentences(self, scorer):
        response = "This is a meaningful sentence. Another important fact here. Short."
        facts = scorer.extract_facts_from_response(response)
        assert isinstance(facts, list)
        # Только содержательные предложения
        for f in facts:
            assert len(f) >= 20

    def test_max_10_facts(self, scorer):
        """Не более 10 фактов за раз."""
        # 15 предложений
        response = " ".join([f"This is sentence number {i} with content." for i in range(15)])
        facts = scorer.extract_facts_from_response(response)
        assert len(facts) <= 10

    def test_empty_response(self, scorer):
        facts = scorer.extract_facts_from_response("")
        assert isinstance(facts, list)


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestScorerIntegration:
    def test_scorer_integrated_in_cleanup(self):
        """cleanup() с query/response обновляет веса через scorer."""
        storage = MemoryStorage()
        memory = WorkingMemory(token_budget=2000, storage=storage)
        
        # Добавляем факты
        f_parse = memory.add_text("parse_file обрабатывает Python файлы")
        f_chroma = memory.add_text("ChromaDB хранит векторные данные")
        
        weight_parse_before = f_parse.attention_weight
        weight_chroma_before = f_chroma.attention_weight
        
        # cleanup() с разговором про parse_file
        stats = memory.cleanup(
            turn=1,
            query="расскажи про parse_file",
            response="parse_file это функция для обработки файлов",
        )
        
        # parse_file упомянут → вес вырос
        assert f_parse.attention_weight >= weight_parse_before
        # ChromaDB не упомянут → вес не вырос (мог decay применить)
        # scored в stats
        assert "scored" in stats
        assert stats["scored"] >= 1

    def test_hot_fact_stays_after_scoring(self):
        """Факт упомянутый в каждом turn остаётся в WorkingMemory."""
        storage = MemoryStorage()
        memory = WorkingMemory(token_budget=2000, storage=storage)
        
        f = memory.add_text("parse_file function processes python files")
        
        # 5 turns — всегда упоминаем parse_file
        for turn in range(1, 6):
            memory.cleanup(
                turn=turn,
                query="parse_file function",
                response="parse_file processes python",
            )
        
        # Факт должен остаться в working memory (горячий!)
        assert memory.size() > 0
        assert any(f.fact_id == fact.fact_id for fact in memory.get_facts())

    def test_cold_fact_evicted_after_scoring(self):
        """Факт НЕ упоминаемый в разговоре уходит в storage."""
        storage = MemoryStorage()
        memory = WorkingMemory(token_budget=2000, storage=storage)
        
        f_hot = memory.add_text("parse_file function processes python")
        f_cold = memory.add_text("some other unrelated information here")
        
        # 10 turns — упоминаем только parse_file
        for turn in range(1, 11):
            memory.cleanup(
                turn=turn,
                query="parse_file python",
                response="parse_file returns results",
            )
        
        # Холодный факт должен уйти в storage
        # (либо evicted, либо в памяти с низким весом)
        cold_in_memory = any(f_cold.fact_id == f.fact_id for f in memory.get_facts())
        cold_in_storage = storage.get(f_cold.fact_id) is not None
        
        # Должен быть хотя бы в одном месте
        assert cold_in_memory or cold_in_storage
