"""Тесты для ImportanceTagger — P3 LLM-based importance tagging."""

import pytest
from pure_intellect.core.memory.tagger import (
    ImportanceTagger,
    TaggingResult,
)


@pytest.fixture
def tagger_no_llm():
    """Tagger с отключенным LLM (только rule-based)."""
    t = ImportanceTagger()
    t._llm_available = False  # Принудительно отключаем LLM
    return t


@pytest.fixture
def tagger():
    """Tagger с автоопределением LLM."""
    return ImportanceTagger()


class TestTaggingResult:
    """Тесты для TaggingResult dataclass."""
    
    def test_empty_result(self):
        result = TaggingResult()
        assert result.anchors == []
        assert result.facts == []
        assert result.transient == []
        assert result.total == 0
    
    def test_total_count(self):
        result = TaggingResult(
            anchors=["Александр", "pure-intellect"],
            facts=["Python 3.13"],
            transient=["вопрос про attention", "рассуждение"]
        )
        assert result.total == 5
    
    def test_method_default(self):
        result = TaggingResult()
        assert result.method == "llm"
    
    def test_rule_based_method(self):
        result = TaggingResult(method="rule_based")
        assert result.method == "rule_based"


class TestImportanceTaggerInit:
    """Тесты инициализации ImportanceTagger."""
    
    def test_default_init(self):
        tagger = ImportanceTagger()
        assert tagger._model == "qwen2.5:3b"
        assert tagger._llm_available is None
        assert tagger._llm_calls == 0
        assert tagger._fallback_calls == 0
    
    def test_custom_model(self):
        tagger = ImportanceTagger(model="llama3.2:3b")
        assert tagger._model == "llama3.2:3b"
    
    def test_repr(self):
        tagger = ImportanceTagger()
        r = repr(tagger)
        assert "ImportanceTagger" in r
        assert "qwen2.5:3b" in r


class TestRuleBasedTagging:
    """Тесты rule-based fallback классификации."""
    
    def test_name_detection(self, tagger_no_llm):
        result = tagger_no_llm.tag(
            query="Меня зовут Александр",
            response="Привет, Александр!"
        )
        assert result.method == "rule_based"
        # Имя должно попасть в anchors
        anchors_lower = [a.lower() for a in result.anchors]
        assert any("александр" in a for a in anchors_lower)
    
    def test_technical_terms_in_facts(self, tagger_no_llm):
        result = tagger_no_llm.tag(
            query="Расскажи про FastAPI",
            response="FastAPI это современный фреймворк для Python."
        )
        assert result.method == "rule_based"
        # Технические термины должны попасть в facts
        assert result.anchors is not None
        assert result.facts is not None
    
    def test_empty_inputs(self, tagger_no_llm):
        result = tagger_no_llm.tag("", "")
        assert result.method == "rule_based"
        assert isinstance(result.anchors, list)
        assert isinstance(result.facts, list)
    
    def test_returns_tagging_result(self, tagger_no_llm):
        result = tagger_no_llm.tag(
            query="Тест запрос",
            response="Тест ответ"
        )
        assert isinstance(result, TaggingResult)
    
    def test_max_anchors_3(self, tagger_no_llm):
        result = tagger_no_llm._tag_rule_based(
            "Меня зовут Александр Иванов Петров",
            "Понял, Александр Иванов Петров"
        )
        assert len(result.anchors) <= 3
    
    def test_max_facts_3(self, tagger_no_llm):
        result = tagger_no_llm._tag_rule_based(
            "Расскажи про FastAPI ChromaDB SQLAlchemy Redis Kafka",
            "FastAPI ChromaDB SQLAlchemy Redis Kafka — популярные библиотеки Python"
        )
        assert len(result.facts) <= 3
    
    def test_fallback_called(self, tagger_no_llm):
        tagger_no_llm.tag("запрос", "ответ")
        assert tagger_no_llm._fallback_calls == 1
        assert tagger_no_llm._llm_calls == 0


class TestJsonParsing:
    """Тесты парсинга JSON из ответов LLM."""
    
    def test_direct_json(self, tagger):
        raw = '{"anchors": ["Александр"], "facts": ["Python"], "transient": []}'
        result = tagger._parse_json(raw)
        assert result is not None
        assert result["anchors"] == ["Александр"]
        assert result["facts"] == ["Python"]
    
    def test_json_with_preamble(self, tagger):
        raw = 'Вот JSON ответ:\n{"anchors": ["test"], "facts": [], "transient": []}'
        result = tagger._parse_json(raw)
        assert result is not None
        assert result["anchors"] == ["test"]
    
    def test_markdown_json(self, tagger):
        raw = '```json\n{"anchors": [], "facts": ["FastAPI"], "transient": []}\n```'
        result = tagger._parse_json(raw)
        assert result is not None
        assert result["facts"] == ["FastAPI"]
    
    def test_invalid_json_returns_none(self, tagger):
        result = tagger._parse_json("это не JSON")
        assert result is None
    
    def test_empty_string_returns_none(self, tagger):
        result = tagger._parse_json("")
        assert result is None
    
    def test_partial_json(self, tagger):
        raw = '{"anchors": ["test"]'  # незакрытый JSON
        result = tagger._parse_json(raw)
        assert result is None
    
    def test_nested_json(self, tagger):
        raw = '{"anchors": ["проект pure-intellect"], "facts": ["Python 3.13"], "transient": ["вопрос"]  }'
        result = tagger._parse_json(raw)
        assert result is not None
        assert len(result["anchors"]) == 1


class TestStats:
    """Тесты статистики ImportanceTagger."""
    
    def test_initial_stats(self, tagger):
        stats = tagger.stats()
        assert stats["llm_calls"] == 0
        assert stats["fallback_calls"] == 0
        assert stats["model"] == "qwen2.5:3b"
    
    def test_fallback_stats_increment(self, tagger_no_llm):
        tagger_no_llm.tag("запрос 1", "ответ 1")
        tagger_no_llm.tag("запрос 2", "ответ 2")
        stats = tagger_no_llm.stats()
        assert stats["fallback_calls"] == 2
        assert stats["llm_calls"] == 0


class TestTaggingOutput:
    """Тесты корректности вывода тегирования."""
    
    def test_anchors_are_strings(self, tagger_no_llm):
        result = tagger_no_llm.tag("Меня зовут Александр", "Понял")
        assert all(isinstance(a, str) for a in result.anchors)
    
    def test_facts_are_strings(self, tagger_no_llm):
        result = tagger_no_llm.tag(
            "Расскажи про FastAPI",
            "FastAPI это Python фреймворк"
        )
        assert all(isinstance(f, str) for f in result.facts)
    
    def test_no_empty_strings_in_anchors(self, tagger_no_llm):
        result = tagger_no_llm.tag("", "")
        assert all(len(a.strip()) > 0 for a in result.anchors)
    
    def test_tagging_result_has_method(self, tagger_no_llm):
        result = tagger_no_llm.tag("запрос", "ответ")
        assert result.method in ("llm", "rule_based")
