"""Тесты для IntentDetector — rule-based и JSON parsing."""

import pytest
from pure_intellect.core.intent import IntentDetector, IntentType, IntentResult


@pytest.fixture
def detector():
    """Создать детектор без модели (rule-based режим)."""
    return IntentDetector(model_manager=None)


class TestRuleBasedDetection:
    """Тесты rule-based определения intent."""

    def test_code_generation_ru(self, detector):
        result = detector.detect("напиши функцию для сортировки массива")
        assert result.intent == IntentType.CODE_GENERATION
        assert result.confidence > 0

    def test_debug_ru(self, detector):
        result = detector.detect("почему программа падает с ошибкой")
        assert result.intent == IntentType.DEBUG
        assert result.confidence > 0

    def test_refactor_ru(self, detector):
        result = detector.detect("рефактори этот код, он слишком сложный")
        assert result.intent == IntentType.REFACTOR
        assert result.confidence > 0

    def test_explain_ru(self, detector):
        result = detector.detect("объясни как работает этот класс")
        assert result.intent == IntentType.CODE_EXPLAIN
        assert result.confidence > 0

    def test_search_ru(self, detector):
        result = detector.detect("найди где определена функция parse")
        assert result.intent == IntentType.SEARCH
        assert result.confidence > 0

    def test_chat_fallback(self, detector):
        """Неизвестный запрос → дефолтный CHAT."""
        result = detector.detect("просто привет")
        assert result.intent == IntentType.CHAT

    def test_returns_intent_result(self, detector):
        """detect() всегда возвращает IntentResult."""
        result = detector.detect("любой запрос")
        assert isinstance(result, IntentResult)
        assert isinstance(result.intent, IntentType)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.entities, list)
        assert isinstance(result.keywords, list)


class TestJsonParsing:
    """Тесты _parse_json_response."""

    VALID_JSON = {
        "intent": "debug",
        "confidence": 0.9,
        "entities": ["main.py"],
        "keywords": ["error"],
        "reasoning": "user asks about error",
        "suggested_context": ["logs"]
    }

    def test_clean_json(self, detector):
        """Чистый JSON парсится корректно."""
        import json
        text = json.dumps(self.VALID_JSON)
        result = detector._parse_json_response(text)
        assert result is not None
        assert result["intent"] == "debug"

    def test_json_wrapped_in_text(self, detector):
        """JSON обёрнутый в текст парсится корректно."""
        import json
        text = f'Here is the analysis: {json.dumps(self.VALID_JSON)} Hope this helps!'
        result = detector._parse_json_response(text)
        assert result is not None
        assert result["intent"] == "debug"

    def test_json_in_markdown(self, detector):
        """JSON в markdown блоке парсится корректно."""
        import json
        text = f'```json\n{json.dumps(self.VALID_JSON)}\n```'
        result = detector._parse_json_response(text)
        assert result is not None
        assert result["intent"] == "debug"

    def test_empty_response(self, detector):
        """Пустой ответ → None без исключения."""
        assert detector._parse_json_response("") is None
        assert detector._parse_json_response(None) is None

    def test_invalid_json_returns_none(self, detector):
        """Невалидный JSON → None без исключения."""
        result = detector._parse_json_response("это не JSON вообще")
        assert result is None

    def test_nested_json_entities(self, detector):
        """JSON с вложенными списками парсится корректно."""
        import json
        data = dict(self.VALID_JSON)
        data["entities"] = ["file.py", "ClassA", "method_x"]
        result = detector._parse_json_response(json.dumps(data))
        assert result is not None
        assert len(result["entities"]) == 3
