"""Тесты для DualModelRouter — P6: двойная дистилляция."""

import pytest
from unittest.mock import patch, MagicMock

from src.pure_intellect.core.dual_model import (
    DualModelRouter,
    COORDINATOR_MODEL,
    GENERATOR_MODEL,
    OLLAMA_BASE_URL,
)


# ── Инициализация ─────────────────────────────────────────────────────────

class TestDualModelRouterInit:
    def test_default_models(self):
        with patch('src.pure_intellect.core.dual_model._load_models_from_config',
                   return_value=(COORDINATOR_MODEL, GENERATOR_MODEL)):
            router = DualModelRouter()
        assert router.coordinator_model == COORDINATOR_MODEL
        assert router.generator_model == GENERATOR_MODEL

    def test_custom_models(self):
        router = DualModelRouter(
            coordinator_model="llama3:3b",
            generator_model="llama3:70b",
        )
        assert router.coordinator_model == "llama3:3b"
        assert router.generator_model == "llama3:70b"

    def test_initial_stats_zero(self):
        router = DualModelRouter()
        stats = router.stats()
        assert stats["coordinator_calls"] == 0
        assert stats["generator_calls"] == 0
        assert stats["coordinator_tokens"] == 0
        assert stats["generator_tokens"] == 0
        assert stats["total_calls"] == 0

    def test_generator_available_none_initially(self):
        router = DualModelRouter()
        assert router._generator_available is None

    def test_repr_pending(self):
        with patch('src.pure_intellect.core.dual_model._load_models_from_config',
                   return_value=(COORDINATOR_MODEL, GENERATOR_MODEL)):
            router = DualModelRouter()
        r = repr(router)
        assert COORDINATOR_MODEL in r
        assert GENERATOR_MODEL in r
        assert "⏳" in r

    def test_repr_available(self):
        router = DualModelRouter()
        router._generator_available = True
        r = repr(router)
        assert "✅" in r

    def test_repr_unavailable(self):
        router = DualModelRouter()
        router._generator_available = False
        r = repr(router)
        assert "❌" in r


# ── Проверка доступности generator ────────────────────────────────────────

class TestCheckGeneratorAvailable:
    def test_cached_true(self):
        """Если _generator_available=True — не делаем HTTP запрос."""
        router = DualModelRouter()
        router._generator_available = True
        result = router._check_generator_available()
        assert result is True

    def test_cached_false(self):
        """Если _generator_available=False — не делаем HTTP запрос."""
        router = DualModelRouter()
        router._generator_available = False
        result = router._check_generator_available()
        assert result is False

    def test_generator_available_on_success(self):
        """Если Ollama отвечает с modelfile — generator доступен."""
        router = DualModelRouter()
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"modelfile": "FROM qwen2.5:7b", "details": {}}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            result = router._check_generator_available()

        assert result is True
        assert router._generator_available is True

    def test_generator_unavailable_on_http_error(self):
        """Если Ollama недоступна — generator недоступен, fallback на coordinator."""
        router = DualModelRouter()
        import urllib.error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("Connection refused")):
            result = router._check_generator_available()

        assert result is False
        assert router._generator_available is False

    def test_generator_unavailable_on_empty_response(self):
        """Если ответ не содержит ключей modelfile/details — недоступен."""
        router = DualModelRouter()
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"error": "model not found"}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            result = router._check_generator_available()

        assert result is False

    def test_refresh_clears_cache(self):
        """refresh_generator_check() сбрасывает кеш и проверяет снова."""
        router = DualModelRouter()
        router._generator_available = True

        import urllib.error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("timeout")):
            result = router.refresh_generator_check()

        assert result is False


# ── Coordinate (3B) ────────────────────────────────────────────────────────

class TestCoordinate:
    def _make_mock_response(self, content: str):
        """Создать mock HTTP response для chat completions."""
        import json
        body = json.dumps({
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_coordinate_returns_content(self):
        router = DualModelRouter()
        messages = [{"role": "user", "content": "Тестовый разговор"}]

        with patch('urllib.request.urlopen', return_value=self._make_mock_response("Координата: тест")):
            result = router.coordinate(messages)

        assert result == "Координата: тест"

    def test_coordinate_uses_coordinator_model(self):
        """coordinate() всегда использует coordinator_model."""
        router = DualModelRouter(coordinator_model="test-coordinator:3b")
        messages = [{"role": "user", "content": "test"}]
        captured_payload = []

        import json
        original_urlopen = None

        def mock_urlopen(req, timeout=None):
            captured_payload.append(json.loads(req.data))
            return self._make_mock_response("ok")

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            router.coordinate(messages)

        assert captured_payload[0]["model"] == "test-coordinator:3b"

    def test_coordinate_increments_stats(self):
        router = DualModelRouter()
        messages = [{"role": "user", "content": "test"}]

        with patch('urllib.request.urlopen', return_value=self._make_mock_response("result")):
            router.coordinate(messages)

        assert router._coordinator_calls == 1
        assert router._coordinator_tokens == 150  # 100 + 50

    def test_coordinate_returns_empty_on_error(self):
        router = DualModelRouter()
        messages = [{"role": "user", "content": "test"}]
        import urllib.error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("timeout")):
            result = router.coordinate(messages)

        assert result == ""
        assert router._coordinator_calls == 0  # Не считаем упавшие вызовы


# ── Generate (7B → 3B fallback) ────────────────────────────────────────────

class TestGenerate:
    def _make_mock_response(self, content: str):
        import json
        body = json.dumps({
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100},
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_generate_uses_generator_when_available(self):
        router = DualModelRouter(generator_model="test-gen:7b")
        router._generator_available = True
        messages = [{"role": "user", "content": "test"}]
        captured = []

        import json
        def mock_urlopen(req, timeout=None):
            captured.append(json.loads(req.data))
            return self._make_mock_response("generated answer")

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            text, pt, ct = router.generate(messages)

        assert text == "generated answer"
        assert captured[0]["model"] == "test-gen:7b"

    def test_generate_fallback_to_coordinator(self):
        """Если generator недоступен — используем coordinator как fallback."""
        router = DualModelRouter(
            coordinator_model="test-coord:3b",
            generator_model="test-gen:7b",
        )
        router._generator_available = False
        messages = [{"role": "user", "content": "test"}]
        captured = []

        import json
        def mock_urlopen(req, timeout=None):
            captured.append(json.loads(req.data))
            return self._make_mock_response("fallback answer")

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            text, pt, ct = router.generate(messages)

        assert text == "fallback answer"
        assert captured[0]["model"] == "test-coord:3b"  # fallback!

    def test_generate_returns_tokens(self):
        router = DualModelRouter()
        router._generator_available = True
        messages = [{"role": "user", "content": "test"}]

        with patch('urllib.request.urlopen', return_value=self._make_mock_response("response")):
            text, pt, ct = router.generate(messages)

        assert pt == 200
        assert ct == 100

    def test_generate_increments_stats(self):
        router = DualModelRouter()
        router._generator_available = True
        messages = [{"role": "user", "content": "test"}]

        with patch('urllib.request.urlopen', return_value=self._make_mock_response("ok")):
            router.generate(messages)

        assert router._generator_calls == 1
        assert router._generator_tokens == 300  # 200 + 100

    def test_generate_returns_empty_on_error(self):
        router = DualModelRouter()
        router._generator_available = False
        messages = [{"role": "user", "content": "test"}]
        import urllib.error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("timeout")):
            text, pt, ct = router.generate(messages)

        assert text == ""
        assert pt == 0
        assert ct == 0


# ── Stats ──────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_structure(self):
        router = DualModelRouter()
        stats = router.stats()

        assert "coordinator_model" in stats
        assert "generator_model" in stats
        assert "generator_available" in stats
        assert "coordinator_calls" in stats
        assert "generator_calls" in stats
        assert "total_calls" in stats
        assert "total_tokens" in stats

    def test_stats_totals(self):
        router = DualModelRouter()
        router._coordinator_calls = 5
        router._generator_calls = 3
        router._coordinator_tokens = 500
        router._generator_tokens = 900

        stats = router.stats()
        assert stats["total_calls"] == 8
        assert stats["total_tokens"] == 1400

    def test_stats_models(self):
        router = DualModelRouter(
            coordinator_model="coord:3b",
            generator_model="gen:7b",
        )
        stats = router.stats()
        assert stats["coordinator_model"] == "coord:3b"
        assert stats["generator_model"] == "gen:7b"
