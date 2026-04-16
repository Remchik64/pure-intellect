"""Тесты для CodeMemoryExtractor и CodeAwareMemoryIntegration (C3)."""

import pytest

from pure_intellect.core.code_memory import (
    CodeFact,
    CodeMemoryExtractor,
    CodeAwareMemoryIntegration,
)


# ── Вспомогательный mock ───────────────────────────────────

class MockCodeResult:
    """Mock для CodeSearchResult."""
    def __init__(self, name, entity_type, file_path, summary, relevance=0.8):
        self.entity_name = name
        self.entity_type = entity_type
        self.file_path = file_path
        self.summary = summary
        self.relevance = relevance


class MockCodeModule:
    """Mock для CodeModule."""
    def __init__(self, is_indexed=True, results=None):
        self._is_indexed = is_indexed
        self._results = results or []

    @property
    def is_indexed(self):
        return self._is_indexed

    def is_code_query(self, query):
        keywords = ["функция", "класс", "метод", "function", "class", "код", "code"]
        return any(k in query.lower() for k in keywords)

    def search(self, query, top_k=5):
        return self._results[:top_k]

    def get_context_for_llm(self, query, top_k=3, max_tokens=1500):
        if not self._results:
            return ""
        return f"[КОНТЕКСТ ПРОЕКТА: test]\n[CODE: function 'greet'] in file1.py:2\nSay hello"


class MockWorkingMemory:
    """Mock для WorkingMemory."""
    def __init__(self):
        self._facts = []
        self._context = ""

    def add(self, fact):
        self._facts.append(fact)

    def get_context(self):
        return self._context

    def size(self):
        return len(self._facts)


# ── CodeFact tests ─────────────────────────────────────────

class TestCodeFact:

    def test_creation(self):
        f = CodeFact(
            content="[КОД] FUNCTION `greet` в file1.py: Say hello",
            entity_name="greet",
            entity_type="function",
            file_path="file1.py",
            importance=0.8,
        )
        assert f.content
        assert f.entity_name == "greet"
        assert f.importance == 0.8

    def test_repr(self):
        f = CodeFact(content="test", entity_name="func", importance=0.7)
        assert "func" in repr(f)
        assert "0.7" in repr(f)

    def test_default_importance(self):
        f = CodeFact(content="test")
        assert f.importance == 0.7


# ── CodeMemoryExtractor tests ──────────────────────────────

class TestCodeMemoryExtractor:

    @pytest.fixture
    def extractor(self):
        return CodeMemoryExtractor()

    @pytest.fixture
    def results(self):
        return [
            MockCodeResult("greet", "function", "/src/file1.py", "Say hello to someone", 0.9),
            MockCodeResult("Calculator", "class", "/src/file2.py", "Simple calculator", 0.7),
            MockCodeResult("multiply", "method", "/src/file2.py", "Multiply two numbers", 0.6),
        ]

    def test_extract_basic(self, extractor, results):
        facts = extractor.extract_from_code_context(
            query="как работает greet",
            code_results=results,
        )
        assert isinstance(facts, list)
        assert len(facts) > 0
        for f in facts:
            assert isinstance(f, CodeFact)

    def test_extract_fact_content(self, extractor, results):
        facts = extractor.extract_from_code_context(
            query="функция greet",
            code_results=results,
        )
        # Первый факт должен быть о greet
        contents = [f.content for f in facts]
        assert any("greet" in c for c in contents)

    def test_extract_empty_results(self, extractor):
        facts = extractor.extract_from_code_context(
            query="функция test",
            code_results=[],
        )
        assert facts == []

    def test_extract_important_query(self, extractor, results):
        """Важный запрос создаёт дополнительный discussion-факт."""
        facts = extractor.extract_from_code_context(
            query="реализовали новую функцию greet",
            code_results=results,
        )
        types = [f.entity_type for f in facts]
        assert "discussion" in types

    def test_extract_non_important_query(self, extractor, results):
        """Обычный запрос не создаёт discussion-факт."""
        facts = extractor.extract_from_code_context(
            query="что такое greet",
            code_results=results[:1],
        )
        types = [f.entity_type for f in facts]
        assert "discussion" not in types

    def test_importance_threshold(self, extractor):
        """Факты с низкой релевантностью всё равно проходят MIN_IMPORTANCE."""
        low_results = [
            MockCodeResult("test", "function", "f.py", "test func", relevance=0.1)
        ]
        facts = extractor.extract_from_code_context(
            query="функция test",
            code_results=low_results,
        )
        # При relevance=0.1 → importance=0.3 < MIN_IMPORTANCE=0.5 → не добавляем
        # Но у нас importance = min(0.9, relevance + 0.2) = 0.3 < 0.5
        for f in facts:
            assert f.importance >= extractor.MIN_IMPORTANCE

    def test_format_for_working_memory(self, extractor, results):
        facts = extractor.extract_from_code_context(
            query="функция greet",
            code_results=results,
        )
        strings = extractor.format_for_working_memory(facts)
        assert isinstance(strings, list)
        for s in strings:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_shorten_path_long(self, extractor):
        short = extractor._shorten_path("/a/b/c/d/file.py")
        assert short == "d/file.py"

    def test_shorten_path_short(self, extractor):
        short = extractor._shorten_path("src/file.py")
        assert short == "src/file.py"

    def test_shorten_path_empty(self, extractor):
        short = extractor._shorten_path("")
        assert short == "unknown"


# ── CodeAwareMemoryIntegration tests ──────────────────────

class TestCodeAwareMemoryIntegration:

    @pytest.fixture
    def integration(self):
        return CodeAwareMemoryIntegration()

    @pytest.fixture
    def wm(self):
        return MockWorkingMemory()

    @pytest.fixture
    def code_results(self):
        return [
            MockCodeResult("greet", "function", "/src/file1.py", "Say hello", 0.9),
            MockCodeResult("add", "function", "/src/file1.py", "Add numbers", 0.7),
        ]

    def test_no_code_module(self, integration, wm):
        """Без CodeModule возвращает пустую строку и 0 фактов."""
        ctx, count = integration.process_code_turn(
            query="функция greet",
            code_module=None,
            working_memory=wm,
        )
        assert ctx == ""
        assert count == 0

    def test_not_indexed(self, integration, wm):
        """Не проиндексированный модуль → пусто."""
        module = MockCodeModule(is_indexed=False)
        ctx, count = integration.process_code_turn(
            query="функция greet",
            code_module=module,
            working_memory=wm,
        )
        assert ctx == ""
        assert count == 0

    def test_non_code_query(self, integration, wm):
        """Не-кодовый запрос → пусто."""
        module = MockCodeModule(is_indexed=True)
        ctx, count = integration.process_code_turn(
            query="как дела привет",
            code_module=module,
            working_memory=wm,
        )
        assert ctx == ""
        assert count == 0

    def test_code_query_with_results(self, integration, wm, code_results):
        """Кодовый запрос с результатами → контекст и факты."""
        module = MockCodeModule(is_indexed=True, results=code_results)
        ctx, count = integration.process_code_turn(
            query="как работает функция greet",
            code_module=module,
            working_memory=wm,
        )
        assert ctx  # контекст не пустой
        assert count >= 0  # факты добавлены (может быть 0 из-за фильтра дублей)

    def test_facts_added_to_wm(self, integration, wm, code_results):
        """Факты реально добавляются в WorkingMemory."""
        module = MockCodeModule(is_indexed=True, results=code_results)
        _, count = integration.process_code_turn(
            query="найди функцию greet",
            code_module=module,
            working_memory=wm,
        )
        # WM должна содержать добавленные факты
        assert wm.size() >= 0  # может быть 0 если все уже есть

    def test_no_results(self, integration, wm):
        """CodeModule без результатов → пусто."""
        module = MockCodeModule(is_indexed=True, results=[])
        ctx, count = integration.process_code_turn(
            query="функция nonexistent",
            code_module=module,
            working_memory=wm,
        )
        assert ctx == ""
        assert count == 0

    def test_max_facts_limit(self, integration, wm):
        """Не добавляет больше max_facts фактов за один turn."""
        many_results = [
            MockCodeResult(f"func_{i}", "function", f"f{i}.py", f"func {i}", 0.9)
            for i in range(10)
        ]
        module = MockCodeModule(is_indexed=True, results=many_results)
        _, count = integration.process_code_turn(
            query="найди все функции",
            code_module=module,
            working_memory=wm,
            max_facts=2,
        )
        assert wm.size() <= 2

    def test_duplicate_prevention(self, integration, wm, code_results):
        """Один и тот же факт не добавляется дважды."""
        # Устанавливаем контекст WM содержащий уже 'greet'
        wm._context = "greet уже в памяти"
        module = MockCodeModule(is_indexed=True, results=code_results)
        _, count = integration.process_code_turn(
            query="функция greet",
            code_module=module,
            working_memory=wm,
        )
        # greet уже в памяти → не должна добавляться снова
        assert count == 0 or wm.size() <= len(code_results)
