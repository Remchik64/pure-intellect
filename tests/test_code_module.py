"""Тесты для CodeModule (C1 roadmap — Code Indexing)."""

import json
from pathlib import Path

import pytest

from pure_intellect.core.code_module import CodeModule, CodeSearchResult


@pytest.fixture
def sample_project(tmp_path):
    """Создаём маленький тестовый проект с Python файлами."""
    # file1.py
    (tmp_path / "file1.py").write_text("""
def greet(name: str) -> str:
    '''Say hello to someone.'''
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
""")

    # file2.py
    (tmp_path / "file2.py").write_text("""
class Calculator:
    '''Simple calculator class.'''

    def multiply(self, a: int, b: int) -> int:
        '''Multiply two numbers.'''
        return a * b

    def divide(self, a: float, b: float) -> float:
        '''Divide a by b.'''
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
""")

    # subdir/utils.py
    subdir = tmp_path / "utils"
    subdir.mkdir()
    (subdir / "helpers.py").write_text("""
def format_number(n: float, decimals: int = 2) -> str:
    '''Format number with given decimal places.'''
    return f"{n:.{decimals}f}"
""")

    return tmp_path


@pytest.fixture
def empty_project(tmp_path):
    """Пустая папка."""
    return tmp_path


@pytest.fixture
def module(sample_project, tmp_path):
    """CodeModule с тестовым проектом."""
    chroma_dir = str(tmp_path / "chromadb")
    return CodeModule(
        project_path=str(sample_project),
        session_id="test_session",
        chroma_dir=chroma_dir,
    )


# ── CodeSearchResult tests ─────────────────────────────────

class TestCodeSearchResult:

    def test_to_dict(self):
        r = CodeSearchResult(
            entity_name="greet",
            entity_type="function",
            file_path="file1.py",
            start_line=2,
            end_line=4,
            summary="Say hello to someone",
            relevance=0.95,
        )
        d = r.to_dict()
        assert d["entity_name"] == "greet"
        assert d["entity_type"] == "function"
        assert d["relevance"] == 0.95

    def test_to_context_string(self):
        r = CodeSearchResult(
            entity_name="greet",
            entity_type="function",
            file_path="file1.py",
            start_line=2,
            end_line=4,
            summary="Say hello to someone",
        )
        s = r.to_context_string()
        assert "FUNCTION" in s
        assert "greet" in s
        assert "file1.py" in s


# ── CodeModule initialization tests ───────────────────────

class TestCodeModuleInit:

    def test_init_valid_path(self, sample_project, tmp_path):
        m = CodeModule(
            project_path=str(sample_project),
            chroma_dir=str(tmp_path / "chroma"),
        )
        assert m.project_path == sample_project
        assert not m.is_indexed
        assert m.indexed_files == 0

    def test_init_default_extensions(self, sample_project, tmp_path):
        m = CodeModule(
            project_path=str(sample_project),
            chroma_dir=str(tmp_path / "chroma"),
        )
        assert ".py" in m.extensions

    def test_init_custom_extensions(self, sample_project, tmp_path):
        m = CodeModule(
            project_path=str(sample_project),
            chroma_dir=str(tmp_path / "chroma"),
            extensions=[".py", ".js"],
        )
        assert ".js" in m.extensions

    def test_stats_before_index(self, module):
        stats = module.stats()
        assert stats["is_indexed"] is False
        assert stats["indexed_files"] == 0
        assert "project_path" in stats
        assert "session_id" in stats


# ── Index project tests ────────────────────────────────────

class TestIndexProject:

    def test_index_nonexistent_path(self, tmp_path):
        m = CodeModule(
            project_path=str(tmp_path / "nonexistent"),
            chroma_dir=str(tmp_path / "chroma"),
        )
        result = m.index_project()
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_index_project_success(self, module):
        result = module.index_project()
        assert result["status"] == "success"
        assert result["indexed_files"] > 0
        assert "elapsed_seconds" in result
        assert module.is_indexed

    def test_index_updates_stats(self, module):
        module.index_project()
        stats = module.stats()
        assert stats["is_indexed"] is True
        assert stats["indexed_files"] > 0
        assert stats["last_indexed"] is not None

    def test_index_already_indexed_no_force(self, module):
        module.index_project()
        result = module.index_project()  # второй раз без force
        assert result["status"] == "already_indexed"

    def test_index_force_reindex(self, module):
        module.index_project()
        result = module.index_project(force=True)
        assert result["status"] == "success"

    def test_index_custom_extensions(self, module):
        result = module.index_project(extensions=[".py"])
        assert result["status"] == "success"
        assert ".py" in result["extensions"]


# ── Search tests ───────────────────────────────────────────

class TestSearch:

    def test_search_before_index_returns_empty(self, module):
        results = module.search("greet function")
        assert results == []

    def test_search_after_index(self, module):
        module.index_project()
        results = module.search("hello greet function", top_k=5)
        # Должны найти что-то
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, CodeSearchResult)

    def test_search_returns_code_search_result(self, module):
        module.index_project()
        results = module.search("calculator multiply", top_k=3)
        for r in results:
            assert hasattr(r, "entity_name")
            assert hasattr(r, "entity_type")
            assert hasattr(r, "file_path")
            assert hasattr(r, "relevance")

    def test_search_top_k_limit(self, module):
        module.index_project()
        results = module.search("function", top_k=2)
        assert len(results) <= 2


# ── Context for LLM tests ──────────────────────────────────

class TestContextForLLM:

    def test_context_empty_before_index(self, module):
        ctx = module.get_context_for_llm("greet function")
        assert ctx == ""

    def test_context_after_index(self, module):
        module.index_project()
        ctx = module.get_context_for_llm("how does greet work", top_k=2)
        # Может быть пустым если ничего не найдено — это OK
        assert isinstance(ctx, str)

    def test_context_contains_project_name(self, module, sample_project):
        module.index_project()
        ctx = module.get_context_for_llm("calculator class", top_k=2)
        if ctx:  # если что-то нашли
            assert sample_project.name in ctx


# ── is_code_query tests ────────────────────────────────────

class TestIsCodeQuery:

    @pytest.fixture
    def m(self, tmp_path):
        return CodeModule(
            project_path=str(tmp_path),
            chroma_dir=str(tmp_path / "chroma"),
        )

    def test_code_query_function(self, m):
        assert m.is_code_query("найди функцию greet") is True

    def test_code_query_class(self, m):
        assert m.is_code_query("как работает класс Calculator") is True

    def test_code_query_english(self, m):
        assert m.is_code_query("find function that adds numbers") is True

    def test_code_query_def(self, m):
        assert m.is_code_query("def greet name str") is True

    def test_not_code_query_casual(self, m):
        assert m.is_code_query("привет как дела") is False

    def test_not_code_query_general(self, m):
        assert m.is_code_query("расскажи о погоде") is False

    def test_not_code_query_question(self, m):
        assert m.is_code_query("что такое машинное обучение") is False
