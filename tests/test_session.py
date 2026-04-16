"""Тесты для SessionPersistence — P5: persistence между сессиями."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.pure_intellect.core.session import SessionPersistence
from src.pure_intellect.core.memory.working_memory import WorkingMemory
from src.pure_intellect.core.memory.storage import MemoryStorage
from src.pure_intellect.core.memory.fact import Fact


@pytest.fixture
def tmp_dir():
    """Временная директория для тестов."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def session(tmp_dir):
    """SessionPersistence с временной директорией."""
    return SessionPersistence(base_dir=tmp_dir, session_id="test")


@pytest.fixture
def working_memory():
    storage = MemoryStorage(use_semantic=False)
    return WorkingMemory(token_budget=2000, storage=storage)


@pytest.fixture
def storage():
    return MemoryStorage(use_semantic=False)


# ── Инициализация ─────────────────────────────────────────────────────────

class TestSessionInit:
    def test_session_created(self, session, tmp_dir):
        assert session.session_id == "test"
        assert session.session_dir.exists()

    def test_new_session_not_exists(self, session):
        assert not session.exists

    def test_info_new_session(self, session):
        info = session.info()
        assert info["exists"] is False
        assert info["session_id"] == "test"

    def test_repr(self, session):
        r = repr(session)
        assert "test" in r
        assert "exists=False" in r


# ── Сохранение ────────────────────────────────────────────────────────────

class TestSessionSave:
    def test_save_creates_files(self, session, working_memory, storage):
        working_memory.add_text("Меня зовут Александр", source="test")
        session.save(working_memory, storage, [], turn=1)

        assert session._wm_path.exists()
        assert session._storage_path.exists()
        assert session._history_path.exists()
        assert session._meta_path.exists()

    def test_save_marks_exists(self, session, working_memory, storage):
        session.save(working_memory, storage, [], turn=1)
        assert session.exists

    def test_save_metadata(self, session, working_memory, storage):
        chat = [{"role": "user", "content": "hello"}]
        session.save(working_memory, storage, chat, turn=5)

        info = session.info()
        assert info["exists"] is True
        assert info["turn"] == 5
        assert info["chat_history_messages"] == 1

    def test_save_working_memory_facts(self, session, working_memory, storage):
        working_memory.add_text("факт 1", source="test")
        working_memory.add_text("факт 2", source="test")
        session.save(working_memory, storage, [], turn=1)

        info = session.info()
        assert info["working_memory_facts"] == 2

    def test_save_chat_history(self, session, working_memory, storage):
        chat = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        session.save(working_memory, storage, chat, turn=1)

        import json
        data = json.loads(session._history_path.read_text())
        assert data["count"] == 2
        assert len(data["messages"]) == 2

    def test_save_anchor_fact(self, session, working_memory, storage):
        working_memory.add_anchor("Координата: проект pure-intellect", source="coordinate")
        session.save(working_memory, storage, [], turn=3)

        info = session.info()
        assert info["working_memory_facts"] == 1

    def test_save_with_extra_meta(self, session, working_memory, storage):
        session.save(working_memory, storage, [], turn=1, extra_meta={"model": "qwen2.5:3b"})
        info = session.info()
        assert info.get("model") == "qwen2.5:3b"


# ── Загрузка ──────────────────────────────────────────────────────────────

class TestSessionLoad:
    def test_load_no_session(self, session, working_memory, storage):
        result = session.load(working_memory, storage)
        assert result["loaded"] is False
        assert result["chat_history"] == []
        assert result["turn"] == 0

    def test_load_restores_turn(self, session, working_memory, storage):
        session.save(working_memory, storage, [], turn=7)

        wm2 = WorkingMemory(token_budget=2000)
        st2 = MemoryStorage(use_semantic=False)
        result = session.load(wm2, st2)

        assert result["loaded"] is True
        assert result["turn"] == 7

    def test_load_restores_working_memory(self, session, working_memory, storage):
        working_memory.add_text("Меня зовут Александр", source="test")
        working_memory.add_text("Проект pure-intellect", source="test")
        session.save(working_memory, storage, [], turn=1)

        wm2 = WorkingMemory(token_budget=2000)
        st2 = MemoryStorage(use_semantic=False)
        session.load(wm2, st2)

        assert wm2.size() == 2
        contents = [f.content for f in wm2.get_facts()]
        assert "Меня зовут Александр" in contents
        assert "Проект pure-intellect" in contents

    def test_load_restores_anchor_facts(self, session, working_memory, storage):
        working_memory.add_anchor("Координата сессии", source="coordinate")
        session.save(working_memory, storage, [], turn=1)

        wm2 = WorkingMemory(token_budget=2000)
        st2 = MemoryStorage(use_semantic=False)
        session.load(wm2, st2)

        assert wm2.size() == 1
        fact = wm2.get_facts()[0]
        assert fact.is_anchor is True

    def test_load_restores_chat_history(self, session, working_memory, storage):
        chat = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        session.save(working_memory, storage, chat, turn=1)

        wm2 = WorkingMemory(token_budget=2000)
        st2 = MemoryStorage(use_semantic=False)
        result = session.load(wm2, st2)

        assert len(result["chat_history"]) == 2
        assert result["chat_history"][0]["content"] == "hello"

    def test_load_restores_storage_facts(self, session, working_memory, storage):
        fact = Fact(content="долгосрочный факт", source="test")
        storage.store(fact)
        session.save(working_memory, storage, [], turn=1)

        wm2 = WorkingMemory(token_budget=2000)
        st2 = MemoryStorage(use_semantic=False)
        session.load(wm2, st2)

        assert st2.size() == 1

    def test_multiple_save_load_cycles(self, session, working_memory, storage):
        """Несколько циклов сохранения/загрузки — данные не дублируются."""
        working_memory.add_text("факт А", source="test")
        session.save(working_memory, storage, [], turn=1)

        wm2 = WorkingMemory(token_budget=2000)
        st2 = MemoryStorage(use_semantic=False)
        session.load(wm2, st2)
        assert wm2.size() == 1

        # Второй цикл
        wm2.add_text("факт Б", source="test")
        session.save(wm2, st2, [], turn=2)

        wm3 = WorkingMemory(token_budget=2000)
        st3 = MemoryStorage(use_semantic=False)
        result = session.load(wm3, st3)
        assert wm3.size() == 2
        assert result["turn"] == 2


# ── Удаление ──────────────────────────────────────────────────────────────

class TestSessionDelete:
    def test_delete_removes_files(self, session, working_memory, storage):
        session.save(working_memory, storage, [], turn=1)
        assert session.exists

        session.delete()
        assert not session.exists
        assert not session._meta_path.exists()
        assert not session._wm_path.exists()

    def test_delete_nonexistent_session(self, session):
        """Удаление несуществующей сессии не падает."""
        session.delete()  # Не должно бросать исключение

    def test_load_after_delete(self, session, working_memory, storage):
        working_memory.add_text("факт", source="test")
        session.save(working_memory, storage, [], turn=5)
        session.delete()

        wm2 = WorkingMemory(token_budget=2000)
        st2 = MemoryStorage(use_semantic=False)
        result = session.load(wm2, st2)
        assert result["loaded"] is False
        assert wm2.size() == 0


# ── WorkingMemory save/load ────────────────────────────────────────────────

class TestWorkingMemorySaveLoad:
    def test_save_load_roundtrip(self, tmp_dir):
        path = Path(tmp_dir) / "wm.json"
        wm = WorkingMemory(token_budget=1000)
        wm.add_text("факт 1", source="test")
        wm.add_anchor("якорный факт", source="coord")
        wm.current_turn = 10

        wm.save_state(path)
        assert path.exists()

        wm2 = WorkingMemory(token_budget=1000)
        loaded = wm2.load_state(path)

        assert loaded is True
        assert wm2.current_turn == 10
        assert wm2.size() == 2

    def test_load_nonexistent_returns_false(self, tmp_dir):
        wm = WorkingMemory(token_budget=1000)
        loaded = wm.load_state(Path(tmp_dir) / "nonexistent.json")
        assert loaded is False

    def test_anchor_preserved_after_load(self, tmp_dir):
        path = Path(tmp_dir) / "wm.json"
        wm = WorkingMemory(token_budget=1000)
        wm.add_anchor("якорь", source="coord")
        wm.save_state(path)

        wm2 = WorkingMemory(token_budget=1000)
        wm2.load_state(path)

        facts = wm2.get_facts()
        assert len(facts) == 1
        assert facts[0].is_anchor is True
        assert facts[0].content == "якорь"
