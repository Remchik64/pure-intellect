"""Тесты для SessionManager (F1 roadmap — Multi-session)."""

import json
from pathlib import Path

import pytest

from pure_intellect.core.session_manager import (
    SessionInfo,
    SessionManager,
    SESSION_TYPE_CHAT,
    SESSION_TYPE_PROJECT,
    DEFAULT_SESSION,
)


@pytest.fixture
def base_dir(tmp_path):
    """Временная директория для сессий."""
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture
def manager(base_dir):
    """SessionManager с временной директорией."""
    return SessionManager(base_dir=str(base_dir))


# ── SessionInfo tests ──────────────────────────────────────

class TestSessionInfo:

    def test_to_dict_chat(self):
        info = SessionInfo(session_id="test", display_name="Test Chat")
        d = info.to_dict()
        assert d["session_id"] == "test"
        assert d["display_name"] == "Test Chat"
        assert d["session_type"] == SESSION_TYPE_CHAT
        assert d["icon"] == "💬"
        assert d["project_path"] is None

    def test_to_dict_project(self):
        info = SessionInfo(
            session_id="proj",
            display_name="My Project",
            session_type=SESSION_TYPE_PROJECT,
            project_path="/path/to/project",
        )
        d = info.to_dict()
        assert d["icon"] == "📁"
        assert d["project_path"] == "/path/to/project"
        assert d["session_type"] == SESSION_TYPE_PROJECT

    def test_from_dict_roundtrip(self):
        info = SessionInfo(
            session_id="s1",
            display_name="Test",
            session_type=SESSION_TYPE_CHAT,
            turn=5,
        )
        restored = SessionInfo.from_dict(info.to_dict())
        assert restored.session_id == info.session_id
        assert restored.display_name == info.display_name
        assert restored.turn == info.turn

    def test_from_session_dir_with_meta(self, tmp_path):
        """Загрузка из папки с session_meta.json."""
        session_dir = tmp_path / "my_session"
        session_dir.mkdir()
        meta = {
            "session_id": "my_session",
            "display_name": "Мой чат",
            "session_type": SESSION_TYPE_CHAT,
            "turn": 7,
        }
        (session_dir / "session_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False)
        )
        info = SessionInfo.from_session_dir(session_dir)
        assert info is not None
        assert info.display_name == "Мой чат"
        assert info.turn == 7

    def test_from_session_dir_without_meta(self, tmp_path):
        """Загрузка из папки без session_meta.json — базовая информация."""
        session_dir = tmp_path / "bare_session"
        session_dir.mkdir()
        info = SessionInfo.from_session_dir(session_dir)
        assert info is not None
        assert info.session_id == "bare_session"


# ── SessionManager basic tests ────────────────────────────

class TestSessionManagerBasic:

    def test_init_creates_default_session(self, manager, base_dir):
        assert (base_dir / DEFAULT_SESSION).exists()
        assert manager.active_session_id == DEFAULT_SESSION

    def test_list_sessions_has_default(self, manager):
        sessions = manager.list_sessions()
        ids = [s.session_id for s in sessions]
        assert DEFAULT_SESSION in ids

    def test_session_exists_default(self, manager):
        assert manager.session_exists(DEFAULT_SESSION)

    def test_session_not_exists(self, manager):
        assert not manager.session_exists("nonexistent_session")

    def test_stats_structure(self, manager):
        stats = manager.stats()
        assert "total_sessions" in stats
        assert "active_session" in stats
        assert "sessions" in stats
        assert "chat_sessions" in stats
        assert "project_sessions" in stats


# ── Create session tests ───────────────────────────────────

class TestCreateSession:

    def test_create_chat_session(self, manager, base_dir):
        info = manager.create_session(display_name="Новый чат")
        assert info.session_type == SESSION_TYPE_CHAT
        assert info.display_name == "Новый чат"
        assert (base_dir / info.session_id).exists()
        assert (base_dir / info.session_id / "session_meta.json").exists()

    def test_create_project_session(self, manager):
        info = manager.create_session(
            display_name="Мой проект",
            session_type=SESSION_TYPE_PROJECT,
            project_path="/path/to/project",
        )
        assert info.session_type == SESSION_TYPE_PROJECT
        assert info.project_path == "/path/to/project"

    def test_create_without_name(self, manager):
        info = manager.create_session()
        assert info.session_id  # ID сгенерирован
        assert manager.session_exists(info.session_id)

    def test_create_unique_ids_for_same_name(self, manager):
        info1 = manager.create_session(display_name="Тест")
        info2 = manager.create_session(display_name="Тест")
        assert info1.session_id != info2.session_id

    def test_slug_generation(self, manager):
        """Имена с пробелами и спецсимволами преобразуются в slug."""
        info = manager.create_session(display_name="Мой чат о Python!")
        # session_id должен быть безопасным для файловой системы
        assert " " not in info.session_id
        assert "!" not in info.session_id

    def test_session_appears_in_list(self, manager):
        info = manager.create_session(display_name="Новый")
        sessions = manager.list_sessions()
        ids = [s.session_id for s in sessions]
        assert info.session_id in ids


# ── Auto-name tests ────────────────────────────────────────

class TestAutoName:

    def test_auto_name_from_message(self, manager):
        info = manager.create_session()
        session_id = info.session_id
        name = manager.auto_name_from_message(
            "Как написать FastAPI приложение", session_id
        )
        assert "Как" in name or "как" in name.lower()
        assert len(name) <= 40

    def test_auto_name_updates_display_name(self, manager):
        info = manager.create_session()
        session_id = info.session_id
        manager.auto_name_from_message("Привет мир тест", session_id)

        updated = manager.get_session_info(session_id)
        assert updated is not None
        assert updated.display_name != session_id  # изменилось

    def test_auto_name_short_message(self, manager):
        info = manager.create_session()
        name = manager.auto_name_from_message("Ок", info.session_id)
        assert name  # не пустой

    def test_auto_name_truncates_long_message(self, manager):
        info = manager.create_session()
        long_msg = "Это очень длинное сообщение которое превышает все разумные границы" * 3
        name = manager.auto_name_from_message(long_msg, info.session_id)
        assert len(name) <= 40


# ── Switch session tests ───────────────────────────────────

class TestSwitchSession:

    def test_switch_to_existing(self, manager):
        info = manager.create_session(display_name="Вторая сессия")
        result = manager.switch_to(info.session_id)
        assert result is not None
        assert manager.active_session_id == info.session_id

    def test_switch_to_nonexistent(self, manager):
        result = manager.switch_to("nonexistent")
        assert result is None
        assert manager.active_session_id == DEFAULT_SESSION

    def test_switch_persists_to_disk(self, base_dir):
        """Активная сессия сохраняется между перезапусками."""
        m1 = SessionManager(base_dir=str(base_dir))
        info = m1.create_session(display_name="Тест")
        m1.switch_to(info.session_id)

        # Новый экземпляр должен знать активную сессию
        m2 = SessionManager(base_dir=str(base_dir))
        assert m2.active_session_id == info.session_id

    def test_active_session_first_in_list(self, manager):
        info = manager.create_session(display_name="Активная")
        manager.switch_to(info.session_id)
        sessions = manager.list_sessions()
        assert sessions[0].session_id == info.session_id


# ── Rename session tests ───────────────────────────────────

class TestRenameSession:

    def test_rename_session(self, manager):
        info = manager.create_session(display_name="Старое имя")
        ok = manager.rename_session(info.session_id, "Новое имя")
        assert ok is True
        updated = manager.get_session_info(info.session_id)
        assert updated.display_name == "Новое имя"

    def test_rename_nonexistent_session(self, manager):
        ok = manager.rename_session("nonexistent", "Новое")
        assert ok is False

    def test_rename_does_not_change_session_id(self, manager):
        info = manager.create_session(display_name="Тест")
        old_id = info.session_id
        manager.rename_session(old_id, "Другое имя")
        assert manager.session_exists(old_id)  # папка не переименована


# ── Delete session tests ───────────────────────────────────

class TestDeleteSession:

    def test_delete_session(self, manager, base_dir):
        info = manager.create_session(display_name="Удаляемая")
        session_id = info.session_id
        ok = manager.delete_session(session_id)
        assert ok is True
        assert not manager.session_exists(session_id)
        assert not (base_dir / session_id).exists()

    def test_cannot_delete_default(self, manager):
        ok = manager.delete_session(DEFAULT_SESSION)
        assert ok is False
        assert manager.session_exists(DEFAULT_SESSION)

    def test_cannot_delete_active(self, manager):
        info = manager.create_session(display_name="Активная")
        manager.switch_to(info.session_id)
        ok = manager.delete_session(info.session_id)
        assert ok is False

    def test_delete_nonexistent(self, manager):
        ok = manager.delete_session("nonexistent")
        assert ok is False

    def test_deleted_not_in_list(self, manager):
        info = manager.create_session(display_name="Удаляемая")
        manager.delete_session(info.session_id)
        ids = [s.session_id for s in manager.list_sessions()]
        assert info.session_id not in ids


# ── Stats tests ────────────────────────────────────────────

class TestStats:

    def test_stats_counts(self, manager):
        manager.create_session(session_type=SESSION_TYPE_CHAT)
        manager.create_session(session_type=SESSION_TYPE_PROJECT)
        stats = manager.stats()
        assert stats["total_sessions"] >= 3  # default + 2 новых
        assert stats["project_sessions"] >= 1
        assert stats["chat_sessions"] >= 2  # default + 1

    def test_stats_active_session(self, manager):
        stats = manager.stats()
        assert stats["active_session"] == DEFAULT_SESSION
