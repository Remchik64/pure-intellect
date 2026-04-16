"""Тесты для MetaCoordinator (R1 roadmap — управление ростом координат)."""

import json
import tempfile
from pathlib import Path

import pytest

from pure_intellect.core.memory.meta_coordinator import CoordinateRecord, MetaCoordinator


@pytest.fixture
def tmp_session_dir(tmp_path):
    """Временная директория для сессии."""
    session_dir = tmp_path / "test_session"
    session_dir.mkdir()
    return session_dir


@pytest.fixture
def coordinator(tmp_session_dir):
    """MetaCoordinator с meta_every=4."""
    return MetaCoordinator(session_dir=tmp_session_dir, meta_every=4)


# ── CoordinateRecord tests ────────────────────────────────

class TestCoordinateRecord:
    """Тесты CoordinateRecord dataclass."""

    def test_to_dict(self):
        rec = CoordinateRecord(content="Test", turn=5)
        d = rec.to_dict()
        assert d["content"] == "Test"
        assert d["turn"] == 5
        assert d["is_meta"] is False
        assert "created_at" in d

    def test_from_dict(self):
        data = {"content": "Hello", "turn": 10, "created_at": "2026-01-01", "is_meta": True}
        rec = CoordinateRecord.from_dict(data)
        assert rec.content == "Hello"
        assert rec.turn == 10
        assert rec.is_meta is True

    def test_roundtrip(self):
        original = CoordinateRecord(content="Test content", turn=7, is_meta=False)
        restored = CoordinateRecord.from_dict(original.to_dict())
        assert restored.content == original.content
        assert restored.turn == original.turn
        assert restored.is_meta == original.is_meta


# ── MetaCoordinator basic tests ──────────────────────────

class TestMetaCoordinatorBasic:
    """Базовые тесты MetaCoordinator."""

    def test_init_empty(self, coordinator):
        assert coordinator.needs_meta() is False
        assert coordinator.get_context_for_prompt() == ""
        stats = coordinator.stats()
        assert stats["active_coordinates"] == 0
        assert stats["has_meta"] is False

    def test_add_coordinate(self, coordinator):
        coordinator.add_coordinate(content="Координата 1", turn=5)
        assert coordinator.stats()["active_coordinates"] == 1
        assert coordinator.needs_meta() is False

    def test_needs_meta_threshold(self, coordinator):
        """needs_meta() True когда активных >= meta_every."""
        for i in range(3):
            coordinator.add_coordinate(f"Координата {i}", turn=i * 10)
            assert coordinator.needs_meta() is False

        coordinator.add_coordinate("Координата 4", turn=40)
        assert coordinator.needs_meta() is True

    def test_meta_every_config(self, tmp_session_dir):
        """meta_every=2 означает мета после 2 координат."""
        c = MetaCoordinator(session_dir=tmp_session_dir, meta_every=2)
        c.add_coordinate("First", turn=1)
        assert c.needs_meta() is False
        c.add_coordinate("Second", turn=2)
        assert c.needs_meta() is True


# ── Context for prompt tests ──────────────────────────────

class TestGetContextForPrompt:
    """Тесты get_context_for_prompt()."""

    def test_empty_returns_empty_string(self, coordinator):
        assert coordinator.get_context_for_prompt() == ""

    def test_single_coordinate_in_context(self, coordinator):
        coordinator.add_coordinate("УЧАСТНИК: Александр | ПРОЕКТ: pure-intellect", turn=5)
        context = coordinator.get_context_for_prompt()
        assert "Александр" in context
        assert "pure-intellect" in context
        assert "turn 5" in context.lower() or "5" in context

    def test_only_last_active_in_context(self, coordinator):
        """В prompt попадает только последняя координата (не все активные)."""
        coordinator.add_coordinate("Первая координата", turn=5)
        coordinator.add_coordinate("Вторая координата", turn=10)
        coordinator.add_coordinate("Третья координата", turn=15)
        context = coordinator.get_context_for_prompt()
        # Только последняя в prompt
        assert "Третья координата" in context
        # Предыдущие НЕ в prompt (стабильный размер)
        assert "Первая координата" not in context

    def test_meta_plus_last_in_context(self, coordinator):
        """После consolidate: мета + последняя координата."""
        for i in range(4):
            coordinator.add_coordinate(f"Координата {i+1}", turn=(i+1)*10)

        # Consolidate
        coordinator.consolidate(meta_content="МЕТА: всё важное", turn=50)
        # Добавляем новую координату после мета
        coordinator.add_coordinate("Координата после мета", turn=60)

        context = coordinator.get_context_for_prompt()
        assert "МЕТА: всё важное" in context
        assert "Координата после мета" in context
        # Старые координаты НЕ в prompt
        assert "Координата 1" not in context

    def test_stable_context_size(self, coordinator):
        """Размер контекста стабилен независимо от числа координат."""
        coordinator.add_coordinate("Первая", turn=5)
        size_1 = len(coordinator.get_context_for_prompt())

        coordinator.consolidate("Мета 1", turn=40)
        for i in range(4):
            coordinator.add_coordinate(f"После мета {i}", turn=50+i)
        coordinator.consolidate("Мета 2", turn=90)
        coordinator.add_coordinate("Финальная", turn=100)

        size_100 = len(coordinator.get_context_for_prompt())
        # Размер контекста не должен расти многократно
        assert size_100 < size_1 * 5  # максимум в 5 раз больше одной координаты


# ── Consolidation tests ───────────────────────────────────

class TestConsolidate:
    """Тесты consolidate()."""

    def test_consolidate_clears_active(self, coordinator):
        for i in range(4):
            coordinator.add_coordinate(f"Координата {i}", turn=i*10)

        coordinator.consolidate("Мета-координата", turn=50)
        assert coordinator.stats()["active_coordinates"] == 0
        assert coordinator.stats()["has_meta"] is True

    def test_consolidate_archives_to_disk(self, coordinator, tmp_session_dir):
        """После consolidate старые координаты сохраняются в archive_dir."""
        for i in range(4):
            coordinator.add_coordinate(f"Координата {i}", turn=i*10)

        coordinator.consolidate("Мета", turn=50)

        archive_dir = tmp_session_dir / "coordinate_archive"
        archive_files = list(archive_dir.glob("*.json"))
        assert len(archive_files) >= 1

        # Проверяем содержимое архива
        with open(archive_files[0], 'r') as f:
            archived = json.load(f)
        assert len(archived) >= 4

    def test_consolidate_preserves_meta_as_new_meta(self, coordinator):
        """Meta после consolidate → новая мета создаётся."""
        for i in range(4):
            coordinator.add_coordinate(f"Координата {i}", turn=i*10)
        coordinator.consolidate("Первая мета", turn=50)

        # Добавляем ещё и снова консолидируем
        for i in range(4):
            coordinator.add_coordinate(f"Вторая волна {i}", turn=60+i*10)
        coordinator.consolidate("Вторая мета", turn=100)

        # Только последняя мета остаётся активной
        context = coordinator.get_context_for_prompt()
        assert "Вторая мета" in context
        assert "Первая мета" not in context  # архивирована

    def test_consolidate_on_empty_does_nothing(self, coordinator):
        """consolidate() без активных координат — не падает."""
        coordinator.consolidate("Мета", turn=1)  # активных нет
        # Мета не создана (нечего консолидировать)
        assert coordinator.stats()["has_meta"] is False


# ── get_all_active_contents tests ─────────────────────────

class TestGetAllActiveContents:
    """Тесты get_all_active_contents()."""

    def test_returns_all_active(self, coordinator):
        coordinator.add_coordinate("Первая", turn=1)
        coordinator.add_coordinate("Вторая", turn=2)
        coordinator.add_coordinate("Третья", turn=3)

        contents = coordinator.get_all_active_contents()
        combined = " ".join(contents)
        assert "Первая" in combined
        assert "Вторая" in combined
        assert "Третья" in combined

    def test_includes_meta_if_exists(self, coordinator):
        """Если есть мета — включает её в contents для создания новой мета."""
        for i in range(4):
            coordinator.add_coordinate(f"Координата {i}", turn=i*10)
        coordinator.consolidate("Предыдущая мета", turn=50)

        coordinator.add_coordinate("Новая координата", turn=60)
        contents = coordinator.get_all_active_contents()
        combined = " ".join(contents)
        assert "Предыдущая мета" in combined
        assert "Новая координата" in combined


# ── Persistence tests ─────────────────────────────────────

class TestPersistence:
    """Тесты сохранения/загрузки состояния MetaCoordinator."""

    def test_saves_state_on_add(self, coordinator, tmp_session_dir):
        coordinator.add_coordinate("Тест координата", turn=5)
        state_file = tmp_session_dir / "meta_coordinator.json"
        assert state_file.exists()

    def test_loads_state_on_init(self, tmp_session_dir):
        """После перезапуска MetaCoordinator восстанавливает состояние."""
        # Создаём и добавляем координату
        c1 = MetaCoordinator(session_dir=tmp_session_dir, meta_every=4)
        c1.add_coordinate("Сохранённая координата", turn=7)

        # Создаём новый экземпляр — должен восстановить состояние
        c2 = MetaCoordinator(session_dir=tmp_session_dir, meta_every=4)
        assert c2.stats()["active_coordinates"] == 1
        assert "Сохранённая" in c2.get_context_for_prompt()

    def test_meta_persists_across_restarts(self, tmp_session_dir):
        """Мета-координата сохраняется между перезапусками."""
        c1 = MetaCoordinator(session_dir=tmp_session_dir, meta_every=2)
        c1.add_coordinate("Первая", turn=1)
        c1.add_coordinate("Вторая", turn=2)
        c1.consolidate("Сохранённая мета", turn=10)

        c2 = MetaCoordinator(session_dir=tmp_session_dir, meta_every=2)
        assert c2.stats()["has_meta"] is True
        context = c2.get_context_for_prompt()
        assert "Сохранённая мета" in context

    def test_reset_clears_state(self, coordinator, tmp_session_dir):
        coordinator.add_coordinate("Координата", turn=1)
        coordinator.reset()

        assert coordinator.stats()["active_coordinates"] == 0
        assert coordinator.stats()["has_meta"] is False
        # Файл удалён
        state_file = tmp_session_dir / "meta_coordinator.json"
        assert not state_file.exists()

    def test_corrupted_file_loads_empty(self, tmp_session_dir):
        """При битом файле — не падает, загружает пустое состояние."""
        state_file = tmp_session_dir / "meta_coordinator.json"
        state_file.write_text("{ invalid json {{")

        c = MetaCoordinator(session_dir=tmp_session_dir, meta_every=4)
        assert c.stats()["active_coordinates"] == 0


# ── Stats tests ───────────────────────────────────────────

class TestStats:
    """Тесты stats()."""

    def test_stats_structure(self, coordinator):
        stats = coordinator.stats()
        assert "active_coordinates" in stats
        assert "has_meta" in stats
        assert "meta_every" in stats
        assert "needs_meta_now" in stats
        assert "archived_batches" in stats
        assert "prompt_tokens_estimate" in stats

    def test_stats_after_operations(self, coordinator):
        for i in range(4):
            coordinator.add_coordinate(f"Коорд {i}", turn=i*10)
        coordinator.consolidate("Мета", turn=50)
        coordinator.add_coordinate("После мета", turn=60)

        stats = coordinator.stats()
        assert stats["active_coordinates"] == 1
        assert stats["has_meta"] is True
        assert stats["needs_meta_now"] is False
        assert stats["archived_batches"] >= 1
        assert stats["prompt_tokens_estimate"] > 0
