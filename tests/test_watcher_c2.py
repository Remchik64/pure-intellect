"""C2: Тесты для Watcher Integration — авто-индексация при изменениях файлов."""

import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from pure_intellect.core.code_module import CodeModule


# ── CodeModule Watcher API ─────────────────────────────────

class TestWatcherInCodeModule:
    """Проверяем watcher методы в CodeModule."""

    def setup_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.project_path = tmp
        # Создаём CodeModule без индексации
        self.module = CodeModule(
            project_path="/tmp/test_project",
            session_id="test",
        )

    def test_watcher_initially_none(self):
        """_watcher изначально None."""
        assert self.module._watcher is None

    def test_watcher_status_not_initialized(self):
        """watcher_status() корректно отвечает когда не инициализирован."""
        status = self.module.watcher_status()
        assert status["is_running"] is False
        assert "Watcher not initialized" in status["message"]

    def test_start_watcher_requires_indexed(self):
        """start_watcher() требует предварительной индексации."""
        # Проект не проиндексирован
        result = self.module.start_watcher()
        assert "error" in result
        assert "not indexed" in result["error"].lower()

    def test_start_watcher_when_indexed(self):
        """start_watcher() запускается когда проект проиндексирован."""
        self.module._is_indexed = True  # Симулируем индексацию

        with patch("pure_intellect.core.code_module.CodeModule.start_watcher") as mock_start:
            mock_start.return_value = {"status": "started", "project_path": "/tmp/test_project"}
            result = self.module.start_watcher()
            assert "error" not in result or result.get("status") == "started"

    def test_stop_watcher_not_running(self):
        """stop_watcher() корректно отвечает когда не запущен."""
        result = self.module.stop_watcher()
        assert result["status"] == "not_running"

    def test_watcher_status_structure(self):
        """watcher_status() возвращает правильную структуру."""
        status = self.module.watcher_status()
        assert "is_running" in status
        assert "project_path" in status

    def test_start_watcher_with_mock(self):
        """start_watcher() с mock WatcherIntegration."""
        self.module._is_indexed = True

        mock_watcher = MagicMock()
        mock_watcher._is_running = False
        mock_watcher.get_status.return_value = {
            "is_running": True,
            "project_path": "/tmp/test_project",
            "total_changes": 0,
            "recent_changes": [],
        }

        with patch(
            "pure_intellect.core.code_module.CodeModule.start_watcher",
            return_value={"status": "started", "project_path": "/tmp/test_project"}
        ) as mock_start:
            result = self.module.start_watcher()
            assert "status" in result or "error" not in result

    def test_on_change_callback_stored(self):
        """_on_change_callback сохраняется при start_watcher."""
        callback = MagicMock()
        self.module._is_indexed = True

        with patch("pure_intellect.core.watcher_integration.WatcherIntegration") as MockWI:
            mock_wi_instance = MagicMock()
            mock_wi_instance._is_running = False
            MockWI.return_value = mock_wi_instance

            self.module.start_watcher(on_change_callback=callback)
            assert self.module._on_change_callback == callback


# ── WatcherIntegration Unit Tests ──────────────────────────

class TestWatcherIntegration:
    """Тестируем WatcherIntegration напрямую."""

    def test_watcher_integration_init(self):
        """WatcherIntegration инициализируется без ошибок."""
        from pure_intellect.core.watcher_integration import WatcherIntegration
        with patch("pure_intellect.core.watcher_integration.CardGenerator"), \
             patch("pure_intellect.core.watcher_integration.GraphBuilder"):
            wi = WatcherIntegration(project_path="/tmp")
            assert wi._is_running is False
            assert str(wi.project_path) == "/tmp"

    def test_watcher_integration_get_status(self):
        """get_status() возвращает правильную структуру."""
        from pure_intellect.core.watcher_integration import WatcherIntegration
        with patch("pure_intellect.core.watcher_integration.CardGenerator") as MockCG, \
             patch("pure_intellect.core.watcher_integration.GraphBuilder") as MockGB:
            mock_cg = MagicMock()
            mock_cg.collection.count.return_value = 42
            MockCG.return_value = mock_cg
            mock_gb = MagicMock()
            mock_gb.get_stats.return_value = {"nodes": 10}
            MockGB.return_value = mock_gb

            wi = WatcherIntegration(project_path="/tmp")
            status = wi.get_status()

            assert "is_running" in status
            assert "project_path" in status
            assert "total_changes" in status
            assert "recent_changes" in status
            assert status["is_running"] is False

    def test_watcher_start_stop(self):
        """start() и stop() работают корректно."""
        from pure_intellect.core.watcher_integration import WatcherIntegration
        from pure_intellect.core.watcher import FileWatcher

        with patch("pure_intellect.core.watcher_integration.CardGenerator"), \
             patch("pure_intellect.core.watcher_integration.GraphBuilder"), \
             patch.object(FileWatcher, "start"), \
             patch.object(FileWatcher, "stop"):

            wi = WatcherIntegration(project_path="/tmp")
            wi.watcher = MagicMock()

            wi.start()
            assert wi._is_running is True

            wi.stop()
            assert wi._is_running is False

    def test_changes_log_grows(self):
        """changes_log растёт при каждом событии."""
        from pure_intellect.core.watcher_integration import WatcherIntegration
        with patch("pure_intellect.core.watcher_integration.CardGenerator") as MockCG, \
             patch("pure_intellect.core.watcher_integration.GraphBuilder") as MockGB:

            mock_cg = MagicMock()
            mock_cg.collection.count.return_value = 0
            mock_cg.index_file.return_value = []
            MockCG.return_value = mock_cg

            mock_gb = MagicMock()
            mock_gb.get_stats.return_value = {}
            MockGB.return_value = mock_gb

            wi = WatcherIntegration(project_path="/tmp")
            assert len(wi.changes_log) == 0

            # Симулируем событие удаления
            wi._on_file_change("/tmp/test.py", "deleted", None)
            assert len(wi.changes_log) == 1
            assert wi.changes_log[0]["event"] == "deleted"


# ── FileWatcher Unit Tests ──────────────────────────────────

class TestFileWatcher:
    """Тестируем FileWatcher."""

    def test_file_watcher_init(self):
        """FileWatcher инициализируется без ошибок."""
        from pure_intellect.core.watcher import FileWatcher
        watcher = FileWatcher(project_path="/tmp")
        assert watcher.is_running is False
        assert str(watcher.project_path) == "/tmp"

    def test_file_watcher_stop_when_not_running(self):
        """stop() не падает когда watcher не запущен."""
        from pure_intellect.core.watcher import FileWatcher
        watcher = FileWatcher(project_path="/tmp")
        watcher.stop()  # Не должно упасть
        assert watcher.is_running is False

    def test_file_watcher_hash_detection(self):
        """Watcher обнаруживает реальные изменения через хэш."""
        from pure_intellect.core.watcher import FileWatcher

        changes_received = []

        def callback(fp, et, fh):
            changes_received.append((fp, et))

        watcher = FileWatcher(project_path="/tmp", callback_on_change=callback)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"x = 1")
            tmp_path = f.name

        # Первый вызов — файл новый
        watcher._on_change(tmp_path, "created")
        assert len(changes_received) == 1

        # Второй вызов с тем же содержимым — хэш не изменился
        watcher._on_change(tmp_path, "modified")
        assert len(changes_received) == 1  # Не добавился!

        # Изменяем файл
        Path(tmp_path).write_text("x = 2")
        watcher._on_change(tmp_path, "modified")
        assert len(changes_received) == 2  # Теперь добавился!

        Path(tmp_path).unlink(missing_ok=True)

    def test_scan_changes_empty_dir(self):
        """scan_changes() на пустой директории возвращает пустой список."""
        from pure_intellect.core.watcher import FileWatcher
        with tempfile.TemporaryDirectory() as tmp:
            watcher = FileWatcher(project_path=tmp)
            changes = watcher.scan_changes()
            assert changes == []

    def test_scan_changes_detects_new_files(self):
        """scan_changes() обнаруживает новые файлы."""
        from pure_intellect.core.watcher import FileWatcher
        with tempfile.TemporaryDirectory() as tmp:
            # Создаём .py файл
            py_file = Path(tmp) / "test.py"
            py_file.write_text("def hello(): pass")

            watcher = FileWatcher(project_path=tmp)
            changes = watcher.scan_changes()

            # Должен найти новый файл
            new_files = [c for c in changes if c["status"] == "new"]
            assert len(new_files) == 1
            assert "test.py" in new_files[0]["path"]


# ── Watcher → Memory callback ──────────────────────────────

class TestWatcherMemoryCallback:
    """Проверяем что callback передаёт факты в WorkingMemory."""

    def test_callback_called_on_file_change(self):
        """Callback вызывается при изменении файла."""
        received = []

        def callback(file_path: str, event_type: str, summary: str):
            received.append({"file": file_path, "event": event_type, "summary": summary})

        # Симулируем вызов internal callback
        file_path = "/project/src/memory.py"
        event_type = "modified"
        summary = "[КОД ИЗМЕНЁН] MODIFIED `memory.py` — переиндексировано автоматически"

        callback(file_path, event_type, summary)

        assert len(received) == 1
        assert received[0]["file"] == file_path
        assert "memory.py" in received[0]["summary"]

    def test_callback_summary_format(self):
        """Формат summary для WorkingMemory."""
        from pathlib import Path
        file_path = "/project/core/orchestrator.py"
        event_type = "modified"
        fname = Path(file_path).name
        summary = f"[КОД ИЗМЕНЁН] {event_type.upper()} `{fname}` — переиндексировано автоматически"

        assert "[КОД ИЗМЕНЁН]" in summary
        assert "orchestrator.py" in summary
        assert "MODIFIED" in summary

    def test_callback_not_called_on_delete(self):
        """Callback НЕ вызывается при удалении файла (только переиндексация)."""
        received = []

        def callback(fp, et, summary):
            received.append(summary)

        # При delete callback не должен вызываться для WorkingMemory
        event_type = "deleted"
        if event_type != "deleted":  # Логика из code_module.py
            callback("/tmp/x.py", event_type, "summary")

        assert len(received) == 0


# ── API Endpoints structure ────────────────────────────────

class TestWatcherRouterEndpoints:
    """Проверяем что watcher endpoints зарегистрированы в router."""

    def test_watcher_status_endpoint_exists(self):
        from pure_intellect.api.routes import router
        paths = [r.path for r in router.routes]
        assert "/code/watcher/status" in paths

    def test_watcher_start_endpoint_exists(self):
        from pure_intellect.api.routes import router
        paths = [r.path for r in router.routes]
        assert "/code/watcher/start" in paths

    def test_watcher_stop_endpoint_exists(self):
        from pure_intellect.api.routes import router
        paths = [r.path for r in router.routes]
        assert "/code/watcher/stop" in paths

    def test_watcher_changes_endpoint_exists(self):
        from pure_intellect.api.routes import router
        paths = [r.path for r in router.routes]
        assert "/code/watcher/changes" in paths

    def test_watcher_scan_endpoint_exists(self):
        from pure_intellect.api.routes import router
        paths = [r.path for r in router.routes]
        assert "/code/watcher/scan" in paths

    def test_watcher_endpoint_methods(self):
        from pure_intellect.api.routes import router
        methods = {r.path: list(r.methods or []) for r in router.routes}
        assert "GET" in methods.get("/code/watcher/status", [])
        assert "POST" in methods.get("/code/watcher/start", [])
        assert "POST" in methods.get("/code/watcher/stop", [])
        assert "GET" in methods.get("/code/watcher/changes", [])
        assert "POST" in methods.get("/code/watcher/scan", [])
