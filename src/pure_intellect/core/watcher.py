"""File Watcher — мониторинг изменений файлов проекта."""

import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pure_intellect.config import settings
from pure_intellect.utils.hashing import file_sha256
from pure_intellect.utils.logger import get_logger

logger = get_logger("watcher")


class ProjectFileHandler(FileSystemEventHandler):
    """Обработчик событий файловой системы."""

    def __init__(self, callback_on_change=None):
        super().__init__()
        self.callback = callback_on_change
        self._debounce: dict[str, float] = {}  # Антидребезг

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path, "modified")

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path, "created")

    def on_deleted(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path, "deleted")

    def _handle_event(self, file_path: str, event_type: str):
        path = Path(file_path)

        # Фильтрация по расширениям
        if path.suffix not in settings.supported_extensions:
            return

        # Фильтрация по игнорируемым директориям
        if any(ignored in path.parts for ignored in settings.ignore_dirs):
            return

        # Антидребезг (debounce 500ms)
        now = time.time()
        if file_path in self._debounce:
            if now - self._debounce[file_path] < 0.5:
                return
        self._debounce[file_path] = now

        logger.info(f"Watcher: {event_type} -> {file_path}")

        if self.callback:
            self.callback(file_path, event_type)


class FileWatcher:
    """Мониторинг изменений в проекте."""

    def __init__(self, project_path: Path | str, callback_on_change=None):
        self.project_path = Path(project_path)
        self.callback = callback_on_change
        self.observer = Observer()
        self._is_running = False
        # Хранилище хэшей файлов для обнаружения реальных изменений
        self._file_hashes: dict[str, str] = {}

    def start(self):
        """Запустить мониторинг."""
        if self._is_running:
            return

        handler = ProjectFileHandler(callback_on_change=self._on_change)
        self.observer.schedule(handler, str(self.project_path), recursive=True)
        self.observer.start()
        self._is_running = True
        logger.info(f"Watcher: мониторинг {self.project_path} запущен")

    def stop(self):
        """Остановить мониторинг."""
        if not self._is_running:
            return

        self.observer.stop()
        self.observer.join()
        self._is_running = False
        logger.info("Watcher: мониторинг остановлен")

    def _on_change(self, file_path: str, event_type: str):
        """Обработка изменения файла."""
        path = Path(file_path)

        if event_type == "deleted":
            # Файл удалён — помечаем как stale
            self._file_hashes.pop(file_path, None)
            if self.callback:
                self.callback(file_path, event_type, None)
            return

        # Проверяем реальное изменение через хэш
        if not path.exists():
            return

        new_hash = file_sha256(path)
        old_hash = self._file_hashes.get(file_path)

        if old_hash == new_hash:
            return  # Хэш не изменился — ложное срабатывание

        self._file_hashes[file_path] = new_hash

        if self.callback:
            self.callback(file_path, event_type, new_hash)

    def scan_changes(self) -> list[dict]:
        """Одноразовое сканирование изменений (без непрерывного мониторинга)."""
        changes = []

        for file_path in self.project_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in settings.supported_extensions:
                continue
            if any(ignored in file_path.parts for ignored in settings.ignore_dirs):
                continue

            str_path = str(file_path)
            current_hash = file_sha256(file_path)
            stored_hash = self._file_hashes.get(str_path)

            if stored_hash is None:
                changes.append({"path": str_path, "status": "new", "hash": current_hash})
            elif stored_hash != current_hash:
                changes.append({"path": str_path, "status": "modified", "hash": current_hash})

            self._file_hashes[str_path] = current_hash

        # Проверяем удалённые файлы
        existing = {str(p) for p in self.project_path.rglob("*") if p.is_file()}
        for stored_path in list(self._file_hashes.keys()):
            if stored_path not in existing:
                changes.append({"path": stored_path, "status": "deleted", "hash": None})
                del self._file_hashes[stored_path]

        return changes

    @property
    def is_running(self) -> bool:
        return self._is_running
