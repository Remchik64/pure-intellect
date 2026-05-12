"""File Watcher Integration — связывает watcher с CardGenerator и GraphBuilder."""

import time
from pathlib import Path
from typing import Optional, List, Callable

from ..config import settings
from ..utils.logger import get_logger
from .card_generator import CardGenerator
from .graph_builder import GraphBuilder
from .watcher import FileWatcher

logger = get_logger("watcher_integration")


class WatcherIntegration:
    """Интеграция FileWatcher с CardGenerator и GraphBuilder."""
    
    def __init__(self, project_path: str = "./src"):
        self.project_path = Path(project_path)
        self.card_generator = CardGenerator()
        self.graph_builder = GraphBuilder()
        self.watcher: Optional[FileWatcher] = None
        self.changes_log: List[dict] = []
        self._is_running = False
    
    def _on_file_change(self, file_path: str, event_type: str, file_hash: Optional[str]):
        """Callback при изменении файла."""
        logger.info(f"File changed: {event_type} -> {file_path}")
        
        change_record = {
            "file": file_path,
            "event": event_type,
            "time": time.time(),
            "hash": file_hash,
        }
        self.changes_log.append(change_record)
        
        if event_type == "deleted":
            # Удаляем из ChromaDB и Graph
            self.card_generator.collection.delete(where={"file_path": file_path})
            self.graph_builder.graph.remove_file(file_path)
            logger.info(f"Removed from index: {file_path}")
        else:
            # Переиндексируем файл
            path = Path(file_path)
            if path.exists():
                # Card Generator
                cards = self.card_generator.index_file(path)
                logger.info(f"Re-indexed {len(cards)} cards from {file_path}")
                
                # Graph Builder — перестроить только для файла
                self.graph_builder.graph.remove_file(file_path)
                self.graph_builder._process_file(path)
                self.graph_builder.graph.save()
                logger.info(f"Graph updated for {file_path}")
    
    def start(self):
        """Запустить мониторинг."""
        if self._is_running:
            logger.warning("Watcher already running")
            return
        
        self.watcher = FileWatcher(
            project_path=self.project_path,
            callback_on_change=self._on_file_change
        )
        self.watcher.start()
        self._is_running = True
        logger.info(f"Watcher integration started for {self.project_path}")
    
    def stop(self):
        """Остановить мониторинг."""
        if not self._is_running or not self.watcher:
            return
        
        self.watcher.stop()
        self._is_running = False
        logger.info("Watcher integration stopped")
    
    def scan_now(self) -> List[dict]:
        """Одноразовое сканирование изменений."""
        if not self.watcher:
            self.watcher = FileWatcher(project_path=self.project_path)
        
        changes = self.watcher.scan_changes()
        
        # Обрабатываем изменения
        for change in changes:
            self._on_file_change(change["path"], change["status"], change.get("hash"))
        
        return changes
    
    def get_status(self) -> dict:
        """Статус watcher."""
        return {
            "is_running": self._is_running,
            "project_path": str(self.project_path),
            "total_changes": len(self.changes_log),
            "recent_changes": self.changes_log[-10:] if self.changes_log else [],
            "cards_indexed": self.card_generator.collection.count(),
            "graph_stats": self.graph_builder.get_stats(),
        }
