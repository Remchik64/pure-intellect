"""Shared state, globals and helpers for API modules."""

import collections
import datetime
import threading
import logging
from ..engine import ModelManager

logger = logging.getLogger(__name__)

# Singleton pipeline — сохраняет память между запросами
_pipeline = None
_pipeline_lock = threading.Lock()

# ── Download progress tracking ────────────────────────────────────────────────
# model_name → {"status": str, "percent": int, "speed": str, "error": str|None}
download_progress: dict[str, dict] = {}

# ── In-memory log buffer ──────────────────────────────────────────────────────
LOG_BUFFER: collections.deque = collections.deque(maxlen=2000)
LOG_LOCK = threading.Lock()

class PIMemoryHandler(logging.Handler):
    """Перехватывает все logging записи в LOG_BUFFER."""
    def emit(self, record: logging.LogRecord):
        try:
            ts = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
            line = f"[{ts}] {record.levelname:<8} {record.name}: {record.getMessage()}"
            if record.exc_info:
                import traceback as _tb
                line += "\n" + "".join(_tb.format_exception(*record.exc_info))
            with LOG_LOCK:
                LOG_BUFFER.append({"ts": ts, "level": record.levelname, "name": record.name, "line": line})
        except Exception:
            pass

# Прикрепляем к root logger чтобы перехватывать ВСЕ логи
_mem_handler = PIMemoryHandler()
_mem_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_mem_handler)


def get_model_manager() -> ModelManager:
    """Получить thread-safe singleton ModelManager."""
    return ModelManager.get_instance(cache_dir="./models")


def get_pipeline():
    """Получить thread-safe singleton OrchestratorPipeline."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from ..core import OrchestratorPipeline
                _pipeline = OrchestratorPipeline(model_manager=get_model_manager())
    return _pipeline
