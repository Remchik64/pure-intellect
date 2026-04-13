"""Archive — хранение и управление историей бесед."""

import json
import time
import gzip
from pathlib import Path
from pure_intellect.utils.logger import get_logger

logger = get_logger("archive")


class Archive:
    """Архив сессий с сжатым хранением."""

    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./storage/archive")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        # Временное хранилище активных сессий (in-memory)
        self._sessions: dict[str, dict] = {}

    def create_session(self, session_id: str | None = None) -> str:
        """Создать новую сессию."""
        sid = session_id or f"session_{int(time.time())}"
        self._sessions[sid] = {
            "id": sid,
            "created_at": time.time(),
            "pairs": [],
            "summary": "",
        }
        return sid

    def add_pair(self, session_id: str, query: str, response: str, metadata: dict | None = None):
        """Добавить пару (запрос, ответ) в сессию."""
        if session_id not in self._sessions:
            self.create_session(session_id)

        pair = {
            "timestamp": time.time(),
            "query": query,
            "response": response,
            "metadata": metadata or {},
        }
        self._sessions[session_id]["pairs"].append(pair)

    def get_pairs(self, session_id: str, limit: int = 50) -> list[dict]:
        """Получить последние N пар из сессии."""
        if session_id not in self._sessions:
            return []
        pairs = self._sessions[session_id]["pairs"]
        return pairs[-limit:] if len(pairs) > limit else pairs

    def get_conversation_summary(self, session_id: str) -> str:
        """Получить summary сессии."""
        if session_id not in self._sessions:
            return ""
        return self._sessions[session_id].get("summary", "")

    def set_summary(self, session_id: str, summary: str):
        """Установить summary для сессии."""
        if session_id not in self._sessions:
            self.create_session(session_id)
        self._sessions[session_id]["summary"] = summary

    def trim_pairs(self, session_id: str, keep_last: int = 50):
        """Удалить старые пары, оставив последние N."""
        if session_id not in self._sessions:
            return
        pairs = self._sessions[session_id]["pairs"]
        if len(pairs) > keep_last:
            self._sessions[session_id]["pairs"] = pairs[-keep_last:]

    def save_session(self, session_id: str):
        """Сохранить сессию на диск (сжатый JSON)."""
        if session_id not in self._sessions:
            return

        session_data = self._sessions[session_id]
        file_path = self.storage_path / f"{session_id}.json.gz"

        try:
            compressed = gzip.compress(
                json.dumps(session_data, ensure_ascii=False).encode("utf-8")
            )
            file_path.write_bytes(compressed)
            logger.info(f"Archive: session {session_id} saved ({len(session_data['pairs'])} pairs)")
        except Exception as e:
            logger.error(f"Archive save error: {e}")

    def load_session(self, session_id: str) -> bool:
        """Загрузить сессию с диска."""
        file_path = self.storage_path / f"{session_id}.json.gz"

        if not file_path.exists():
            return False

        try:
            compressed = file_path.read_bytes()
            data = json.loads(gzip.decompress(compressed).decode("utf-8"))
            self._sessions[session_id] = data
            logger.info(f"Archive: session {session_id} loaded ({len(data.get('pairs', []))} pairs)")
            return True
        except Exception as e:
            logger.error(f"Archive load error: {e}")
            return False

    def list_sessions(self) -> list[str]:
        """Список сохранённых сессий."""
        sessions = []
        for f in self.storage_path.glob("*.json.gz"):
            sessions.append(f.stem)
        return sessions

    def get_stats(self) -> dict:
        """Статистика архива."""
        total_pairs = sum(
            len(s.get("pairs", [])) for s in self._sessions.values()
        )
        return {
            "active_sessions": len(self._sessions),
            "saved_sessions": len(list(self.storage_path.glob("*.json.gz"))),
            "total_pairs": total_pairs,
        }
