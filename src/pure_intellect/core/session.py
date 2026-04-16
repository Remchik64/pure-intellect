"""SessionPersistence — сохранение и восстановление сессий Pure Intellect.

P5: Persistence между перезапусками сервера.

Сохраняет:
  - WorkingMemory (горячие факты + anchor facts)
  - MemoryStorage (long-term факты + embeddings)
  - chat_history (rolling window последних turns)
  - Session metadata (turn, timestamp, session_id)

Структура файлов:
  storage/sessions/{session_id}/
    ├── working_memory.json
    ├── storage.json
    ├── chat_history.json
    └── session_meta.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.working_memory import WorkingMemory
    from .memory.storage import MemoryStorage

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "default"


class SessionPersistence:
    """Координатор persistence для всех компонентов сессии.
    
    Использование:
        session = SessionPersistence(base_dir='storage/sessions')
        
        # Сохранить
        session.save(working_memory, storage, chat_history, turn)
        
        # Загрузить
        result = session.load(working_memory, storage)
        chat_history = result['chat_history']
        turn = result['turn']
    """
    
    def __init__(
        self,
        base_dir: str = "storage/sessions",
        session_id: str = DEFAULT_SESSION_ID,
    ):
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Пути к файлам
        self._wm_path = self.session_dir / "working_memory.json"
        self._storage_path = self.session_dir / "storage.json"
        self._history_path = self.session_dir / "chat_history.json"
        self._meta_path = self.session_dir / "session_meta.json"
    
    @property
    def exists(self) -> bool:
        """Есть ли сохранённая сессия."""
        return self._meta_path.exists()
    
    def save(
        self,
        working_memory: 'WorkingMemory',
        storage: 'MemoryStorage',
        chat_history: list[dict],
        turn: int,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """Сохранить полное состояние сессии."""
        try:
            # 1. WorkingMemory
            working_memory.save_state(self._wm_path)
            
            # 2. MemoryStorage (с явным путём)
            storage.storage_path = self._storage_path
            storage.save()
            
            # 3. chat_history
            self._save_chat_history(chat_history)
            
            # 4. Metadata
            meta = {
                "session_id": self.session_id,
                "turn": turn,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "working_memory_facts": working_memory.size(),
                "storage_facts": storage.size(),
                "chat_history_messages": len(chat_history),
            }
            if extra_meta:
                meta.update(extra_meta)
            self._meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )
            
            logger.info(
                f"Session saved: turn={turn}, "
                f"wm={working_memory.size()} facts, "
                f"storage={storage.size()} facts, "
                f"history={len(chat_history)} msgs"
            )
        except Exception as e:
            logger.error(f"Session save failed: {e}")
    
    def load(
        self,
        working_memory: 'WorkingMemory',
        storage: 'MemoryStorage',
    ) -> dict:
        """Загрузить состояние сессии.
        
        Returns:
            dict с ключами:
            - chat_history: list[dict]
            - turn: int
            - loaded: bool (была ли найдена сессия)
            - meta: dict (метаданные сессии)
        """
        result = {
            "chat_history": [],
            "turn": 0,
            "loaded": False,
            "meta": {},
        }
        
        if not self.exists:
            logger.info(f"No saved session found at {self.session_dir}")
            return result
        
        try:
            # Metadata первым — чтобы знать что загружаем
            meta = json.loads(self._meta_path.read_text())
            result["meta"] = meta
            result["turn"] = meta.get("turn", 0)
            
            # WorkingMemory
            wm_loaded = working_memory.load_state(self._wm_path)
            
            # MemoryStorage
            storage.storage_path = self._storage_path
            storage_loaded = False
            if self._storage_path.exists():
                storage._load()
                storage_loaded = True
            
            # chat_history
            result["chat_history"] = self._load_chat_history()
            
            result["loaded"] = wm_loaded or storage_loaded
            
            if result["loaded"]:
                logger.info(
                    f"Session loaded: turn={result['turn']}, "
                    f"wm={working_memory.size()} facts, "
                    f"storage={storage.size()} facts, "
                    f"history={len(result['chat_history'])} msgs "
                    f"(saved at {meta.get('saved_at', 'unknown')})"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Session load failed: {e}")
            return result
    
    def _save_chat_history(self, chat_history: list[dict]) -> None:
        """Сохранить chat_history в JSON."""
        data = {
            "version": "1.0",
            "count": len(chat_history),
            "messages": chat_history,
        }
        self._history_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2)
        )
    
    def _load_chat_history(self) -> list[dict]:
        """Загрузить chat_history из JSON."""
        if not self._history_path.exists():
            return []
        try:
            data = json.loads(self._history_path.read_text())
            return data.get("messages", [])
        except Exception as e:
            logger.error(f"chat_history load failed: {e}")
            return []
    
    def delete(self) -> None:
        """Удалить сохранённую сессию."""
        for path in [
            self._wm_path, self._storage_path,
            self._history_path, self._meta_path,
        ]:
            if path.exists():
                path.unlink()
        logger.info(f"Session deleted: {self.session_id}")
    
    def info(self) -> dict:
        """Информация о сохранённой сессии (без загрузки)."""
        if not self.exists:
            return {"exists": False, "session_id": self.session_id}
        try:
            meta = json.loads(self._meta_path.read_text())
            return {"exists": True, **meta}
        except Exception:
            return {"exists": True, "session_id": self.session_id}
    
    def __repr__(self) -> str:
        return f"SessionPersistence(id={self.session_id!r}, exists={self.exists})"
