"""SessionManager — управление несколькими именованными сессиями.

F1 дорожной карты: Multi-session + Project binding.

Возможности:
- Несколько независимых сессий (чаты и проекты)
- Автоматическое имя из первого сообщения
- Переключение между сессиями
- Привязка к папке проекта
- Переименование и удаление

UX как в ChatGPT/Claude:
  [+ Новый чат]  [📂 Открыть проект]
  💬 как_написать_fastapi
  📁 pure-intellect
  💬 мои_вопросы
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SESSION = "default"
SESSION_TYPE_CHAT = "chat"
SESSION_TYPE_PROJECT = "project"


class SessionInfo:
    """Метаданные одной сессии."""

    def __init__(
        self,
        session_id: str,
        display_name: str,
        session_type: str = SESSION_TYPE_CHAT,
        project_path: Optional[str] = None,
        indexed_files: int = 0,
        created_at: Optional[str] = None,
        turn: int = 0,
        last_active: Optional[str] = None,
    ):
        self.session_id = session_id
        self.display_name = display_name
        self.session_type = session_type
        self.project_path = project_path
        self.indexed_files = indexed_files
        self.created_at = created_at or datetime.now().isoformat()
        self.turn = turn
        self.last_active = last_active or self.created_at

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "display_name": self.display_name,
            "session_type": self.session_type,
            "project_path": self.project_path,
            "indexed_files": self.indexed_files,
            "created_at": self.created_at,
            "turn": self.turn,
            "last_active": self.last_active,
            "icon": "📁" if self.session_type == SESSION_TYPE_PROJECT else "💬",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionInfo":
        return cls(
            session_id=data["session_id"],
            display_name=data.get("display_name", data["session_id"]),
            session_type=data.get("session_type", SESSION_TYPE_CHAT),
            project_path=data.get("project_path"),
            indexed_files=data.get("indexed_files", 0),
            created_at=data.get("created_at"),
            turn=data.get("turn", 0),
            last_active=data.get("last_active"),
        )

    @classmethod
    def from_session_dir(cls, session_dir: Path) -> Optional["SessionInfo"]:
        """Загрузить SessionInfo из папки сессии."""
        meta_path = session_dir / "session_meta.json"
        session_id = session_dir.name

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return cls(
                    session_id=session_id,
                    display_name=meta.get("display_name", session_id),
                    session_type=meta.get("session_type", SESSION_TYPE_CHAT),
                    project_path=meta.get("project_path"),
                    indexed_files=meta.get("indexed_files", 0),
                    created_at=meta.get("created_at"),
                    turn=meta.get("turn", 0),
                    last_active=meta.get("timestamp") or meta.get("last_active"),
                )
            except Exception:
                pass

        # Нет meta файла — создаём базовую информацию
        return cls(
            session_id=session_id,
            display_name=session_id,
        )


class SessionManager:
    """Управляет несколькими именованными сессиями.

    Алгоритм:
    1. list_sessions() — список всех сессий из storage/sessions/
    2. create_session() — создать новую сессию
    3. auto_name() — автоматическое имя из первого сообщения
    4. switch_session() — переключить активную сессию
    5. rename_session() — переименовать
    6. delete_session() — удалить сессию и файлы
    """

    def __init__(self, base_dir: str = "storage/sessions"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._active_session_id: str = DEFAULT_SESSION
        self._active_file = self._base_dir / ".active_session"

        # Загружаем активную сессию из файла если есть
        if self._active_file.exists():
            try:
                self._active_session_id = self._active_file.read_text().strip() or DEFAULT_SESSION
            except Exception:
                self._active_session_id = DEFAULT_SESSION

        # Убеждаемся что default сессия существует
        default_dir = self._base_dir / DEFAULT_SESSION
        default_dir.mkdir(exist_ok=True)

    # ── Публичный API ────────────────────────────────────────

    @property
    def active_session_id(self) -> str:
        return self._active_session_id

    def list_sessions(self) -> list[SessionInfo]:
        """Список всех сессий отсортированных по активности (новые сначала)."""
        sessions = []
        for path in sorted(self._base_dir.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                info = SessionInfo.from_session_dir(path)
                if info:
                    sessions.append(info)

        # Сортируем: активная первая, потом по last_active
        sessions.sort(
            key=lambda s: (
                0 if s.session_id == self._active_session_id else 1,
                s.last_active or "",
            ),
            reverse=True,
        )
        # Корректируем: активная должна быть первой
        sessions.sort(key=lambda s: 0 if s.session_id == self._active_session_id else 1)
        return sessions

    def create_session(
        self,
        display_name: Optional[str] = None,
        session_type: str = SESSION_TYPE_CHAT,
        project_path: Optional[str] = None,
    ) -> SessionInfo:
        """Создать новую сессию.

        Args:
            display_name: Человекочитаемое имя (авто-генерируется из slug если не задано)
            session_type: 'chat' или 'project'
            project_path: Путь к папке проекта (для type='project')

        Returns:
            SessionInfo новой сессии
        """
        # Генерируем уникальный session_id
        timestamp = datetime.now().strftime("%m%d_%H%M")
        base_name = self._to_slug(display_name or f"new_chat_{timestamp}")
        session_id = self._unique_id(base_name)

        # Создаём директорию
        session_dir = self._base_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # Создаём метаданные
        info = SessionInfo(
            session_id=session_id,
            display_name=display_name or session_id,
            session_type=session_type,
            project_path=project_path,
            indexed_files=0,
        )

        # Сохраняем session_meta.json
        meta_path = session_dir / "session_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            f"[session_manager] Created session: {session_id} "
            f"(type={session_type}, name={display_name!r})"
        )
        return info

    def auto_name_from_message(self, message: str, session_id: str) -> str:
        """Автоматически назвать сессию из первого сообщения.

        Берёт первые 4 слова, делает slug, обновляет display_name.
        session_id (папка) остаётся прежним.

        Returns:
            Новый display_name
        """
        # Берём первые 4 значимых слова
        words = message.strip().split()
        meaningful = [w for w in words if len(w) > 2][:4]
        if not meaningful:
            meaningful = words[:4]

        display_name = " ".join(meaningful)
        if len(display_name) > 40:
            display_name = display_name[:40]

        # Обновляем display_name в session_meta.json
        meta_path = self._base_dir / session_id / "session_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["display_name"] = display_name
                meta["last_active"] = datetime.now().isoformat()
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[session_manager] auto_name failed: {e}")

        logger.info(f"[session_manager] Auto-named session {session_id!r}: {display_name!r}")
        return display_name

    def switch_to(self, session_id: str) -> Optional[SessionInfo]:
        """Переключить активную сессию.

        Returns:
            SessionInfo если сессия найдена, None если не существует
        """
        session_dir = self._base_dir / session_id
        if not session_dir.exists():
            logger.warning(f"[session_manager] Session not found: {session_id}")
            return None

        self._active_session_id = session_id
        self._active_file.write_text(session_id)

        info = SessionInfo.from_session_dir(session_dir)
        logger.info(f"[session_manager] Switched to session: {session_id}")
        return info

    def rename_session(self, session_id: str, new_display_name: str) -> bool:
        """Переименовать сессию (только display_name, папка не меняется).

        Returns:
            True если успешно
        """
        meta_path = self._base_dir / session_id / "session_meta.json"
        if not meta_path.exists():
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["display_name"] = new_display_name
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info(f"[session_manager] Renamed {session_id!r} → {new_display_name!r}")
            return True
        except Exception as e:
            logger.error(f"[session_manager] Rename failed: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """Удалить сессию и все её файлы.

        Нельзя удалить активную сессию или default.

        Returns:
            True если удалено успешно
        """
        if session_id == DEFAULT_SESSION:
            logger.warning("[session_manager] Cannot delete default session")
            return False

        if session_id == self._active_session_id:
            logger.warning(f"[session_manager] Cannot delete active session: {session_id}")
            return False

        session_dir = self._base_dir / session_id
        if not session_dir.exists():
            return False

        try:
            shutil.rmtree(session_dir)
            logger.info(f"[session_manager] Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"[session_manager] Delete failed: {e}")
            return False

    def update_meta(
        self,
        session_id: str,
        turn: Optional[int] = None,
        indexed_files: Optional[int] = None,
    ) -> None:
        """Обновить метаданные сессии."""
        meta_path = self._base_dir / session_id / "session_meta.json"
        if not meta_path.exists():
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if turn is not None:
                meta["turn"] = turn
            if indexed_files is not None:
                meta["indexed_files"] = indexed_files
            meta["last_active"] = datetime.now().isoformat()
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[session_manager] update_meta failed: {e}")

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Получить информацию о конкретной сессии."""
        session_dir = self._base_dir / session_id
        if not session_dir.exists():
            return None
        return SessionInfo.from_session_dir(session_dir)

    def session_exists(self, session_id: str) -> bool:
        """Проверить существование сессии."""
        return (self._base_dir / session_id).exists()

    def stats(self) -> dict:
        """Статистика менеджера сессий."""
        sessions = self.list_sessions()
        return {
            "total_sessions": len(sessions),
            "active_session": self._active_session_id,
            "chat_sessions": sum(1 for s in sessions if s.session_type == SESSION_TYPE_CHAT),
            "project_sessions": sum(1 for s in sessions if s.session_type == SESSION_TYPE_PROJECT),
            "sessions": [s.to_dict() for s in sessions],
        }

    # ── Внутренние методы ────────────────────────────────────

    def _to_slug(self, text: str) -> str:
        """Преобразовать текст в slug для имени папки."""
        # Убираем спецсимволы, оставляем буквы цифры и подчёркивания
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[\s-]+', '_', slug)
        slug = slug.strip('_')
        # Ограничиваем длину
        return slug[:50] if slug else "session"

    def _unique_id(self, base: str) -> str:
        """Генерировать уникальный session_id."""
        if not (self._base_dir / base).exists():
            return base
        # Добавляем суффикс
        i = 2
        while (self._base_dir / f"{base}_{i}").exists():
            i += 1
        return f"{base}_{i}"
