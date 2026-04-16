"""MetaCoordinator — управление ростом координат.

R1 дорожной карты: предотвращает бесконтрольный рост anchor facts
которые накапливаются после каждого soft reset.

Проблема:
  Каждый soft reset создаёт новую координату (~300 токенов).
  После 20 reset'ов = 6000 токенов только на координаты.
  Это уничтожает экономию которую мы получили от soft reset.

Решение — Мета-координата:
  Каждые N координат → 3B создаёт одну мета-координату:
  «Краткое эссе обо всём что было» (300 токенов)
  Старые N координат → archive на диск (освобождаем RAM)
  В prompt: 1 мета-координата + 1 последняя координата

Результат:
  Независимо от длины сессии:
  - В prompt: ~600 токенов на координаты (стабильно)
  - В RAM:    2 anchor facts (стабильно)
  - На диске: полная история координат (для аудита)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CoordinateRecord:
    """Запись одной координаты."""
    content: str
    turn: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_meta: bool = False  # True если это мета-координата

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "turn": self.turn,
            "created_at": self.created_at,
            "is_meta": self.is_meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CoordinateRecord":
        return cls(
            content=data["content"],
            turn=data["turn"],
            created_at=data.get("created_at", ""),
            is_meta=data.get("is_meta", False),
        )


class MetaCoordinator:
    """Управляет жизненным циклом координат.

    Алгоритм:
    1. Каждый soft_reset создаёт новую координату → add_coordinate()
    2. Когда координат >= meta_every → needs_meta() возвращает True
    3. Orchestrator создаёт мета-координату и вызывает consolidate()
    4. consolidate() архивирует старые координаты на диск
    5. В prompt попадают только: [мета-координата] + [последняя координата]
    """

    def __init__(
        self,
        session_dir: Path | str,
        meta_every: int = 4,
    ):
        self._session_dir = Path(session_dir)
        self._meta_every = meta_every
        self._archive_dir = self._session_dir / "coordinate_archive"
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        # Активные координаты (в RAM, малое количество)
        self._active: list[CoordinateRecord] = []
        # Текущая мета-координата (если есть)
        self._meta: CoordinateRecord | None = None

        # Загружаем состояние если есть
        self._load()

    # ── Публичный API ────────────────────────────────────────

    def add_coordinate(self, content: str, turn: int) -> None:
        """Добавить новую координату после soft reset."""
        record = CoordinateRecord(content=content, turn=turn)
        self._active.append(record)
        self._save()
        logger.info(
            f"[meta_coord] Added coordinate #{len(self._active)} at turn {turn}"
        )

    def needs_meta(self) -> bool:
        """Нужна ли мета-координата?

        True если активных координат >= meta_every.
        """
        return len(self._active) >= self._meta_every

    def consolidate(self, meta_content: str, turn: int) -> None:
        """Архивировать активные координаты и установить новую мета-координату.

        Вызывается после того как orchestrator создал мета-координату.
        """
        if not self._active:
            return

        # Архивируем текущие активные координаты
        self._archive_coordinates(self._active)

        # Если была предыдущая мета — тоже архивируем
        if self._meta:
            self._archive_coordinates([self._meta])

        # Устанавливаем новую мета
        self._meta = CoordinateRecord(
            content=meta_content,
            turn=turn,
            is_meta=True,
        )

        # Очищаем активные
        self._active.clear()
        self._save()

        logger.info(
            f"[meta_coord] Consolidated {self._meta_every} coordinates "
            f"into meta at turn {turn}"
        )

    def get_context_for_prompt(self) -> str:
        """Получить текст координат для system prompt.

        Возвращает:
        - Если есть мета: мета + последняя активная координата
        - Если только активные: последние 2
        - Стабильный размер ~600 токенов независимо от длины сессии
        """
        parts = []

        if self._meta:
            parts.append(
                f"[ИСТОРИЯ СЕССИИ — сводка]\n{self._meta.content}"
            )

        if self._active:
            last = self._active[-1]
            parts.append(
                f"[ПОСЛЕДНЯЯ КООРДИНАТА — turn {last.turn}]\n{last.content}"
            )
        elif not self._meta:
            return ""  # Ещё нет координат

        return "\n\n".join(parts)

    def get_all_active_contents(self) -> list[str]:
        """Все активные координаты для создания мета."""
        contents = []
        if self._meta:
            contents.append(f"[Предыдущая мета]\n{self._meta.content}")
        for rec in self._active:
            contents.append(f"[Turn {rec.turn}]\n{rec.content}")
        return contents

    def stats(self) -> dict:
        """Статистика MetaCoordinator."""
        archive_count = len(list(self._archive_dir.glob("*.json")))
        return {
            "active_coordinates": len(self._active),
            "has_meta": self._meta is not None,
            "meta_every": self._meta_every,
            "needs_meta_now": self.needs_meta(),
            "archived_batches": archive_count,
            "prompt_tokens_estimate": len(self.get_context_for_prompt()) // 4,
        }

    def reset(self) -> None:
        """Полный сброс координат (при удалении сессии)."""
        self._active.clear()
        self._meta = None
        state_file = self._session_dir / "meta_coordinator.json"
        if state_file.exists():
            state_file.unlink()

    # ── Персистентность ──────────────────────────────────────

    def _save(self) -> None:
        """Сохранить состояние на диск."""
        state = {
            "active": [r.to_dict() for r in self._active],
            "meta": self._meta.to_dict() if self._meta else None,
            "meta_every": self._meta_every,
        }
        state_file = self._session_dir / "meta_coordinator.json"
        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[meta_coord] Save failed: {e}")

    def _load(self) -> None:
        """Загрузить состояние с диска."""
        state_file = self._session_dir / "meta_coordinator.json"
        if not state_file.exists():
            return
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._active = [
                CoordinateRecord.from_dict(r)
                for r in state.get("active", [])
            ]
            meta_data = state.get("meta")
            self._meta = CoordinateRecord.from_dict(meta_data) if meta_data else None
            logger.info(
                f"[meta_coord] Loaded: {len(self._active)} active, "
                f"meta={'yes' if self._meta else 'no'}"
            )
        except Exception as e:
            logger.warning(f"[meta_coord] Load failed: {e}")

    def _archive_coordinates(self, records: list[CoordinateRecord]) -> None:
        """Архивировать координаты на диск (cold storage)."""
        if not records:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self._archive_dir / f"batch_{timestamp}.json"
        try:
            with open(archive_file, "w", encoding="utf-8") as f:
                json.dump(
                    [r.to_dict() for r in records],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(
                f"[meta_coord] Archived {len(records)} coordinates "
                f"→ {archive_file.name}"
            )
        except Exception as e:
            logger.error(f"[meta_coord] Archive failed: {e}")
