"""Fact — атом памяти в системе Pure Intellect.

Fact это минимальная единица информации которую система хранит.
Каждый факт имеет жизненный цикл: создание → использование → компрессия → хранение.
"""

import uuid
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class CompressionLevel(IntEnum):
    """Уровни компрессии факта.
    
    Чем выше уровень — тем больше сжат, тем меньше токенов занимает.
    """
    RAW = 1         # 100% fidelity — оригинальный контент
    SUMMARIZED = 2  # ~30% — сжатый LLM, суть сохранена
    ENTITY_ONLY = 3 # ~10% — только имя/ссылка
    ARCHIVED = 4    # Удалён из active storage, может быть восстановлен


@dataclass
class Fact:
    """Атом памяти — единица информации с lifecycle управлением.
    
    Attributes:
        content: Содержимое факта
        fact_id: Уникальный идентификатор
        created_at: Время создания (turn number)
        last_referenced: Последний раз использован (turn number)
        attention_weight: Вес важности (0.0 — не нужен, 1.0 — критичен)
        compression_level: Уровень компрессии
        stability: Насколько факт стабилен (0.0 — меняется, 1.0 — постоянен)
        is_anchor: Якорный факт — не decay, не evict из WorkingMemory
        reference_count: Сколько раз был запрошен
        source: Откуда взялся факт (entity_name, file_path, etc)
        metadata: Произвольные метаданные
    """
    
    content: str
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_referenced: int = 0
    attention_weight: float = 0.5
    compression_level: CompressionLevel = CompressionLevel.RAW
    stability: float = 0.0
    reference_count: int = 0
    source: str = ""
    metadata: dict = field(default_factory=dict)
    is_anchor: bool = False  # Якорный факт — защищён от decay и eviction
    
    def touch(self, turn: int) -> None:
        """Обновить время использования и увеличить вес."""
        self.last_referenced = turn
        self.reference_count += 1
        # Каждое использование увеличивает вес
        self.attention_weight = min(1.0, self.attention_weight + 0.1)
    
    def decay(self, current_turn: int, decay_rate: float = 0.05) -> None:
        """Снизить вес если факт не используется.
        
        Anchor факты не decay — они всегда остаются актуальными.
        Чем дольше не используется — тем сильнее decay.
        """
        if self.is_anchor:
            return  # Anchor facts защищены от decay
        turns_since_use = current_turn - self.last_referenced
        if turns_since_use > 0:
            self.attention_weight = max(0.0, self.attention_weight - decay_rate * turns_since_use)
    
    def update_stability(self) -> None:
        """Обновить stability на основе reference_count.
        
        Факт стабилен если его часто используют и не обновляют.
        """
        # Больше использований → выше stability
        self.stability = min(1.0, self.reference_count / 10.0)
    
    def is_hot(self, threshold: float = 0.6) -> bool:
        """Факт 'горячий' — часто используется, должен оставаться в WorkingMemory."""
        return self.attention_weight >= threshold
    
    def is_cold(self, threshold: float = 0.1, min_age: int = 5) -> bool:
        """Факт 'холодный' — давно не использовался, можно компрессировать."""
        return (
            self.attention_weight < threshold
            and self.reference_count >= 0
        )
    
    def is_stable(self, threshold: float = 0.7) -> bool:
        """Факт стабилен — можно безопасно переместить в Storage."""
        return self.stability >= threshold
    
    def token_size(self) -> int:
        """Приблизительный размер факта в токенах."""
        # Грубая оценка: ~4 символа на токен
        return len(self.content) // 4
    
    def to_dict(self) -> dict:
        """Сериализовать в словарь для хранения."""
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "created_at": self.created_at,
            "last_referenced": self.last_referenced,
            "attention_weight": self.attention_weight,
            "compression_level": int(self.compression_level),
            "stability": self.stability,
            "reference_count": self.reference_count,
            "source": self.source,
            "metadata": self.metadata,
            "is_anchor": self.is_anchor,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Fact':
        """Десериализовать из словаря."""
        return cls(
            content=data["content"],
            fact_id=data["fact_id"],
            created_at=data["created_at"],
            last_referenced=data["last_referenced"],
            attention_weight=data["attention_weight"],
            compression_level=CompressionLevel(data["compression_level"]),
            stability=data["stability"],
            reference_count=data.get("reference_count", 0),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
            is_anchor=data.get("is_anchor", False),
        )
    
    def __repr__(self) -> str:
        return (
            f"Fact(id={self.fact_id[:8]}..., "
            f"weight={self.attention_weight:.2f}, "
            f"level={self.compression_level.name}, "
            f"tokens≈{self.token_size()})"
        )
