"""MemoryStorage — долгосрочное хранилище фактов Pure Intellect.

Long-term хранилище куда перемещаются факты из WorkingMemory.
Факты не удаляются — только сжимаются.
Можно восстановить любой факт обратно в WorkingMemory.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from .fact import Fact, CompressionLevel

logger = logging.getLogger(__name__)

# Пороги для оптимизации
HOT_PROMOTION_THRESHOLD = 5   # Если факт запрошен N раз — вернуть в WorkingMemory
COLD_ARCHIVE_THRESHOLD = 0.05  # Если вес ниже — кандидат на архивирование


class MemoryStorage:
    """Долгосрочное хранилище фактов.
    
    Принципы:
    - Факты не удаляются безвозвратно — только сжимаются (compression_level ↑)
    - Поиск по ключевым словам для retrieve()
    - Периодический анализ: горячие → promote в WorkingMemory, холодные → сжать
    - JSON persistence для сохранения между сессиями
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self._facts: dict[str, Fact] = {}  # fact_id → Fact
        self.storage_path = Path(storage_path) if storage_path else None
        self._retrieve_count: int = 0  # статистика
        
        # Загружаем из файла если есть
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def store(self, fact: Fact) -> None:
        """Сохранить факт в хранилище.
        
        Если факт с таким id уже есть — обновляем (берём версию с большим весом).
        """
        if fact.fact_id in self._facts:
            existing = self._facts[fact.fact_id]
            # Сохраняем версию с большим reference_count
            if fact.reference_count >= existing.reference_count:
                self._facts[fact.fact_id] = fact
        else:
            self._facts[fact.fact_id] = fact
            logger.debug(f"Stored fact: {fact.fact_id[:8]}... (weight={fact.attention_weight:.2f})")
    
    def retrieve(self, query: str, top_k: int = 5) -> list[Fact]:
        """Найти релевантные факты по запросу.
        
        Текущая реализация: простой keyword matching.
        Будущее: semantic embeddings для точного поиска.
        """
        if not self._facts:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored: list[tuple[float, Fact]] = []
        
        for fact in self._facts.values():
            if fact.compression_level == CompressionLevel.ARCHIVED:
                continue  # Архивированные не возвращаем
            
            # Простой keyword scoring — поиск по подстрокам
            content_lower = fact.content.lower()
            
            # Проверяем каждое слово запроса
            matches = sum(1 for word in query_words if word in content_lower)
            if matches == 0:
                continue
            
            # Score = доля совпавших слов + attention_weight бонус
            keyword_score = matches / max(len(query_words), 1)
            attention_bonus = fact.attention_weight * 0.2
            score = keyword_score + attention_bonus
            
            scored.append((score, fact))
        
        # Сортируем по score
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [fact for _, fact in scored[:top_k]]
        
        # Обновляем статистику использования
        for fact in results:
            fact.reference_count += 1
            self._retrieve_count += 1
        
        logger.debug(f"Retrieved {len(results)} facts for query: '{query[:40]}...'")
        return results
    
    def get(self, fact_id: str) -> Optional[Fact]:
        """Получить факт по ID."""
        return self._facts.get(fact_id)
    
    def get_hot_facts(self, threshold: int = HOT_PROMOTION_THRESHOLD) -> list[Fact]:
        """Вернуть факты которые часто запрашиваются (кандидаты на promotion)."""
        return [
            fact for fact in self._facts.values()
            if fact.reference_count >= threshold
            and fact.compression_level != CompressionLevel.ARCHIVED
        ]
    
    def get_cold_facts(self, threshold: float = COLD_ARCHIVE_THRESHOLD) -> list[Fact]:
        """Вернуть факты с низким весом (кандидаты на архивирование)."""
        return [
            fact for fact in self._facts.values()
            if fact.attention_weight <= threshold
            and fact.compression_level != CompressionLevel.ARCHIVED
        ]
    
    def compress(self, fact_id: str) -> bool:
        """Повысить уровень компрессии факта (уменьшить размер).
        
        Returns:
            True если факт был сжат, False если уже максимально сжат.
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            return False
        
        if fact.compression_level == CompressionLevel.ARCHIVED:
            return False
        
        next_level = CompressionLevel(int(fact.compression_level) + 1)
        
        if next_level == CompressionLevel.SUMMARIZED:
            # Простая компрессия без LLM: первые 2 предложения
            sentences = fact.content.split('.')
            fact.content = '. '.join(sentences[:2]).strip() + '.'
            fact.compression_level = next_level
            logger.debug(f"Compressed fact {fact_id[:8]}... to SUMMARIZED")
            return True
        
        elif next_level == CompressionLevel.ENTITY_ONLY:
            # Только имя источника
            fact.content = fact.source if fact.source else fact.content[:50]
            fact.compression_level = next_level
            logger.debug(f"Compressed fact {fact_id[:8]}... to ENTITY_ONLY")
            return True
        
        elif next_level == CompressionLevel.ARCHIVED:
            fact.compression_level = CompressionLevel.ARCHIVED
            logger.info(f"Archived fact {fact_id[:8]}...")
            return True
        
        return False
    
    def delete(self, fact_id: str) -> bool:
        """Мягкое удаление — переводит в ARCHIVED, не удаляет физически."""
        fact = self._facts.get(fact_id)
        if fact is None:
            return False
        fact.compression_level = CompressionLevel.ARCHIVED
        return True
    
    def purge_archived(self) -> int:
        """Физически удалить заархивированные факты. Необратимо!"""
        to_delete = [
            fid for fid, f in self._facts.items()
            if f.compression_level == CompressionLevel.ARCHIVED
        ]
        for fid in to_delete:
            del self._facts[fid]
        logger.info(f"Purged {len(to_delete)} archived facts")
        return len(to_delete)
    
    def size(self) -> int:
        """Количество фактов в хранилище."""
        return len(self._facts)
    
    def stats(self) -> dict:
        """Статистика хранилища."""
        by_level = {level.name: 0 for level in CompressionLevel}
        for fact in self._facts.values():
            by_level[fact.compression_level.name] += 1
        
        return {
            "total_facts": len(self._facts),
            "by_compression_level": by_level,
            "total_retrieve_calls": self._retrieve_count,
            "hot_facts": len(self.get_hot_facts()),
            "cold_facts": len(self.get_cold_facts()),
        }
    
    def save(self) -> None:
        """Сохранить хранилище на диск."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "facts": {fid: fact.to_dict() for fid, fact in self._facts.items()}
        }
        self.storage_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.debug(f"Saved {len(self._facts)} facts to {self.storage_path}")
    
    def _load(self) -> None:
        """Загрузить хранилище с диска."""
        try:
            data = json.loads(self.storage_path.read_text())
            for fid, fact_data in data.get("facts", {}).items():
                self._facts[fid] = Fact.from_dict(fact_data)
            logger.info(f"Loaded {len(self._facts)} facts from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load memory storage: {e}")
    
    def __repr__(self) -> str:
        return f"MemoryStorage(facts={len(self._facts)}, retrieves={self._retrieve_count})"
