"""MemoryStorage — долгосрочное хранилище фактов Pure Intellect.

Long-term хранилище куда перемещаются факты из WorkingMemory.
Факты не удаляются — только сжимаются.
Можно восстановить любой факт обратно в WorkingMemory.

P1 улучшение: Semantic retrieval через Ollama embeddings.
Fallback на BM25 keyword matching если Ollama недоступен.
"""

import json
import logging
import math
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from .fact import Fact, CompressionLevel

logger = logging.getLogger(__name__)

# Пороги для оптимизации
HOT_PROMOTION_THRESHOLD = 5   # Если факт запрошен N раз — вернуть в WorkingMemory
COLD_ARCHIVE_THRESHOLD = 0.05  # Если вес ниже — кандидат на архивирование

# Ollama embedding настройки
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "qwen2.5:3b"
EMBED_TIMEOUT = 10  # секунд


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity между двумя векторами."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_ollama_embedding(text: str) -> list[float] | None:
    """Получить embedding текста через Ollama API.
    
    Returns:
        Вектор embedding или None если Ollama недоступен.
    """
    try:
        payload = json.dumps({
            "model": OLLAMA_EMBED_MODEL,
            "prompt": text[:2000]  # Ограничиваем длину
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read())
            embedding = data.get("embedding", [])
            if embedding:
                return embedding
            return None
    except (urllib.error.URLError, TimeoutError, Exception) as e:
        logger.debug(f"Ollama embedding unavailable: {e}")
        return None


class MemoryStorage:
    """Долгосрочное хранилище фактов.
    
    Принципы:
    - Факты не удаляются безвозвратно — только сжимаются (compression_level ↑)
    - Semantic retrieval через Ollama embeddings (P1 улучшение)
    - Fallback на BM25 keyword matching если Ollama недоступен
    - Периодический анализ: горячие → promote в WorkingMemory, холодные → сжать
    - JSON persistence для сохранения между сессиями
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        use_semantic: bool = True,
        ollama_url: str = OLLAMA_BASE_URL,
        embed_model: str = OLLAMA_EMBED_MODEL,
    ):
        self._facts: dict[str, Fact] = {}  # fact_id → Fact
        self._embeddings: dict[str, list[float]] = {}  # fact_id → embedding
        self.storage_path = Path(storage_path) if storage_path else None
        self._retrieve_count: int = 0  # статистика
        self._semantic_hits: int = 0   # сколько раз semantic нашёл результат
        self._bm25_hits: int = 0       # сколько раз fallback BM25 использовался
        
        # Семантический поиск
        self._use_semantic = use_semantic
        self._ollama_url = ollama_url
        self._embed_model = embed_model
        self._semantic_available: bool | None = None  # None = не проверяли
        
        # Загружаем из файла если есть
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def _check_semantic_available(self) -> bool:
        """Проверить доступность Ollama для embeddings (кешируем результат)."""
        if self._semantic_available is not None:
            return self._semantic_available
        
        if not self._use_semantic:
            self._semantic_available = False
            return False
        
        test_embedding = _get_ollama_embedding("test")
        self._semantic_available = test_embedding is not None and len(test_embedding) > 0
        
        if self._semantic_available:
            logger.info(f"Semantic retrieval: Ollama embeddings available (dim={len(test_embedding)})")
        else:
            logger.warning("Semantic retrieval: Ollama unavailable, using BM25 fallback")
        
        return self._semantic_available
    
    def _embed_fact(self, fact: Fact) -> None:
        """Вычислить и сохранить embedding для факта."""
        if not self._check_semantic_available():
            return
        
        if fact.fact_id in self._embeddings:
            return  # Уже проиндексирован
        
        embedding = _get_ollama_embedding(fact.content)
        if embedding:
            self._embeddings[fact.fact_id] = embedding
    
    def store(self, fact: Fact) -> None:
        """Сохранить факт в хранилище.
        
        Если факт с таким id уже есть — обновляем (берём версию с большим весом).
        Автоматически индексируем embedding для semantic retrieval.
        """
        if fact.fact_id in self._facts:
            existing = self._facts[fact.fact_id]
            # Сохраняем версию с большим reference_count
            if fact.reference_count >= existing.reference_count:
                self._facts[fact.fact_id] = fact
                # Переиндексируем если контент изменился
                if fact.content != existing.content:
                    self._embeddings.pop(fact.fact_id, None)
                    self._embed_fact(fact)
        else:
            self._facts[fact.fact_id] = fact
            self._embed_fact(fact)  # Индексируем новый факт
            logger.debug(f"Stored fact: {fact.fact_id[:8]}... (weight={fact.attention_weight:.2f})")
    
    def retrieve(self, query: str, top_k: int = 5) -> list[Fact]:
        """Найти релевантные факты по запросу.
        
        Стратегия:
        1. Semantic retrieval через Ollama embeddings (если доступен)
        2. Fallback: BM25 keyword matching
        """
        if not self._facts:
            return []
        
        if self._check_semantic_available():
            results = self._retrieve_semantic(query, top_k)
            if results:
                self._semantic_hits += 1
                self._retrieve_count += 1
                for fact in results:
                    fact.reference_count += 1
                logger.debug(f"Semantic retrieval: {len(results)} facts for '{query[:40]}'")
                return results
        
        # Fallback на BM25
        self._bm25_hits += 1
        results = self._retrieve_bm25(query, top_k)
        self._retrieve_count += 1
        for fact in results:
            fact.reference_count += 1
        logger.debug(f"BM25 retrieval: {len(results)} facts for '{query[:40]}'")
        return results
    
    def _retrieve_semantic(self, query: str, top_k: int) -> list[Fact]:
        """Semantic retrieval через cosine similarity embeddings."""
        query_embedding = _get_ollama_embedding(query)
        if not query_embedding:
            return []
        
        scored: list[tuple[float, Fact]] = []
        
        for fact in self._facts.values():
            if fact.compression_level == CompressionLevel.ARCHIVED:
                continue
            
            # Получаем embedding факта (если нет — вычисляем)
            if fact.fact_id not in self._embeddings:
                self._embed_fact(fact)
            
            fact_embedding = self._embeddings.get(fact.fact_id)
            if not fact_embedding:
                continue
            
            # Cosine similarity
            sim = _cosine_similarity(query_embedding, fact_embedding)
            
            # Добавляем attention weight бонус
            score = sim + fact.attention_weight * 0.1
            
            if sim > 0.3:  # Минимальный порог релевантности
                scored.append((score, fact))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]
    
    def _retrieve_bm25(self, query: str, top_k: int) -> list[Fact]:
        """BM25 keyword matching — fallback если Ollama недоступен."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored: list[tuple[float, Fact]] = []
        
        for fact in self._facts.values():
            if fact.compression_level == CompressionLevel.ARCHIVED:
                continue
            
            content_lower = fact.content.lower()
            matches = sum(1 for word in query_words if word in content_lower)
            if matches == 0:
                continue
            
            keyword_score = matches / max(len(query_words), 1)
            attention_bonus = fact.attention_weight * 0.2
            score = keyword_score + attention_bonus
            scored.append((score, fact))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]
    
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
            sentences = fact.content.split('.')
            fact.content = '. '.join(sentences[:2]).strip() + '.'
            fact.compression_level = next_level
            # Переиндексируем embedding после сжатия
            self._embeddings.pop(fact_id, None)
            self._embed_fact(fact)
            logger.debug(f"Compressed fact {fact_id[:8]}... to SUMMARIZED")
            return True
        
        elif next_level == CompressionLevel.ENTITY_ONLY:
            fact.content = fact.source if fact.source else fact.content[:50]
            fact.compression_level = next_level
            self._embeddings.pop(fact_id, None)
            self._embed_fact(fact)
            logger.debug(f"Compressed fact {fact_id[:8]}... to ENTITY_ONLY")
            return True
        
        elif next_level == CompressionLevel.ARCHIVED:
            fact.compression_level = CompressionLevel.ARCHIVED
            self._embeddings.pop(fact_id, None)  # Embedding не нужен для архивных
            logger.info(f"Archived fact {fact_id[:8]}...")
            return True
        
        return False
    
    def delete(self, fact_id: str) -> bool:
        """Мягкое удаление — переводит в ARCHIVED, не удаляет физически."""
        fact = self._facts.get(fact_id)
        if fact is None:
            return False
        fact.compression_level = CompressionLevel.ARCHIVED
        self._embeddings.pop(fact_id, None)
        return True
    
    def purge_archived(self) -> int:
        """Физически удалить заархивированные факты. Необратимо!"""
        to_delete = [
            fid for fid, f in self._facts.items()
            if f.compression_level == CompressionLevel.ARCHIVED
        ]
        for fid in to_delete:
            del self._facts[fid]
            self._embeddings.pop(fid, None)
        logger.info(f"Purged {len(to_delete)} archived facts")
        return len(to_delete)
    
    def size(self) -> int:
        """Количество фактов в хранилище."""
        return len(self._facts)
    
    def reindex_all(self) -> int:
        """Переиндексировать все факты (полезно после загрузки с диска).
        
        Returns:
            Количество проиндексированных фактов.
        """
        if not self._check_semantic_available():
            return 0
        
        indexed = 0
        for fact in self._facts.values():
            if fact.fact_id not in self._embeddings:
                if fact.compression_level != CompressionLevel.ARCHIVED:
                    self._embed_fact(fact)
                    indexed += 1
        
        logger.info(f"Reindexed {indexed} facts")
        return indexed
    
    def stats(self) -> dict:
        """Статистика хранилища."""
        by_level = {level.name: 0 for level in CompressionLevel}
        for fact in self._facts.values():
            by_level[fact.compression_level.name] += 1
        
        return {
            "total_facts": len(self._facts),
            "by_compression_level": by_level,
            "total_retrieve_calls": self._retrieve_count,
            "semantic_hits": self._semantic_hits,
            "bm25_hits": self._bm25_hits,
            "indexed_embeddings": len(self._embeddings),
            "semantic_available": self._semantic_available,
            "hot_facts": len(self.get_hot_facts()),
            "cold_facts": len(self.get_cold_facts()),
        }
    
    def save(self) -> None:
        """Сохранить хранилище на диск."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.0",
            "facts": {fid: fact.to_dict() for fid, fact in self._facts.items()},
            "embeddings": self._embeddings,  # Сохраняем embeddings
        }
        self.storage_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.debug(f"Saved {len(self._facts)} facts + {len(self._embeddings)} embeddings to {self.storage_path}")
    
    def _load(self) -> None:
        """Загрузить хранилище с диска."""
        try:
            data = json.loads(self.storage_path.read_text())
            for fid, fact_data in data.get("facts", {}).items():
                self._facts[fid] = Fact.from_dict(fact_data)
            
            # Загружаем embeddings если есть (v2.0+)
            self._embeddings = data.get("embeddings", {})
            
            logger.info(
                f"Loaded {len(self._facts)} facts + "
                f"{len(self._embeddings)} embeddings from {self.storage_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load memory storage: {e}")
    
    def __repr__(self) -> str:
        return (
            f"MemoryStorage(facts={len(self._facts)}, "
            f"embeddings={len(self._embeddings)}, "
            f"retrieves={self._retrieve_count})"
        )
