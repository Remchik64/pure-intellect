"""MemoryStorage — долгосрочное хранилище фактов Pure Intellect.

Long-term хранилище куда перемещаются факты из WorkingMemory.
Факты не удаляются — только сжимаются.
Можно восстановить любой факт обратно в WorkingMemory.

P1 улучшение: Semantic retrieval через embeddings.
P4 улучшение: SentenceTransformer (CUDA) как primary embedding provider.

Embedding провайдеры (по приоритету):
  1. SentenceTransformer all-MiniLM-L6-v2 (GPU если доступен, ~5ms/факт)
  2. Ollama HTTP embeddings (~200-500ms/факт, fallback)
  3. BM25 keyword matching (всегда работает)
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

# Ollama embedding настройки (fallback)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "qwen2.5:3b"
EMBED_TIMEOUT = 10  # секунд

# SentenceTransformer настройки
ST_MODEL_NAME = "all-MiniLM-L6-v2"
ST_EMBEDDING_DIM = 384

# ── Singleton SentenceTransformer (загружается один раз) ──────────────────
_st_model = None          # Глобальный singleton
_st_available: bool | None = None  # None = не проверяли
_st_device: str = "cpu"   # Устройство после загрузки


def _get_st_model():
    """Получить singleton SentenceTransformer модель (lazy init)."""
    global _st_model, _st_available, _st_device
    
    if _st_available is not None:
        return _st_model if _st_available else None
    
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _st_model = SentenceTransformer(ST_MODEL_NAME, device=device)
        _st_device = device
        _st_available = True
        
        logger.info(
            f"SentenceTransformer loaded: {ST_MODEL_NAME} on {device} "
            f"(dim={ST_EMBEDDING_DIM})"
        )
    except ImportError:
        _st_available = False
        logger.info("sentence-transformers not installed, falling back to Ollama")
    except Exception as e:
        _st_available = False
        logger.warning(f"SentenceTransformer init failed: {e}, falling back to Ollama")
    
    return _st_model if _st_available else None


def _get_st_embedding(text: str) -> list[float] | None:
    """Получить embedding через SentenceTransformer (primary, fast)."""
    model = _get_st_model()
    if model is None:
        return None
    try:
        embedding = model.encode([text[:512]], normalize_embeddings=True)
        return embedding[0].tolist()
    except Exception as e:
        logger.debug(f"SentenceTransformer encode failed: {e}")
        return None


def _get_st_embeddings_batch(texts: list[str]) -> list[list[float]] | None:
    """Батчевое получение embeddings через SentenceTransformer."""
    model = _get_st_model()
    if model is None or not texts:
        return None
    try:
        truncated = [t[:512] for t in texts]
        embeddings = model.encode(truncated, normalize_embeddings=True, batch_size=64)
        return [e.tolist() for e in embeddings]
    except Exception as e:
        logger.debug(f"SentenceTransformer batch encode failed: {e}")
        return None


def _get_ollama_embedding(text: str) -> list[float] | None:
    """Получить embedding через Ollama API (fallback)."""
    try:
        payload = json.dumps({
            "model": OLLAMA_EMBED_MODEL,
            "prompt": text[:2000]
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
            return embedding if embedding else None
    except (urllib.error.URLError, TimeoutError, Exception) as e:
        logger.debug(f"Ollama embedding unavailable: {e}")
        return None


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


class MemoryStorage:
    """Долгосрочное хранилище фактов.
    
    Принципы:
    - Факты не удаляются безвозвратно — только сжимаются (compression_level ↑)
    - P4: SentenceTransformer CUDA как primary embedding provider (5ms/факт)
    - P1: Ollama embeddings как fallback (200-500ms/факт)
    - BM25 keyword matching как финальный fallback
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
        self._retrieve_count: int = 0
        self._semantic_hits: int = 0
        self._bm25_hits: int = 0
        self._st_hits: int = 0      # сколько раз SentenceTransformer использовался
        self._ollama_hits: int = 0  # сколько раз Ollama использовался
        
        # Семантический поиск
        self._use_semantic = use_semantic
        self._ollama_url = ollama_url
        self._embed_model = embed_model
        self._semantic_available: bool | None = None
        self._provider: str = "none"  # "st", "ollama", "bm25"
        self._embed_dim: int = 0      # размерность embedding
        
        # Загружаем из файла если есть
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def _init_provider(self) -> str:
        """Инициализировать embedding провайдер (один раз).
        
        Порядок: SentenceTransformer → Ollama → BM25
        Returns: 'st', 'ollama', или 'bm25'
        """
        if self._provider != "none":
            return self._provider
        
        if not self._use_semantic:
            self._provider = "bm25"
            self._semantic_available = False
            return self._provider
        
        # Пробуем SentenceTransformer
        st_model = _get_st_model()
        if st_model is not None:
            self._provider = "st"
            self._embed_dim = ST_EMBEDDING_DIM
            self._semantic_available = True
            logger.info(f"Embedding provider: SentenceTransformer ({_st_device}, dim={ST_EMBEDDING_DIM})")
            # Если загружены embeddings другой размерности — сбрасываем индекс
            if self._embeddings:
                sample = next(iter(self._embeddings.values()))
                if len(sample) != ST_EMBEDDING_DIM:
                    logger.warning(
                        f"Embedding dim mismatch: stored={len(sample)}, "
                        f"expected={ST_EMBEDDING_DIM} — reindexing"
                    )
                    self._embeddings = {}
            return self._provider
        
        # Пробуем Ollama
        test_emb = _get_ollama_embedding("test")
        if test_emb is not None and len(test_emb) > 0:
            self._provider = "ollama"
            self._embed_dim = len(test_emb)
            self._semantic_available = True
            logger.info(f"Embedding provider: Ollama ({OLLAMA_EMBED_MODEL}, dim={self._embed_dim})")
            return self._provider
        
        # Fallback BM25
        self._provider = "bm25"
        self._semantic_available = False
        logger.warning("Embedding provider: BM25 (semantic unavailable)")
        return self._provider
    
    def _check_semantic_available(self) -> bool:
        """Проверить доступность semantic retrieval."""
        provider = self._init_provider()
        return provider in ("st", "ollama")
    
    def _get_embedding(self, text: str) -> list[float] | None:
        """Получить embedding текста через активный провайдер."""
        provider = self._init_provider()
        
        if provider == "st":
            emb = _get_st_embedding(text)
            if emb is not None:
                self._st_hits += 1
                return emb
            # ST упал — пробуем Ollama
            logger.debug("ST failed, trying Ollama fallback")
        
        if provider in ("st", "ollama"):
            emb = _get_ollama_embedding(text)
            if emb is not None:
                self._ollama_hits += 1
                return emb
        
        return None
    
    def _embed_fact(self, fact: Fact) -> None:
        """Вычислить и сохранить embedding для факта."""
        if not self._check_semantic_available():
            return
        if fact.fact_id in self._embeddings:
            return  # Уже проиндексирован
        
        embedding = self._get_embedding(fact.content)
        if embedding:
            self._embeddings[fact.fact_id] = embedding
    
    def reindex_all_batch(self) -> int:
        """Батчевая переиндексация всех фактов (эффективно для SentenceTransformer).
        
        Использует batch encode для максимальной производительности на GPU.
        Returns: количество проиндексированных фактов.
        """
        provider = self._init_provider()
        if provider not in ("st", "ollama"):
            return 0
        
        # Собираем факты без embedding
        to_index = [
            fact for fact in self._facts.values()
            if fact.fact_id not in self._embeddings
            and fact.compression_level != CompressionLevel.ARCHIVED
        ]
        
        if not to_index:
            return 0
        
        if provider == "st":
            # Батчевое индексирование на GPU
            texts = [fact.content for fact in to_index]
            embeddings = _get_st_embeddings_batch(texts)
            if embeddings:
                for fact, emb in zip(to_index, embeddings):
                    self._embeddings[fact.fact_id] = emb
                logger.info(f"Batch reindexed {len(to_index)} facts via SentenceTransformer")
                return len(to_index)
        
        # Fallback: по одному
        indexed = 0
        for fact in to_index:
            emb = self._get_embedding(fact.content)
            if emb:
                self._embeddings[fact.fact_id] = emb
                indexed += 1
        
        logger.info(f"Reindexed {indexed} facts via {provider}")
        return indexed
    
    def store(self, fact: Fact) -> None:
        """Сохранить факт в хранилище."""
        if fact.fact_id in self._facts:
            existing = self._facts[fact.fact_id]
            if fact.reference_count >= existing.reference_count:
                self._facts[fact.fact_id] = fact
                if fact.content != existing.content:
                    self._embeddings.pop(fact.fact_id, None)
                    self._embed_fact(fact)
        else:
            self._facts[fact.fact_id] = fact
            self._embed_fact(fact)
            logger.debug(f"Stored fact: {fact.fact_id[:8]}... (weight={fact.attention_weight:.2f})")
    
    def retrieve(self, query: str, top_k: int = 5) -> list[Fact]:
        """Найти релевантные факты по запросу.
        
        Использует активный embedding провайдер (ST → Ollama → BM25).
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
                logger.debug(
                    f"[{self._provider}] retrieval: {len(results)} facts "
                    f"for '{query[:40]}'"
                )
                return results
        
        # Fallback BM25
        self._bm25_hits += 1
        results = self._retrieve_bm25(query, top_k)
        self._retrieve_count += 1
        for fact in results:
            fact.reference_count += 1
        logger.debug(f"BM25 retrieval: {len(results)} facts for '{query[:40]}'")
        return results
    
    def _retrieve_semantic(self, query: str, top_k: int) -> list[Fact]:
        """Semantic retrieval через cosine similarity."""
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        scored: list[tuple[float, Fact]] = []
        
        for fact in self._facts.values():
            if fact.compression_level == CompressionLevel.ARCHIVED:
                continue
            
            if fact.fact_id not in self._embeddings:
                self._embed_fact(fact)
            
            fact_embedding = self._embeddings.get(fact.fact_id)
            if not fact_embedding:
                continue
            
            # Проверяем совместимость размерностей
            if len(fact_embedding) != len(query_embedding):
                # Несовместимые embeddings (сменился провайдер) — пропускаем
                continue
            
            sim = _cosine_similarity(query_embedding, fact_embedding)
            score = sim + fact.attention_weight * 0.1
            
            if sim > 0.3:
                scored.append((score, fact))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]
    
    def _retrieve_bm25(self, query: str, top_k: int) -> list[Fact]:
        """BM25 keyword matching."""
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
            scored.append((keyword_score + attention_bonus, fact))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]
    
    def get(self, fact_id: str) -> Optional[Fact]:
        return self._facts.get(fact_id)
    
    def get_hot_facts(self, threshold: int = HOT_PROMOTION_THRESHOLD) -> list[Fact]:
        return [
            fact for fact in self._facts.values()
            if fact.reference_count >= threshold
            and fact.compression_level != CompressionLevel.ARCHIVED
        ]
    
    def get_cold_facts(self, threshold: float = COLD_ARCHIVE_THRESHOLD) -> list[Fact]:
        return [
            fact for fact in self._facts.values()
            if fact.attention_weight <= threshold
            and fact.compression_level != CompressionLevel.ARCHIVED
        ]
    
    def compress(self, fact_id: str) -> bool:
        fact = self._facts.get(fact_id)
        if fact is None or fact.compression_level == CompressionLevel.ARCHIVED:
            return False
        
        next_level = CompressionLevel(int(fact.compression_level) + 1)
        
        if next_level == CompressionLevel.SUMMARIZED:
            sentences = fact.content.split('.')
            fact.content = '. '.join(sentences[:2]).strip() + '.'
            fact.compression_level = next_level
            self._embeddings.pop(fact_id, None)
            self._embed_fact(fact)
            return True
        elif next_level == CompressionLevel.ENTITY_ONLY:
            fact.content = fact.source if fact.source else fact.content[:50]
            fact.compression_level = next_level
            self._embeddings.pop(fact_id, None)
            self._embed_fact(fact)
            return True
        elif next_level == CompressionLevel.ARCHIVED:
            fact.compression_level = CompressionLevel.ARCHIVED
            self._embeddings.pop(fact_id, None)
            return True
        
        return False
    
    def delete(self, fact_id: str) -> bool:
        fact = self._facts.get(fact_id)
        if fact is None:
            return False
        fact.compression_level = CompressionLevel.ARCHIVED
        self._embeddings.pop(fact_id, None)
        return True
    
    def purge_archived(self) -> int:
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
        return len(self._facts)
    
    def reindex_all(self) -> int:
        """Переиндексировать все факты (использует batch для ST)."""
        return self.reindex_all_batch()
    
    def stats(self) -> dict:
        provider = self._init_provider()
        by_level = {level.name: 0 for level in CompressionLevel}
        for fact in self._facts.values():
            by_level[fact.compression_level.name] += 1
        
        return {
            "total_facts": len(self._facts),
            "by_compression_level": by_level,
            "total_retrieve_calls": self._retrieve_count,
            "semantic_hits": self._semantic_hits,
            "bm25_hits": self._bm25_hits,
            "st_hits": self._st_hits,
            "ollama_hits": self._ollama_hits,
            "indexed_embeddings": len(self._embeddings),
            "semantic_available": self._semantic_available,
            "embedding_provider": provider,
            "embedding_dim": self._embed_dim,
            "embedding_device": _st_device if provider == "st" else "http",
            "hot_facts": len(self.get_hot_facts()),
            "cold_facts": len(self.get_cold_facts()),
        }
    
    def save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.1",
            "embedding_provider": self._provider,
            "embedding_dim": self._embed_dim,
            "facts": {fid: fact.to_dict() for fid, fact in self._facts.items()},
            "embeddings": self._embeddings,
        }
        self.storage_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.debug(
            f"Saved {len(self._facts)} facts + "
            f"{len(self._embeddings)} embeddings [{self._provider}] "
            f"to {self.storage_path}"
        )
    
    def _load(self) -> None:
        try:
            data = json.loads(self.storage_path.read_text())
            for fid, fact_data in data.get("facts", {}).items():
                self._facts[fid] = Fact.from_dict(fact_data)
            self._embeddings = data.get("embeddings", {})
            saved_dim = data.get("embedding_dim", 0)
            
            # Если размерность сохранённых embeddings не совпадает с текущим провайдером
            # — сбрасываем индекс (переиндексация произойдёт при первом retrieve)
            if saved_dim and saved_dim != ST_EMBEDDING_DIM and _get_st_model() is not None:
                logger.warning(
                    f"Embedding dim changed ({saved_dim} → {ST_EMBEDDING_DIM}), "
                    f"clearing index for reindexing"
                )
                self._embeddings = {}
            
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
            f"provider={self._provider}, "
            f"retrieves={self._retrieve_count})"
        )
