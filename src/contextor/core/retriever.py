"""RAG Retrieval — поиск релевантного контекста."""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import chromadb
from chromadb.config import Settings

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("retriever")


@dataclass
class RetrievalResult:
    """Результат поиска контекста."""
    card_id: str
    entity_name: str
    entity_type: str
    file_path: str
    start_line: int
    end_line: int
    summary: str
    distance: float
    relevance_score: float = 0.0
    
    @classmethod
    def from_chroma(cls, item: dict) -> 'RetrievalResult':
        """Создать из результата ChromaDB."""
        metadata = item.get('metadata', {})
        distance = item.get('distance', 1.0)
        return cls(
            card_id=item.get('id', ''),
            entity_name=metadata.get('entity_name', ''),
            entity_type=metadata.get('entity_type', ''),
            file_path=metadata.get('file_path', ''),
            start_line=metadata.get('start_line', 0),
            end_line=metadata.get('end_line', 0),
            summary=cls._extract_summary(item.get('document', '')),
            distance=distance,
            relevance_score=max(0, 1 - distance),
        )
    
    @staticmethod
    def _extract_summary(document: str) -> str:
        """Извлечь summary из YAML-документа."""
        for line in document.split('\n'):
            if line.startswith('summary:'):
                return line.replace('summary:', '').strip()
        return ''
    
    def to_context_string(self) -> str:
        """Форматировать для вставки в контекст LLM."""
        lines = [
            f"### {self.entity_type.upper()}: {self.entity_name}",
            f"File: {self.file_path}:{self.start_line}-{self.end_line}",
            f"Summary: {self.summary}",
        ]
        return '\n'.join(lines)


class Retriever:
    """RAG Retriever — поиск релевантного контекста для запроса."""
    
    def __init__(self, chroma_dir: str = "./storage/chromadb"):
        self.chroma_dir = Path(chroma_dir)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="code_cards",
            metadata={"hnsw:space": "cosine"}
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: Optional[str] = None,
        file_path: Optional[str] = None,
        threshold: float = 0.5,
    ) -> List[RetrievalResult]:
        """Поиск релевантных карточек."""
        try:
            kwargs = {
                "query_texts": [query],
                "n_results": top_k,
            }
            
            where = {}
            if entity_type:
                where["entity_type"] = entity_type
            if file_path:
                where["file_path"] = file_path
            if where:
                kwargs["where"] = where
            
            results = self.collection.query(**kwargs)
            
            if not results or not results.get("documents") or not results["documents"][0]:
                return []
            
            items = []
            for i in range(len(results["documents"][0])):
                item = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 1.0,
                }
                result = RetrievalResult.from_chroma(item)
                
                if result.distance < threshold:
                    items.append(result)
            
            if not items and results["documents"][0]:
                for i in range(min(2, len(results["documents"][0]))):
                    item = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": results["distances"][0][i] if results.get("distances") else 1.0,
                    }
                    items.append(RetrievalResult.from_chroma(item))
            
            items.sort(key=lambda x: x.distance)
            logger.info(f"RAG: found {len(items)} cards for '{query[:50]}...'")
            return items
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    def multi_query_search(
        self,
        queries: List[str],
        top_k: int = 3,
        threshold: float = 0.5,
    ) -> List[RetrievalResult]:
        """Поиск по нескольким запросам с дедупликацией."""
        all_results = {}
        
        for query in queries:
            results = self.search(query, top_k=top_k, threshold=threshold)
            for r in results:
                if r.card_id not in all_results or r.distance < all_results[r.card_id].distance:
                    all_results[r.card_id] = r
        
        sorted_results = sorted(all_results.values(), key=lambda x: x.distance)
        return sorted_results[:top_k * 2]
    
    def search_by_intent(self, intent: str, entities: List[str] = None) -> List[RetrievalResult]:
        """Поиск контекста на основе intent и entities."""
        queries = [intent]
        
        if entities:
            for entity in entities:
                queries.append(entity)
        
        intent_queries = {
            "debug": ["error", "exception", "fix", "bug"],
            "code_generation": ["function", "class", "implementation"],
            "code_explain": ["documentation", "docstring", "comment"],
            "refactor": ["pattern", "structure", "clean"],
            "architecture": ["design", "module", "dependency"],
        }
        
        if intent in intent_queries:
            queries.extend(intent_queries[intent])
        
        return self.multi_query_search(queries, top_k=5)
    
    def count(self) -> int:
        """Количество документов в коллекции."""
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def format_context(self, results: List[RetrievalResult], max_tokens: int = 2000) -> str:
        """Форматировать результаты для вставки в контекст LLM."""
        if not results:
            return "No relevant code context found."
        
        parts = ["[RELEVANT CODE CONTEXT]"]
        total_chars = 0
        max_chars = max_tokens * 4
        
        for r in results:
            context_str = r.to_context_string()
            if total_chars + len(context_str) > max_chars:
                break
            parts.append(context_str)
            total_chars += len(context_str)
        
        parts.append(f"[/RELEVANT CODE CONTEXT] ({len(results)} cards found)")
        return '\n\n'.join(parts)