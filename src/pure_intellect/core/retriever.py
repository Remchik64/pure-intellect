"""RAG Retrieval — поиск релевантного контекста."""

from pure_intellect.config import settings
from pure_intellect.utils.logger import get_logger

logger = get_logger("retriever")


class Retriever:
    """Поиск релевантных карточек кода через ChromaDB."""

    def __init__(self, collection):
        self.collection = collection

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """
        Найти релевантные карточки для запроса.

        Args:
            query: Текстовый запрос
            top_k: Количество результатов
            filter_metadata: Фильтр по метаданным (project, entity_type и т.д.)

        Returns:
            Список словарей с ключами: document, metadata, distance, id
        """
        top_k = top_k or settings.max_rag_chunks

        try:
            kwargs = {
                "query_texts": [query],
                "n_results": top_k,
            }
            if filter_metadata:
                kwargs["where"] = filter_metadata

            results = self.collection.query(**kwargs)

            if not results or not results.get("documents") or not results["documents"][0]:
                return []

            items = []
            for i in range(len(results["documents"][0])):
                items.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    "id": results["ids"][0][i] if results.get("ids") else "",
                })

            # Фильтруем по порогу релевантности (cosine distance < 0.5)
            relevant = [item for item in items if item["distance"] < 0.5]

            if not relevant:
                # Если ничего не прошло порог — возвращаем топ-2 с минимальной дистанцией
                relevant = sorted(items, key=lambda x: x["distance"])[:2]

            logger.info(f"RAG: найдено {len(relevant)} релевантных карточек для '{query[:50]}...'")
            return relevant

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []

    def count(self) -> int:
        """Количество документов в коллекции."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_by_file(self, file_path: str) -> list[dict]:
        """Получить все карточки для файла."""
        try:
            results = self.collection.get(
                where={"file": file_path}
            )
            items = []
            if results and results.get("documents"):
                for i in range(len(results["documents"])):
                    items.append({
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                        "id": results["ids"][i] if results.get("ids") else "",
                    })
            return items
        except Exception as e:
            logger.error(f"Retriever.get_by_file error: {e}")
            return []
