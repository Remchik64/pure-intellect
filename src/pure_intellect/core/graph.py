"""Граф знаний — NetworkX DiGraph для связей между сущностями."""

import json
from pathlib import Path
import networkx as nx

from pure_intellect.utils.logger import get_logger

logger = get_logger("graph")


class KnowledgeGraph:
    """Граф знаний проекта."""

    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.graph = nx.DiGraph()
        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        file_path: str = "",
        summary: str = "",
        **extra,
    ):
        """Добавить сущность в граф."""
        self.graph.add_node(
            entity_id,
            name=name,
            type=entity_type,
            file=file_path,
            summary=summary,
            **extra,
        )

    def add_relation(self, source_id: str, target_id: str, relation: str = "uses"):
        """Добавить связь между сущностями."""
        self.graph.add_edge(source_id, target_id, relation=relation)

    def get_related(self, entity_id: str, depth: int = 1) -> list[dict]:
        """Получить связанные сущности."""
        if entity_id not in self.graph:
            return []

        related = []
        # Исходящие связи
        for successor in nx.descendants_at_distance(self.graph, entity_id, depth):
            if successor in self.graph.nodes:
                data = self.graph.nodes[successor]
                related.append({"id": successor, **dict(data)})

        # Входящие связи
        for predecessor in self.graph.predecessors(entity_id):
            if predecessor not in [r["id"] for r in related]:
                data = self.graph.nodes[predecessor]
                related.append({"id": predecessor, **dict(data)})

        return related

    def search_by_name(self, query: str, limit: int = 10) -> list[dict]:
        """Поиск узлов по имени."""
        query_lower = query.lower()
        results = []
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            summary = data.get("summary", "").lower()
            if query_lower in name or query_lower in summary:
                results.append({"id": node_id, **dict(data)})
            if len(results) >= limit:
                break
        return results

    def get_file_entities(self, file_path: str) -> list[dict]:
        """Получить все сущности файла."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("file") == file_path:
                results.append({"id": node_id, **dict(data)})
        return results

    def remove_file(self, file_path: str):
        """Удалить все узлы файла."""
        to_remove = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("file") == file_path
        ]
        self.graph.remove_nodes_from(to_remove)
        logger.info(f"Graph: удалено {len(to_remove)} узлов для {file_path}")

    def get_stats(self) -> dict:
        """Статистика графа."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4),
            "types": self._count_types(),
        }

    def _count_types(self) -> dict:
        """Подсчёт узлов по типам."""
        types = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        return types

    def to_json(self) -> str:
        """Сериализовать граф в JSON."""
        return json.dumps(nx.node_link_data(self.graph), indent=2, ensure_ascii=False)

    def save(self):
        """Сохранить граф на диск."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(self.to_json(), encoding="utf-8")
            logger.info(f"Graph: сохранён ({self.graph.number_of_nodes()} узлов)")

    def load(self):
        """Загрузить граф с диска."""
        if self.storage_path and self.storage_path.exists():
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.graph = nx.node_link_graph(data)
            logger.info(f"Graph: загружен ({self.graph.number_of_nodes()} узлов)")
