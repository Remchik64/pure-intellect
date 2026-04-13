"""Graph Builder — строит граф связей из карточек кода."""

import hashlib
from pathlib import Path
from typing import Optional, List

from ..config import settings
from ..utils.logger import get_logger
from ..parsers.base import CodeEntity, CodeCard
from ..parsers.python_parser import PythonParser
from .card_generator import CardGenerator
from .graph import KnowledgeGraph

logger = get_logger("graph_builder")


class GraphBuilder:
    """Строит граф знаний из карточек кода."""
    
    def __init__(self, graph: Optional[KnowledgeGraph] = None):
        self.graph = graph or KnowledgeGraph(settings.graph_file)
        self.card_generator = CardGenerator()
    
    def build_from_directory(self, directory: Path, extensions: List[str] = ['.py']) -> dict:
        """Построить граф из всех файлов директории."""
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return self.graph.get_stats()
        
        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if any(part in ['__pycache__', '.git', 'venv', 'node_modules', '.venv'] 
                       for part in file_path.parts):
                    continue
                self._process_file(file_path)
        
        self.graph.save()
        stats = self.graph.get_stats()
        logger.info(f"Graph built: {stats['nodes']} nodes, {stats['edges']} edges")
        return stats
    
    def _process_file(self, file_path: Path):
        """Обработать один файл — добавить сущности и связи в граф."""
        parser = PythonParser()
        entities = parser.parse_file(file_path)
        
        # Имена сущностей в этом файле для связей calls
        file_entity_names = {e.name for e in entities}
        
        for entity in entities:
            entity_id = self._make_id(entity)
            
            # Добавляем узел
            self.graph.add_entity(
                entity_id=entity_id,
                name=entity.name,
                entity_type=entity.type,
                file_path=entity.file_path,
                start_line=entity.start_line,
                end_line=entity.end_line,
                summary=entity.docstring[:100] if entity.docstring else '',
            )
            
            # Связь parent (method -> class)
            if entity.parent:
                parent_entity = next((e for e in entities if e.name == entity.parent), None)
                if parent_entity:
                    parent_id = self._make_id(parent_entity)
                    self.graph.add_relation(parent_id, entity_id, "contains")
            
            # Связи calls (функция вызывает другую функцию)
            for called in entity.calls:
                # Ищем called в этом же файле
                called_entity = next((e for e in entities if e.name == called), None)
                if called_entity:
                    called_id = self._make_id(called_entity)
                    self.graph.add_relation(entity_id, called_id, "calls")
                else:
                    # Внешний вызов — создаём placeholder узел
                    ext_id = hashlib.md5(f"external:{called}".encode()).hexdigest()
                    self.graph.add_entity(
                        entity_id=ext_id,
                        name=called,
                        entity_type="external",
                        file_path="",
                        summary=f"External reference: {called}",
                    )
                    self.graph.add_relation(entity_id, ext_id, "calls")
            
            # Связи imports
            if entity.type == 'import':
                # Import связывает файл с импортированным модулем
                for imported_name in entity.name.split(', '):
                    imp_id = hashlib.md5(f"import:{imported_name}".encode()).hexdigest()
                    self.graph.add_entity(
                        entity_id=imp_id,
                        name=imported_name,
                        entity_type="module",
                        file_path="",
                        summary=f"Imported module: {imported_name}",
                    )
                    self.graph.add_relation(entity_id, imp_id, "imports")
    
    def _make_id(self, entity: CodeEntity) -> str:
        """Генерировать уникальный ID для сущности."""
        key = f"{entity.file_path}:{entity.name}:{entity.start_line}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_stats(self) -> dict:
        """Статистика графа."""
        return self.graph.get_stats()
    
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Поиск узлов по имени."""
        return self.graph.search_by_name(query, limit)
    
    def get_related(self, entity_id: str, depth: int = 1) -> list[dict]:
        """Получить связанные сущности."""
        return self.graph.get_related(entity_id, depth)
    
    def get_file_graph(self, file_path: str) -> dict:
        """Получить подграф для файла."""
        entities = self.graph.get_file_entities(file_path)
        edges = []
        for entity in entities:
            related = self.graph.get_related(entity['id'], depth=1)
            for r in related:
                edges.append({
                    "source": entity['id'],
                    "target": r['id'],
                    "relation": "related",
                })
        return {
            "file": file_path,
            "entities": entities,
            "edges": edges,
        }
