"""Base parser interface for code parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class CodeEntity:
    """Represents a code entity (function, class, etc.)."""
    name: str
    type: str  # 'function', 'class', 'method', 'import', 'variable'
    file_path: str
    start_line: int
    end_line: int
    source_code: str
    docstring: Optional[str] = None
    parent: Optional[str] = None  # Parent class or module
    calls: List[str] = None  # Functions/methods called within this entity
    
    def __post_init__(self):
        if self.calls is None:
            self.calls = []


@dataclass
class CodeCard:
    """Card representation of a code entity for RAG."""
    card_id: str  # Unique identifier
    entity: CodeEntity
    summary: str  # Short summary
    embedding: Optional[List[float]] = None
    metadata: Optional[dict] = None
    
    def to_yaml(self) -> str:
        """Convert card to YAML-like format for LLM consumption."""
        lines = [
            f"---",
            f"card: {self.card_id}",
            f"file: {self.entity.file_path}:{self.entity.start_line}-{self.entity.end_line}",
            f"type: {self.entity.type}",
            f"name: {self.entity.name}",
        ]
        
        if self.entity.parent:
            lines.append(f"parent: {self.entity.parent}")
        
        if self.entity.calls:
            lines.append(f"calls: [{', '.join(self.entity.calls)}]")
        
        if self.entity.docstring:
            lines.append(f"docstring: |")
            lines.append(f"  {self.entity.docstring.strip()}")
        
        lines.append(f"summary: {self.summary}")
        lines.append(f"---")
        return "\n".join(lines)


class BaseParser(ABC):
    """Abstract base class for code parsers."""
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse a file and extract code entities."""
        pass
    
    @abstractmethod
    def supports_extension(self, extension: str) -> bool:
        """Check if parser supports given file extension."""
        pass
