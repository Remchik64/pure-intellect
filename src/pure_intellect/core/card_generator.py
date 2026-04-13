"""Card Generator - converts parsed code entities to cards for RAG."""

import hashlib
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings

from ..parsers.base import CodeEntity, CodeCard, BaseParser
from ..parsers.python_parser import PythonParser
from ..utils.hashing import file_hash

logger = logging.getLogger(__name__)


class CardGenerator:
    """Generates and manages code cards for RAG."""
    
    def __init__(self, chroma_dir: str = "./storage/chromadb"):
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection for code cards
        self.collection = self.client.get_or_create_collection(
            name="code_cards",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize parsers
        self.parsers: List[BaseParser] = [PythonParser()]
        
        # File hash cache to detect changes
        self.file_hashes: Dict[str, str] = {}
    
    def _get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """Get appropriate parser for file extension."""
        extension = file_path.suffix
        for parser in self.parsers:
            if parser.supports_extension(extension):
                return parser
        return None
    
    def generate_card_id(self, entity: CodeEntity) -> str:
        """Generate unique card ID from entity."""
        key = f"{entity.file_path}:{entity.name}:{entity.start_line}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def generate_summary(self, entity: CodeEntity) -> str:
        """Generate short summary for entity."""
        if entity.docstring:
            # Use first sentence of docstring
            first_sentence = entity.docstring.split('.')[0].strip()
            return f"{entity.type.capitalize()} {entity.name}: {first_sentence}."
        
        # Fallback summary
        if entity.type == 'function' or entity.type == 'method':
            return f"{entity.type.capitalize()} {entity.name} defined in {entity.file_path}"
        elif entity.type == 'class':
            return f"Class {entity.name} defined in {entity.file_path}"
        elif entity.type == 'import':
            return f"Imports: {entity.name}"
        else:
            return f"{entity.type.capitalize()} {entity.name}"
    
    def create_card(self, entity: CodeEntity) -> CodeCard:
        """Create a code card from entity."""
        card_id = self.generate_card_id(entity)
        summary = self.generate_summary(entity)
        
        return CodeCard(
            card_id=card_id,
            entity=entity,
            summary=summary,
            metadata={
                "file_path": entity.file_path,
                "entity_type": entity.type,
                "entity_name": entity.name,
                "start_line": entity.start_line,
                "end_line": entity.end_line,
                "parent": entity.parent,
            }
        )
    
    def index_file(self, file_path: Path) -> List[CodeCard]:
        """Index a single file: parse it and create cards."""
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        # Check if file changed
        current_hash = file_hash(file_path)
        if str(file_path) in self.file_hashes and self.file_hashes[str(file_path)] == current_hash:
            logger.debug(f"File unchanged: {file_path}")
            return []  # No changes, skip
        
        parser = self._get_parser(file_path)
        if not parser:
            logger.debug(f"No parser for: {file_path}")
            return []
        
        logger.info(f"Indexing file: {file_path}")
        
        # Parse file
        entities = parser.parse_file(file_path)
        cards = []
        
        for entity in entities:
            card = self.create_card(entity)
            cards.append(card)
            
            # Store in ChromaDB
            try:
                # For now, store the YAML representation as document
                # Later, we'll add embeddings via embedding function
                self.collection.upsert(
                    ids=[card.card_id],
                    documents=[card.to_yaml()],
                    metadatas=[card.metadata],
                )
                logger.debug(f"Stored card: {card.card_id}")
            except Exception as e:
                logger.error(f"Failed to store card {card.card_id}: {e}")
        
        # Update file hash
        self.file_hashes[str(file_path)] = current_hash
        
        logger.info(f"Indexed {len(cards)} cards from {file_path}")
        return cards
    
    def index_directory(self, directory: Path, extensions: List[str] = ['.py']) -> int:
        """Index all supported files in directory recursively."""
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return 0
        
        total_cards = 0
        
        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                # Skip common non-source directories
                if any(part in ['__pycache__', '.git', 'venv', 'node_modules', '.venv'] 
                       for part in file_path.parts):
                    continue
                
                cards = self.index_file(file_path)
                total_cards += len(cards)
        
        logger.info(f"Indexed {total_cards} cards from {directory}")
        return total_cards
    
    def search_cards(self, query: str, top_k: int = 5) -> List[CodeCard]:
        """Search cards by semantic similarity."""
        # For now, simple keyword search
        # Later, integrate with embeddings
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
            )
            
            cards = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                card_id = results['ids'][0][i]
                
                # Reconstruct card (simplified)
                cards.append(CodeCard(
                    card_id=card_id,
                    entity=CodeEntity(
                        name=metadata.get('entity_name', ''),
                        type=metadata.get('entity_type', ''),
                        file_path=metadata.get('file_path', ''),
                        start_line=metadata.get('start_line', 0),
                        end_line=metadata.get('end_line', 0),
                        source_code='',  # Not stored in metadata
                    ),
                    summary=doc.split('summary: ')[-1].split('\n')[0] if 'summary: ' in doc else '',
                    metadata=metadata,
                ))
            
            return cards
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_card_by_id(self, card_id: str) -> Optional[CodeCard]:
        """Get card by ID."""
        try:
            result = self.collection.get(ids=[card_id])
            if result['documents']:
                metadata = result['metadatas'][0] if result['metadatas'] else {}
                return CodeCard(
                    card_id=card_id,
                    entity=CodeEntity(
                        name=metadata.get('entity_name', ''),
                        type=metadata.get('entity_type', ''),
                        file_path=metadata.get('file_path', ''),
                        start_line=metadata.get('start_line', 0),
                        end_line=metadata.get('end_line', 0),
                        source_code='',
                    ),
                    summary=result['documents'][0].split('summary: ')[-1].split('\n')[0] if 'summary: ' in result['documents'][0] else '',
                    metadata=metadata,
                )
        except Exception as e:
            logger.error(f"Failed to get card {card_id}: {e}")
        return None
