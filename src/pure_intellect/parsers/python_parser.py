"""Python parser using tree-sitter."""

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from typing import List, Optional
from pathlib import Path

from .base import BaseParser, CodeEntity


class PythonParser(BaseParser):
    """Parser for Python files using tree-sitter."""
    
    def __init__(self):
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
    
    def supports_extension(self, extension: str) -> bool:
        return extension.lower() in ['.py', '.pyi', '.pyx']
    
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse Python file and extract entities."""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = self.parser.parse(source_code)
            root_node = tree.root_node
            
            entities = []
            self._extract_entities(root_node, source_code, str(file_path), entities)
            return entities
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def _extract_entities(self, node: Node, source: bytes, file_path: str, 
                         entities: List[CodeEntity], parent: Optional[str] = None):
        """Recursively extract entities from AST."""
        if node.type == 'function_definition':
            entity = self._parse_function(node, source, file_path, parent)
            if entity:
                entities.append(entity)
                # Parse nested functions/methods
                for child in node.children:
                    if child.type == 'block':
                        self._extract_entities(child, source, file_path, entities, entity.name)
        
        elif node.type == 'class_definition':
            entity = self._parse_class(node, source, file_path, parent)
            if entity:
                entities.append(entity)
                # Parse methods inside class
                for child in node.children:
                    if child.type == 'block':
                        self._extract_entities(child, source, file_path, entities, entity.name)
        
        elif node.type == 'import_statement' or node.type == 'import_from_statement':
            entity = self._parse_import(node, source, file_path)
            if entity:
                entities.append(entity)
        
        else:
            # Recurse into children
            for child in node.children:
                self._extract_entities(child, source, file_path, entities, parent)
    
    def _parse_function(self, node: Node, source: bytes, file_path: str, 
                       parent: Optional[str]) -> Optional[CodeEntity]:
        """Parse function definition node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        source_code = source[node.start_byte:node.end_byte].decode('utf-8')
        
        # Extract docstring if present
        docstring = self._extract_docstring(node, source)
        
        # Extract function calls within this function
        calls = self._extract_calls(node, source)
        
        return CodeEntity(
            name=name,
            type='method' if parent else 'function',
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            docstring=docstring,
            parent=parent,
            calls=calls
        )
    
    def _parse_class(self, node: Node, source: bytes, file_path: str, 
                    parent: Optional[str]) -> Optional[CodeEntity]:
        """Parse class definition node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        source_code = source[node.start_byte:node.end_byte].decode('utf-8')
        
        docstring = self._extract_docstring(node, source)
        
        return CodeEntity(
            name=name,
            type='class',
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            docstring=docstring,
            parent=parent
        )
    
    def _parse_import(self, node: Node, source: bytes, file_path: str) -> Optional[CodeEntity]:
        """Parse import statement."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        source_code = source[node.start_byte:node.end_byte].decode('utf-8')
        
        # Extract imported names
        imported = []
        for child in node.children:
            if child.type == 'dotted_name' or child.type == 'aliased_import':
                imported.append(source[child.start_byte:child.end_byte].decode('utf-8'))
        
        name = ', '.join(imported) if imported else 'unknown'
        
        return CodeEntity(
            name=name,
            type='import',
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code
        )
    
    def _extract_docstring(self, node: Node, source: bytes) -> Optional[str]:
        """Extract docstring from function or class."""
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr in stmt.children:
                            if expr.type == 'string':
                                return source[expr.start_byte:expr.end_byte].decode('utf-8').strip('"\'')
        return None
    
    def _extract_calls(self, node: Node, source: bytes) -> List[str]:
        """Extract function calls within a node."""
        calls = []
        
        def traverse(n: Node):
            if n.type == 'call':
                func_node = n.child_by_field_name('function')
                if func_node:
                    if func_node.type == 'identifier':
                        calls.append(source[func_node.start_byte:func_node.end_byte].decode('utf-8'))
                    elif func_node.type == 'attribute':
                        # method call like obj.method()
                        obj = func_node.child_by_field_name('object')
                        method = func_node.child_by_field_name('attribute')
                        if obj and method:
                            obj_name = source[obj.start_byte:obj.end_byte].decode('utf-8')
                            method_name = source[method.start_byte:method.end_byte].decode('utf-8')
                            calls.append(f"{obj_name}.{method_name}")
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(calls))  # Remove duplicates
