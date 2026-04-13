"""Hashing utilities for file change detection."""

import hashlib
from pathlib import Path


def file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (sha256, md5, etc.)
        
    Returns:
        Hex string of the file hash
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def content_hash(content: str, algorithm: str = "sha256") -> str:
    """Calculate hash of a string content.
    
    Args:
        content: String content to hash
        algorithm: Hash algorithm
        
    Returns:
        Hex string of the content hash
    """
    hash_func = hashlib.new(algorithm)
    hash_func.update(content.encode('utf-8'))
    return hash_func.hexdigest()
