"""Утилиты для хеширования файлов."""

import hashlib
from pathlib import Path


def file_sha256(file_path: Path | str) -> str:
    """Вычислить SHA256 хэш файла."""
    path = Path(file_path)
    if not path.exists():
        return ""
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def content_sha256(content: str) -> str:
    """Вычислить SHA256 хэш строки."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunked_hash(file_path: Path | str, chunk_size: int = 8192) -> str:
    """Хэширование файла блоками (для больших файлов)."""
    path = Path(file_path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
