"""Управление моделями: скачивание, кэширование, загрузка."""

import os
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, scan_cache_dir

from .registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelManager:
    """Управление моделями: скачивание, кэширование, загрузка."""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_model = None
    
    def list_available(self) -> dict:
        """Список всех доступных моделей."""
        return MODEL_REGISTRY
    
    def list_downloaded(self) -> list[str]:
        """Список скачанных моделей."""
        downloaded = []
        for key, info in MODEL_REGISTRY.items():
            model_path = self.cache_dir / info["file"]
            if model_path.exists():
                downloaded.append(key)
        return downloaded
    
    def download(self, model_key: str, force: bool = False) -> str:
        """Скачать модель с HuggingFace."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
        
        info = MODEL_REGISTRY[model_key]
        logger.info(f"Downloading {info['name']} ({info['size_gb']} GB)...")
        
        # Проверяем, существует ли файл уже
        model_path = self.cache_dir / info["file"]
        if model_path.exists() and not force:
            logger.info(f"Model already exists: {model_path}")
            return str(model_path)
        
        path = hf_hub_download(
            repo_id=info["repo"],
            filename=info["file"],
            cache_dir=str(self.cache_dir),
            force_download=force,
        )
        
        logger.info(f"Model ready: {path}")
        return path
    
    def load(self, model_key: str, n_gpu_layers: int = -1) -> 'Llama':
        """Загрузить модель в память."""
        from llama_cpp import Llama
        
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}")
        
        info = MODEL_REGISTRY[model_key]
        
        # Скачать если нет
        model_path = self.cache_dir / info["file"]
        if not model_path.exists():
            model_path = Path(self.download(model_key))
        
        logger.info(f"Loading {info['name']} with {n_gpu_layers} GPU layers...")
        
        self.loaded_model = Llama(
            model_path=str(model_path),
            n_ctx=info["context"],
            n_gpu_layers=n_gpu_layers,
            n_batch=512,
            verbose=True,  # видеть GPU detection
        )
        
        logger.info(f"Model loaded: {info['name']}")
        return self.loaded_model
    
    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """Отправить сообщение модели."""
        if self.loaded_model is None:
            raise RuntimeError("No model loaded. Call load() first.")
        
        response = self.loaded_model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        return response["choices"][0]["message"]["content"]
