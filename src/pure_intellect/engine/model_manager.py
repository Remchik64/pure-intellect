"""Управление моделями: скачивание, кэширование, загрузка."""

import logging
import threading
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download

from .registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelManager:
    """Управление моделями: скачивание, кэширование, загрузка.
    
    Thread-safe singleton с явным lifecycle управлением.
    Вызывайте dispose() перед загрузкой новой модели во избежание утечки VRAM.
    """
    
    _instance: Optional['ModelManager'] = None
    _instance_lock: threading.Lock = threading.Lock()
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_model = None
        self._loaded_key: Optional[str] = None
        self._model_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, cache_dir: str = "./models") -> 'ModelManager':
        """Получить singleton экземпляр ModelManager (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:  # double-checked locking
                    cls._instance = cls(cache_dir=cache_dir)
        return cls._instance
    
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
    
    def dispose(self) -> None:
        """Освободить текущую загруженную модель и VRAM.
        
        Вызывайте перед загрузкой новой модели или при завершении работы.
        """
        with self._model_lock:
            if self.loaded_model is not None:
                logger.info(f"Disposing model: {self._loaded_key}")
                try:
                    # llama-cpp освобождает VRAM при удалении объекта
                    del self.loaded_model
                except Exception as e:
                    logger.warning(f"Error during model disposal: {e}")
                finally:
                    self.loaded_model = None
                    self._loaded_key = None
                    logger.info("Model disposed, VRAM released")
    
    def load(self, model_key: str, n_gpu_layers: int = -1) -> 'Llama':
        """Загрузить модель в память.
        
        Если другая модель уже загружена — она будет выгружена автоматически.
        """
        from llama_cpp import Llama
        
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}")
        
        with self._model_lock:
            # Если запрошена та же модель — возвращаем без перезагрузки
            if self._loaded_key == model_key and self.loaded_model is not None:
                logger.info(f"Model already loaded: {model_key}")
                return self.loaded_model
            
            # Выгрузить предыдущую модель если есть (предотвращает утечку VRAM)
            if self.loaded_model is not None:
                logger.info(f"Unloading previous model: {self._loaded_key}")
                try:
                    del self.loaded_model
                except Exception as e:
                    logger.warning(f"Error unloading previous model: {e}")
                self.loaded_model = None
                self._loaded_key = None
            
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
                verbose=True,
            )
            self._loaded_key = model_key
            
            logger.info(f"Model loaded: {info['name']}")
            return self.loaded_model
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Отправить сообщение модели."""
        if self.loaded_model is None:
            raise RuntimeError("No model loaded. Call load() first.")
        
        response = self.loaded_model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]
    
    def is_loaded(self) -> bool:
        """Проверить загружена ли модель."""
        return self.loaded_model is not None
    
    def loaded_model_key(self) -> Optional[str]:
        """Вернуть ключ загруженной модели."""
        return self._loaded_key
