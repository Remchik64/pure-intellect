"""Структурированное логирование."""

import logging
import sys
from pure_intellect.config import settings


def get_logger(name: str) -> logging.Logger:
    """Получить настроенный логгер."""
    logger = logging.getLogger(f"pure_intellect.{name}")
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        if settings.log_format == "json":
            formatter = logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    return logger
