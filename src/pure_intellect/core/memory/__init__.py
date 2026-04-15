"""Memory subsystem для Pure Intellect.

Иерархическая самообновляемая память:
- Fact: атом памяти с lifecycle
- WorkingMemory: рабочий буфер (всегда маленький, чистый)
- MemoryStorage: long-term хранилище (безграничное)
"""

from .fact import Fact, CompressionLevel
from .working_memory import WorkingMemory
from .storage import MemoryStorage

__all__ = [
    "Fact",
    "CompressionLevel",
    "WorkingMemory",
    "MemoryStorage",
]
