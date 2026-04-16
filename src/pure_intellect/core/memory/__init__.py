"""Memory subsystem для Pure Intellect.

Иерархическая самообновляемая память:
- Fact: атом памяти с lifecycle
- WorkingMemory: рабочий буфер (всегда маленький, чистый)
- MemoryStorage: long-term хранилище (безграничное)
- AttentionScorer: оценка важности фактов по тексту разговора
- MemoryOptimizer: фоновая оптимизация (promote/compress/archive)
- CCITracker: Context Coherence Index (отслеживание связности контекста)
"""

from .fact import Fact, CompressionLevel
from .working_memory import WorkingMemory
from .storage import MemoryStorage
from .scorer import AttentionScorer, ScoreResult
from .optimizer import MemoryOptimizer, OptimizationStats
from .cci import CCITracker, CoherenceEntry, CoherenceResult

__all__ = [
    "Fact",
    "CompressionLevel",
    "WorkingMemory",
    "MemoryStorage",
    "AttentionScorer",
    "ScoreResult",
    "MemoryOptimizer",
    "OptimizationStats",
    "CCITracker",
    "CoherenceEntry",
    "CoherenceResult",
]
