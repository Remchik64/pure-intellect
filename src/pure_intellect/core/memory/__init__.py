"""Memory subsystem для Pure Intellect.

Иерархическая самообновляемая память:
- Fact: атом памяти с lifecycle
- WorkingMemory: рабочий буфер (всегда маленький, чистый)
- MemoryStorage: long-term хранилище (безграничное)
- AttentionScorer: оценка важности фактов по тексту разговора
- MemoryOptimizer: фоновая оптимизация (promote/compress/archive)
- CCITracker: Context Coherence Index (отслеживание связности контекста)
- ImportanceTagger: LLM-based классификация важности фактов (P3)
- MetaCoordinator: управление ростом координат (R1 roadmap)
"""

from .fact import Fact, CompressionLevel
from .working_memory import WorkingMemory
from .storage import MemoryStorage
from .scorer import AttentionScorer, ScoreResult
from .optimizer import MemoryOptimizer, OptimizationStats
from .cci import CCITracker, CoherenceEntry, CoherenceResult
from .tagger import ImportanceTagger, TaggingResult
from .meta_coordinator import MetaCoordinator, CoordinateRecord

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
    "ImportanceTagger",
    "TaggingResult",
    "MetaCoordinator",
    "CoordinateRecord",
]
