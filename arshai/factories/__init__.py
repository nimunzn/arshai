"""
Factory components for creating different services.

This module provides factory classes for creating various 
components in the Arshai framework, such as:
- LLMs
- Memory managers
- Retrievers
- Rerankers
- Agents
- Embeddings
- Vector databases
"""

# Core factories (always available)
from .llm_factory import LLMFactory
from .agent_factory import AgentFactory
from .embedding_factory import EmbeddingFactory

# Optional factories (depend on optional dependencies)
try:
    from .memory_factory import MemoryFactory
except ImportError:
    MemoryFactory = None

try:
    from .reranker_factory import RerankerFactory
except ImportError:
    RerankerFactory = None

try:
    from .vector_db_factory import VectorDBFactory
except ImportError:
    VectorDBFactory = None

try:
    from .search_factory import SearchFactory
except ImportError:
    SearchFactory = None

try:
    from .speech_factory import SpeechFactory
except ImportError:
    SpeechFactory = None

# Build __all__ dynamically
__all__ = ["LLMFactory", "AgentFactory", "EmbeddingFactory"]

if MemoryFactory:
    __all__.append("MemoryFactory")
if RerankerFactory:
    __all__.append("RerankerFactory")
if VectorDBFactory:
    __all__.append("VectorDBFactory")
if SearchFactory:
    __all__.append("SearchFactory")
if SpeechFactory:
    __all__.append("SpeechFactory") 