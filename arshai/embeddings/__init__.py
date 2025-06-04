"""
Embedding services for vector representations.

This module provides embedding capabilities for the Arshai framework,
converting text into vector representations for semantic search and retrieval.
"""

from .openai_embeddings import OpenAIEmbedding
from .mgte_embeddings import MGTEEmbedding

__all__ = ["OpenAIEmbedding", "MGTEEmbedding"] 