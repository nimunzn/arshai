"""
Optional utilities for creating embedding components.

These are convenience functions for developers who want simplified embedding creation.
Direct instantiation is always preferred, but these utilities can help in some cases.
"""

from typing import Dict, Any, Type
from arshai.core.interfaces.iembedding import IEmbedding, EmbeddingConfig
from ..embeddings.openai_embeddings import OpenAIEmbedding
from ..embeddings.mgte_embeddings import MGTEEmbedding
from ..embeddings.voyageai_embedding import VoyageAIEmbedding


# Registry of embedding providers
_EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbedding,
    "mgte": MGTEEmbedding,
    "voyage": VoyageAIEmbedding
}


def register_embedding_provider(name: str, provider_class: Type[IEmbedding]) -> None:
    """
    Register a new embedding provider for use with create_embedding().
    
    Args:
        name: Provider name
        provider_class: Class implementing IEmbedding
    """
    _EMBEDDING_PROVIDERS[name.lower()] = provider_class


def create_embedding(provider: str, config: Dict[str, Any]) -> IEmbedding:
    """
    Optional utility to create an embedding instance.
    
    Note: Direct instantiation is preferred (e.g., OpenAIEmbedding(config)).
    Use this utility only when you need dynamic provider selection.
    
    Args:
        provider: Provider type (e.g., "openai")
        config: Provider-specific non-sensitive configuration
        
    Returns:
        An instance implementing IEmbedding
        
    Raises:
        ValueError: If provider is not supported
        
    Example:
        # ✅ PREFERRED: Direct instantiation
        from arshai.embeddings.openai_embeddings import OpenAIEmbedding
        from arshai.core.interfaces.iembedding import EmbeddingConfig
        
        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            batch_size=16
        )
        embedding = OpenAIEmbedding(config)
        
        # ⚠️ OPTIONAL: Utility function for dynamic selection
        embedding = create_embedding("openai", {
            "model_name": "text-embedding-3-small",
            "batch_size": 16
        })
    """
    provider = provider.lower()
    
    if provider not in _EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: {', '.join(_EMBEDDING_PROVIDERS.keys())}"
        )

    embedding_config = EmbeddingConfig(
        model_name=config.get("model_name"),
        additional_params=config.copy(),
        batch_size=config.get("batch_size", 16)
    )

    provider_class = _EMBEDDING_PROVIDERS[provider]
    # Each embedding implementation reads its own API keys/sensitive data
    # from environment variables
    return provider_class(embedding_config)


def get_available_embedding_providers() -> Dict[str, Type[IEmbedding]]:
    """
    Get all registered embedding providers.
    
    Returns:
        Dictionary mapping provider names to their classes
    """
    return _EMBEDDING_PROVIDERS.copy()