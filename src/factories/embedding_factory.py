"""
Factory for creating embedding components.
"""

from typing import Dict, Any, Type, Optional
from arshai.core.interfaces.iembedding import IEmbedding, EmbeddingConfig
from ..embeddings.openai_embeddings import OpenAIEmbedding
from ..embeddings.mgte_embeddings import MGTEEmbedding
from ..embeddings.voyageai_embedding import VoyageAIEmbedding

class EmbeddingFactory:
    """Factory for creating embedding instances."""
    
    # Registry of embedding providers
    _providers = {
        "openai": OpenAIEmbedding,
        "mgte": MGTEEmbedding,
        "voyage": VoyageAIEmbedding
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[IEmbedding]) -> None:
        """
        Register a new embedding provider.
        
        Args:
            name: Provider name
            provider_class: Class implementing IEmbedding
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> IEmbedding:
        """
        Create an embedding instance based on provider type.
        
        Args:
            provider: Provider type (e.g., "openai")
            config: Provider-specific non-sensitive configuration
            
        Returns:
            An instance implementing IEmbedding
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        


        embedding_config = EmbeddingConfig(model_name=config.get("model_name"),
                                           additional_params=config.copy(),
                                           batch_size=config.get("batch_size", 16))

        provider_class = cls._providers[provider]
        # Each embedding implementation reads its own API keys/sensitive data
        # from environment variables
        return provider_class(embedding_config) 