"""
Factory for creating reranker components.
"""

from typing import Dict, Any, Type
from arshai.core.interfaces.ireranker import IReranker
from ..rerankers.flashrank_reranker import FlashRankReranker
from ..rerankers.voyage_reranker import VoyageReranker

class RerankerFactory:
    """Factory for creating reranker instances."""
    
    # Registry of reranker providers
    _providers = {
        "flashrank": FlashRankReranker,
        "voyage": VoyageReranker,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[IReranker]) -> None:
        """
        Register a new reranker provider.
        
        Args:
            name: Provider name
            provider_class: Class implementing IReranker
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> IReranker:
        """
        Create a reranker instance based on provider type.
        
        Args:
            provider: Provider type (e.g., "flashrank", "voyage")
            config: Provider-specific non-sensitive configuration
            
        Returns:
            An instance implementing IReranker
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported reranker provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider]
        
        # Handle provider-specific configuration - only pass structural configuration
        if provider == "flashrank":
            return provider_class(
                model_name=config.get("model_name", "rank-T5-flan"),
                top_k=config.get("top_k")
            )
        elif provider == "voyage":
            return provider_class(
                model_name=config.get("model_name", "rerank-2"),
                top_k=config.get("top_k")
            )
        
        # Default case for future providers
        return provider_class(**config) 