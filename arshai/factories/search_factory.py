"""
Factory for creating search component implementations.
"""

from typing import Dict, Any, Type, Optional
from arshai.core.interfaces.iwebsearch import IWebSearchClient
from arshai.core.interfaces.ireranker import IReranker
from arshai.web_search.searxng import SearxNGClient
from arshai.core.interfaces.isetting import ISetting
from arshai.factories.reranker_factory import RerankerFactory
import logging

logger = logging.getLogger(__name__)

class WebSearchFactory:
    """Factory for creating search engine instances."""
    
    # Registry of search engine providers
    _providers: Dict[str, Type[IWebSearchClient]] = {
        "searxng": SearxNGClient,  # SearxNG implementation
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[IWebSearchClient]) -> None:
        """
        Register a new search engine provider.
        
        Args:
            name: Provider name
            provider_class: Class implementing search functionality
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def _create_reranker(cls, config: Dict[str, Any], settings: ISetting) -> Optional[IReranker]:
        """
        Create a reranker from configuration if specified.
        
        Args:
            config: Search configuration dictionary
            settings: Settings instance
            
        Returns:
            Optional[IReranker]: Reranker instance or None if not configured
        """
        # Check if reranking is disabled in config
        if config.get('use_reranker') is False:
            return None
            
        # Try to get reranker from settings
        reranker = settings.create_reranker()
        if reranker:
            logger.info("Created reranker from settings")
            return reranker
            
        # Try to use reranker config from search config
        reranker_config = config.get('reranker')
        if reranker_config and isinstance(reranker_config, dict):
            provider = reranker_config.get('provider')
            if provider:
                try:
                    reranker = RerankerFactory.create(provider, reranker_config)
                    logger.info(f"Created reranker with provider {provider} from search config")
                    return reranker
                except Exception as e:
                    logger.error(f"Error creating reranker from search config: {str(e)}")
        
        return None
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any], settings: Optional[ISetting] = None) -> IWebSearchClient:
        """
        Create a search engine instance based on provider type.
        
        Args:
            provider: Provider type (e.g., "searxng")
            config: Provider-specific configuration
            settings: Optional Settings instance
            
        Returns:
            An instance implementing IWebSearchClient
            
        Raises:
            ValueError: If provider is not supported
        """
        # Default to searxng if not specified
        provider = provider.lower() if provider else "searxng"
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported search provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider]
        
        # Create search provider with config and reranker
        search_client = provider_class(config=config)
        
        
        return search_client 