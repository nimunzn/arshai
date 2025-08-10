"""
Factory for creating Language Model (LLM) instances.
"""

from typing import Optional, Dict, Any, Type, Union
from pathlib import Path
from arshai.core.interfaces import ILLM, ILLMConfig
from ..llms.openai import OpenAIClient
from ..llms.azure import AzureClient

# Import observability components
try:
    from arshai.observability import ObservabilityConfig, ObservableFactory
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

class LLMFactory:
    """Factory for creating LLM instances."""
    
    # Registry of available LLM providers
    _providers = {
        "openai": OpenAIClient,
        "azure": AzureClient,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[ILLM]) -> None:
        """
        Register a new LLM provider.
        
        Args:
            name: Name of the provider
            provider_class: Class implementing the ILLM interface
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(
        cls,
        provider: str,
        config: ILLMConfig,
        **kwargs
    ) -> ILLM:
        """
        Create an LLM provider instance based on the specified provider type.
        
        Args:
            provider: The provider type (e.g., 'openai', 'azure')
            config: Configuration for the LLM
            **kwargs: Additional non-sensitive configuration parameters
            
        Returns:
            An instance of the specified LLM provider
            
        Raises:
            ValueError: If provider is not supported or required parameters are missing
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        
        if not isinstance(config, ILLMConfig):
            raise ValueError("config must be an instance of ILLMConfig")
        
        provider_class = cls._providers[provider]
        
        # All LLM implementations read their sensitive data from environment variables
        return provider_class(config)
    
    @classmethod
    def create_with_observability(
        cls,
        provider: str,
        config: ILLMConfig,
        observability_config: Optional[ObservabilityConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> ILLM:
        """
        Create an LLM provider instance with automatic observability.
        
        Args:
            provider: The provider type (e.g., 'openai', 'azure')
            config: Configuration for the LLM
            observability_config: Optional observability configuration
            config_path: Optional path to configuration file
            **kwargs: Additional non-sensitive configuration parameters
            
        Returns:
            An instrumented instance of the specified LLM provider
            
        Raises:
            ValueError: If provider is not supported or required parameters are missing
            ImportError: If observability components are not available
        """
        if not OBSERVABILITY_AVAILABLE:
            raise ImportError(
                "Observability components not available. "
                "Please install observability dependencies or use create() method instead."
            )
        
        # Create observable factory
        observable_factory = ObservableFactory(
            cls,
            observability_config=observability_config,
            config_path=config_path
        )
        
        return observable_factory.create(provider, config, **kwargs)
    
    @classmethod
    def get_observable_factory(
        cls,
        observability_config: Optional[ObservabilityConfig] = None,
        config_path: Optional[Union[str, Path]] = None
    ) -> 'ObservableFactory':
        """
        Get an observable version of this factory.
        
        Args:
            observability_config: Optional observability configuration
            config_path: Optional path to configuration file
            
        Returns:
            ObservableFactory instance
            
        Raises:
            ImportError: If observability components are not available
        """
        if not OBSERVABILITY_AVAILABLE:
            raise ImportError(
                "Observability components not available. "
                "Please install observability dependencies."
            )
        
        return ObservableFactory(
            cls,
            observability_config=observability_config,
            config_path=config_path
        ) 