"""
Factory for creating Language Model (LLM) instances.
"""

from typing import Optional, Dict, Any, Type
from arshai.core.interfaces.illm import ILLM, ILLMConfig
from ..llms.openai import OpenAIClient
from ..llms.azure import AzureClient
from ..llms.openrouter import OpenRouterClient

class LLMFactory:
    """Factory for creating LLM instances."""
    
    # Registry of available LLM providers
    _providers = {
        "openai": OpenAIClient,
        "azure": AzureClient,
        "openrouter": OpenRouterClient,
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