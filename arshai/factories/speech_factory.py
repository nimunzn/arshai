"""
Factory for creating speech processing service instances.
"""

from typing import Dict, Any, Type
from arshai.core.interfaces.ispeech import ISpeechProcessor, ISpeechConfig
from arshai.speech.openai import OpenAISpeechClient
from ..speech.azure import AzureSpeechClient

class SpeechFactory:
    """Factory for creating speech processor instances."""
    
    # Registry of available speech processor providers
    _providers = {
        "openai": OpenAISpeechClient,
        "azure": AzureSpeechClient,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[ISpeechProcessor]) -> None:
        """
        Register a new speech processor provider.
        
        Args:
            name: Name of the provider
            provider_class: Class implementing the ISpeechProcessor interface
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(
        cls,
        provider: str,
        config: ISpeechConfig,
        **kwargs
    ) -> ISpeechProcessor:
        """
        Create a speech processor instance based on the specified provider type.
        
        Args:
            provider: The provider type (e.g., 'openai', 'azure')
            config: Configuration for the speech processor
            **kwargs: Additional non-sensitive configuration parameters
            
        Returns:
            An instance of the specified speech processor provider
            
        Raises:
            ValueError: If provider is not supported or required parameters are missing
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported speech provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        
        # Create a copy of the config with the provider field set
        config_dict = config.model_dump()
        config_dict["provider"] = provider
        speech_config = ISpeechConfig(**config_dict)
        
        provider_class = cls._providers[provider]
        
        # All speech processor implementations read their sensitive data from environment variables
        return provider_class(speech_config) 