"""
Optional utilities for creating Language Model (LLM) instances.

These are convenience functions for developers who want simplified LLM creation.
Direct instantiation is always preferred, but these utilities can help in some cases.
"""

from typing import Optional, Dict, Any, Type
from arshai.core.interfaces.illm import ILLM, ILLMConfig
from ..llms.openai import OpenAIClient
from ..llms.azure import AzureClient
from ..llms.openrouter import OpenRouterClient
from ..llms.google_genai import GeminiClient


# Registry of available LLM providers
_LLM_PROVIDERS = {
    "openai": OpenAIClient,
    "azure": AzureClient,
    "openrouter": OpenRouterClient,
    "google": GeminiClient,
}


def register_llm_provider(name: str, provider_class: Type[ILLM]) -> None:
    """
    Register a new LLM provider for use with create_llm_client().
    
    Args:
        name: Name of the provider
        provider_class: Class implementing the ILLM interface
    """
    _LLM_PROVIDERS[name.lower()] = provider_class


def create_llm_client(
    provider: str,
    config: ILLMConfig,
    **kwargs
) -> ILLM:
    """
    Optional utility to create an LLM provider instance.
    
    Note: Direct instantiation is preferred (e.g., OpenAIClient(config)).
    Use this utility only when you need dynamic provider selection.
    
    Args:
        provider: The provider type (e.g., 'openai', 'azure')
        config: Configuration for the LLM
        **kwargs: Additional non-sensitive configuration parameters
        
    Returns:
        An instance of the specified LLM provider
        
    Raises:
        ValueError: If provider is not supported or required parameters are missing
        
    Example:
        # ✅ PREFERRED: Direct instantiation
        config = ILLMConfig(model="gpt-4o")
        client = OpenAIClient(config)
        
        # ⚠️ OPTIONAL: Utility function for dynamic selection
        client = create_llm_client("openai", config)
    """
    provider = provider.lower()
    
    if provider not in _LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(_LLM_PROVIDERS.keys())}"
        )
    
    if not isinstance(config, ILLMConfig):
        raise ValueError("config must be an instance of ILLMConfig")
    
    provider_class = _LLM_PROVIDERS[provider]
    
    # All LLM implementations read their sensitive data from environment variables
    return provider_class(config)


def get_available_providers() -> Dict[str, Type[ILLM]]:
    """
    Get all registered LLM providers.
    
    Returns:
        Dictionary mapping provider names to their classes
    """
    return _LLM_PROVIDERS.copy()