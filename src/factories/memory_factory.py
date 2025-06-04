"""
Factory for creating memory management components.
"""

from typing import Dict, Any, Type, Optional
from arshai.core.interfaces import IMemoryManager
from ..memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
from ..memory.working_memory.in_memory_manager import InMemoryManager

class MemoryFactory:
    """Factory for creating memory management components."""
    
    # Registry of working memory providers
    _working_memory_providers = {
        "redis": RedisWorkingMemoryManager,
        "in_memory": InMemoryManager,
    }
    
    @classmethod
    def register_working_memory_provider(cls, name: str, provider_class: Type[IMemoryManager]) -> None:
        """
        Register a new working memory provider.
        
        Args:
            name: Provider name
            provider_class: Class implementing IMemoryManager
        """
        cls._working_memory_providers[name.lower()] = provider_class
    
    @classmethod
    def create_working_memory(cls, provider: str, **kwargs) -> IMemoryManager:
        """
        Create a working memory manager based on provider type.
        
        Args:
            provider: Provider type (e.g., "redis", "in_memory")
            **kwargs: Provider-specific configuration (non-sensitive configuration only)
            
        Returns:
            An instance implementing IMemoryManager
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._working_memory_providers:
            raise ValueError(
                f"Unsupported working memory provider: {provider}. "
                f"Supported providers: {', '.join(cls._working_memory_providers.keys())}"
            )
        
        provider_class = cls._working_memory_providers[provider]
        
        # Create the provider instance with the provided configuration
        # Sensitive data like storage_url should be read from environment variables
        # in the implementation itself
        return provider_class(**kwargs)
    
    @classmethod
    def create_memory_manager_service(cls, config: Dict[str, Any]):
        """
        Create a memory manager service with the given configuration.
        
        Args:
            config: Configuration dictionary with memory settings
            
        Returns:
            MemoryManagerService instance
        """
        # Import here to avoid circular imports
        from ..memory.memory_manager import MemoryManagerService
        return MemoryManagerService(config) 