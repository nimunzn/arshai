"""
Optional utilities for creating memory management components.

These are convenience functions for developers who want simplified memory creation.
Direct instantiation is always preferred, but these utilities can help in some cases.
"""

from typing import Dict, Any, Type
from arshai.core.interfaces.imemorymanager import IMemoryManager
from ..memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
from ..memory.working_memory.in_memory_manager import InMemoryManager


# Registry of working memory providers
_WORKING_MEMORY_PROVIDERS = {
    "redis": RedisWorkingMemoryManager,
    "in_memory": InMemoryManager,
}


def register_memory_provider(name: str, provider_class: Type[IMemoryManager]) -> None:
    """
    Register a new working memory provider for use with create_memory_manager().
    
    Args:
        name: Provider name
        provider_class: Class implementing IMemoryManager
    """
    _WORKING_MEMORY_PROVIDERS[name.lower()] = provider_class


def create_memory_manager(provider: str, **kwargs) -> IMemoryManager:
    """
    Optional utility to create a working memory manager.
    
    Note: Direct instantiation is preferred (e.g., RedisMemoryManager(redis_url="...")).
    Use this utility only when you need dynamic provider selection.
    
    Args:
        provider: Provider type (e.g., "redis", "in_memory")
        **kwargs: Provider-specific configuration (non-sensitive configuration only)
        
    Returns:
        An instance implementing IMemoryManager
        
    Raises:
        ValueError: If provider is not supported
        
    Example:
        # ✅ PREFERRED: Direct instantiation
        from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
        
        memory = RedisMemoryManager(
            redis_url="redis://localhost:6379",
            ttl=3600
        )
        
        # ⚠️ OPTIONAL: Utility function for dynamic selection
        memory = create_memory_manager("redis", redis_url="redis://localhost:6379", ttl=3600)
    """
    provider = provider.lower()
    
    if provider not in _WORKING_MEMORY_PROVIDERS:
        raise ValueError(
            f"Unsupported working memory provider: {provider}. "
            f"Supported providers: {', '.join(_WORKING_MEMORY_PROVIDERS.keys())}"
        )
    
    provider_class = _WORKING_MEMORY_PROVIDERS[provider]
    
    # Create the provider instance with the provided configuration
    # Sensitive data like storage_url should be read from environment variables
    # in the implementation itself
    return provider_class(**kwargs)


def create_memory_manager_service(config: Dict[str, Any]):
    """
    Optional utility to create a memory manager service.
    
    Note: Consider direct instantiation for better control.
    
    Args:
        config: Configuration dictionary with memory settings
        
    Returns:
        MemoryManagerService instance
        
    Example:
        # ✅ PREFERRED: Direct instantiation
        from arshai.memory.memory_manager import MemoryManagerService
        service = MemoryManagerService(config)
        
        # ⚠️ OPTIONAL: Utility function
        service = create_memory_manager_service(config)
    """
    # Import here to avoid circular imports
    from ..memory.memory_manager import MemoryManagerService
    return MemoryManagerService(config)


def get_available_memory_providers() -> Dict[str, Type[IMemoryManager]]:
    """
    Get all registered memory providers.
    
    Returns:
        Dictionary mapping provider names to their classes
    """
    return _WORKING_MEMORY_PROVIDERS.copy()