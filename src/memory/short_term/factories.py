from typing import Dict, Any
from arshai.core.interfaces import IMemoryManager

class ShortTermMemoryFactory:
    """Factory for creating short-term memory manager instances"""
    
    @staticmethod
    def create_short_term_manager(provider: str, storage_url: str, **kwargs) -> IMemoryManager:
        """
        Create and return a short-term memory manager instance.

        Args:
            provider: Type of storage provider ("redis", "mongodb", etc.)
            storage_url: URL for the storage backend
            **kwargs: Additional provider-specific configuration

        Returns:
            An instance implementing IMemoryManager

        Raises:
            ValueError: If the provider type is not supported
        """
        # Will be implemented when we add concrete implementations
        raise NotImplementedError("No short-term memory providers implemented yet") 