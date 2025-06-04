"""
Factory for creating vector database client components.
"""

import logging
from typing import Dict, Any, Type, Optional, Callable, Tuple
from arshai.core.interfaces.isetting import ISetting
from arshai.core.interfaces.ivector_db_client import IVectorDBClient, ICollectionConfig
from arshai.vector_db.milvus_client import MilvusClient

logger = logging.getLogger(__name__)

class VectorDBFactory:
    """Factory for creating vector database client instances."""
    
    # Registry of vector DB providers
    _providers = {
        "milvus": MilvusClient,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[IVectorDBClient]) -> None:
        """
        Register a new vector database provider.
        
        Args:
            name: Provider name
            provider_class: Class implementing IVectorDBClient
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(cls, embedding_dimension: int, settings: ISetting) -> Tuple[Optional[IVectorDBClient], Optional[ICollectionConfig]]:
        """
        Create a vector database client instance based on provider type.
        
        Args:
            config: Provider-specific configuration
            settings: Optional settings instance (for accessing embedding models)
            
        Returns:
            An instance implementing IVectorDBClient
            
        Raises:
            ValueError: If provider is not supported
        """
        collection_config = settings.get("db_collection")
        provider = settings.get("vector_db").get("provider").lower()
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unsupported vector database provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )


        # Special handling for different providers
        if provider.lower() == "milvus":
            try:
                
                dense_dim = embedding_dimension
                text_field = collection_config.get("text_field")
                collection_name = collection_config.get("collection_name")
                is_hybrid = collection_config.get("is_hybrid")

                collection_config = ICollectionConfig(dense_dim=dense_dim,
                                                            text_field=text_field,
                                                            collection_name=collection_name,
                                                            is_hybrid=is_hybrid)
                
                # Create and initialize the Milvus client
                vector_db_client = MilvusClient()
                
                
                logger.info(f"Created Milvus vector database client")

                return vector_db_client, collection_config
            
            except Exception as e:
                logger.error(f"Error creating Milvus client: {e}")
                raise ValueError(f"Error creating Milvus client")

        else:
            raise ValueError(f"Unsupported vector database provider: {provider}")