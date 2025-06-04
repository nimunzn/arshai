from typing import Dict, List, Optional, Any, Type, Callable
from pydantic import Field
from .idto import IDTO
from .idatabase_client import IDatabaseClient


class IVectorDBConfig(IDTO):
    """
    Base configuration for vector database connections.
    This is a generalized configuration that can be extended for specific vector databases.
    """
    host: str = Field(description="Database host address")
    port: str = Field(description="Database port")
    db_name: str = Field(description="Database name")
    batch_size: int = Field(default=50, description="Batch size for bulk operations")
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional database-specific parameters")

class ICollectionConfig(IDTO):
    """
    Base configuration for vector database collections.
    This is a generalized configuration that can be extended for specific vector databases.
    """
    collection_name: str = Field(description="Name of the collection")
    dense_dim: int = Field(description="Dimension of dense vector embeddings")
    text_field: str = Field(description="Field name for text content")
    pk_field: str = Field(default="doc_id", description="Primary key field name")
    dense_field: str = Field(default="dense_vector", description="Field name for dense vector data")
    sparse_field: Optional[str] = Field(default="sparse_vector", description="Field name for sparse vector data")
    metadata_field: str = Field(default="metadata", description="Field name for metadata")
    schema_model: Optional[Type[IDTO]] = Field(default=None, description="Optional Pydantic model for schema validation")
    is_hybrid: bool = Field(default=False, description="Whether to enable hybrid search capabilities")

class IVectorDBClient(IDatabaseClient):
    """
    Interface for vector database clients.
    Extends the base database client interface with vector-specific operations.
    """
    
    def connect(self) -> None:
        """
        Establish connection to the vector database.
        """
        ...
    
    def disconnect(self) -> None:
        """
        Close the vector database connection.
        """
        ...
    
    def get_or_create_collection(self, config: ICollectionConfig) -> Any:
        """
        Get an existing collection or create a new one if it doesn't exist.
        
        Args:
            config: Collection configuration
            
        Returns:
            Collection object
        """
        ...
    
    def get_collection_stats(self, config: ICollectionConfig) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            config: Collection configuration
            
        Returns:
            Dictionary containing collection statistics
        """
        ...
    
    def insert_entity(
        self,
        config: ICollectionConfig,
        entity: Dict[str, Any],
        documents_embedding: Dict[str, Any]
    ) -> None:
        """
        Insert a single entity with embeddings into collection.
        
        Args:
            config: Collection configuration
            entity: Dictionary containing document content and metadata
            documents_embedding: Dictionary containing dense and/or sparse embeddings
        """
        ...
    
    def insert_entities(
        self,
        config: ICollectionConfig,
        data: List[Dict[str, Any]],
        documents_embedding: Dict[str, Any]
    ) -> None:
        """
        Insert multiple entities with embeddings into collection.
        
        Args:
            config: Collection configuration
            data: List of dictionaries containing document content and metadata
            documents_embedding: Dictionary containing dense and/or sparse embeddings
        """
        ...
    
    def search_by_vector(
        self,
        config: ICollectionConfig,
        query_vectors: List[List[float]],
        search_field: Optional[str] = None,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            config: Collection configuration
            query_vectors: Query vectors to search for
            search_field: Optional field to search in (defaults to dense_field)
            expr: Optional filter expression
            output_fields: Optional list of fields to return
            limit: Maximum number of results to return
            search_params: Optional search parameters
            
        Returns:
            List of search results with distances and metadata
        """
        ...
    
    def hybrid_search(
        self,
        config: ICollectionConfig,
        dense_vectors: Optional[List[List[float]]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        weights: Optional[List[float]] = None,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both dense and sparse vectors.
        
        Args:
            config: Collection configuration
            dense_vectors: Dense vectors for search
            sparse_vectors: Sparse vectors for search
            weights: List of weights for dense and sparse vectors [dense_weight, sparse_weight]
            expr: Optional filter expression
            output_fields: Optional list of fields to return
            limit: Maximum number of results to return
            search_params: Optional search parameters
            
        Returns:
            List of search results with distances and metadata
        """
        ...
    
    def delete_entity(
        self,
        config: ICollectionConfig,
        filter_expr: str
    ) -> Any:
        """
        Delete entities from collection based on filter expression.
        
        Args:
            config: Collection configuration
            filter_expr: Filter expression to identify entities to delete
            
        Returns:
            Delete operation result
        """
        ... 