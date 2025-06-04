from typing import Protocol, List, Dict, Any, Optional
from pydantic import Field
from .idto import IDTO

class EmbeddingConfig(IDTO):
    """Configuration for embedding services."""
    model_name: Optional[str] = Field(default=None, description="Name of the embedding model to use")
    batch_size: int = Field(default=16, description="Batch size for processing documents")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model-specific parameters")

class IEmbedding(Protocol):
    """
    Interface defining the contract for embedding services.
    Any embedding service implementation must conform to this interface.
    """
    
    @property
    def dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            int: Dimension of the embedding vectors
        """
        ...
    
    def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            Dictionary containing embeddings with keys for different vector types (e.g., 'dense', 'sparse')
        """
        ...
        
    def embed_document(self, text: str) -> Dict[str, Any]:
        """
        Generate embeddings for a single document.
        
        Args:
            text: Text document to embed
            
        Returns:
            Dictionary containing embeddings with keys for different vector types (e.g., 'dense', 'sparse')
        """
        ... 