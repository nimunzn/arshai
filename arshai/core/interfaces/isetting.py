from typing import Protocol, Any, Optional, Dict, TypeVar
from ..interfaces.illm import ILLM
from ..interfaces.imemorymanager import IMemoryManager
from ..interfaces.ireranker import IReranker
from ..interfaces.ivector_db_client import IVectorDBClient

T = TypeVar('T')

class ISetting(Protocol):
    """
    Generic interface for application settings
    
    This protocol defines the minimum interface that any settings implementation
    must provide, representing the actual usage in the codebase.
    """
    
    # Core Component Properties
    @property
    def llm_model(self) -> ILLM:
        """Get the LLM model configured for the application"""
        ...
    
    @property
    def context_manager(self) -> Optional[IMemoryManager]:
        """Get the memory/context management component if configured"""
        ...
    
    @property
    def reranker(self) -> Optional[IReranker]:
        """Get the reranker component if configured"""
        ...
    
    @property
    def embedding_model(self) -> Any:
        """Get the embedding model configured for the application"""
        ...
    
    @property
    def vector_db(self) -> Optional[IVectorDBClient]:
        """Get the vector database client if configured"""
        ...
    
    # Configuration Properties
    @property
    def chatbot_context(self) -> str:
        """Get the context/instructions for the chatbot"""
        ...
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL from environment or default"""
        ...
    
    # Configuration methods
    def get_value(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get a configuration value by key with optional default
        
        Args:
            key: The configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        ...
    
    def load_from_path(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a specified path
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Dict containing the loaded configuration
        """
        ...
    
    def model_dump(self) -> Dict[str, Any]:
        """
        Convert the settings object to a serializable dictionary
        
        This method is used to allow serialization of the settings object
        for passing configuration between components or services.
        
        Returns:
            Dict containing the serialized settings
        """
        ...
    
    # Helper methods for creating components
    def _create_context_manager(self) -> IMemoryManager:
        """Create a context manager instance"""
        ...
    
    def _create_llm_model(self) -> ILLM:
        """Create an LLM model instance"""
        ...
    
    def _create_embedding_model(self) -> Any:
        """Create an embedding model instance"""
        ...
    
    def _create_reranker(self) -> Optional[IReranker]:
        """Create a reranker instance"""
        ...
    
    def _create_vector_db(self) -> Optional[IVectorDBClient]:
        """Create a vector database client instance"""
        ...

