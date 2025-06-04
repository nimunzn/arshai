"""
OpenAI embeddings implementation.
"""

import logging
from typing import List, Dict, Any, Optional
import os
from arshai.core.interfaces.iembedding import EmbeddingConfig, IEmbedding

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


logger = logging.getLogger(__name__)

class OpenAIEmbedding(IEmbedding):
    """
    OpenAI embedding implementation using their API.
    
    This class provides access to OpenAI's text embedding models.
    """
    
    _EMBEDDING_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the OpenAI embedding service.
        
        Args:
            config: Configuration for the embedding service
        """
        if not HAS_OPENAI:
            raise ImportError(
                "The openai package is required to use OpenAIEmbedding. "
                "Please install it with 'pip install openai'."
            )
        
        self.model_name = config.model_name
        self.batch_size = config.batch_size
        
        # Set dimension based on model if not provided
        self._dimension = self._EMBEDDING_DIMENSIONS.get(self.model_name, 1536)
        
        # Get API key directly from environment variables
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("EMBEDDINGS_OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set either OPENAI_API_KEY or EMBEDDINGS_OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.logger = logger
        
        logger.info(
            f"Initialized OpenAI embedding function with model: {self.model_name}, "
            f"dimension: {self._dimension}, batch_size: {self.batch_size}"
        )
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        return self._dimension
    
    def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for multiple documents using OpenAI API.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            Dictionary containing embeddings with 'dense' vectors
        """
        if not texts:
            return {"dense": []}
        
        # Process in batches to avoid API limitations
        dense_vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                batch_embeddings = [data.embedding for data in response.data]
                dense_vectors.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
                raise e
        
        return {"dense": dense_vectors}
    
    def embed_document(self, text: str) -> Dict[str, Any]:
        """
        Generate embeddings for a single document.
        
        Args:
            text: Text document to embed
            
        Returns:
            Dictionary containing embeddings with 'dense' vectors
        """
        embeddings = self.embed_documents([text])
        return {"dense": embeddings["dense"][0]}
    
    async def aembed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        Asynchronously generate embeddings for multiple documents using OpenAI API.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            Dictionary containing embeddings with 'dense' vectors
        """
        if not texts:
            return {"dense": []}
        
        # Process in batches to avoid API limitations
        dense_vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = await self.async_client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                batch_embeddings = [data.embedding for data in response.data]
                dense_vectors.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
                raise e
        
        return {"dense": dense_vectors} 