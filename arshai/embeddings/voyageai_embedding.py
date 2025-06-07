"""
OpenAI embeddings implementation.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import os
from arshai.core.interfaces.iembedding import EmbeddingConfig, IEmbedding
import voyageai
from PIL.Image import Image

logger = logging.getLogger(__name__)

class VoyageAIEmbedding(IEmbedding):
    """
    OpenAI embedding implementation using their API.
    
    This class provides access to OpenAI's text embedding models.
    """
    
    # Model dimension configurations based on Voyage AI documentation
    MODEL_DIMENSIONS = {
        # Models with flexible dimensions
        "voyage-3-large": {"default": 1024, "allowed": [256, 512, 1024, 2048]},
        "voyage-3.5": {"default": 1024, "allowed": [256, 512, 1024, 2048]},
        "voyage-3.5-lite": {"default": 1024, "allowed": [256, 512, 1024, 2048]},
        "voyage-code-3": {"default": 1024, "allowed": [256, 512, 1024, 2048]},
        
        # Models with fixed dimensions
        "voyage-finance-2": {"default": 1024, "allowed": [1024]},
        "voyage-law-2": {"default": 1024, "allowed": [1024]},
        "voyage-code-2": {"default": 1536, "allowed": [1536]},
        "voyage-3": {"default": 1024, "allowed": [1024]},
        "voyage-3-lite": {"default": 512, "allowed": [512]},
        "voyage-multilingual-2": {"default": 1024, "allowed": [1024]},
        "voyage-large-2-instruct": {"default": 1024, "allowed": [1024]},
        "voyage-large-2": {"default": 1536, "allowed": [1536]},
        "voyage-2": {"default": 1024, "allowed": [1024]},
    }
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the OpenAI embedding service.
        
        Args:
            config: Configuration for the embedding service
        """
        
        self.model_name = config.model_name
        self._dimension = getattr(config, "dimension", None)
        
        # Validate model name
        if self.model_name not in self.MODEL_DIMENSIONS:
            logger.warning(f"Model {self.model_name} not found in known models. Using default dimension of 1024.")
            self.default_dimension = 1024
            self.allowed_dimensions = [1024]
        else:
            self.default_dimension = self.MODEL_DIMENSIONS[self.model_name]["default"]
            self.allowed_dimensions = self.MODEL_DIMENSIONS[self.model_name]["allowed"]
        
        # Validate dimension if provided
        if self._dimension is not None and self._dimension not in self.allowed_dimensions:
            raise ValueError(
                f"Invalid dimension {self._dimension} for model {self.model_name}. "
                f"Allowed dimensions are: {self.allowed_dimensions}"
            )
        
        # Get API key directly from environment variables
        api_key = os.environ.get("VOYAGE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "VoyageAI API key not found. Please set VOYAGE_API_KEY environment variable."
            )
        
        self.client = voyageai.Client(api_key=api_key)
        self.logger = logger
        
        logger.info(
            f"Initialized VoyageAI embedding function with model: {self.model_name}, "
            f"dimension: {self._dimension or self.default_dimension}"
        )
    
    @property
    def dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            int: Dimension of the embedding vectors
        """
        return self._dimension or self.default_dimension
    
    def embed_document(self, text: str) -> Dict[str, Any]:
        """
        Generate embeddings for a single text document.
        
        Args:
            text: The text to embed
            
        Returns:
            Dictionary containing the embedding and metadata
        """
        try:
            result = self.client.embed(
                texts=[text],
                model=self.model_name,
                truncation=True,
                output_dimension=self.dimension
            )
            
            return result.embeddings[0]
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for multiple text documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Dictionary containing the embeddings and metadata
        """
        try:
            result = self.client.embed(
                texts=texts,
                model=self.model_name,
                truncation=True,
                output_dimension=self.dimension
            )
            
            return result.embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def multimodel_embed(self, input: List[Union[str, Image]]) -> Any:
        """
        Generate multimodal embeddings for text and optional image.
        
        Args:
            text: The text to embed
            image: Optional PIL image to embed alongside text
            
        Returns:
            Dictionary containing the embedding and metadata
        """
        try:

            result = self.client.multimodal_embed(
                inputs=[input],
                model=self.model_name,
                truncation=True
            )
            
            return result.embeddings[0]
        except Exception as e:
            self.logger.error(f"Error generating multimodal embedding: {e}")
            raise
    
    def multimodel_embed_bulk(self, inputs: Union[List[Dict], List[List[Union[str, Image]]]]) -> List[Any]:
        """
        Generate multimodal embeddings for multiple inputs.
        
        Args:
            inputs: List of inputs, where each input can be:
                   - A string (text only)
                   - A PIL Image (image only)
                   - A tuple of (text, image)
            
        Returns:
            Dictionary containing the embeddings and metadata
        """
        try:
            # Format inputs for the multimodal_embed API
            formatted_inputs = []
            for item in inputs:
                if isinstance(item, str):
                    formatted_inputs.append([item])
                elif isinstance(item, Image):
                    formatted_inputs.append([item])
                elif isinstance(item, tuple) and len(item) == 2:
                    text, image = item
                    if text and image:
                        formatted_inputs.append([text, image])
                    elif image:
                        formatted_inputs.append([image])
                    else:
                        formatted_inputs.append([text])
                else:
                    raise ValueError(f"Invalid input type: {type(item)}. Expected str, Image, or Tuple[str, Image]")
                
            result = self.client.multimodal_embed(
                inputs=formatted_inputs,
                model=self.model_name,
                truncation=True
            )
            
            return result.embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating multimodal embeddings: {e}")
            raise
