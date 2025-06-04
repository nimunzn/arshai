"""
MGTE embedding implementation.
"""

import logging
from typing import List, Dict, Any, Optional
from milvus_model.hybrid import MGTEEmbeddingFunction
from arshai.core.interfaces.iembedding import EmbeddingConfig, IEmbedding

logger = logging.getLogger(__name__)

class MGTEEmbedding(IEmbedding):
    """
    MGTE embedding implementation for hybrid dense and sparse vectors.
    
    This class provides access to Milvus General Text Embeddings models for
    generating both dense and sparse vector representations.
    """
    # Hardcoded dimension value for MGTE
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the MGTE embedding service.
        
        Args:
            config: Configuration for the embedding service
            settings: Optional settings for the embedding service
        """

                
        # Configure MGTE model
        self.use_fp16 = config.additional_params.get("use_fp16", False)
        self.device = config.additional_params.get("device", "cpu")
        self.batch_size = config.batch_size
        self.model_name = config.model_name
        
        self.embedding_function = MGTEEmbeddingFunction(
            model_name=self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
            batch_size=self.batch_size
        )
    
        # Set batch size from config

        logger.info(
            f"Initialized MGTE embedding function with dimension: {self.embedding_function.dim['dense']}, "
            f"device: {self.device}, use_fp16: {self.use_fp16}"
        )
    
    @property
    def dimension(self) -> int:
        return self.embedding_function.dim["dense"]
    
    def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for multiple documents using MGTE.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of dense embedding vectors
        """
        if not texts:
            return []
        
        # Process in batches to manage memory
        results = {}
        try:
            # MGTE generates both dense and sparse embeddings, but we'll use dense by default
            embeddings = self.embedding_function(texts)

            dense_vectors = []
            sparse_vectors = []
            for i in range(len(embeddings["dense"])):
                dense_vectors.append(embeddings["dense"][i].tolist())
                sparse_vectors.append(embeddings["sparse"][[i]])

            results["dense"] = dense_vectors
            results["sparse"] = sparse_vectors

        except Exception as e:
            logger.error(f"Error generating embeddings with MGTE: {str(e)}")
            # Return empty embeddings for failed batch
            raise e
    
        return results

    
    def embed_document(self, text: str) -> Dict[str, Any]:

        embeddings = self.embed_documents([text])
        results = {
                    'dense': embeddings['dense'][0],
                    'sparse': embeddings['sparse'][0]
        }
        return results
    