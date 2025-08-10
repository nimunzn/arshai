from PIL import Image
from typing import Dict, Any, List, Optional, Union
from arshai.core.interfaces.itool import ITool
from arshai.core.interfaces.ivector_db_client import IVectorDBClient, ICollectionConfig
from arshai.core.interfaces.iembedding import IEmbedding
from arshai.core.interfaces.ireranker import IReranker
from arshai.core.interfaces.idocument import Document
import logging
import traceback

class MultimodalKnowledgeBaseRetrievalTool(ITool):
    """Tool for retrieving knowledge from the vector database using both semantic and keyword-based search"""
    
    def __init__(self, 
                 vector_db: IVectorDBClient, 
                 embedding_model: IEmbedding,
                 collection_config: ICollectionConfig,
                 reranker: Optional[IReranker] = None,
                 search_limit: int = 3):
        """
        Initialize the multimodal knowledge base retrieval tool.
        
        Args:
            vector_db: Vector database client for storing/retrieving embeddings
            embedding_model: Embedding model for converting queries to vectors
            collection_config: Complete collection configuration including field names and settings
            reranker: Optional reranker for improving search result quality
            search_limit: Maximum number of results to return (default: 3)
        
        Example:
            from arshai.vector_db.milvus_client import MilvusClient
            from arshai.embeddings.openai_embeddings import OpenAIEmbedding
            from arshai.rerankers.voyage_reranker import VoyageReranker
            from arshai.core.interfaces.iembedding import EmbeddingConfig
            from arshai.core.interfaces.ivector_db_client import ICollectionConfig
            
            # Create components directly
            vector_db = MilvusClient(host="localhost", port=19530)
            
            embedding_config = EmbeddingConfig(model_name="text-embedding-3-small")
            embedding_model = OpenAIEmbedding(embedding_config)
            
            reranker = VoyageReranker(model="rerank-lite-1")  # Optional
            
            collection_config = ICollectionConfig(
                collection_name="multimodal_kb",
                dense_dim=1536,
                text_field="content", 
                metadata_field="metadata",
                is_hybrid=True  # Multimodal often uses hybrid search
            )
            
            # Create multimodal knowledge base tool
            kb_tool = MultimodalKnowledgeBaseRetrievalTool(
                vector_db=vector_db,
                embedding_model=embedding_model,
                collection_config=collection_config,
                reranker=reranker,
                search_limit=5
            )
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.collection_config = collection_config
        self.reranker = reranker
        self.search_limit = search_limit
        self.logger = logging.getLogger('MultimodalKnowledgeBaseRetrievalTool')
        
        # Validate required components
        if not vector_db:
            self.logger.error("Vector database client not provided")
        if not embedding_model:
            self.logger.error("Embedding model not provided")
        if not collection_config:
            self.logger.error("Collection configuration not provided")
        
        if vector_db and embedding_model and collection_config:
            self.logger.info(f"Initialized multimodal KB tool - Collection: {collection_config.collection_name}, "
                           f"Reranker: {'enabled' if reranker else 'disabled'}, "
                           f"Search limit: {search_limit}")
            

    @property
    def function_definition(self) -> Dict:
        """Get the function definition for the LLM"""
        return {
            "name": "retrieve_knowledge",
            "description": "Retrieve relevant knowledge from the vector database using semantic and keyword search. The query MUST be self-contained and include all necessary context without relying on conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A standalone text query to search for relevant images. The query should be self-contained and specific. "
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }

    def _search_results_to_documents(self, search_results: List[Any]) -> List[Document]:
        """
        Convert search results to a list of Document objects for reranking
        
        Args:
            search_results: List of search results from vector database
            
        Returns:
            List[Document]: List of Document objects
        """
        documents = []

        # Process each hit within the search results
        for hits in search_results:
            for hit in hits:
                try:
                    # Extract text content
                    image_string = hit.get(self.collection_config.text_field)
                    if image_string is None:
                        raise ValueError(f"Text field is None for hit ID: {hit.id}")
                    
                    # Extract metadata if available
                    metadata_field = self.collection_config.metadata_field
                    metadata = hit.get(metadata_field) if metadata_field in hit.fields else {}
                    if metadata is None:
                        metadata = {}
                    
                    # Add distance score to metadata
                    metadata["distance"] = hit.distance
                    metadata["id"] = hit.id
                    
                    # Create Document object
                    documents.append(Document(
                        page_content=image_string,
                        metadata=metadata
                    ))
                except Exception as e:
                    self.logger.error(f"Error processing hit for document conversion: {str(e)}")
                    self.logger.error(traceback.format_exc())
        
        return documents

    def _format_image_results(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format search results into a list of image objects for the LLM
        
        Args:
            documents: List of Document objects
            
        Returns:
            List[Dict[str, Any]]: List of image objects in the format required by the LLM
        """
        if not documents:
            return []
            
        formatted_results = []
        
        # Process each document
        for document in documents:
            try:
                # Format as required for LLM consumption
                description = {
                    "type": "text",
                    "text": f"The following image is with {document.metadata['id']} id in database"
                }
                result = {
                    "type": "image_url",
                    "image_url": {
                        "url": document.page_content
                    }
                }
                formatted_results.append(description)
                formatted_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error formatting document: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        return formatted_results

    async def aexecute(self, query: str) -> Union[List[Dict[str, Any]]]:
        """
        Asynchronous execution of the knowledge retrieval using vector search
        
        Args:
            query: A standalone question containing all required context
            
        Returns:
            List[Dict[str, Any]]: List of image objects for LLM
        """

        try:
            # Generate embeddings for the query
            query_embeddings = self.embedding_model.multimodel_embed(input=[query])

            self.logger.debug(f"query_embeddings: {query_embeddings}")
            # Use dense vector search only
            search_results = self.vector_db.search_by_vector(
                config=self.collection_config,
                query_vectors=[query_embeddings],
                limit=self.search_limit,
                output_fields = [self.collection_config.text_field, self.collection_config.metadata_field]
            )
            self.logger.debug(f"search_results: {search_results}")

            # Convert search results to documents and then to list format
            if search_results and len(search_results) > 0:
                documents = self._search_results_to_documents(search_results)
                return "assistant", self._format_image_results(documents)
            else:
                return "assistant", []
        
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "assistant", []

    def execute(self, query: str) -> Union[List[Dict[str, Any]]]:
        """
        Synchronous execution of the knowledge retrieval using vector search
        
        Args:
            query: A standalone question containing all required context
            
        Returns:
            List[Dict[str, Any]]: List of image objects for LLM
        """

        try:
            # Generate embeddings for the query
            query_embeddings = self.embedding_model.multimodel_embed(input=[query])
            
            # Perform vector search
            search_results = self.vector_db.search_by_vector(
                config=self.collection_config,
                query_vectors=[query_embeddings],
                limit=self.search_limit,
                output_fields = [self.collection_config.text_field, self.collection_config.metadata_field]
            )
            
            # Convert search results to documents and then to list format
            if search_results and len(search_results) > 0:
                documents = self._search_results_to_documents(search_results)
                return "assistant", self._format_image_results(documents)
            else:
                return "assistant", []
                
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "assistant", []
