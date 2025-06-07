from PIL import Image
from typing import Dict, Any, List, Optional, Union
from arshai.core.interfaces.itool import ITool
from arshai.core.interfaces.isetting import ISetting
from arshai.core.interfaces.idocument import Document
import logging

class MultimodalKnowledgeBaseRetrievalTool(ITool):
    """Tool for retrieving knowledge from the vector database using both semantic and keyword-based search"""
    
    def __init__(self, settings: ISetting):
        self.settings = settings
        self.logger = logging.getLogger('KnowledgeBaseRetrievalTool')
        
        # Log availability of required components
        if not settings:
            self.logger.error("Settings not provided")
            return
            
        # Get the vector db client and embedding model from settings
        self.vector_db, self.collection_config, self.embedding_model = self.settings.create_vector_db()
        
        self.reranker = self.settings.create_reranker()
        
        # Get search parameters from config
        self.search_limit = self.settings.get("search_limit", 3)
        
        self.logger.info(f"search_limit: {self.search_limit}")
        # Check if components are available
        if self.vector_db is None:
            self.logger.error("Vector database client not available")
            return False
        if self.embedding_model is None:
            self.logger.error("Embedding model not available")
            return False
        if self.collection_config is None:
            self.logger.error("Collection configuration not available")
            return False
            

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

            self.logger.info(f"query_embeddings: {query_embeddings}")
            # Use dense vector search only
            search_results = self.vector_db.search_by_vector(
                config=self.collection_config,
                query_vectors=[query_embeddings],
                limit=self.search_limit,
                output_fields = [self.collection_config.text_field, self.collection_config.metadata_field]
            )
            self.logger.info(f"search_results: {search_results}")

            # Convert search results to documents and then to list format
            if search_results and len(search_results) > 0:
                documents = self._search_results_to_documents(search_results)
                return self._format_image_results(documents)
            else:
                return []
        
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            return []

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
                return self._format_image_results(documents)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            return []
