from typing import Dict, Any, List, Optional
from arshai.core.interfaces import ITool
from arshai.core.interfaces import ISetting
import logging

class KnowledgeBaseRetrievalTool(ITool):
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
                        "description": "A standalone, self-contained search query that includes all necessary context. The query will be processed using both dense and sparse vectors for semantic and keyword matching. Example of a good query: 'What are the fees for transferring money internationally using Taloan?' Instead of just 'What are the fees?'"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }

    def _format_search_results(self, search_results: List[Any]) -> str:
        """
        Format search results into a string for response
        
        Args:
            search_results: List of search results from vector database (Milvus Hits objects)
            
        Returns:
            str: Formatted string of relevant knowledge
        """
        if not search_results:
            return "No relevant information found."
            
        formatted_results = []
        
        # Each search_result is a Hits object containing multiple Hit objects
        for hits in search_results:
            # Log the number of hits in this result set
            self.logger.info(f"Processing result set with {len(hits)} hits")
            
            # Process each hit within the hits collection
            for hit in hits:
                try:
                    # Extract information from the hit
                    self.logger.info(f"Processing hit with ID: {hit.id}, distance: {hit.distance}")
                    
                    # Extract text content - directly access the field value using hit.get()
                    text = hit.get(self.collection_config.text_field)
                    if text is None:
                        raise ValueError(f"Text field is None for hit ID: {hit.id}")
                    
                    # Extract metadata if available
                    metadata_field = self.collection_config.metadata_field
                    metadata = hit.get(metadata_field) if metadata_field in hit.fields else {}
                    if metadata is None:
                        metadata = {}
                    
                    source = metadata.get('source', 'unknown') if isinstance(metadata, dict) else 'unknown'
                    
                    # Format the result
                    formatted_results.append(f"Source: {source}\nContent: {text}\n")
                except Exception as e:
                    self.logger.error(f"Error processing hit: {str(e)}")
        
        if not formatted_results:
            return "No relevant information could be extracted from the search results."
            
        return "\n".join(formatted_results)

    async def aexecute(self, query: str) -> str:
        """
        Asynchronous execution of the knowledge retrieval using vector search
        
        Args:
            query: A standalone question containing all required context
            
        Returns:
            str: Retrieved relevant knowledge from the vector database
        """

        try:
            # Generate embeddings for the query
            query_embeddings = self.embedding_model.embed_document(query)
            
            # Perform vector search
            if self.collection_config.is_hybrid and 'sparse' in query_embeddings:
                # Use hybrid search if configuration supports it
                search_results = self.vector_db.hybrid_search(
                    config=self.collection_config,
                    dense_vectors=[query_embeddings['dense']],
                    sparse_vectors=[query_embeddings['sparse']],
                    limit=self.search_limit
                )
            else:
                # Use dense vector search only
                search_results = self.vector_db.search_by_vector(
                    config=self.collection_config,
                    query_vectors=[query_embeddings['dense']],
                    limit=self.search_limit
                )
            
            # Format the search results
            if search_results and len(search_results) > 0:
                return self._format_search_results(search_results)
            else:
                return "No relevant information found."
                
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            return f"Error retrieving knowledge: {str(e)}"

    def execute(self, query: str) -> str:
        """
        Synchronous execution of the knowledge retrieval using vector search
        
        Args:
            query: A standalone question containing all required context
            
        Returns:
            str: Retrieved relevant knowledge from the vector database
        """

        try:
            # Generate embeddings for the query
            query_embeddings = self.embedding_model.embed_document(query)
            
            # Perform vector search
            if self.collection_config.is_hybrid and 'sparse' in query_embeddings:
                # Use hybrid search if configuration supports it
                search_results = self.vector_db.hybrid_search(
                    config=self.collection_config,
                    dense_vectors=[query_embeddings['dense']],
                    sparse_vectors=[query_embeddings['sparse']],
                    limit=self.search_limit
                )
            else:
                # Use dense vector search only
                search_results = self.vector_db.search_by_vector(
                    config=self.collection_config,
                    query_vectors=[query_embeddings['dense']],
                    limit=self.search_limit
                )
            
            # Format the search results
            if search_results and len(search_results) > 0:
                self.logger.info(f"search_results: {len(search_results)}")
                return self._format_search_results(search_results)
            else:
                return "No relevant information found."
                
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            return f"Error retrieving knowledge: {str(e)}"
