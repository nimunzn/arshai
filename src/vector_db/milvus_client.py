from dataclasses import dataclass
import os
from typing import List, Dict, Any, Type, Optional, Callable
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from pymilvus import WeightedRanker, AnnSearchRequest
from pydantic import BaseModel
from arshai.core.interfaces import ICollectionConfig, IVectorDBClient
import logging
import traceback

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)




class MilvusClient(IVectorDBClient):
    def __init__(self):

        self.host = os.getenv("MILVUS_HOST")
        self.port = os.getenv("MILVUS_PORT")
        self.db_name = os.getenv("MILVUS_DB_NAME")
        self.logger = logging.getLogger('MilvusClient')
    
    # ----------------------
    # Private Helper Methods
    # ----------------------
    
    def _with_collection(self, config: ICollectionConfig, operation_func: Callable, *args, **kwargs):
        """Helper method to perform operations with a collection
        
        Args:
            config: Collection configuration
            operation_func: Function to execute with the collection
            args: Additional positional arguments for the operation function
            kwargs: Additional keyword arguments for the operation function
            
        Returns:
            Result of the operation function
        """
        try:
            # Get or create the collection
            collection = self.get_or_create_collection(config)
            
            # Ensure collection is loaded
            self._ensure_collection_loaded(collection)
            
            # Execute the operation function
            return operation_func(collection, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in operation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def _ensure_collection_loaded(self, collection):
        """Ensure collection is loaded before querying"""
        try:
            # Replace the is_loaded attribute check which doesn't exist in this PyMilvus version
            self.logger.info(f"Loading collection {collection.name}")
            collection.load()
            self.logger.info(f"Collection {collection.name} loaded successfully")
        except Exception as e:
            # If the collection is already loaded, load() may throw an exception
            self.logger.info(f"Collection might already be loaded: {str(e)}")
    
    def _get_default_output_fields(self, collection):
        """Get default output fields (all non-vector fields)"""
        output_fields = []
        for field in collection.schema.fields:
            # Skip vector fields by default as they're usually large
            if field.dtype not in [DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR]:
                output_fields.append(field.name)
        self.logger.info(f"Default output fields: {output_fields}")
        return output_fields
        
    def _get_field_schema(self, field_name: str, field_type: Any) -> Optional[FieldSchema]:
        """Convert Python type to Milvus FieldSchema"""
        from datetime import datetime
        from uuid import UUID
        
        type_mapping = {
            str: (DataType.VARCHAR, {"max_length": 65535}),
            int: (DataType.INT64, {}),
            float: (DataType.FLOAT, {}),
            bool: (DataType.BOOL, {}),
            dict: (DataType.JSON, {}),
            list: (DataType.JSON, {}),
            UUID: (DataType.VARCHAR, {"max_length": 40}),
            datetime: (DataType.JSON, {})
        }
        
        # Get the base type from typing annotations if needed
        if hasattr(field_type, "__origin__"):
            origin_type = field_type.__origin__
            if origin_type in (list, dict):
                return FieldSchema(name=field_name, dtype=DataType.JSON)
        
        # Map standard Python types
        if field_type in type_mapping:
            dtype, kwargs = type_mapping[field_type]
            return FieldSchema(name=field_name, dtype=dtype, **kwargs)
            
        # Default to JSON for complex types
        return FieldSchema(name=field_name, dtype=DataType.JSON)
    
    def _get_stats(self, collection, config: ICollectionConfig):
        """Get collection statistics"""
        try:
            stats = {
                "name": config.collection_name,
                "num_entities": collection.num_entities,
                "schema": collection.schema,
                "indexes": collection.indexes
            }
            self.logger.info(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return None
    
    def _insert_single(self, collection, config: ICollectionConfig, entity: dict, documents_embedding: dict):
        """Insert a single entity into collection"""
        self.logger.info(f"Inserting entity into collection {collection.name}")
        
        # Create entity with explicit field names
        entity_data = {}
        
        # Add text/content field
        if 'content' in entity:
            entity_data[config.text_field] = entity['content']
        elif config.text_field in entity:
            entity_data[config.text_field] = entity[config.text_field]
        else:
            self.logger.error(f"Cannot find text field. Available keys: {entity.keys()}")
            raise KeyError(f"Text field 'content' or '{config.text_field}' not found in data.")
        
        # Add metadata
        if 'metadata' in entity:
            # Convert UUIDs in metadata to strings
            metadata = {}
            for k, v in entity['metadata'].items():
                metadata[k] = str(v) if hasattr(v, 'hex') else v
            entity_data[config.metadata_field] = metadata
        elif config.metadata_field in entity:
            # Convert UUIDs in metadata to strings
            metadata = {}
            for k, v in entity[config.metadata_field].items():
                metadata[k] = str(v) if hasattr(v, 'hex') else v
            entity_data[config.metadata_field] = metadata
        else:
            entity_data[config.metadata_field] = {}
        
        # Add dense vector
        if 'dense' in documents_embedding:
            entity_data[config.dense_field] = documents_embedding['dense']
            
        # Add sparse vector only if hybrid search is enabled
        if config.is_hybrid and 'sparse' in documents_embedding:
            entity_data[config.sparse_field] = documents_embedding['sparse']
        
        # Add any additional fields from schema model
        if config.schema_model:
            model_fields = config.schema_model.__annotations__
            for field_name in model_fields:
                if field_name not in [config.pk_field, config.text_field, 
                                    config.metadata_field, config.sparse_field, 
                                    config.dense_field]:
                    if field_name in entity:
                        # Convert UUID to string if needed
                        value = entity[field_name]
                        if hasattr(value, 'hex'):  # Check if it's a UUID
                            entity_data[field_name] = str(value)
                        # Convert datetime to ISO format string
                        elif hasattr(value, 'isoformat'):  # Check if it's a datetime
                            entity_data[field_name] = value.isoformat()
                        else:
                            entity_data[field_name] = value

        # Insert the entity into the collection
        collection.insert([entity_data])
        collection.flush()
        self.logger.info(f"Successfully inserted entity into collection {collection.name}")
    
    def _insert_batch(self, collection, config: ICollectionConfig, data: list[dict[str, str | dict]], documents_embedding):
        """Insert multiple entities into collection"""
        self.logger.info(f"Inserting {len(data)} entities into collection {collection.name}")
        self.logger.info(f"First data item keys: {data[0].keys() if data else 'No data'}")
        self.logger.info(f"Embedding keys: {documents_embedding.keys() if documents_embedding else 'No embeddings'}")
        
        if 'dense' in documents_embedding:
            self.logger.info(f"Dense vector type: {type(documents_embedding['dense'][0])}")
        if 'sparse' in documents_embedding:
            self.logger.info(f"Sparse vector type: {type(documents_embedding['sparse'][0])}")
        
        # Prepare all the entities as a list of dictionaries
        entities = []
        
        for i, doc in enumerate(data):
            # Create entity with explicit field names
            entity = {}
            
            # Add text/content field
            if 'content' in doc:
                entity[config.text_field] = doc['content']
            elif config.text_field in doc:
                entity[config.text_field] = doc[config.text_field]
            else:
                self.logger.error(f"Cannot find text field. Available keys: {doc.keys()}")
                raise KeyError(f"Text field 'content' or '{config.text_field}' not found in data.")
            
            # Add metadata
            if 'metadata' in doc:
                # Convert UUIDs in metadata to strings
                metadata = {}
                for k, v in doc['metadata'].items():
                    metadata[k] = str(v) if hasattr(v, 'hex') else v
                entity[config.metadata_field] = metadata
            elif config.metadata_field in doc:
                # Convert UUIDs in metadata to strings
                metadata = {}
                for k, v in doc[config.metadata_field].items():
                    metadata[k] = str(v) if hasattr(v, 'hex') else v
                entity[config.metadata_field] = metadata
            else:
                entity[config.metadata_field] = {}
            
            # Add dense vector
            if 'dense' in documents_embedding:
                entity[config.dense_field] = documents_embedding['dense'][i]
                
            # Add sparse vector only if hybrid search is enabled
            if config.is_hybrid and 'sparse' in documents_embedding:
                entity[config.sparse_field] = documents_embedding['sparse'][i]
            
            # Add any additional fields from schema model
            if config.schema_model:
                model_fields = config.schema_model.__annotations__
                for field_name in model_fields:
                    if field_name not in [config.pk_field, config.text_field, 
                                        config.metadata_field, config.sparse_field, 
                                        config.dense_field]:
                        if field_name in doc:
                            # Convert UUID to string if needed
                            value = doc[field_name]
                            if hasattr(value, 'hex'):  # Check if it's a UUID
                                entity[field_name] = str(value)
                            # Convert datetime to ISO format string
                            elif hasattr(value, 'isoformat'):  # Check if it's a datetime
                                entity[field_name] = value.isoformat()
                            else:
                                entity[field_name] = value
            
            entities.append(entity)
        
        # Insert in batches
        batch_size = self.config.batch_size  # Default to 50 if batch_size not in config
        for i in range(0, len(entities), batch_size):
            batch_end = min(i + batch_size, len(entities))
            current_batch = entities[i:batch_end]
            current_batch_size = len(current_batch)
            
            self.logger.info(f"Inserting batch {i//batch_size + 1}, size: {current_batch_size}")
            self.logger.info(f"First entity keys: {current_batch[0].keys() if current_batch else 'Empty batch'}")
            
            # Insert the batch
            collection.insert(current_batch)
            
        collection.flush()
        self.logger.info(f"Successfully added {len(entities)} new documents to collection {collection.name}")
        self.logger.info(f"Total entities in collection: {collection.num_entities}")
    
    def _query(self, collection, config: ICollectionConfig, expr: str, output_fields=None, consistency_level="Eventually"):
        """Execute a query on the collection"""
        self.logger.info(f"Querying collection {collection.name} with expression: {expr}")
        
        # Use default output fields if not specified
        if output_fields is None:
            output_fields = self._get_default_output_fields(collection)
        
        # Execute query with expression
        results = collection.query(
            expr=expr,
            output_fields=output_fields,
            consistency_level=consistency_level
        )
        
        self.logger.info(f"Query returned {len(results)} results")
        return results
    
    def _search(self, collection, config: ICollectionConfig, query_vectors, search_field=None,
                expr=None, output_fields=None, limit=3, search_params=None, 
                consistency_level="Eventually"):
        """Search the collection using vector similarity"""
        self.logger.info(f"Vector searching in collection {collection.name}")
        if expr:
            self.logger.info(f"With filter expression: {expr}")
        
        # Determine which field to search in
        if search_field is None:
            search_field = config.dense_field
            
        # Use default output fields if not specified
        if output_fields is None:
            output_fields = self._get_default_output_fields(collection)
        
        # Use default search parameters if not specified
        if search_params is None:
            search_params = {"metric_type": "IP", "params": {}}
        
        # Execute search
        self.logger.info(f"Search params: {search_params}")
        self.logger.info(f"Searching in field: {search_field}, limit: {limit}")
        search_results = collection.search(
            data=query_vectors,
            anns_field=search_field,
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields
        )
        
        self.logger.info(f"Search returned {len(search_results)} result sets")
        return search_results
    
    def _hybrid_search(self, collection, config: ICollectionConfig, dense_vectors=None, sparse_vectors=None,
                       expr=None, output_fields=None, limit=3, search_params=None):
        """Perform hybrid search on the collection"""
        self.logger.info(f"Performing hybrid search in collection {collection.name}")
        
        # Check if hybrid search is enabled in configuration
        if not config.is_hybrid:
            error_msg = "Hybrid search is not enabled in the database configuration. Set is_hybrid=True in VectoreDatabaseConfig."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Use default output fields if not specified
        if output_fields is None:
            output_fields = self._get_default_output_fields(collection)
        
        if not search_params:
            search_params = [
                {"metric_type": "IP", "params": {"nprobe": 10}},  # For dense
                {"metric_type": "IP"}  # For sparse
            ]

        rerank = WeightedRanker(0.5, 0.5)
        self.logger.info(f"Using reranker: {type(rerank).__name__}, limit: {limit}")

        dense_req = AnnSearchRequest(
            data=dense_vectors,
            anns_field=config.dense_field,
            param=search_params[0],
            limit=limit,
            expr=expr
        )
        self.logger.info(f"Dense request: {dense_req}")
        
        sparse_req = AnnSearchRequest(
            data=sparse_vectors,
            anns_field=config.sparse_field,
            param=search_params[1],
            limit=limit,
            expr=expr
        )
        # Execute hybrid search with reranker
        hybrid_results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=rerank,
            limit=limit,
            output_fields=output_fields,
        )
            
        self.logger.info(f"Hybrid search returned {len(hybrid_results)} result sets")
        return hybrid_results
    
    def _delete_by_expr(self, collection, expr: str):
        """Delete entities from collection using expression filter"""
        self.logger.info(f"Deleting entities from collection {collection.name} with expression: {expr}")
        try:
            # Execute the delete operation
            expr = expr.strip()
            if not expr:
                raise ValueError("Expression cannot be empty for delete operation")
                
            delete_result = collection.delete(expr)
            self.logger.info(f"Deleted entities result: {delete_result}")
            return delete_result
        except Exception as e:
            self.logger.error(f"Error deleting entities: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    # ----------------------
    # Public Methods
    # ----------------------
    
    def connect(self):
        """Connect to Milvus server"""
        self.logger.info(f"Connecting to Milvus at {self.host}:{self.port}, db: {self.db_name}")
        try:
            connections.connect(
                host=self.host,
                port=self.port,
                db_name=self.db_name
            )
            self.logger.info("Successfully connected to Milvus")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
        
    def disconnect(self):
        """Disconnect from Milvus server"""
        self.logger.info(f"Disconnecting from Milvus (db: {self.db_name})")
        try:
            connections.disconnect(alias=self.db_name)
            self.logger.info("Successfully disconnected from Milvus")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Milvus: {str(e)}")
        
    def create_schema(self, config: ICollectionConfig) -> CollectionSchema:
        """Create collection schema with appropriate fields
        
        Args:
            config: Collection configuration with schema model
            
        Returns:
            CollectionSchema: Milvus collection schema
        """
        self.logger.info("Creating collection schema")
        # Default fields for vector search functionality
        fields = [
            FieldSchema(
                name=config.pk_field, dtype=DataType.VARCHAR, 
                is_primary=True, auto_id=True, max_length=100
            ),
            FieldSchema(name=config.text_field, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=config.metadata_field, dtype=DataType.JSON),
            FieldSchema(name=config.dense_field, dtype=DataType.FLOAT_VECTOR, 
                       dim=config.dense_dim),
        ]
        
        # Only add sparse vector field if hybrid search is enabled
        if config.is_hybrid:
            fields.append(FieldSchema(name=config.sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR))
            self.logger.info(f"Added sparse vector field: {config.sparse_field}")
        
        # Add additional fields from the Pydantic model if provided
        if config.schema_model:
            self.logger.info(f"Adding fields from schema model: {config.schema_model.__name__}")
            model_fields = config.schema_model.__annotations__
            for field_name, field_type in model_fields.items():
                # Skip fields that are already defined in default fields
                if field_name in [config.pk_field, config.text_field, 
                                config.metadata_field, config.sparse_field, 
                                config.dense_field]:
                    continue
                
                # Map Python types to Milvus DataType
                field_schema = self._get_field_schema(field_name, field_type)
                if field_schema:
                    self.logger.info(f"Adding field: {field_name}, type: {field_type}")
                    fields.append(field_schema)
        
        schema = CollectionSchema(fields)
        self.logger.info(f"Created schema with fields: {[f.name for f in fields]}")
        return schema
    
    def get_or_create_collection(self, config: ICollectionConfig) -> Collection:
        """Get existing collection or create new one if it doesn't exist
        
        Args:
            config: Collection configuration
            
        Returns:
            Collection instance
        """
        try:
            # Ensure we're connected
            self.connect()
            collection_name = config.collection_name
            # Check if collection exists
            if utility.has_collection(collection_name):
                self.logger.info(f"Collection {collection_name} already exists")
                return Collection(collection_name)

            # Create new collection if it doesn't exist
            self.logger.info(f"Creating new collection: {collection_name}")
            schema = self.create_schema(config)
            collection = Collection(name=collection_name, schema=schema)
            
            self.logger.info(f"Creating indices for collection: {collection_name}")
            
            # Only create sparse index if hybrid search is enabled
            if config.is_hybrid:
                sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
                collection.create_index(config.sparse_field, sparse_index)
                self.logger.info(f"Created sparse index on {config.sparse_field}")
            
            dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
            collection.create_index(config.dense_field, dense_index)
            self.logger.info(f"Created dense index on {config.dense_field}")

            return collection

        except Exception as e:
            self.logger.error(f"Error in get_or_create_collection: {str(e)}")
            raise
    
    def get_collection_stats(self, config: ICollectionConfig):
        """Get collection statistics
        
        Args:
            config: Collection configuration
            
        Returns:
            Dictionary with collection statistics
        """
        return self._with_collection(config, self._get_stats, config)
    
    def insert_entity(self, config: ICollectionConfig, entity: dict, documents_embedding: dict):
        """Insert a single entity into collection
        
        Args:
            config: Collection configuration
            entity: Dictionary representing the entity to insert
            documents_embedding: Document embeddings with sparse and dense vectors
        """
        return self._with_collection(config, self._insert_single, config, entity, documents_embedding)
            
    def insert_entities(self, config: ICollectionConfig, data: list[dict[str, str | dict]], documents_embedding):
        """Insert documents with embeddings into collection
        
        Args:
            config: Collection configuration
            data: List of documents with content and metadata
            documents_embedding: Document embeddings with sparse and dense vectors
        """
        return self._with_collection(config, self._insert_batch, config, data, documents_embedding)
    
    def query_by_expr(self, config: ICollectionConfig, expr: str, output_fields=None, 
                   consistency_level="Eventually"):
        """Query documents from collection using attribute filter expression
        
        Args:
            config: Collection configuration
            expr: Filter expression (e.g. "metadata['source'] == 'file1.txt'")
            output_fields: List of fields to return in the query results
            consistency_level: Consistency level for the query
            
        Returns:
            List of matched documents
        """
        return self._with_collection(config, self._query, config, expr, output_fields, consistency_level)
    
    def search_by_vector(self, config: ICollectionConfig, query_vectors, search_field=None,
                      expr=None, output_fields=None, limit=3, search_params=None):
        """Search documents using vector similarity
        
        Args:
            config: Collection configuration
            query_vectors: Query vector(s) for search
            search_field: Field to search in (defaults to dense_field)
            expr: Optional filter expression
            output_fields: List of fields to return
            limit: Maximum number of results to return
            search_params: Optional search parameters
            consistency_level: Consistency level for the query
            
        Returns:
            List of search results with distances and entities
        """
        return self._with_collection(config, self._search, config, query_vectors, search_field, 
                                   expr, output_fields, limit, search_params)
            
    def hybrid_search(self, config: ICollectionConfig, dense_vectors=None, sparse_vectors=None,
                    expr=None, output_fields=None, limit=3, search_params=None):
        """Perform hybrid search using both dense and sparse vectors or using a reranker
        
        Args:
            config: Collection configuration
            dense_vectors: Dense vectors for search
            sparse_vectors: Sparse vectors for search
            expr: Optional filter expression
            output_fields: List of fields to return
            limit: Maximum number of results to return
            consistency_level: Consistency level for the query
            
        Returns:
            List of search results with distances and entities
        """
        return self._with_collection(config, self._hybrid_search, config, dense_vectors, sparse_vectors, 
                                   expr, output_fields, limit, search_params)

    def delete_entity(self, config: ICollectionConfig, filter_expr: str):
        """Delete entities from collection based on filter expression
        
        Args:
            config: Collection configuration
            filter_expr: Filter expression to identify entities to delete (e.g., "thread_id == '123'")
            
        Returns:
            Delete operation result
        """
        self.logger.info(f"Delete entity with filter: {filter_expr}")
        return self._with_collection(config, self._delete_by_expr, filter_expr)
