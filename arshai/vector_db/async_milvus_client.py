"""
AsyncMilvusClient implementation using native PyMilvus async support.

This module provides a wrapper around PyMilvus AsyncMilvusClient to maintain
interface compatibility while offering true async operations.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Callable
from pymilvus import AsyncMilvusClient, MilvusClient, connections, utility
from pymilvus import CollectionSchema, DataType, FieldSchema
from arshai.core.interfaces.ivector_db_client import ICollectionConfig, IVectorDBClient
import traceback

logger = logging.getLogger(__name__)


class AsyncMilvusClientWrapper(IVectorDBClient):
    """
    Async wrapper for Milvus operations using native AsyncMilvusClient.
    
    Provides both sync and async methods for backward compatibility while
    leveraging native async support for better performance.
    """
    
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = int(os.getenv("MILVUS_PORT", "19530"))
        self.db_name = os.getenv("MILVUS_DB_NAME", "default")
        self.batch_size = int(os.getenv("MILVUS_BATCH_SIZE", "50"))
        self.logger = logger
        
        # Connection URI for AsyncMilvusClient
        self._connection_uri = f"http://{self.host}:{self.port}"
        
        # Clients (lazy initialization)
        self._async_client: Optional[AsyncMilvusClient] = None
        self._sync_client: Optional[MilvusClient] = None
        
        self.logger.info(f"Initialized AsyncMilvusClient for {self._connection_uri}, db: {self.db_name}")
    
    async def _get_async_client(self) -> AsyncMilvusClient:
        """Get or create async client with connection pooling."""
        if self._async_client is None:
            self._async_client = AsyncMilvusClient(
                uri=self._connection_uri,
                db_name=self.db_name
            )
            self.logger.info("Created new AsyncMilvusClient instance")
        return self._async_client
    
    def _get_sync_client(self) -> MilvusClient:
        """Get or create sync client for backward compatibility."""
        if self._sync_client is None:
            self._sync_client = MilvusClient(
                uri=self._connection_uri,
                db_name=self.db_name
            )
            self.logger.info("Created new MilvusClient instance")
        return self._sync_client
    
    # ----------------------
    # Schema and Collection Management (Sync Only - Limitation of AsyncMilvusClient)
    # ----------------------
    
    def create_schema(self, config: ICollectionConfig) -> CollectionSchema:
        """Create collection schema - sync only due to AsyncMilvusClient limitation."""
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
        
        # Add additional fields from schema model
        if config.schema_model:
            self.logger.info(f"Adding fields from schema model: {config.schema_model.__name__}")
            model_fields = config.schema_model.__annotations__
            for field_name, field_type in model_fields.items():
                # Skip fields that are already defined
                if field_name in [config.pk_field, config.text_field, 
                                config.metadata_field, config.sparse_field, 
                                config.dense_field]:
                    continue
                
                field_schema = self._get_field_schema(field_name, field_type)
                if field_schema:
                    self.logger.info(f"Adding field: {field_name}, type: {field_type}")
                    fields.append(field_schema)
        
        schema = CollectionSchema(fields)
        self.logger.info(f"Created schema with fields: {[f.name for f in fields]}")
        return schema
    
    def _get_field_schema(self, field_name: str, field_type: Any) -> Optional[FieldSchema]:
        """Convert Python type to Milvus FieldSchema."""
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
        
        # Handle typing annotations
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
    
    def get_or_create_collection(self, config: ICollectionConfig):
        """
        Get or create collection using sync client (required for schema operations).
        
        NOTE: Collection creation must use sync client due to AsyncMilvusClient limitations.
        Data operations can still use async client.
        """
        try:
            sync_client = self._get_sync_client()
            collection_name = config.collection_name
            
            # Check if collection exists
            if sync_client.has_collection(collection_name):
                self.logger.info(f"Collection {collection_name} already exists")
                return sync_client.get_collection(collection_name)

            # Create new collection
            self.logger.info(f"Creating new collection: {collection_name}")
            
            # For sync MilvusClient, we need to create schema differently
            # This is a limitation we work around
            schema = self.create_schema(config)
            
            # Create collection using low-level API for complex schemas
            from pymilvus import Collection
            connections.connect(
                host=self.host,
                port=self.port,
                db_name=self.db_name
            )
            
            collection = Collection(name=collection_name, schema=schema)
            
            # Create indices
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
            self.logger.error(traceback.format_exc())
            raise
    
    # ----------------------
    # Sync Methods (Backward Compatibility)
    # ----------------------
    
    def insert_entity(self, config: ICollectionConfig, entity: dict, documents_embedding: dict):
        """Sync insert - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.insert_entity_async(config, entity, documents_embedding))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.insert_entity_async(config, entity, documents_embedding))
    
    def insert_entities(self, config: ICollectionConfig, data: List[dict], documents_embedding):
        """Sync batch insert - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.insert_entities_async(config, data, documents_embedding))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.insert_entities_async(config, data, documents_embedding))
    
    def search_by_vector(self, config: ICollectionConfig, query_vectors, search_field=None,
                        expr=None, output_fields=None, limit=3, search_params=None):
        """Sync vector search - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.search_by_vector_async(
                config, query_vectors, search_field, expr, output_fields, limit, search_params
            ))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.search_by_vector_async(
                config, query_vectors, search_field, expr, output_fields, limit, search_params
            ))
    
    def hybrid_search(self, config: ICollectionConfig, dense_vectors=None, sparse_vectors=None,
                     expr=None, output_fields=None, limit=3, search_params=None):
        """Sync hybrid search - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.hybrid_search_async(
                config, dense_vectors, sparse_vectors, expr, output_fields, limit, search_params
            ))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.hybrid_search_async(
                config, dense_vectors, sparse_vectors, expr, output_fields, limit, search_params
            ))
    
    def query_by_expr(self, config: ICollectionConfig, expr: str, output_fields=None, 
                     consistency_level="Eventually"):
        """Sync query - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.query_by_expr_async(
                config, expr, output_fields, consistency_level
            ))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.query_by_expr_async(
                config, expr, output_fields, consistency_level
            ))
    
    def delete_entity(self, config: ICollectionConfig, filter_expr: str):
        """Sync delete - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.delete_entity_async(config, filter_expr))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.delete_entity_async(config, filter_expr))
    
    def get_collection_stats(self, config: ICollectionConfig):
        """Sync stats - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_collection_stats_async(config))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.get_collection_stats_async(config))
    
    # ----------------------
    # Async Methods (New - Native Performance)
    # ----------------------
    
    async def insert_entity_async(self, config: ICollectionConfig, entity: dict, documents_embedding: dict):
        """Async insert single entity - true async, no executor needed."""
        self.logger.info(f"Async inserting entity into collection {config.collection_name}")
        
        try:
            client = await self._get_async_client()
            
            # Ensure collection exists (using sync method for schema operations)
            await asyncio.to_thread(self.get_or_create_collection, config)
            
            # Prepare entity data
            entity_data = self._prepare_entity_data(config, entity, documents_embedding, 0)
            
            # True async insert
            result = await client.insert(
                collection_name=config.collection_name,
                data=[entity_data]
            )
            
            self.logger.info(f"Successfully inserted entity async: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async insert_entity: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def insert_entities_async(self, config: ICollectionConfig, data: List[dict], documents_embedding):
        """Async batch insert - true async, no executor needed."""
        self.logger.info(f"Async inserting {len(data)} entities into collection {config.collection_name}")
        
        try:
            client = await self._get_async_client()
            
            # Ensure collection exists (using sync method for schema operations)
            await asyncio.to_thread(self.get_or_create_collection, config)
            
            # Prepare all entities
            entities = []
            for i, doc in enumerate(data):
                entity = self._prepare_entity_data(config, doc, documents_embedding, i)
                entities.append(entity)
            
            # Insert in async batches
            results = []
            for i in range(0, len(entities), self.batch_size):
                batch_end = min(i + self.batch_size, len(entities))
                current_batch = entities[i:batch_end]
                
                self.logger.info(f"Inserting async batch {i//self.batch_size + 1}, size: {len(current_batch)}")
                
                # True async batch insert
                result = await client.insert(
                    collection_name=config.collection_name,
                    data=current_batch
                )
                results.append(result)
            
            self.logger.info(f"Successfully inserted {len(entities)} entities async")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in async insert_entities: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def search_by_vector_async(self, config: ICollectionConfig, query_vectors, search_field=None,
                                   expr=None, output_fields=None, limit=3, search_params=None):
        """Async vector search - true async, no executor needed."""
        self.logger.info(f"Async vector search in collection {config.collection_name}")
        
        try:
            client = await self._get_async_client()
            
            # Determine search field
            if search_field is None:
                search_field = config.dense_field
            
            # Use default output fields if not specified
            if output_fields is None:
                output_fields = ["*"]
            
            # Default search parameters
            if search_params is None:
                search_params = {"metric_type": "IP", "params": {}}
            
            self.logger.info(f"Searching field: {search_field}, limit: {limit}")
            
            # True async search
            results = await client.search(
                collection_name=config.collection_name,
                data=query_vectors,
                anns_field=search_field,
                search_params=search_params,
                limit=limit,
                filter=expr,
                output_fields=output_fields
            )
            
            self.logger.info(f"Async search returned {len(results)} result sets")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in async search_by_vector: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def hybrid_search_async(self, config: ICollectionConfig, dense_vectors=None, sparse_vectors=None,
                                expr=None, output_fields=None, limit=3, search_params=None):
        """Async hybrid search - true async, no executor needed."""
        self.logger.info(f"Async hybrid search in collection {config.collection_name}")
        
        if not config.is_hybrid:
            raise ValueError("Hybrid search not enabled in configuration")
        
        try:
            client = await self._get_async_client()
            
            # Use default output fields if not specified
            if output_fields is None:
                output_fields = ["*"]
            
            # Default search parameters for hybrid search
            if search_params is None:
                search_params = [
                    {"metric_type": "IP", "params": {"nprobe": 10}},  # Dense
                    {"metric_type": "IP"}  # Sparse
                ]
            
            # True async hybrid search
            results = await client.hybrid_search(
                collection_name=config.collection_name,
                reqs=[
                    {
                        "data": dense_vectors,
                        "anns_field": config.dense_field,
                        "param": search_params[0],
                        "limit": limit,
                        "filter": expr
                    },
                    {
                        "data": sparse_vectors, 
                        "anns_field": config.sparse_field,
                        "param": search_params[1],
                        "limit": limit,
                        "filter": expr
                    }
                ],
                rerank={"strategy": "weighted", "params": {"weights": [0.5, 0.5]}},
                limit=limit,
                output_fields=output_fields
            )
            
            self.logger.info(f"Async hybrid search returned {len(results)} result sets")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in async hybrid_search: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def query_by_expr_async(self, config: ICollectionConfig, expr: str, output_fields=None, 
                                consistency_level="Eventually"):
        """Async query - true async, no executor needed."""
        self.logger.info(f"Async query in collection {config.collection_name} with expr: {expr}")
        
        try:
            client = await self._get_async_client()
            
            # Use default output fields if not specified
            if output_fields is None:
                output_fields = ["*"]
            
            # True async query
            results = await client.query(
                collection_name=config.collection_name,
                filter=expr,
                output_fields=output_fields
            )
            
            self.logger.info(f"Async query returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in async query_by_expr: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def delete_entity_async(self, config: ICollectionConfig, filter_expr: str):
        """Async delete - true async, no executor needed."""
        self.logger.info(f"Async deleting entities from {config.collection_name} with filter: {filter_expr}")
        
        try:
            client = await self._get_async_client()
            
            # True async delete
            result = await client.delete(
                collection_name=config.collection_name,
                filter=filter_expr
            )
            
            self.logger.info(f"Async delete result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async delete_entity: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def get_collection_stats_async(self, config: ICollectionConfig):
        """Async collection stats - true async, no executor needed."""
        try:
            client = await self._get_async_client()
            
            # Get collection info asynchronously
            info = await client.get_collection_stats(config.collection_name)
            
            stats = {
                "name": config.collection_name,
                "row_count": info.get("row_count", 0),
                "data_size": info.get("data_size", 0),
            }
            
            self.logger.info(f"Async collection stats: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting async collection stats: {str(e)}")
            return {"name": config.collection_name, "row_count": 0, "data_size": 0}
    
    # ----------------------
    # Helper Methods
    # ----------------------
    
    def _prepare_entity_data(self, config: ICollectionConfig, entity: dict, documents_embedding, index: int) -> dict:
        """Prepare entity data for insertion."""
        entity_data = {}
        
        # Add text/content field
        if 'content' in entity:
            entity_data[config.text_field] = entity['content']
        elif config.text_field in entity:
            entity_data[config.text_field] = entity[config.text_field]
        else:
            raise KeyError(f"Text field 'content' or '{config.text_field}' not found in data.")
        
        # Add metadata
        if 'metadata' in entity:
            # Convert UUIDs to strings
            metadata = {}
            for k, v in entity['metadata'].items():
                metadata[k] = str(v) if hasattr(v, 'hex') else v
            entity_data[config.metadata_field] = metadata
        elif config.metadata_field in entity:
            # Convert UUIDs to strings
            metadata = {}
            for k, v in entity[config.metadata_field].items():
                metadata[k] = str(v) if hasattr(v, 'hex') else v
            entity_data[config.metadata_field] = metadata
        else:
            entity_data[config.metadata_field] = {}
        
        # Add dense vector
        if 'dense' in documents_embedding:
            if isinstance(documents_embedding['dense'], list) and len(documents_embedding['dense']) > index:
                entity_data[config.dense_field] = documents_embedding['dense'][index]
            else:
                entity_data[config.dense_field] = documents_embedding['dense']
                
        # Add sparse vector if hybrid search enabled
        if config.is_hybrid and 'sparse' in documents_embedding:
            if isinstance(documents_embedding['sparse'], list) and len(documents_embedding['sparse']) > index:
                entity_data[config.sparse_field] = documents_embedding['sparse'][index]
            else:
                entity_data[config.sparse_field] = documents_embedding['sparse']
        
        # Add additional fields from schema model
        if config.schema_model:
            model_fields = config.schema_model.__annotations__
            for field_name in model_fields:
                if field_name not in [config.pk_field, config.text_field, 
                                    config.metadata_field, config.sparse_field, 
                                    config.dense_field]:
                    if field_name in entity:
                        # Convert special types
                        value = entity[field_name]
                        if hasattr(value, 'hex'):  # UUID
                            entity_data[field_name] = str(value)
                        elif hasattr(value, 'isoformat'):  # datetime
                            entity_data[field_name] = value.isoformat()
                        else:
                            entity_data[field_name] = value
        
        return entity_data
    
    # ----------------------
    # Connection Management (Legacy compatibility)
    # ----------------------
    
    def connect(self):
        """Legacy connect method - AsyncMilvusClient handles connections automatically."""
        self.logger.info(f"AsyncMilvusClient connection is handled automatically")
        
    def disconnect(self):
        """Legacy disconnect method - cleanup async client if needed."""
        if self._async_client:
            # AsyncMilvusClient doesn't have explicit disconnect
            self._async_client = None
        if self._sync_client:
            self._sync_client = None
        self.logger.info("Disconnected from Milvus")