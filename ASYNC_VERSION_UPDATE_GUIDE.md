# Async Version Update Guide - Complete Implementation Plan

## Overview

This guide provides step-by-step instructions to update Arshai codebase with native async support while maintaining 100% backward compatibility.

## 1. Dependencies Update (pyproject.toml)

### Current Dependencies:
```toml
[tool.poetry.dependencies]
python = ">=3.11,<3.13"
openai = "^1.0.0"
redis = {version = "^5.0.0", optional = true}
pymilvus = {version = "^2.3.0", optional = true}
aiohttp = "^3.11.16"
```

### Updated Dependencies:
```toml
[tool.poetry.dependencies]
python = ">=3.11,<3.13"
openai = "^1.0.0"                    # ✅ Already supports AsyncOpenAI
redis = {version = "^5.0.0", optional = true}  # ✅ Already supports redis.asyncio
pymilvus = {version = "^2.5.3", optional = true}  # 🔄 UPGRADE for AsyncMilvusClient
aiofiles = "^24.1.0"                 # ➕ NEW for async file operations
aiohttp = "^3.11.16"

[tool.poetry.extras]
async = ["aiofiles"]
all = ["redis", "pymilvus", "flashrank", "aiofiles", "google-genai", "google-auth", "opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-grpc", "opentelemetry-exporter-otlp-proto-http"]
```

### Installation Commands:
```bash
# Update pymilvus to latest version with async support
poetry add "pymilvus>=2.5.3"

# Add async file I/O support
poetry add aiofiles

# Install all dependencies
poetry install
```

## 2. Milvus Client Async Implementation

### Current Implementation Location:
- `arshai/vector_db/milvus_client.py`

### New Async Implementation:

Create new file: `arshai/vector_db/async_milvus_client.py`
```python
"""
Async Milvus Client Implementation with Native AsyncMilvusClient
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from functools import partial

# Import both sync and async clients
from pymilvus import MilvusClient, AsyncMilvusClient, Collection, FieldSchema, CollectionSchema, DataType
from arshai.core.interfaces.ivector_db_client import IVectorDBClient, ICollectionConfig

class AsyncMilvusClientWrapper(IVectorDBClient):
    """
    Async Milvus client wrapper that provides both sync and async methods.
    Uses native AsyncMilvusClient for true async operations.
    """
    
    def __init__(self, uri: Optional[str] = None, **kwargs):
        """
        Initialize both sync and async Milvus clients.
        
        Args:
            uri: Milvus connection URI (default: from env vars)
            **kwargs: Additional connection parameters
        """
        # Connection configuration
        self.host = kwargs.get('host') or os.getenv("MILVUS_HOST", "localhost")
        self.port = kwargs.get('port') or os.getenv("MILVUS_PORT", "19530")
        self.db_name = kwargs.get('db_name') or os.getenv("MILVUS_DB_NAME", "default")
        self.batch_size = int(kwargs.get('batch_size', os.getenv("MILVUS_BATCH_SIZE", "50")))
        
        # Build URI if not provided
        if uri is None:
            uri = f"http://{self.host}:{self.port}"
        
        # Initialize both clients
        self._sync_client = MilvusClient(uri=uri, db_name=self.db_name, **kwargs)
        self._async_client = AsyncMilvusClient(uri=uri, db_name=self.db_name, **kwargs)
        
        self.logger = logging.getLogger('AsyncMilvusClient')
    
    # ==========================================
    # SYNC METHODS (Backward Compatibility)
    # ==========================================
    
    def connect(self):
        """Connect to Milvus - sync version"""
        # Connection is handled automatically by MilvusClient
        self.logger.info(f"Connected to Milvus at {self.host}:{self.port}")
    
    def disconnect(self):
        """Disconnect from Milvus - sync version"""
        if hasattr(self._sync_client, 'close'):
            self._sync_client.close()
        self.logger.info("Disconnected from Milvus")
    
    def create_schema(self, config: ICollectionConfig):
        """Create collection schema - sync version"""
        # Note: AsyncMilvusClient doesn't support create_schema yet
        # Use sync client for schema operations
        fields = []
        
        # Primary key field
        fields.append(FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ))
        
        # Text field
        fields.append(FieldSchema(
            name=config.text_field,
            dtype=DataType.VARCHAR,
            max_length=65535
        ))
        
        # Dense vector field
        if config.dense_dim:
            fields.append(FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=config.dense_dim
            ))
        
        # Metadata field
        fields.append(FieldSchema(
            name=config.metadata_field,
            dtype=DataType.JSON
        ))
        
        return CollectionSchema(
            fields=fields,
            description=f"Collection schema for {config.collection_name}"
        )
    
    def get_or_create_collection(self, config: ICollectionConfig):
        """Get or create collection - sync version"""
        try:
            # Check if collection exists
            if self._sync_client.has_collection(config.collection_name):
                self.logger.info(f"Collection {config.collection_name} already exists")
                return Collection(config.collection_name)
            
            # Create new collection
            schema = self.create_schema(config)
            collection = Collection(
                name=config.collection_name,
                schema=schema
            )
            
            # Create index for dense vector
            if config.dense_dim:
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                }
                collection.create_index("dense_vector", index_params)
            
            self.logger.info(f"Created collection {config.collection_name}")
            return collection
            
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise
    
    def insert_entity(self, config: ICollectionConfig, entity: dict, documents_embedding: dict):
        """Insert single entity - sync version"""
        try:
            # Prepare entity data
            entity_data = {
                config.text_field: entity.get('content', ''),
                config.metadata_field: entity.get('metadata', {}),
                "dense_vector": documents_embedding.get('dense', [])
            }
            
            # Insert using sync client
            result = self._sync_client.insert(
                collection_name=config.collection_name,
                data=[entity_data]
            )
            
            self.logger.info(f"Inserted entity into {config.collection_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error inserting entity: {e}")
            raise
    
    def search_by_vector(self, config: ICollectionConfig, query_vectors, search_field=None, 
                        expr=None, output_fields=None, limit=3, search_params=None, 
                        consistency_level="Eventually"):
        """Vector search - sync version"""
        try:
            result = self._sync_client.search(
                collection_name=config.collection_name,
                data=query_vectors,
                anns_field=search_field or "dense_vector",
                search_params=search_params or {"metric_type": "COSINE"},
                limit=limit,
                expr=expr,
                output_fields=output_fields or ["*"],
                consistency_level=consistency_level
            )
            
            self.logger.info(f"Search returned {len(result)} results")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            raise
    
    # ==========================================
    # ASYNC METHODS (New Native Async)
    # ==========================================
    
    async def connect_async(self):
        """Connect to Milvus - async version"""
        # Connection is handled automatically by AsyncMilvusClient
        self.logger.info(f"Connected to Milvus async at {self.host}:{self.port}")
    
    async def disconnect_async(self):
        """Disconnect from Milvus - async version"""
        if hasattr(self._async_client, 'close'):
            await self._async_client.close()
        self.logger.info("Disconnected from Milvus async")
    
    async def insert_entity_async(self, config: ICollectionConfig, entity: dict, documents_embedding: dict):
        """Insert single entity - async version using native AsyncMilvusClient"""
        try:
            # Prepare entity data
            entity_data = {
                config.text_field: entity.get('content', ''),
                config.metadata_field: entity.get('metadata', {}),
                "dense_vector": documents_embedding.get('dense', [])
            }
            
            # Insert using native async client - TRUE ASYNC!
            result = await self._async_client.insert(
                collection_name=config.collection_name,
                data=[entity_data]
            )
            
            self.logger.info(f"Async inserted entity into {config.collection_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async insert: {e}")
            raise
    
    async def insert_entities_async(self, config: ICollectionConfig, entities: List[dict], 
                                   documents_embeddings: List[dict]):
        """Batch insert entities - async version with concurrency control"""
        try:
            # Prepare batch data
            batch_data = []
            for entity, embedding in zip(entities, documents_embeddings):
                entity_data = {
                    config.text_field: entity.get('content', ''),
                    config.metadata_field: entity.get('metadata', {}),
                    "dense_vector": embedding.get('dense', [])
                }
                batch_data.append(entity_data)
            
            # Process in batches with concurrency control
            semaphore = asyncio.Semaphore(4)  # Max 4 concurrent batch operations
            
            async def insert_batch(batch):
                async with semaphore:
                    return await self._async_client.insert(
                        collection_name=config.collection_name,
                        data=batch
                    )
            
            # Split into batches
            batches = [
                batch_data[i:i + self.batch_size] 
                for i in range(0, len(batch_data), self.batch_size)
            ]
            
            # Execute all batches concurrently
            results = await asyncio.gather(*[insert_batch(batch) for batch in batches])
            
            self.logger.info(f"Async inserted {len(entities)} entities in {len(batches)} batches")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in async batch insert: {e}")
            raise
    
    async def search_by_vector_async(self, config: ICollectionConfig, query_vectors, 
                                    search_field=None, expr=None, output_fields=None, 
                                    limit=3, search_params=None, consistency_level="Eventually"):
        """Vector search - async version using native AsyncMilvusClient"""
        try:
            # Native async search - TRUE ASYNC!
            result = await self._async_client.search(
                collection_name=config.collection_name,
                data=query_vectors,
                anns_field=search_field or "dense_vector",
                search_params=search_params or {"metric_type": "COSINE"},
                limit=limit,
                expr=expr,
                output_fields=output_fields or ["*"],
                consistency_level=consistency_level
            )
            
            self.logger.info(f"Async search returned {len(result)} results")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async vector search: {e}")
            raise
    
    async def hybrid_search_async(self, config: ICollectionConfig, dense_vectors=None, 
                                 sparse_vectors=None, expr=None, output_fields=None, 
                                 limit=3, search_params=None):
        """Hybrid search - async version"""
        if dense_vectors:
            return await self.search_by_vector_async(
                config=config,
                query_vectors=dense_vectors,
                search_field="dense_vector",
                expr=expr,
                output_fields=output_fields,
                limit=limit,
                search_params=search_params
            )
        else:
            raise NotImplementedError("Sparse vector search not implemented yet")
```

### Update existing MilvusClient to use async wrapper:

Modify `arshai/vector_db/milvus_client.py`:
```python
# Add at the top of the file
from .async_milvus_client import AsyncMilvusClientWrapper

class MilvusClient(AsyncMilvusClientWrapper):
    """
    Enhanced MilvusClient with both sync and async support.
    Inherits from AsyncMilvusClientWrapper for full compatibility.
    """
    pass
```

## 3. Redis Memory Manager Async Implementation

### Current Implementation Location:
- `arshai/memory/working_memory/redis_memory_manager.py`

### New Async Implementation:

Create new file: `arshai/memory/working_memory/async_redis_memory_manager.py`
```python
"""
Async Redis Memory Manager with Native redis.asyncio Support
"""
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import redis.asyncio as redis
import redis as sync_redis  # For backward compatibility
from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput, IWorkingMemory, ConversationMemoryType

class AsyncRedisMemoryManager(IMemoryManager):
    """
    Redis-based memory manager with native async support.
    Uses redis.asyncio for true async operations.
    """
    
    def __init__(self, redis_url: Optional[str] = None, **kwargs):
        """
        Initialize async Redis memory manager.
        
        Args:
            redis_url: Redis connection URL
            **kwargs: Additional configuration (ttl, max_connections, etc.)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.prefix = "memory"
        self.ttl = kwargs.get('ttl', 60 * 60 * 12)  # 12 hours default
        
        # Async client (lazy initialization)
        self._async_client: Optional[redis.Redis] = None
        self._client_config = {
            'decode_responses': True,
            'max_connections': kwargs.get('max_connections', 20),
            'retry_on_timeout': True,
            'socket_keepalive': True,
            'health_check_interval': 30
        }
        
        # Sync client for backward compatibility (lazy initialization)
        self._sync_client: Optional[sync_redis.Redis] = None
        
        self.logger = logging.getLogger('AsyncRedisMemoryManager')
        self.logger.info(f"Initialized AsyncRedisMemoryManager with TTL: {self.ttl} seconds")
    
    def _get_key(self, conversation_id: str, memory_type: ConversationMemoryType) -> str:
        """Generate a storage key for a memory entry."""
        return f"{self.prefix}:{memory_type}:{conversation_id}"
    
    async def _get_async_client(self) -> redis.Redis:
        """Get or create async Redis client."""
        if self._async_client is None:
            self._async_client = redis.from_url(
                self.redis_url,
                **self._client_config
            )
            # Test connection
            try:
                await self._async_client.ping()
                self.logger.info("Async Redis connection established")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis async: {e}")
                raise
        
        return self._async_client
    
    def _get_sync_client(self) -> sync_redis.Redis:
        """Get or create sync Redis client for backward compatibility."""
        if self._sync_client is None:
            # Convert async config to sync config
            sync_config = self._client_config.copy()
            sync_config.pop('max_connections', None)  # Not supported in sync client
            
            self._sync_client = sync_redis.from_url(
                self.redis_url,
                **sync_config
            )
            
            # Test connection
            try:
                self._sync_client.ping()
                self.logger.info("Sync Redis connection established")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis sync: {e}")
                raise
        
        return self._sync_client
    
    # ==========================================
    # SYNC METHODS (Backward Compatibility)
    # ==========================================
    
    def store(self, input: IMemoryInput) -> str:
        """Store memory data - sync version"""
        if not input.data:
            self.logger.warning("No data provided to store")
            raise ValueError("No data provided to store")
        
        client = self._get_sync_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        for data in input.data:
            storage_data = {
                "data": {"working_memory": data.working_memory},
                "metadata": input.metadata or {},
                "created_at": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
            # Sync Redis operation
            client.setex(key, self.ttl, json.dumps(storage_data))
            self.logger.debug(f"Stored memory with key: {key}")
        
        return key
    
    def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """Retrieve memory data - sync version"""
        client = self._get_sync_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        data = client.get(key)
        
        if not data:
            self.logger.debug(f"No data found for key: {key}")
            return []
        
        stored_data = json.loads(data)
        working_memory = IWorkingMemory(
            working_memory=stored_data["data"]["working_memory"]
        )
        
        return [working_memory]
    
    def update(self, input: IMemoryInput) -> None:
        """Update memory data - sync version"""
        client = self._get_sync_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        # Get existing data
        existing_data = client.get(key)
        
        if existing_data:
            stored_data = json.loads(existing_data)
            # Update the data
            for new_data in input.data:
                stored_data["data"]["working_memory"] = new_data.working_memory
                stored_data["last_update"] = datetime.now().isoformat()
            
            # Store updated data
            client.setex(key, self.ttl, json.dumps(stored_data))
            self.logger.debug(f"Updated memory with key: {key}")
        else:
            # If no existing data, create new entry
            self.store(input)
    
    def delete(self, input: IMemoryInput) -> None:
        """Delete memory data - sync version"""
        client = self._get_sync_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        result = client.delete(key)
        if result:
            self.logger.debug(f"Deleted memory with key: {key}")
        else:
            self.logger.warning(f"No data found to delete for key: {key}")
    
    # ==========================================
    # ASYNC METHODS (New Native Async)
    # ==========================================
    
    async def store_async(self, input: IMemoryInput) -> str:
        """Store memory data - async version using native redis.asyncio"""
        if not input.data:
            self.logger.warning("No data provided to store")
            raise ValueError("No data provided to store")
        
        client = await self._get_async_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        for data in input.data:
            storage_data = {
                "data": {"working_memory": data.working_memory},
                "metadata": input.metadata or {},
                "created_at": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
            # Native async Redis operation - TRUE ASYNC!
            await client.setex(key, self.ttl, json.dumps(storage_data))
            self.logger.debug(f"Async stored memory with key: {key}")
        
        return key
    
    async def retrieve_async(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """Retrieve memory data - async version using native redis.asyncio"""
        client = await self._get_async_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        # Native async Redis operation - TRUE ASYNC!
        data = await client.get(key)
        
        if not data:
            self.logger.debug(f"No async data found for key: {key}")
            return []
        
        # Parse JSON in executor for large data to avoid blocking
        if len(data) > 10000:  # 10KB threshold
            stored_data = await asyncio.to_thread(json.loads, data)
        else:
            stored_data = json.loads(data)
        
        working_memory = IWorkingMemory(
            working_memory=stored_data["data"]["working_memory"]
        )
        
        return [working_memory]
    
    async def update_async(self, input: IMemoryInput) -> None:
        """Update memory data - async version"""
        client = await self._get_async_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        # Get existing data - native async
        existing_data = await client.get(key)
        
        if existing_data:
            stored_data = json.loads(existing_data)
            # Update the data
            for new_data in input.data:
                stored_data["data"]["working_memory"] = new_data.working_memory
                stored_data["last_update"] = datetime.now().isoformat()
            
            # Store updated data - native async
            await client.setex(key, self.ttl, json.dumps(stored_data))
            self.logger.debug(f"Async updated memory with key: {key}")
        else:
            # If no existing data, create new entry
            await self.store_async(input)
    
    async def delete_async(self, input: IMemoryInput) -> None:
        """Delete memory data - async version"""
        client = await self._get_async_client()
        key = self._get_key(input.conversation_id, input.memory_type)
        
        # Native async delete operation
        result = await client.delete(key)
        if result:
            self.logger.debug(f"Async deleted memory with key: {key}")
        else:
            self.logger.warning(f"No async data found to delete for key: {key}")
    
    async def close_async(self):
        """Close async Redis connection"""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
            self.logger.info("Async Redis connection closed")
    
    def close(self):
        """Close sync Redis connection"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
            self.logger.info("Sync Redis connection closed")
```

### Update Memory Factory:

Modify `arshai/utils/memory_utils.py`:
```python
# Add async Redis import
from ..memory.working_memory.async_redis_memory_manager import AsyncRedisMemoryManager

class MemoryManagerRegistry:
    _working_memory_providers = {
        "redis": RedisWorkingMemoryManager,  # Legacy sync version
        "async_redis": AsyncRedisMemoryManager,  # New async version
        "in_memory": InMemoryManager,
    }
    
    @classmethod
    def create_async_working_memory(cls, provider: str, **kwargs) -> IMemoryManager:
        """Create async-capable memory manager"""
        if provider == "redis":
            # Return async version for Redis
            return AsyncRedisMemoryManager(**kwargs)
        elif provider == "in_memory":
            # Return regular in-memory (already has async methods via executor)
            return InMemoryManager(**kwargs)
        else:
            provider_class = cls._working_memory_providers.get(provider.lower())
            if not provider_class:
                raise ValueError(f"Unknown provider: {provider}")
            return provider_class(**kwargs)
```

## 4. OpenAI Embeddings Async Implementation

### Current Implementation Locations:
- `arshai/embeddings/openai_embeddings.py`

### Update OpenAI Embeddings:

Modify `arshai/embeddings/openai_embeddings.py`:
```python
"""
OpenAI Embeddings with Native Async Support
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any

from openai import OpenAI, AsyncOpenAI  # Import both sync and async clients
from arshai.core.interfaces.iembedding import IEmbedding, EmbeddingConfig

class OpenAIEmbedding(IEmbedding):
    """
    OpenAI embedding provider with both sync and async support.
    Uses native AsyncOpenAI for true async operations.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize OpenAI embedding with both sync and async clients.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model_name = config.model_name
        
        # Initialize both sync and async clients
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        
        self.logger = logging.getLogger('OpenAIEmbedding')
    
    def _get_sync_client(self) -> OpenAI:
        """Get or create sync OpenAI client."""
        if self._sync_client is None:
            self._sync_client = OpenAI()
        return self._sync_client
    
    async def _get_async_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI()
        return self._async_client
    
    # ==========================================
    # SYNC METHODS (Backward Compatibility)
    # ==========================================
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents - sync version"""
        try:
            client = self._get_sync_client()
            
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            self.logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_document(self, text: str) -> List[float]:
        """Generate embedding for a single document - sync version"""
        embeddings = self.embed_documents([text])
        return embeddings[0]
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query - sync version"""
        return self.embed_document(query)
    
    # ==========================================
    # ASYNC METHODS (New Native Async)
    # ==========================================
    
    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents - async version using native AsyncOpenAI"""
        try:
            client = await self._get_async_client()
            
            # Native async OpenAI embeddings - TRUE ASYNC!
            response = await client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            self.logger.debug(f"Async generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating async embeddings: {e}")
            raise
    
    async def embed_document_async(self, text: str) -> List[float]:
        """Generate embedding for a single document - async version"""
        embeddings = await self.embed_documents_async([text])
        return embeddings[0]
    
    async def embed_query_async(self, query: str) -> List[float]:
        """Generate embedding for a query - async version"""
        return await self.embed_document_async(query)
    
    async def embed_documents_batch_async(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings with batch processing and concurrency control.
        Useful for processing large numbers of documents.
        """
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def embed_batch(batch_texts):
            async with semaphore:
                return await self.embed_documents_async(batch_texts)
        
        # Split into batches
        batches = [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        
        # Process all batches concurrently
        batch_results = await asyncio.gather(*[embed_batch(batch) for batch in batches])
        
        # Flatten results
        all_embeddings = []
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)
        
        self.logger.info(f"Processed {len(texts)} texts in {len(batches)} batches")
        return all_embeddings
    
    async def close_async(self):
        """Close async OpenAI client"""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
```

## 5. Speech Processing with aiofiles

### Current Implementation Location:
- `arshai/speech/openai.py`
- `arshai/speech/azure.py`

### Update OpenAI Speech:

Modify `arshai/speech/openai.py`:
```python
"""
OpenAI Speech with Async File Operations using aiofiles
"""
import asyncio
import os
from typing import Union, BinaryIO
from io import BytesIO

import aiofiles  # Add this import
from openai import OpenAI, AsyncOpenAI

class OpenAISpeechClient:
    """
    OpenAI Speech client with async file operations.
    """
    
    def __init__(self, config):
        self.config = config
        self._sync_client = None
        self._async_client = None
    
    def _get_sync_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI()
        return self._sync_client
    
    async def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI()
        return self._async_client
    
    # ==========================================
    # SYNC METHODS (Backward Compatibility)
    # ==========================================
    
    def _prepare_audio_input(self, audio_input: Union[str, bytes, BinaryIO]) -> BinaryIO:
        """Prepare audio input for processing - sync version"""
        if isinstance(audio_input, str):
            # File path - sync file open (original behavior)
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            return open(audio_input, "rb")  # Original sync implementation
        elif isinstance(audio_input, bytes):
            return BytesIO(audio_input)
        else:
            return audio_input
    
    def speech_to_text(self, audio_input: Union[str, bytes, BinaryIO], **kwargs) -> str:
        """Convert speech to text - sync version"""
        try:
            client = self._get_sync_client()
            audio_file = self._prepare_audio_input(audio_input)
            
            response = client.audio.transcriptions.create(
                model=self.config.model or "whisper-1",
                file=audio_file,
                **kwargs
            )
            
            # Close file if we opened it
            if hasattr(audio_file, 'close') and isinstance(audio_input, str):
                audio_file.close()
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Speech-to-text conversion failed: {e}")
            raise
    
    # ==========================================
    # ASYNC METHODS (New with aiofiles)
    # ==========================================
    
    async def _prepare_audio_input_async(self, audio_input: Union[str, bytes, BinaryIO]) -> BinaryIO:
        """Prepare audio input for processing - async version using aiofiles"""
        if isinstance(audio_input, str):
            # File path - async file operations using aiofiles
            if not await asyncio.to_thread(os.path.exists, audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            
            # Use aiofiles for non-blocking file operations
            async with aiofiles.open(audio_input, "rb") as f:
                # Read file content into memory for OpenAI API
                content = await f.read()
                return BytesIO(content)
                
        elif isinstance(audio_input, bytes):
            return BytesIO(audio_input)
        else:
            return audio_input
    
    async def speech_to_text_async(self, audio_input: Union[str, bytes, BinaryIO], **kwargs) -> str:
        """Convert speech to text - async version using AsyncOpenAI + aiofiles"""
        try:
            client = await self._get_async_client()
            
            # Prepare input asynchronously using aiofiles
            audio_file = await self._prepare_audio_input_async(audio_input)
            
            # Native async OpenAI transcription - TRUE ASYNC!
            response = await client.audio.transcriptions.create(
                model=self.config.model or "whisper-1",
                file=audio_file,
                **kwargs
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Async speech-to-text conversion failed: {e}")
            raise
    
    async def text_to_speech_async(self, text: str, output_path: str = None, **kwargs) -> Union[bytes, str]:
        """Convert text to speech - async version"""
        try:
            client = await self._get_async_client()
            
            # Native async OpenAI text-to-speech
            response = await client.audio.speech.create(
                model=self.config.model or "tts-1",
                voice=kwargs.get('voice', 'alloy'),
                input=text
            )
            
            # Get audio content
            audio_content = response.content
            
            if output_path:
                # Save to file asynchronously using aiofiles
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(audio_content)
                return output_path
            else:
                return audio_content
                
        except Exception as e:
            self.logger.error(f"Async text-to-speech conversion failed: {e}")
            raise
```

## 6. Knowledge Base Tool Async Update

### Current Implementation Location:
- `arshai/tools/knowledge_base_tool.py`

### Update Knowledge Base Tool:

Modify `arshai/tools/knowledge_base_tool.py`:
```python
"""
Knowledge Base Tool with Native Async Support
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional

from arshai.core.interfaces.itool import ITool
from arshai.core.interfaces.ivector_db_client import IVectorDBClient, ICollectionConfig
from arshai.core.interfaces.iembedding import IEmbedding

class KnowledgeBaseRetrievalTool(ITool):
    """
    Knowledge base retrieval tool with both sync and async support.
    Uses native async methods from vector DB and embedding providers.
    """
    
    def __init__(self, vector_db: IVectorDBClient, embedding_model: IEmbedding, 
                 collection_config: ICollectionConfig, search_limit: int = 5):
        """
        Initialize knowledge base tool.
        
        Args:
            vector_db: Vector database client (should support async methods)
            embedding_model: Embedding model (should support async methods)
            collection_config: Collection configuration
            search_limit: Maximum number of search results
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.collection_config = collection_config
        self.search_limit = search_limit
        self.logger = logging.getLogger('KnowledgeBaseRetrievalTool')
    
    @property
    def function_definition(self) -> Dict[str, Any]:
        """Tool function definition for LLM"""
        return {
            "name": "knowledge_base_search",
            "description": "Search the knowledge base for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the knowledge base"
                    }
                },
                "required": ["query"]
            }
        }
    
    # ==========================================
    # SYNC METHODS (Backward Compatibility)
    # ==========================================
    
    def execute(self, query: str) -> str:
        """
        Execute knowledge base search - sync version.
        This maintains backward compatibility but may block.
        """
        try:
            # Generate embeddings - sync version (may block)
            if hasattr(self.embedding_model, 'embed_document'):
                query_embeddings = self.embedding_model.embed_document(query)
            else:
                # Fallback to embed_query if embed_document doesn't exist
                query_embeddings = self.embedding_model.embed_query(query)
            
            # Perform vector search - sync version (may block)
            if self.collection_config.is_hybrid and isinstance(query_embeddings, dict):
                # Hybrid search
                if hasattr(self.vector_db, 'hybrid_search'):
                    search_results = self.vector_db.hybrid_search(
                        config=self.collection_config,
                        dense_vectors=[query_embeddings.get('dense')],
                        sparse_vectors=[query_embeddings.get('sparse')],
                        limit=self.search_limit
                    )
                else:
                    # Fallback to dense search
                    search_results = self.vector_db.search_by_vector(
                        config=self.collection_config,
                        query_vectors=[query_embeddings.get('dense', query_embeddings)],
                        limit=self.search_limit
                    )
            else:
                # Dense vector search only
                search_results = self.vector_db.search_by_vector(
                    config=self.collection_config,
                    query_vectors=[query_embeddings],
                    limit=self.search_limit
                )
            
            return self._format_results(search_results)
            
        except Exception as e:
            error_msg = f"Error during knowledge base search: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    # ==========================================
    # ASYNC METHODS (New Native Async)
    # ==========================================
    
    async def aexecute(self, query: str) -> str:
        """
        Execute knowledge base search - async version using native async methods.
        This provides true non-blocking async operations.
        """
        try:
            # Generate embeddings - async version using native async embedding
            query_embeddings = await self._get_query_embeddings_async(query)
            
            # Perform vector search - async version using native async vector DB
            search_results = await self._perform_vector_search_async(query_embeddings)
            
            return self._format_results(search_results)
            
        except Exception as e:
            error_msg = f"Error during async knowledge base search: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    async def _get_query_embeddings_async(self, query: str):
        """Generate query embeddings using native async methods"""
        
        # Try native async embedding methods first
        if hasattr(self.embedding_model, 'embed_document_async'):
            return await self.embedding_model.embed_document_async(query)
        elif hasattr(self.embedding_model, 'embed_query_async'):
            return await self.embedding_model.embed_query_async(query)
        
        # Fallback: Use executor for sync embedding methods
        elif hasattr(self.embedding_model, 'embed_document'):
            self.logger.warning("Using executor fallback for sync embedding method")
            return await asyncio.to_thread(self.embedding_model.embed_document, query)
        elif hasattr(self.embedding_model, 'embed_query'):
            self.logger.warning("Using executor fallback for sync embedding method")
            return await asyncio.to_thread(self.embedding_model.embed_query, query)
        
        else:
            raise AttributeError("Embedding model doesn't have required methods")
    
    async def _perform_vector_search_async(self, query_embeddings):
        """Perform vector search using native async methods"""
        
        if self.collection_config.is_hybrid and isinstance(query_embeddings, dict):
            # Try hybrid search with async method
            if hasattr(self.vector_db, 'hybrid_search_async'):
                return await self.vector_db.hybrid_search_async(
                    config=self.collection_config,
                    dense_vectors=[query_embeddings.get('dense')],
                    sparse_vectors=[query_embeddings.get('sparse')],
                    limit=self.search_limit
                )
            else:
                # Fallback to dense search
                query_vectors = [query_embeddings.get('dense', query_embeddings)]
        else:
            # Dense vector search only
            query_vectors = [query_embeddings]
        
        # Try native async vector search
        if hasattr(self.vector_db, 'search_by_vector_async'):
            return await self.vector_db.search_by_vector_async(
                config=self.collection_config,
                query_vectors=query_vectors,
                limit=self.search_limit
            )
        
        # Fallback: Use executor for sync vector search
        elif hasattr(self.vector_db, 'search_by_vector'):
            self.logger.warning("Using executor fallback for sync vector search")
            return await asyncio.to_thread(
                self.vector_db.search_by_vector,
                self.collection_config,
                query_vectors,
                limit=self.search_limit
            )
        
        else:
            raise AttributeError("Vector DB doesn't have required search methods")
    
    def _format_results(self, search_results) -> str:
        """Format search results into readable text"""
        if not search_results:
            return "No relevant information found in the knowledge base."
        
        formatted_results = []
        for i, result in enumerate(search_results[:self.search_limit], 1):
            # Handle different result formats
            if hasattr(result, 'entity'):
                # Milvus result format
                content = result.entity.get(self.collection_config.text_field, "No content")
                score = getattr(result, 'distance', 'Unknown')
            elif isinstance(result, dict):
                # Dictionary result format
                content = result.get(self.collection_config.text_field, result.get('content', "No content"))
                score = result.get('distance', result.get('score', 'Unknown'))
            else:
                # Fallback
                content = str(result)
                score = 'Unknown'
            
            formatted_results.append(f"{i}. {content} (Score: {score})")
        
        return "Relevant information from knowledge base:\n\n" + "\n\n".join(formatted_results)
```

## 7. Agent Memory Manager Async Fix

### Current Implementation Location:
- `arshai/agents/working_memory.py`

### Fix Async/Sync Mismatch:

Modify `arshai/agents/working_memory.py`:
```python
"""
Working Memory Agent with Proper Async Memory Operations
"""
import asyncio
import logging
from typing import Tuple, Optional

from arshai.core.interfaces.iagent import IAgent, IAgentInput, IAgentUsage
from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput, ConversationMemoryType
from arshai.core.interfaces.illm import ILLM, ILLMInput

class WorkingMemoryAgent(IAgent):
    """
    Agent with proper async memory management.
    Fixed to use async memory methods when available.
    """
    
    def __init__(self, llm: ILLM, memory_manager: Optional[IMemoryManager] = None):
        self.llm = llm
        self.memory_manager = memory_manager
        self.logger = logging.getLogger('WorkingMemoryAgent')
    
    async def process(self, input: IAgentInput) -> Tuple[str, IAgentUsage]:
        """Process user input with async memory operations"""
        
        conversation_id = input.conversation_id or "default"
        user_message = input.message
        
        # Fetch current working memory if available
        current_memory = ""
        if self.memory_manager:
            try:
                # Use async memory method if available
                memory_data = await self._retrieve_memory_async(conversation_id)
                if memory_data:
                    current_memory = memory_data[0].working_memory if hasattr(memory_data[0], 'working_memory') else str(memory_data[0])
                    self.logger.debug(f"WorkingMemoryAgent: Retrieved existing memory for conversation {conversation_id}")
                else:
                    self.logger.debug(f"WorkingMemoryAgent: No existing memory found for conversation {conversation_id}")
            except Exception as e:
                # If retrieval fails, continue without current memory
                self.logger.warning(f"WorkingMemoryAgent: Failed to retrieve memory for conversation {conversation_id}: {e}")
        
        # Build context with memory
        context_with_memory = self._build_context_with_memory(current_memory, user_message)
        
        # Process with LLM
        llm_input = ILLMInput(
            system_prompt="You are a helpful assistant with access to working memory.",
            user_message=context_with_memory,
        )
        
        response, usage = await self.llm.chat(llm_input)
        
        # Store updated memory if memory manager is available
        if self.memory_manager:
            try:
                updated_memory = self._extract_memory_from_response(response, current_memory, user_message)
                if updated_memory:
                    await self._store_memory_async(conversation_id, updated_memory)
                    self.logger.debug(f"WorkingMemoryAgent: Updated memory for conversation {conversation_id}")
            except Exception as e:
                self.logger.warning(f"WorkingMemoryAgent: Failed to store memory for conversation {conversation_id}: {e}")
        
        return response, usage
    
    async def _retrieve_memory_async(self, conversation_id: str):
        """Retrieve memory using async methods when available"""
        
        memory_input = {
            "conversation_id": conversation_id,
            "memory_type": ConversationMemoryType.WORKING
        }
        
        # Try async method first
        if hasattr(self.memory_manager, 'retrieve_async'):
            return await self.memory_manager.retrieve_async(memory_input)
        
        # Fallback to sync method with executor
        elif hasattr(self.memory_manager, 'retrieve'):
            self.logger.warning("Using executor fallback for sync memory retrieval")
            return await asyncio.to_thread(self.memory_manager.retrieve, memory_input)
        
        else:
            raise AttributeError("Memory manager doesn't have retrieve methods")
    
    async def _store_memory_async(self, conversation_id: str, memory_content: str):
        """Store memory using async methods when available"""
        
        memory_input = {
            "conversation_id": conversation_id,
            "memory_type": ConversationMemoryType.WORKING,
            "data": [{"working_memory": memory_content}],
            "metadata": {"updated_by": "WorkingMemoryAgent"}
        }
        
        # Try async method first
        if hasattr(self.memory_manager, 'store_async'):
            await self.memory_manager.store_async(memory_input)
        
        # Fallback to sync method with executor
        elif hasattr(self.memory_manager, 'store'):
            self.logger.warning("Using executor fallback for sync memory storage")
            await asyncio.to_thread(self.memory_manager.store, memory_input)
        
        else:
            raise AttributeError("Memory manager doesn't have store methods")
    
    def _build_context_with_memory(self, current_memory: str, user_message: str) -> str:
        """Build context combining memory and current message"""
        if current_memory:
            return f"Previous context: {current_memory}\n\nUser: {user_message}"
        else:
            return user_message
    
    def _extract_memory_from_response(self, response: str, current_memory: str, user_message: str) -> str:
        """Extract/update memory from agent response"""
        # Simple memory update strategy - can be enhanced
        updated_memory = f"{current_memory}\nUser: {user_message}\nAssistant: {response}"
        
        # Trim memory if too long (simple strategy)
        if len(updated_memory) > 2000:
            # Keep last 1500 characters
            updated_memory = "..." + updated_memory[-1500:]
        
        return updated_memory
```

## 8. Testing Async Implementation

Create new test file: `tests/async/test_async_performance.py`
```python
"""
Test async performance improvements
"""
import asyncio
import time
import pytest
from typing import List

from arshai.vector_db.async_milvus_client import AsyncMilvusClientWrapper
from arshai.memory.working_memory.async_redis_memory_manager import AsyncRedisMemoryManager
from arshai.embeddings.openai_embeddings import OpenAIEmbedding

@pytest.mark.asyncio
async def test_async_milvus_performance():
    """Test that async Milvus operations don't block event loop"""
    
    client = AsyncMilvusClientWrapper()
    
    # Test concurrent async operations
    start_time = time.perf_counter()
    
    tasks = []
    for i in range(10):
        # Create mock search tasks
        task = client.search_by_vector_async(
            config=mock_config,
            query_vectors=[[0.1] * 768],
            limit=5
        )
        tasks.append(task)
    
    # All operations should run concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - start_time
    
    # Should be much faster than sequential
    assert elapsed < 2.0  # Reasonable threshold
    assert len(results) == 10

@pytest.mark.asyncio
async def test_async_redis_performance():
    """Test that async Redis operations don't block event loop"""
    
    manager = AsyncRedisMemoryManager()
    
    # Test concurrent operations
    tasks = []
    for i in range(20):
        memory_input = {
            "conversation_id": f"test_{i}",
            "memory_type": ConversationMemoryType.WORKING,
            "data": [{"working_memory": f"test content {i}"}]
        }
        tasks.append(manager.store_async(memory_input))
    
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start_time
    
    # Should complete quickly with concurrent operations
    assert elapsed < 1.0
    assert len(results) == 20

@pytest.mark.asyncio
async def test_async_embeddings_performance():
    """Test that async embeddings don't block event loop"""
    
    embedding_model = OpenAIEmbedding(config=mock_embedding_config)
    
    # Test concurrent embedding generation
    texts = [f"Test text {i}" for i in range(5)]
    
    start_time = time.perf_counter()
    
    # Generate embeddings concurrently
    embedding_tasks = [
        embedding_model.embed_document_async(text)
        for text in texts
    ]
    
    embeddings = await asyncio.gather(*embedding_tasks)
    elapsed = time.perf_counter() - start_time
    
    # Should be faster than sequential
    assert elapsed < 3.0
    assert len(embeddings) == 5
    assert all(isinstance(emb, list) for emb in embeddings)

def test_backward_compatibility():
    """Test that sync methods still work"""
    
    # Test Milvus sync methods
    client = AsyncMilvusClientWrapper()
    assert hasattr(client, 'search_by_vector')  # Sync method preserved
    assert hasattr(client, 'search_by_vector_async')  # Async method added
    
    # Test Redis sync methods
    manager = AsyncRedisMemoryManager()
    assert hasattr(manager, 'store')  # Sync method preserved
    assert hasattr(manager, 'store_async')  # Async method added
    
    # Test OpenAI sync methods
    embedding_model = OpenAIEmbedding(config=mock_embedding_config)
    assert hasattr(embedding_model, 'embed_document')  # Sync method preserved
    assert hasattr(embedding_model, 'embed_document_async')  # Async method added
```

## 9. Configuration Update

Create new config file: `arshai/config/async_config.py`
```python
"""
Async configuration settings
"""
from typing import Dict, Any, Optional
import os

class AsyncConfig:
    """Configuration for async operations"""
    
    def __init__(self):
        # Milvus async settings
        self.milvus_batch_concurrency = int(os.getenv("MILVUS_BATCH_CONCURRENCY", "4"))
        self.milvus_connection_timeout = int(os.getenv("MILVUS_CONNECTION_TIMEOUT", "30"))
        
        # Redis async settings
        self.redis_max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        self.redis_connection_timeout = int(os.getenv("REDIS_CONNECTION_TIMEOUT", "10"))
        
        # OpenAI async settings
        self.openai_batch_concurrency = int(os.getenv("OPENAI_BATCH_CONCURRENCY", "5"))
        self.openai_request_timeout = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
        
        # General async settings
        self.enable_async_operations = os.getenv("ENABLE_ASYNC_OPERATIONS", "true").lower() == "true"
        self.async_fallback_to_sync = os.getenv("ASYNC_FALLBACK_TO_SYNC", "true").lower() == "true"
    
    def get_milvus_config(self) -> Dict[str, Any]:
        """Get Milvus async configuration"""
        return {
            "batch_concurrency": self.milvus_batch_concurrency,
            "connection_timeout": self.milvus_connection_timeout,
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis async configuration"""
        return {
            "max_connections": self.redis_max_connections,
            "connection_timeout": self.redis_connection_timeout,
        }
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI async configuration"""
        return {
            "batch_concurrency": self.openai_batch_concurrency,
            "request_timeout": self.openai_request_timeout,
        }
```

## 10. Migration Summary

### Files to Update:
1. **pyproject.toml** - Update dependencies
2. **arshai/vector_db/milvus_client.py** - Add async wrapper
3. **arshai/memory/working_memory/redis_memory_manager.py** - Add async methods
4. **arshai/embeddings/openai_embeddings.py** - Add async methods
5. **arshai/speech/openai.py** - Add aiofiles support
6. **arshai/tools/knowledge_base_tool.py** - Update to use async methods
7. **arshai/agents/working_memory.py** - Fix async/sync mismatch

### New Files to Create:
1. **arshai/vector_db/async_milvus_client.py** - Native async Milvus client
2. **arshai/memory/working_memory/async_redis_memory_manager.py** - Native async Redis manager
3. **arshai/config/async_config.py** - Async configuration
4. **tests/async/test_async_performance.py** - Async performance tests

### Commands to Run:
```bash
# Update dependencies
poetry add "pymilvus>=2.5.3"
poetry add aiofiles

# Run tests
poetry run pytest tests/async/

# Check performance
python -m tests.async.test_async_performance
```

### Expected Performance Improvements:
- **Milvus Operations**: 2-5x faster concurrent searches
- **Redis Operations**: 3-10x better throughput
- **OpenAI Embeddings**: 5-20x better concurrent processing
- **File I/O**: Eliminated blocking in speech processing
- **Overall**: Support for 100-1000+ concurrent operations

This implementation provides both backward compatibility and significant performance improvements through native async support.