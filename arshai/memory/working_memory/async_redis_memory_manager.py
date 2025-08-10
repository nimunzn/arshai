"""
AsyncRedisMemoryManager implementation using native redis.asyncio support.

This module provides native async Redis operations for working memory management
while maintaining backward compatibility with sync interfaces.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

# Import both sync and async Redis
import redis
import redis.asyncio as async_redis

from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput, IWorkingMemory
from ..memory_types import ConversationMemoryType

logger = logging.getLogger(__name__)


class AsyncRedisMemoryManager(IMemoryManager):
    """
    Async Redis implementation of working memory management.
    
    Provides both sync and async methods for backward compatibility while
    leveraging native redis.asyncio for better performance.
    """
    
    def __init__(self, storage_url: str = None, **kwargs):
        """
        Initialize the async Redis working memory manager.
        
        Args:
            storage_url: Redis connection URL (if not provided, will be read from REDIS_URL env var)
            **kwargs: Additional configuration parameters
                - ttl: Time to live in seconds (default: 12 hours)
                - max_connections: Max connections in async pool (default: 20)
                - retry_on_timeout: Whether to retry on timeout (default: True)
        """
        # Get Redis URL from parameter or environment variable
        self.storage_url = storage_url or os.environ.get("REDIS_URL", "redis://localhost:6379/1")
        
        # Configuration
        self.prefix = "memory"
        self.ttl = kwargs.get('ttl', 60 * 60 * 12)  # 12 hours default
        self.max_connections = kwargs.get('max_connections', 20)
        self.retry_on_timeout = kwargs.get('retry_on_timeout', True)
        
        # Connection pools (lazy initialization)
        self._async_redis_client: Optional[async_redis.Redis] = None
        self._sync_redis_client: Optional[redis.Redis] = None
        
        # Check if REDIS_URL is set when storage_url is not provided
        if not storage_url and not os.environ.get("REDIS_URL"):
            logger.warning("No REDIS_URL environment variable set, using default: redis://localhost:6379/1")
            
        logger.info(
            f"Initialized AsyncRedisMemoryManager with URL: {self.storage_url}, "
            f"TTL: {self.ttl}s, max_connections: {self.max_connections}"
        )
    
    async def _get_async_redis_client(self) -> async_redis.Redis:
        """
        Get or create async Redis client with connection pooling.
        
        Returns:
            async_redis.Redis: Async Redis client instance
        """
        if self._async_redis_client is None:
            self._async_redis_client = async_redis.from_url(
                self.storage_url,
                decode_responses=True,
                max_connections=self.max_connections,
                retry_on_timeout=self.retry_on_timeout,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info("Created new async Redis client with connection pooling")
        return self._async_redis_client
    
    def _get_sync_redis_client(self) -> redis.Redis:
        """
        Get or create sync Redis client for backward compatibility.
        
        Returns:
            redis.Redis: Sync Redis client instance
        """
        if self._sync_redis_client is None:
            self._sync_redis_client = redis.from_url(
                self.storage_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info("Created new sync Redis client")
        return self._sync_redis_client
    
    def _get_key(self, conversation_id: str, memory_type: ConversationMemoryType) -> str:
        """
        Generate Redis key for a memory entry.
        
        Args:
            conversation_id: ID of the conversation
            memory_type: Type of memory
            
        Returns:
            str: Generated Redis key
        """
        return f"{self.prefix}:{memory_type}:{conversation_id}"
    
    # ----------------------
    # Sync Methods (Backward Compatibility)
    # ----------------------
    
    def store(self, input: IMemoryInput) -> str:
        """Sync store - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.store_async(input))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.store_async(input))
    
    def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """Sync retrieve - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.retrieve_async(input))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.retrieve_async(input))
    
    def update(self, input: IMemoryInput) -> None:
        """Sync update - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.update_async(input))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.update_async(input))
    
    def delete(self, input: IMemoryInput) -> None:
        """Sync delete - runs async version in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.delete_async(input))
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(self.delete_async(input))
    
    # ----------------------
    # Async Methods (New - Native Performance)
    # ----------------------
    
    async def store_async(self, input: IMemoryInput) -> str:
        """
        Async store memory data in Redis - true async, no executor needed.
        
        Args:
            input: Memory input containing data to store
            
        Returns:
            str: Key for the stored data
            
        Raises:
            ValueError: If no data is provided to store
        """
        if not input.data:
            logger.warning("No data provided to store")
            raise ValueError("No data provided to store")
            
        try:
            redis_client = await self._get_async_redis_client()
            key = self._get_key(input.conversation_id, input.memory_type)
            
            for data in input.data:
                # Prepare storage data
                storage_data = {
                    "data": {"working_memory": data.working_memory},
                    "metadata": input.metadata or {},
                    "created_at": datetime.now().isoformat(),
                    "last_update": datetime.now().isoformat()
                }
                
                # True async Redis operation - no blocking!
                await redis_client.setex(
                    key,
                    self.ttl,
                    json.dumps(storage_data)
                )
                logger.debug(f"Async stored memory with key: {key}")
            
            return key
            
        except Exception as e:
            logger.error(f"Error in async store: {str(e)}")
            raise
    
    async def retrieve_async(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """
        Async retrieve memory data from Redis - true async, no executor needed.
        
        Args:
            input: Memory input containing query parameters
            
        Returns:
            List[IWorkingMemory]: List of matching memory objects
        """
        try:
            redis_client = await self._get_async_redis_client()
            key = self._get_key(input.conversation_id, input.memory_type)
            
            # True async Redis operation - no blocking!
            data = await redis_client.get(key)
            
            if not data:
                logger.debug(f"No data found for key: {key}")
                return []
                
            stored_data = json.loads(data)
            working_memory = stored_data["data"]["working_memory"]
            return [IWorkingMemory(working_memory=working_memory)]
            
        except Exception as e:
            logger.error(f"Error in async retrieve: {str(e)}")
            return []
    
    async def update_async(self, input: IMemoryInput) -> None:
        """
        Async update memory data in Redis - true async, no executor needed.
        
        Args:
            input: Memory input containing update data
            
        Raises:
            ValueError: If no data is provided to update
        """
        if not input.data:
            logger.warning("No data provided to update")
            raise ValueError("No data provided to update")
            
        try:
            redis_client = await self._get_async_redis_client()
            key = self._get_key(input.conversation_id, input.memory_type)
            
            # True async Redis operation - no blocking!
            existing_data = await redis_client.get(key)
            
            if existing_data:
                stored_data = json.loads(existing_data)
                
                for data in input.data:
                    stored_data["data"]["working_memory"] = data.working_memory
                    stored_data["last_update"] = datetime.now().isoformat()
                
                # True async Redis operation - no blocking!
                await redis_client.setex(
                    key,
                    self.ttl,
                    json.dumps(stored_data)
                )
                logger.debug(f"Async updated memory with key: {key}")
            else:
                logger.warning(f"No existing data found for key: {key}")
                
        except Exception as e:
            logger.error(f"Error in async update: {str(e)}")
            raise
    
    async def delete_async(self, input: IMemoryInput) -> None:
        """
        Async delete memory data from Redis - true async, no executor needed.
        
        Args:
            input: Memory input identifying data to delete
        """
        try:
            redis_client = await self._get_async_redis_client()
            key = self._get_key(input.conversation_id, input.memory_type)
            
            # True async Redis operation - no blocking!
            result = await redis_client.delete(key)
            
            if result:
                logger.debug(f"Async deleted memory with key: {key}")
            else:
                logger.debug(f"No data found to delete for key: {key}")
                
        except Exception as e:
            logger.error(f"Error in async delete: {str(e)}")
            raise
    
    # ----------------------
    # Batch Operations (Async Performance Boost)
    # ----------------------
    
    async def store_batch_async(self, inputs: List[IMemoryInput]) -> List[str]:
        """
        Async batch store multiple memory entries - leverages Redis pipelining.
        
        Args:
            inputs: List of memory inputs to store
            
        Returns:
            List[str]: List of keys for stored data
        """
        if not inputs:
            return []
            
        try:
            redis_client = await self._get_async_redis_client()
            
            # Use Redis pipeline for batch operations
            pipe = redis_client.pipeline()
            keys = []
            
            for input_data in inputs:
                if not input_data.data:
                    continue
                    
                key = self._get_key(input_data.conversation_id, input_data.memory_type)
                keys.append(key)
                
                for data in input_data.data:
                    storage_data = {
                        "data": {"working_memory": data.working_memory},
                        "metadata": input_data.metadata or {},
                        "created_at": datetime.now().isoformat(),
                        "last_update": datetime.now().isoformat()
                    }
                    
                    pipe.setex(key, self.ttl, json.dumps(storage_data))
            
            # Execute all operations in a single pipeline - much faster!
            await pipe.execute()
            logger.info(f"Async batch stored {len(keys)} memory entries")
            
            return keys
            
        except Exception as e:
            logger.error(f"Error in async batch store: {str(e)}")
            raise
    
    async def retrieve_batch_async(self, inputs: List[IMemoryInput]) -> Dict[str, List[IWorkingMemory]]:
        """
        Async batch retrieve multiple memory entries - leverages Redis pipelining.
        
        Args:
            inputs: List of memory inputs to retrieve
            
        Returns:
            Dict[str, List[IWorkingMemory]]: Dictionary mapping keys to memory objects
        """
        if not inputs:
            return {}
            
        try:
            redis_client = await self._get_async_redis_client()
            
            # Use Redis pipeline for batch operations
            keys = []
            key_to_input = {}
            
            for input_data in inputs:
                key = self._get_key(input_data.conversation_id, input_data.memory_type)
                keys.append(key)
                key_to_input[key] = input_data
            
            # Batch retrieve all keys at once - much faster!
            values = await redis_client.mget(keys)
            
            results = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        stored_data = json.loads(value)
                        working_memory = stored_data["data"]["working_memory"]
                        results[key] = [IWorkingMemory(working_memory=working_memory)]
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error parsing data for key {key}: {str(e)}")
                        results[key] = []
                else:
                    results[key] = []
            
            logger.debug(f"Async batch retrieved {len(keys)} memory entries")
            return results
            
        except Exception as e:
            logger.error(f"Error in async batch retrieve: {str(e)}")
            return {key: [] for key in keys}
    
    # ----------------------
    # Health and Monitoring
    # ----------------------
    
    async def health_check_async(self) -> Dict[str, Any]:
        """
        Async health check for Redis connection.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            redis_client = await self._get_async_redis_client()
            
            # Test connection with ping
            start_time = datetime.now()
            pong = await redis_client.ping()
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await redis_client.info()
            
            return {
                "status": "healthy" if pong else "unhealthy",
                "latency_ms": round(latency, 2),
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": None
            }
    
    async def get_memory_stats_async(self) -> Dict[str, Any]:
        """
        Async get memory usage statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        try:
            redis_client = await self._get_async_redis_client()
            
            # Count keys with our prefix
            pattern = f"{self.prefix}:*"
            keys = await redis_client.keys(pattern)
            
            # Get memory usage info
            info = await redis_client.info("memory")
            
            return {
                "total_keys": len(keys),
                "pattern": pattern,
                "redis_used_memory": info.get("used_memory", 0),
                "redis_used_memory_human": info.get("used_memory_human", "0B"),
                "redis_peak_memory": info.get("used_memory_peak", 0),
                "redis_peak_memory_human": info.get("used_memory_peak_human", "0B")
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}
    
    # ----------------------
    # Connection Management
    # ----------------------
    
    async def close_async(self):
        """Close async Redis connection and cleanup."""
        if self._async_redis_client:
            await self._async_redis_client.close()
            self._async_redis_client = None
            logger.info("Closed async Redis connection")
    
    def close_sync(self):
        """Close sync Redis connection."""
        if self._sync_redis_client:
            self._sync_redis_client.close()
            self._sync_redis_client = None
            logger.info("Closed sync Redis connection")
    
    def __del__(self):
        """Cleanup on object destruction."""
        # Close sync client if it exists
        if self._sync_redis_client:
            try:
                self._sync_redis_client.close()
            except:
                pass