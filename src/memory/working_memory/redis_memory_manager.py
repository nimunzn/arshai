import json
import redis
import os
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from arshai.core.interfaces import IMemoryManager, IMemoryInput, IWorkingMemory
from ..memory_types import ConversationMemoryType
import logging

logger = logging.getLogger(__name__)

class RedisWorkingMemoryManager(IMemoryManager):
    """Redis implementation of working memory management"""
    
    def __init__(self, storage_url: str = None, **kwargs):
        """
        Initialize the Redis working memory manager.
        
        Args:
            storage_url: Redis connection URL (if not provided, will be read from REDIS_URL env var)
            **kwargs: Additional configuration parameters
        """
        # Get Redis URL from parameter or environment variable
        self.storage_url = storage_url or os.environ.get("REDIS_URL", "redis://localhost:6379/1")
        
        # Check if REDIS_URL is set when storage_url is not provided
        if not storage_url and not os.environ.get("REDIS_URL"):
            logger.warning("No REDIS_URL environment variable set, using default: redis://localhost:6379/1")
            
        self.redis_client = redis.from_url(self.storage_url)
        self.prefix = "memory"
        self.ttl = kwargs.get('ttl', 60 * 60 * 12)  # 12 hours default
        logger.info(f"Initialized Redis working memory manager with URL: {self.storage_url}")
    
    def _get_key(self, conversation_id: str, memory_type: ConversationMemoryType) -> str:
        """Generate Redis key for a memory entry"""
        return f"{self.prefix}:{memory_type}:{conversation_id}"
    
    def store(self, input: IMemoryInput) -> str:
        """Store memory data in Redis"""
        if not input.data:
            logger.warning("No data provided to store")
            raise ValueError("No data provided to store")
            
        key = self._get_key(input.conversation_id, input.memory_type)
        
        for data in input.data:
            # Store data
            storage_data = {
                "data": {"working_memory": data.working_memory},
                "metadata": input.metadata or {},
                "created_at": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(storage_data)
            )
            logger.debug(f"Stored memory with key: {key}")
        
        return key

    def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """Retrieve memory data from Redis"""
        key = self._get_key(input.conversation_id, input.memory_type)
        data = self.redis_client.get(key)
        
        if not data:
            logger.debug(f"No data found for key: {key}")
            return []
            
        stored_data = json.loads(data)
        working_memory = stored_data["data"]["working_memory"]
        return [IWorkingMemory(working_memory=working_memory)]

    def update(self, input: IMemoryInput) -> None:
        """Update memory data in Redis"""
        if not input.data:
            logger.warning("No data provided to update")
            raise ValueError("No data provided to update")
            
        key = self._get_key(input.conversation_id, input.memory_type)
        existing_data = self.redis_client.get(key)
        
        if existing_data:
            stored_data = json.loads(existing_data)
            
            for data in input.data:
                stored_data["data"]["working_memory"] = data.working_memory
                stored_data["last_update"] = datetime.now().isoformat()
            
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(stored_data)
            )
            logger.debug(f"Updated memory with key: {key}")
        else:
            logger.warning(f"No existing data found for key: {key}")

    def delete(self, input: IMemoryInput) -> None:
        """Delete memory data from Redis"""
        key = self._get_key(input.conversation_id, input.memory_type)
        result = self.redis_client.delete(key)
        if result:
            logger.debug(f"Deleted memory with key: {key}")
        else:
            logger.debug(f"No data found to delete for key: {key}")
