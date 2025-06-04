"""In-memory implementation of the working memory manager."""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from arshai.core.interfaces import IMemoryManager, IMemoryInput, IWorkingMemory
from ..memory_types import ConversationMemoryType

logger = logging.getLogger(__name__)

class InMemoryManager(IMemoryManager):
    """In-memory implementation of working memory management."""
    
    def __init__(self, **kwargs):
        """
        Initialize the in-memory storage.
        
        Args:
            **kwargs: Optional configuration parameters
                - ttl: Time to live in seconds
        """
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.prefix = "memory"
        
        # Read TTL from config or use default
        self.ttl = kwargs.get('ttl', 60 * 60 * 12)  # 12 hours default
        
        logger.info(f"Initialized InMemoryManager with TTL: {self.ttl} seconds")
    
    def _get_key(self, conversation_id: str, memory_type: ConversationMemoryType) -> str:
        """
        Generate a storage key for a memory entry.
        
        Args:
            conversation_id: ID of the conversation
            memory_type: Type of memory
            
        Returns:
            str: Generated key
        """
        return f"{self.prefix}:{memory_type}:{conversation_id}"
    
    def _clear_expired_memory(self):
        """
        Clean up expired memory entries based on TTL.
        """
        current_time = datetime.now()
        keys_to_delete = []
        
        for key, data in self.storage.items():
            if "created_at" in data:
                try:
                    created_at = datetime.fromisoformat(data["created_at"])
                    if current_time - created_at > timedelta(seconds=self.ttl):
                        keys_to_delete.append(key)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid timestamp format for key {key}")
        
        for key in keys_to_delete:
            del self.storage[key]
            logger.debug(f"Removed expired memory with key: {key}")
    
    def store(self, input: IMemoryInput) -> str:
        """
        Store memory data in the in-memory storage.
        
        Args:
            input: Memory input containing data to store
            
        Returns:
            str: Key for the stored data
        """
        if not input.data:
            logger.warning("No data provided to store")
            raise ValueError("No data provided to store")
        
        # Clear expired entries
        self._clear_expired_memory()
            
        key = self._get_key(input.conversation_id, input.memory_type)
        
        for data in input.data:
            # Store data
            storage_data = {
                "data": {"working_memory": data.working_memory},
                "metadata": input.metadata or {},
                "created_at": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
            self.storage[key] = storage_data
            logger.debug(f"Stored memory with key: {key}")
            
        return key

    def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """
        Retrieve memory data from the in-memory storage.
        
        Args:
            input: Memory input containing query parameters
            
        Returns:
            List[IWorkingMemory]: List of matching memory objects
        """
        # Clear expired entries
        self._clear_expired_memory()
        
        key = self._get_key(input.conversation_id, input.memory_type)
        data = self.storage.get(key)
        
        if not data:
            logger.debug(f"No data found for key: {key}")
            return []
            
        working_memory = data["data"]["working_memory"]
        return [IWorkingMemory(working_memory=working_memory)]

    def update(self, input: IMemoryInput) -> None:
        """
        Update memory data in the in-memory storage.
        
        Args:
            input: Memory input containing update data
        """
        if not input.data:
            logger.warning("No data provided to update")
            raise ValueError("No data provided to update")
            
        key = self._get_key(input.conversation_id, input.memory_type)
        existing_data = self.storage.get(key)
        
        if existing_data:
            for data in input.data:
                existing_data["data"]["working_memory"] = data.working_memory
                existing_data["last_update"] = datetime.now().isoformat()
                
            self.storage[key] = existing_data
            logger.debug(f"Updated memory with key: {key}")
        else:
            logger.warning(f"No existing data found for key: {key}")

    def delete(self, input: IMemoryInput) -> None:
        """
        Delete memory data from the in-memory storage.
        
        Args:
            input: Memory input identifying data to delete
        """
        key = self._get_key(input.conversation_id, input.memory_type)
        if key in self.storage:
            del self.storage[key]
            logger.debug(f"Deleted memory with key: {key}")
        else:
            logger.debug(f"No data found to delete for key: {key}") 