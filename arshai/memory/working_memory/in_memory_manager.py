"""In-memory implementation of the working memory manager."""

import json
import os
import asyncio
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import OrderedDict
import logging
from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput, IWorkingMemory
from ..memory_types import ConversationMemoryType

logger = logging.getLogger(__name__)

class InMemoryManager(IMemoryManager):
    """In-memory implementation of working memory management."""
    
    def __init__(self, **kwargs):
        """
        Initialize the in-memory storage with memory management.
        
        Args:
            **kwargs: Optional configuration parameters
                - ttl: Time to live in seconds
                - max_entries: Maximum number of entries (default: 10000)
                - max_memory_mb: Maximum memory usage in MB (default: 500)
        """
        # Use OrderedDict for LRU eviction
        self.storage: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.prefix = "memory"
        
        # Read configuration from kwargs or use defaults
        self.ttl = kwargs.get('ttl', 60 * 60 * 12)  # 12 hours default
        self.max_entries = kwargs.get('max_entries', 10000)
        self.max_memory_mb = kwargs.get('max_memory_mb', 500)  # 500MB limit
        
        # Memory management tracking
        self._access_times: Dict[str, float] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5 minutes
        
        # Start proactive cleanup if event loop is running
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running yet
            pass
        
        logger.info(
            f"Initialized InMemoryManager with TTL: {self.ttl}s, "
            f"max_entries: {self.max_entries}, max_memory: {self.max_memory_mb}MB"
        )
    
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
        
        # Check limits before storing
        if len(self.storage) >= self.max_entries:
            self._evict_lru_entries()
            
        key = self._get_key(input.conversation_id, input.memory_type)
        
        for data in input.data:
            # Store data
            storage_data = {
                "data": {"working_memory": data.working_memory},
                "metadata": input.metadata or {},
                "created_at": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
            # Update storage with LRU tracking
            self.storage[key] = storage_data
            self.storage.move_to_end(key)  # Mark as recently used
            self._access_times[key] = time.time()
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
        
        # Mark as recently accessed for LRU
        self.storage.move_to_end(key)
        self._access_times[key] = time.time()
            
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
            if key in self._access_times:
                del self._access_times[key]
            logger.debug(f"Deleted memory with key: {key}")
        else:
            logger.debug(f"No data found to delete for key: {key}")
    
    async def _periodic_cleanup(self) -> None:
        """
        Periodically clean up expired entries and enforce memory limits.
        Runs every cleanup_interval seconds.
        """
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                # Clear expired entries
                self._clear_expired_memory()
                
                # Check and enforce memory limits
                if len(self.storage) > self.max_entries * 0.8:  # 80% threshold
                    entries_to_remove = len(self.storage) - int(self.max_entries * 0.7)  # Target 70%
                    self._evict_lru_entries(entries_to_remove)
                    logger.info(f"Evicted {entries_to_remove} LRU entries to maintain memory limits")
                
                # Check memory usage
                estimated_memory_mb = self._estimate_memory_usage()
                if estimated_memory_mb > self.max_memory_mb * 0.8:  # 80% threshold
                    entries_to_remove = max(100, int(len(self.storage) * 0.1))  # Remove 10% or 100 entries
                    self._evict_lru_entries(entries_to_remove)
                    logger.info(f"Evicted {entries_to_remove} entries due to memory pressure ({estimated_memory_mb:.1f}MB)")
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {str(e)}")
    
    def _evict_lru_entries(self, count: int = None) -> None:
        """
        Evict least recently used entries to free memory.
        
        Args:
            count: Number of entries to evict. If None, evicts until under limits.
        """
        if not self.storage:
            return
        
        # Determine how many to evict
        if count is None:
            if len(self.storage) > self.max_entries:
                count = len(self.storage) - self.max_entries
            else:
                return
        
        # Get LRU entries - OrderedDict maintains insertion/access order
        # Items at the beginning are least recently used
        evicted_count = 0
        keys_to_evict = []
        
        for key in list(self.storage.keys()):
            if evicted_count >= count:
                break
            keys_to_evict.append(key)
            evicted_count += 1
        
        # Remove the identified entries
        for key in keys_to_evict:
            if key in self.storage:
                del self.storage[key]
            if key in self._access_times:
                del self._access_times[key]
        
        if keys_to_evict:
            logger.debug(f"Evicted {len(keys_to_evict)} LRU entries")
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage of stored data in MB.
        
        Returns:
            float: Estimated memory usage in megabytes
        """
        if not self.storage:
            return 0.0
        
        # Sample a few entries to estimate average size
        sample_size = min(10, len(self.storage))
        sample_keys = list(self.storage.keys())[:sample_size]
        
        total_sample_size = 0
        for key in sample_keys:
            try:
                # Estimate size of key + value
                key_size = sys.getsizeof(key)
                value_size = sys.getsizeof(str(self.storage[key]))  # Rough estimate
                total_sample_size += key_size + value_size
            except Exception:
                # If estimation fails, use a conservative estimate
                total_sample_size += 1024  # 1KB per entry
        
        # Calculate average size per entry
        avg_size_per_entry = total_sample_size / sample_size if sample_size > 0 else 1024
        
        # Estimate total memory usage
        total_entries = len(self.storage)
        total_size_bytes = total_entries * avg_size_per_entry
        
        # Add overhead for data structures (OrderedDict, access_times dict)
        overhead_bytes = total_entries * 200  # ~200 bytes overhead per entry
        
        total_size_mb = (total_size_bytes + overhead_bytes) / (1024 * 1024)
        
        return total_size_mb 