"""Memory manager entry point for all memory types."""

from datetime import datetime
from typing import Dict, Optional, Type, Any, List
from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput, IWorkingMemory
from .memory_types import ConversationMemoryType
from ..utils.logging import get_logger

logger = get_logger(__name__)

class MemoryManagerService:
    """Service for managing different types of memory."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory manager service with configuration.
        
        Args:
            config: Configuration dictionary with memory settings
        """
        self.config = config
        self.working_memory_provider = config.get("working_memory", {}).get("provider")
        self.working_memory_ttl = config.get("working_memory", {}).get("ttl", 60 * 60 * 12)  # 12 hours
        
        # Initialize working memory manager
        self.working_memory_manager = self._init_working_memory()
        logger.info(f"Initialized memory manager service with provider: {self.working_memory_provider}")
    
    def _init_working_memory(self) -> IMemoryManager:
        """
        Initialize the working memory manager.
        
        This method only passes structural configuration to the factory.
        Sensitive data like connection URLs should be read from environment
        variables directly by the memory implementation.
        """
        try:
            # Import here to avoid circular import
            from ..factories.memory_factory import MemoryFactory
            
            kwargs = {
                "ttl": self.working_memory_ttl
            }
            
            # Don't pass sensitive data, let components read from environment
            return MemoryFactory.create_working_memory(
                provider=self.working_memory_provider,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to initialize working memory manager: {str(e)}")
            # Fall back to in-memory provider
            logger.info("Falling back to in-memory provider")
            
            # Import again inside exception handler to ensure it's available
            from ..factories.memory_factory import MemoryFactory
            return MemoryFactory.create_working_memory(
                provider="in_memory",
                ttl=self.working_memory_ttl
            )
    
    def store_working_memory(self, 
                            conversation_id: str, 
                            memory_data: IWorkingMemory, 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store working memory for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            memory_data: Working memory data to store
            metadata: Optional metadata for the memory
            
        Returns:
            str: Memory ID for the stored data
        """

        logger.info(f"Storing working memory: {memory_data}")

        if not hasattr(memory_data, "working_memory"):
            # Direct conversion to string using string representation
            import json
            try:
                # Try to convert using JSON for clean formatting
                string_working_memory = json.dumps(memory_data, indent=2)
            except (TypeError, ValueError):
                # Fall back to simple string representation if JSON fails
                string_working_memory = str(memory_data)
            
            memory_data = IWorkingMemory(working_memory=string_working_memory)


        memory_input = IMemoryInput(
            conversation_id=conversation_id,
            memory_type=ConversationMemoryType.WORKING_MEMORY,
            data=[memory_data],
            metadata=metadata
        )
        
        return self.working_memory_manager.store(memory_input)
    
    def retrieve_working_memory(self, conversation_id: Optional[str]) -> Optional[IWorkingMemory]:
        """
        Retrieve working memory for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Optional[IWorkingMemory]: Working memory data if found, None otherwise
        """
        if not conversation_id:
            return IWorkingMemory.initialize_memory()
        
        memory_input = IMemoryInput(
            conversation_id=conversation_id,
            memory_type=ConversationMemoryType.WORKING_MEMORY
        )
        
        result = self.working_memory_manager.retrieve(memory_input)
        if result:
            return result[0]
        
        # Initialize new working memory if none exists
        return IWorkingMemory.initialize_memory()
    
    def update_working_memory(self, 
                             conversation_id: str, 
                             memory_data: IWorkingMemory) -> None:
        """
        Update working memory for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            memory_data: Updated working memory data
        """
        memory_input = IMemoryInput(
            conversation_id=conversation_id,
            memory_type=ConversationMemoryType.WORKING_MEMORY,
            data=[memory_data]
        )
        
        self.working_memory_manager.update(memory_input)
    
    def delete_working_memory(self, conversation_id: str) -> None:
        """
        Delete working memory for a conversation.
        
        Args:
            conversation_id: ID of the conversation
        """
        memory_input = IMemoryInput(
            conversation_id=conversation_id,
            memory_type=ConversationMemoryType.WORKING_MEMORY
        )
        
        self.working_memory_manager.delete(memory_input) 