"""Mock memory implementation for testing."""

from typing import Dict, List, Optional, Any

from arshai.core.interfaces import (
    IMemoryManager,
    IMemoryInput,
    IWorkingMemory,
    ConversationMemoryType
)


class MockMemoryManager(IMemoryManager):
    """A mock memory manager implementation for testing."""

    def __init__(self, initial_memory: Optional[Dict[str, List[IWorkingMemory]]] = None):
        """
        Initialize the mock memory manager.
        
        Args:
            initial_memory: Initial memory state, mapping conversation IDs to lists of memory items
        """
        self.memory_store = initial_memory or {}
        self.retrieve_calls = []
        self.store_calls = []
        self.update_calls = []
        self.delete_calls = []
    
    def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """
        Mock implementation of retrieve.
        
        Args:
            input: The memory request
            
        Returns:
            A list of memory items for the given conversation
        """
        self.retrieve_calls.append(input)
        key = f"{input.memory_type}:{input.conversation_id}"
        return self.memory_store.get(key, [])
    
    def store(self, input: IMemoryInput) -> str:
        """
        Mock implementation of store.
        
        Args:
            input: The memory input containing data to store
            
        Returns:
            str: Key for the stored data
        """
        if not input.data:
            raise ValueError("No data provided to store")
            
        self.store_calls.append(input)
        
        key = f"{input.memory_type}:{input.conversation_id}"
        
        if key not in self.memory_store:
            self.memory_store[key] = []
            
        self.memory_store[key].extend(input.data)
        
        return key
    
    def update(self, input: IMemoryInput) -> None:
        """
        Mock implementation of update.
        
        Args:
            input: The memory input containing update data
        """
        if not input.data:
            raise ValueError("No data provided to update")
            
        self.update_calls.append(input)
        
        key = f"{input.memory_type}:{input.conversation_id}"
        
        if key in self.memory_store:
            # Replace existing memory
            self.memory_store[key] = input.data
    
    def delete(self, input: IMemoryInput) -> None:
        """
        Mock implementation of delete.
        
        Args:
            input: The memory input identifying data to delete
        """
        self.delete_calls.append(input)
        
        key = f"{input.memory_type}:{input.conversation_id}"
        
        if key in self.memory_store:
            del self.memory_store[key]
    
    def retrieve_working_memory(self, conversation_id: str) -> IWorkingMemory:
        """
        Retrieve working memory for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            IWorkingMemory: The working memory
        """
        key = f"{ConversationMemoryType.WORKING_MEMORY}:{conversation_id}"
        items = self.memory_store.get(key, [])
        
        if items:
            return items[0]  # Return the first item
        
        # Return initialized memory if none exists
        return IWorkingMemory.initialize_memory()
    
    def store_working_memory(self, conversation_id: str, working_memory: Any) -> None:
        """
        Store working memory for a conversation.
        
        Args:
            conversation_id: The conversation ID
            working_memory: The working memory to store
        """
        key = f"{ConversationMemoryType.WORKING_MEMORY}:{conversation_id}"
        
        # Check if working_memory is already an IWorkingMemory or needs conversion
        if not isinstance(working_memory, IWorkingMemory):
            memory_data = IWorkingMemory(working_memory=working_memory)
        else:
            memory_data = working_memory
            
        if key not in self.memory_store:
            self.memory_store[key] = []
            
        self.memory_store[key] = [memory_data] 