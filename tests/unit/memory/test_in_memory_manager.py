"""Unit tests for the InMemoryManager."""

import pytest
from datetime import datetime, timedelta
from typing import List

from arshai.core.interfaces import IMemoryInput, IWorkingMemory
from arshai.memory.working_memory.in_memory_manager import InMemoryManager
from arshai.memory.memory_types import ConversationMemoryType


class TestInMemoryManager:
    """Tests for the InMemoryManager class."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager for testing."""
        return InMemoryManager(ttl=300)  # 5 minutes TTL for testing
    
    @pytest.fixture
    def sample_memory_data(self):
        """Create sample memory data for testing."""
        return IWorkingMemory(
            working_memory=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
    
    @pytest.fixture
    def memory_input(self, sample_memory_data):
        """Create memory input for testing."""
        return IMemoryInput(
            conversation_id="test-conversation",
            memory_type=ConversationMemoryType.WORKING_MEMORY,
            data=[sample_memory_data],
            metadata={"test_key": "test_value"}
        )
    
    def test_init(self, memory_manager):
        """Test initialization."""
        assert memory_manager.ttl == 300
        assert isinstance(memory_manager.storage, dict)
        assert len(memory_manager.storage) == 0
    
    def test_store_and_retrieve(self, memory_manager, memory_input, sample_memory_data):
        """Test storing and retrieving memory."""
        # Store memory
        key = memory_manager.store(memory_input)
        
        # Verify key format
        assert key.startswith("memory:working_memory:test-conversation")
        
        # Verify storage
        assert key in memory_manager.storage
        assert "data" in memory_manager.storage[key]
        assert "working_memory" in memory_manager.storage[key]["data"]
        assert "metadata" in memory_manager.storage[key]
        assert memory_manager.storage[key]["metadata"]["test_key"] == "test_value"
        
        # Retrieve memory
        retrieved = memory_manager.retrieve(memory_input)
        
        # Verify retrieved data
        assert isinstance(retrieved, list)
        assert len(retrieved) == 1
        assert isinstance(retrieved[0], IWorkingMemory)
        assert retrieved[0].working_memory == sample_memory_data.working_memory
    
    def test_update(self, memory_manager, memory_input):
        """Test updating memory."""
        # Store initial memory
        key = memory_manager.store(memory_input)
        
        # Create updated memory
        updated_memory = IWorkingMemory(
            working_memory=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        )
        
        update_input = IMemoryInput(
            conversation_id="test-conversation",
            memory_type=ConversationMemoryType.WORKING_MEMORY,
            data=[updated_memory]
        )
        
        # Update memory
        memory_manager.update(update_input)
        
        # Retrieve updated memory
        retrieved = memory_manager.retrieve(memory_input)
        
        # Verify retrieved data
        assert len(retrieved) == 1
        assert len(retrieved[0].working_memory) == 3
        assert retrieved[0].working_memory[2]["content"] == "How are you?"
    
    def test_delete(self, memory_manager, memory_input):
        """Test deleting memory."""
        # Store memory
        memory_manager.store(memory_input)
        
        # Verify memory exists
        retrieved_before = memory_manager.retrieve(memory_input)
        assert len(retrieved_before) == 1
        
        # Delete memory
        memory_manager.delete(memory_input)
        
        # Verify memory is deleted
        retrieved_after = memory_manager.retrieve(memory_input)
        assert len(retrieved_after) == 0
    
    def test_expired_memory_cleanup(self, sample_memory_data):
        """Test automatic cleanup of expired memory."""
        # Create manager with very short TTL
        manager = InMemoryManager(ttl=1)  # 1 second TTL
        
        # Create input
        input_data = IMemoryInput(
            conversation_id="test-expiration",
            memory_type=ConversationMemoryType.WORKING_MEMORY,
            data=[sample_memory_data]
        )
        
        # Store memory
        key = manager.store(input_data)
        
        # Verify memory exists
        assert key in manager.storage
        
        # Manually set creation time to the past
        past_time = datetime.now() - timedelta(seconds=10)
        manager.storage[key]["created_at"] = past_time.isoformat()
        
        # Request memory, which should trigger cleanup
        result = manager.retrieve(input_data)
        
        # Verify memory was cleaned up
        assert len(result) == 0
        assert key not in manager.storage
    
    def test_store_no_data(self, memory_manager):
        """Test storing with no data raises an error."""
        empty_input = IMemoryInput(
            conversation_id="test-empty",
            memory_type=ConversationMemoryType.WORKING_MEMORY,
            data=[]
        )
        
        with pytest.raises(ValueError) as excinfo:
            memory_manager.store(empty_input)
        
        assert "No data provided to store" in str(excinfo.value)
    
    def test_update_no_data(self, memory_manager):
        """Test updating with no data raises an error."""
        empty_input = IMemoryInput(
            conversation_id="test-empty-update",
            memory_type=ConversationMemoryType.WORKING_MEMORY,
            data=[]
        )
        
        with pytest.raises(ValueError) as excinfo:
            memory_manager.update(empty_input)
        
        assert "No data provided to update" in str(excinfo.value)
    
    def test_retrieve_nonexistent(self, memory_manager):
        """Test retrieving nonexistent memory returns empty list."""
        nonexistent_input = IMemoryInput(
            conversation_id="nonexistent-conversation",
            memory_type=ConversationMemoryType.WORKING_MEMORY
        )
        
        result = memory_manager.retrieve(nonexistent_input)
        
        assert isinstance(result, list)
        assert len(result) == 0 