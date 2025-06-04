"""Unit tests for Redis working memory manager."""

import pytest
from unittest.mock import MagicMock, patch, ANY
import json
from datetime import datetime

from arshai.core.interfaces import IMemoryInput, IWorkingMemory
from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
from arshai.memory.memory_types import ConversationMemoryType


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    with patch('src.memory.working_memory.redis_memory_manager.redis') as mock_redis:
        # Setup the from_url method to return a mock client
        mock_client = MagicMock()
        mock_redis.from_url.return_value = mock_client
        yield mock_client


@pytest.fixture
def redis_memory_manager(mock_redis):
    """Create a RedisWorkingMemoryManager with mocked Redis client."""
    with patch.dict('os.environ', {'REDIS_URL': 'redis://test:6379/1'}):
        manager = RedisWorkingMemoryManager()
        return manager


@pytest.fixture
def memory_input():
    """Create a basic memory input for testing."""
    return IMemoryInput(
        conversation_id="test-convo-123",
        memory_type=ConversationMemoryType.WORKING_MEMORY,
        data=[IWorkingMemory(working_memory="""
            USER PROFILE:
            Test user with sample data

            CONVERSATION STORY:
            Test conversation
        """)],
        metadata={"user_id": "test-user-123"}
    )


def test_initialization():
    """Test initialization with different parameters."""
    # Test with explicit URL
    with patch('src.memory.working_memory.redis_memory_manager.redis') as mock_redis:
        manager = RedisWorkingMemoryManager(storage_url="redis://custom:6379/1")
        assert manager.storage_url == "redis://custom:6379/1"
        mock_redis.from_url.assert_called_once_with("redis://custom:6379/1")
    
    # Test with environment variable
    with patch('src.memory.working_memory.redis_memory_manager.redis') as mock_redis:
        with patch.dict('os.environ', {'REDIS_URL': 'redis://envtest:6379/1'}):
            manager = RedisWorkingMemoryManager()
            assert manager.storage_url == "redis://envtest:6379/1"
            mock_redis.from_url.assert_called_once_with("redis://envtest:6379/1")
    
    # Test with default when no env var or param
    with patch('src.memory.working_memory.redis_memory_manager.redis') as mock_redis:
        with patch.dict('os.environ', {}, clear=True):
            manager = RedisWorkingMemoryManager()
            assert manager.storage_url == "redis://localhost:6379/1"
            mock_redis.from_url.assert_called_once_with("redis://localhost:6379/1")


def test_get_key(redis_memory_manager):
    """Test key generation."""
    key = redis_memory_manager._get_key("test-123", ConversationMemoryType.WORKING_MEMORY)
    assert key == "memory:WORKING_MEMORY:test-123"
    
    key = redis_memory_manager._get_key("test-456", ConversationMemoryType.SHORT_TERM_MEMORY)
    assert key == "memory:SHORT_TERM_MEMORY:test-456"


def test_store(redis_memory_manager, memory_input, mock_redis):
    """Test storing memory data."""
    key = redis_memory_manager.store(memory_input)
    
    # Verify the correct key was generated
    assert key == "memory:WORKING_MEMORY:test-convo-123"
    
    # Verify Redis client was called with correct arguments
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    
    # Check key
    assert args[0] == key
    
    # Check TTL
    assert args[1] == 60 * 60 * 12  # 12 hours
    
    # Check data format
    stored_data = json.loads(args[2])
    assert "USER PROFILE:" in stored_data["data"]["working_memory"]
    assert stored_data["metadata"] == {"user_id": "test-user-123"}
    assert "created_at" in stored_data
    assert "last_update" in stored_data


def test_store_no_data(redis_memory_manager):
    """Test store with no data raises error."""
    empty_input = IMemoryInput(
        conversation_id="test-convo-123",
        memory_type=ConversationMemoryType.WORKING_MEMORY,
        data=[],
        metadata={}
    )
    
    with pytest.raises(ValueError, match="No data provided to store"):
        redis_memory_manager.store(empty_input)


def test_retrieve_existing(redis_memory_manager, memory_input, mock_redis):
    """Test retrieving existing memory data."""
    # Mock Redis get to return data
    mock_data = {
        "data": {"working_memory": "USER PROFILE:\nTest user with sample data"},
        "metadata": {"user_id": "test-user-123"},
        "created_at": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat()
    }
    mock_redis.get.return_value = json.dumps(mock_data)
    
    result = redis_memory_manager.retrieve(memory_input)
    
    # Verify Redis client was called with correct key
    mock_redis.get.assert_called_once_with("memory:WORKING_MEMORY:test-convo-123")
    
    # Verify result format
    assert len(result) == 1
    assert "USER PROFILE:" in result[0].working_memory


def test_retrieve_nonexistent(redis_memory_manager, memory_input, mock_redis):
    """Test retrieving nonexistent memory data."""
    # Mock Redis get to return None (no data)
    mock_redis.get.return_value = None
    
    result = redis_memory_manager.retrieve(memory_input)
    
    # Verify Redis client was called with correct key
    mock_redis.get.assert_called_once_with("memory:WORKING_MEMORY:test-convo-123")
    
    # Verify empty result
    assert result == []


def test_update_existing(redis_memory_manager, memory_input, mock_redis):
    """Test updating existing memory data."""
    # Mock Redis get to return existing data
    existing_data = {
        "data": {"working_memory": "USER PROFILE:\nOld user data"},
        "metadata": {"user_id": "test-user-123"},
        "created_at": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat()
    }
    mock_redis.get.return_value = json.dumps(existing_data)
    
    # Update with new data
    update_input = IMemoryInput(
        conversation_id="test-convo-123",
        memory_type=ConversationMemoryType.WORKING_MEMORY,
        data=[IWorkingMemory(working_memory="USER PROFILE:\nUpdated user data")],
        metadata={"user_id": "test-user-123"}
    )
    
    redis_memory_manager.update(update_input)
    
    # Verify Redis client was called with correct key for get
    mock_redis.get.assert_called_once_with("memory:WORKING_MEMORY:test-convo-123")
    
    # Verify setex was called with updated data
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    
    # Check key and TTL
    assert args[0] == "memory:WORKING_MEMORY:test-convo-123"
    assert args[1] == 60 * 60 * 12
    
    # Check updated data
    updated_data = json.loads(args[2])
    assert "Updated user data" in updated_data["data"]["working_memory"]
    assert "last_update" in updated_data


def test_update_nonexistent(redis_memory_manager, memory_input, mock_redis):
    """Test updating nonexistent memory data."""
    # Mock Redis get to return None (no existing data)
    mock_redis.get.return_value = None
    
    redis_memory_manager.update(memory_input)
    
    # Verify Redis client was called with correct key for get
    mock_redis.get.assert_called_once_with("memory:WORKING_MEMORY:test-convo-123")
    
    # Verify setex was not called
    mock_redis.setex.assert_not_called()


def test_update_no_data(redis_memory_manager):
    """Test update with no data raises error."""
    empty_input = IMemoryInput(
        conversation_id="test-convo-123",
        memory_type=ConversationMemoryType.WORKING_MEMORY,
        data=[],
        metadata={}
    )
    
    with pytest.raises(ValueError, match="No data provided to update"):
        redis_memory_manager.update(empty_input)


def test_delete_existing(redis_memory_manager, memory_input, mock_redis):
    """Test deleting existing memory data."""
    # Mock Redis delete to return 1 (successful deletion)
    mock_redis.delete.return_value = 1
    
    redis_memory_manager.delete(memory_input)
    
    # Verify Redis client was called with correct key
    mock_redis.delete.assert_called_once_with("memory:WORKING_MEMORY:test-convo-123")


def test_delete_nonexistent(redis_memory_manager, memory_input, mock_redis):
    """Test deleting nonexistent memory data."""
    # Mock Redis delete to return 0 (no deletion)
    mock_redis.delete.return_value = 0
    
    redis_memory_manager.delete(memory_input)
    
    # Verify Redis client was called with correct key
    mock_redis.delete.assert_called_once_with("memory:WORKING_MEMORY:test-convo-123") 