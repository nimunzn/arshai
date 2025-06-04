"""Unit tests for the Memory factory."""

import pytest
from unittest.mock import patch, MagicMock

from arshai.core.interfaces import IMemoryConfig
from arshai.factories.memory_factory import MemoryFactory


class TestMemoryFactory:
    """Tests for the MemoryFactory class."""
    
    def test_create_in_memory_working_memory(self):
        """Test creating an in-memory working memory."""
        # Create memory
        memory = MemoryFactory.create_working_memory("in_memory", ttl=3600)
        
        # Verify correct instance type
        from arshai.memory.working_memory.in_memory_manager import InMemoryManager
        assert isinstance(memory, InMemoryManager)
    
    def test_create_redis_working_memory(self):
        """Test creating a Redis working memory."""
        # Create memory
        memory = MemoryFactory.create_working_memory(
            "redis",
            storage_url="redis://localhost:6379/0",
            ttl=3600
        )
        
        # Verify correct instance type
        from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
        assert isinstance(memory, RedisWorkingMemoryManager)
    
    def test_create_unknown_working_memory(self):
        """Test that creating an unknown working memory raises an error."""
        with pytest.raises(ValueError) as excinfo:
            MemoryFactory.create_working_memory("unknown_provider")
        
        assert "Unsupported working memory provider" in str(excinfo.value)
    
    @patch("src.memory.memory_manager.MemoryManagerService")
    def test_create_memory_manager_service(self, mock_service):
        """Test creating a memory manager service."""
        # Setup mocks
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        
        # Create memory config
        memory_config = {
            "working_memory": {
                "provider": "in_memory",
                "ttl": 3600
            }
        }
        
        # Create memory manager service
        service = MemoryFactory.create_memory_manager_service(memory_config)
        
        # Verify service constructor was called with config
        mock_service.assert_called_once_with(memory_config)
        
        # Verify correct instance was returned
        assert service == mock_service_instance
    
    @patch("src.memory.memory_manager.MemoryManagerService")
    def test_create_memory_manager_service_with_redis(self, mock_service):
        """Test creating a memory manager service with Redis."""
        # Setup mocks
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        
        # Create memory config
        memory_config = {
            "working_memory": {
                "provider": "redis",
                "storage_url": "redis://localhost:6379/0",
                "ttl": 3600
            }
        }
        
        # Create memory manager service
        service = MemoryFactory.create_memory_manager_service(memory_config)
        
        # Verify service constructor was called with config
        mock_service.assert_called_once_with(memory_config)
        
        # Verify correct instance was returned
        assert service == mock_service_instance 