"""Unit tests for the Settings class."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from arshai.core.interfaces import IAgentConfig
from arshai.core.interfaces import ILLMConfig
from arshai.core.interfaces import IMemoryConfig
from arshai.config.settings import Settings


class TestSettings:
    """Tests for the Settings class."""
    
    def test_init_with_defaults(self):
        """Test that Settings initializes with defaults."""
        settings = Settings()
        assert settings.config_manager is not None
    
    def test_init_with_custom_config(self):
        """Test initializing with a custom configuration."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as temp_file:
            temp_file.write("""
            llm:
              provider: test_provider
              api_key: test_key
            """)
            temp_path = temp_file.name
        
        try:
            settings = Settings(config_path=temp_path)
            assert settings.get("llm.provider") == "test_provider"
            assert settings.get("llm.api_key") == "test_key"
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def test_get_value(self):
        """Test getting settings with default values."""
        settings = Settings()
        
        # Set a value directly through config_manager
        settings.config_manager.set("test.key", "value")
        
        # Existing setting
        assert settings.get_value("test.key") == "value"
        assert settings.get("test.key") == "value"  # Test alias
        
        # Non-existing setting with default
        assert settings.get_value("missing.key", "default") == "default"
        
        # Non-existing setting without default
        assert settings.get_value("missing.key") is None
    
    @patch("src.factories.llm_factory.LLMFactory.create")
    def test_create_llm(self, mock_create):
        """Test creating an LLM client."""
        # Setup mock
        mock_llm = MagicMock()
        mock_create.return_value = mock_llm
        
        # Create settings and setup test config directly
        settings = Settings()
        settings.config_manager.set("llm.provider", "openai")
        settings.config_manager.set("llm.model", "gpt-4")
        settings.config_manager.set("llm.temperature", 0.7)
        
        # Create LLM
        llm = settings.create_llm()
        
        # Verify factory was called with correct args
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        assert args[0] == "openai"  # provider
        assert isinstance(args[1], ILLMConfig)
        assert args[1].model == "gpt-4"
        assert args[1].temperature == 0.7
        
        # Verify correct LLM was returned
        assert llm == mock_llm
    
    @patch("src.factories.memory_factory.MemoryFactory.create_memory_manager_service")
    def test_create_memory_manager(self, mock_create):
        """Test creating a memory manager."""
        # Setup mock
        mock_memory = MagicMock()
        mock_create.return_value = mock_memory
        
        # Create settings and set memory config
        settings = Settings()
        memory_config = {
            "working_memory": {
                "provider": "in_memory",
                "ttl": 3600
            }
        }
        settings.config_manager.set("memory", memory_config)
        
        # Create memory manager
        memory = settings.create_memory_manager()
        
        # Verify factory was called with the memory config
        mock_create.assert_called_once_with(memory_config)
        
        # Verify correct memory manager was returned
        assert memory == mock_memory
        
        # Test caching - create again and verify factory not called again
        memory2 = settings.create_memory_manager()
        assert memory2 == mock_memory
        assert mock_create.call_count == 1
    
    @patch("src.factories.agent_factory.AgentFactory.create")
    def test_create_agent(self, mock_create):
        """Test creating an agent."""
        # Setup mock
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        
        # Create settings
        settings = Settings()
        
        # Create agent config
        agent_config = IAgentConfig(
            task_context="Test task",
            tools=[]
        )
        
        # Create agent
        agent = settings.create_agent("operator", agent_config)
        
        # Verify factory was called with correct args
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        assert args[0] == "operator"  # agent type
        assert args[1] == agent_config  # agent config
        assert args[2] == settings  # Settings should be passed as third argument
        
        # Verify correct agent was returned
        assert agent == mock_agent
    
    @patch("src.factories.vector_db_factory.VectorDBFactory.create")
    def test_create_vector_db(self, mock_create):
        """Test creating a vector database client."""
        # Setup mock
        mock_db = MagicMock()
        mock_create.return_value = mock_db
        
        # Create settings and set vector_db config
        settings = Settings()
        settings.config_manager.set("vector_db.provider", "milvus")
        settings.config_manager.set("vector_db.connection_uri", "http://localhost:19530")
        
        # Create vector DB
        db = settings.create_vector_db()
        
        # Verify factory was called
        mock_create.assert_called_once()
        
        # Verify correct DB was returned
        assert db == mock_db
        
        # Test caching - create again and verify factory not called again
        db2 = settings.create_vector_db()
        assert db2 == mock_db
        assert mock_create.call_count == 1 