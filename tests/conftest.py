"""
Global pytest fixtures and configurations.

This module provides fixtures that can be used across all test modules.
"""

import os
import pytest
from unittest.mock import MagicMock

from arshai.core.interfaces import IAgentConfig, IAgentInput
from arshai.core.interfaces import ILLMConfig, ILLMInput
from arshai.core.interfaces import ConversationMemoryType, IMemoryInput, IWorkingMemory, IMemoryManager
from arshai.core.interfaces import ITool


@pytest.fixture
def test_config_path():
    """Return the path to the test config directory."""
    return os.path.join(os.path.dirname(__file__), "test_configs")


@pytest.fixture
def agent_config():
    """Create a basic agent configuration."""
    return IAgentConfig(
        task_context="You are a helpful assistant for testing",
        tools=[]
    )


@pytest.fixture
def base_llm_config():
    """Create a basic LLM configuration for testing."""
    return ILLMConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def memory_config():
    """Create a basic memory configuration."""
    return {
        "working_memory": {
            "provider": "in_memory",
            "ttl": 3600
        }
    }


@pytest.fixture
def agent_input():
    """Create a basic agent input for testing."""
    return IAgentInput(
        message="Hello, this is a test message",
        conversation_id="test-conversation-123",
        stream=False
    )


@pytest.fixture
def llm_input():
    """Create a basic LLM input for testing."""
    return ILLMInput(
        system_prompt="You are a helpful assistant",
        user_message="Hello, this is a test message",
        regular_functions={},
        background_tasks={},
        structure_type=None
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.chat_with_tools.return_value = {
        "llm_response": "This is a mock response",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }
    return mock


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager for testing."""
    mock = MagicMock(spec=IMemoryManager)
    return mock


@pytest.fixture
def mock_settings():
    """Create a mock settings object for testing."""
    settings = MagicMock()
    settings.create_llm.return_value = MagicMock()
    settings.create_memory_manager.return_value = MagicMock()
    settings.get_setting.return_value = "mock_value"
    return settings


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = MagicMock(spec=ITool)
    tool.name = "mock_tool"
    tool.description = "A mock tool for testing"
    tool.function_definition = {
        "name": "mock_tool",
        "description": "A mock tool for testing",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
    }
    tool.execute.return_value = "Mock tool execution result"
    return tool


@pytest.fixture
def tool_args():
    """Create basic tool arguments for testing."""
    return {"param1": "value1"} 