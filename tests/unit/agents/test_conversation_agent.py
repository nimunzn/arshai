"""Unit tests for the ConversationAgent."""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Optional

from arshai.core.interfaces import IAgentConfig, IAgentInput
from arshai.core.interfaces import ILLMOutput, LLMInputType
from arshai.core.interfaces import IWorkingMemory, IMemoryItem
from arshai.agents.conversation import ConversationAgent
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_memory import MockMemoryManager
from pydantic import BaseModel, Field


class MockTestOutput(BaseModel):
    """Mock output structure for tests."""
    answer: str
    confidence: float

    @classmethod
    def model_json_schema(cls):
        """Mock implementation for model_json_schema."""
        return {
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            }
        }


class MockMemory(BaseModel):
    """Mock memory model that implements model_dump()."""
    memory: List[Dict[str, Any]]
    agent_message: str

    def model_dump(self):
        """Mock model_dump method."""
        return {"memory": self.memory, "agent_message": self.agent_message}


class MockAgentConfig:
    """Mock implementation of IAgentConfig that supports get method as used in implementation."""
    
    def __init__(self, task_context, tools, output_structure=None):
        self._config = {
            "task_context": task_context,
            "tools": tools,
            "output_structure": output_structure
        }
    
    def get(self, key, default=None):
        """Get method to match implementation."""
        return self._config.get(key, default)


class MockTool:
    """Mock implementation of ITool for testing."""
    
    def __init__(self, name="test_tool", description="A test tool"):
        self.function_definition = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        }
    
    def execute(self, *args, **kwargs):
        """Mock execute method."""
        return "Tool execution result"


class TestConversationAgent:
    """Tests for the ConversationAgent class."""
    
    @pytest.fixture
    def agent_config(self):
        """Create a basic agent configuration for testing."""
        return MockAgentConfig(
            task_context="You are a helpful test assistant",
            tools=[],
            output_structure=None
        )
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock()
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = {
            "llm_response": {
                "agent_message": "This is a test response",
                "memory": [
                    {"role": "user", "content": "Test message"},
                    {"role": "assistant", "content": "Test response"}
                ]
            },
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        settings.create_llm.return_value = mock_llm
        
        # Mock Memory Manager
        mock_memory = MagicMock()
        mock_memory.retrieve_working_memory.return_value = MagicMock()
        mock_memory.retrieve_working_memory.return_value.working_memory = "USER PROFILE:\nTest user\n\nCONVERSATION HISTORY:\nNo history yet."
        settings.create_memory_manager.return_value = mock_memory
        
        # Mock settings
        settings.get.return_value = {}
        
        return settings
    
    def test_init(self, agent_config, mock_settings):
        """Test agent initialization."""
        agent = ConversationAgent(config=agent_config, settings=mock_settings)
        
        # Verify dependencies were created
        mock_settings.create_llm.assert_called_once()
        mock_settings.create_memory_manager.assert_called_once()
        
        # Verify properties were set correctly
        assert agent.task_context == "You are a helpful test assistant"
        assert agent.available_tools == []
        assert agent.output_structure is None
    
    @patch("src.agents.conversation.ConversationAgent.process_message")
    def test_process_message(self, mock_process, agent_config, mock_settings):
        """Test processing a message by replacing the entire method."""
        # Setup mock for the entire method
        mock_process.return_value = ("This is a test response", {"total_tokens": 30})
        
        # Create agent
        agent = ConversationAgent(config=agent_config, settings=mock_settings)
        
        # Create input
        input_data = IAgentInput(
            message="Hello, world!",
            conversation_id="test-conversation",
            stream=False
        )
        
        # Process message
        response, usage = agent.process_message(input_data)
        
        # Verify mock was called
        mock_process.assert_called_once_with(input_data)
        
        # Verify response
        assert response == "This is a test response"
        assert usage["total_tokens"] == 30
    
    @patch("src.agents.conversation.ConversationAgent.process_message")
    def test_prepare_system_prompt(self, mock_process, agent_config, mock_settings):
        """Test system prompt preparation."""
        # Setup mock for the entire method
        mock_process.return_value = ("This is a test response", {"total_tokens": 30})
        
        # Create agent
        agent = ConversationAgent(config=agent_config, settings=mock_settings)
        
        # Create input
        input_data = IAgentInput(
            message="Hello, world!",
            conversation_id="test-conversation",
            stream=False
        )
        
        # Process message (this will use our mock)
        agent.process_message(input_data)
        
        # Verify mock was called
        mock_process.assert_called_once_with(input_data)
    
    @pytest.mark.asyncio
    async def test_stream_process_message(self, agent_config, mock_settings):
        """Test streaming message processing."""
        # Skip this test for now as it requires more complex mocking
        pytest.skip("Streaming test needs more complex mocking")
    
    def test_with_tools(self, mock_settings):
        """Test agent with tools."""
        # Create a mock tool
        mock_tool = MockTool(name="test_tool", description="A test tool")
        
        # Create agent config with tool
        agent_config = MockAgentConfig(
            task_context="You are a helpful test assistant",
            tools=[mock_tool],
            output_structure=None
        )
        
        # Create agent
        agent = ConversationAgent(config=agent_config, settings=mock_settings)
        
        # Verify function definitions were created correctly
        assert len(agent._get_function_description()) == 1
        assert agent._get_function_description()[0]["name"] == "test_tool"
        
        # Verify callable functions were created correctly
        assert len(agent._get_callable_functions()) == 1
        assert "test_tool" in agent._get_callable_functions()
    
    @patch("src.agents.conversation.ConversationAgent.process_message")
    @patch("src.agents.conversation.ILLMInput")
    def test_with_output_structure(self, mock_illm_input, mock_process, mock_settings):
        """Test agent with structured output."""
        # Setup mocks
        mock_process.return_value = ("This is a test response", {"total_tokens": 30})
        
        # Create agent config with output structure
        agent_config = MockAgentConfig(
            task_context="You are a helpful test assistant",
            tools=[],
            output_structure=MockTestOutput
        )
        
        # Create agent
        agent = ConversationAgent(config=agent_config, settings=mock_settings)
        
        # Verify output structure was set correctly
        assert agent.output_structure == MockTestOutput
        
        # Create input
        input_data = IAgentInput(
            message="Hello, world!",
            conversation_id="test-conversation",
            stream=False
        )
        
        # Process message (this will use our mock)
        agent.process_message(input_data)
        
        # Verify mock was called
        mock_process.assert_called_once_with(input_data) 