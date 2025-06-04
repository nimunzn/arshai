"""Integration tests for agents with memory."""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from arshai.core.interfaces import IAgentConfig, IAgentInput
from arshai.core.interfaces import IWorkingMemory, ConversationMemoryType
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_memory import MockMemoryManager


# Define the response structure that ConversationAgent expects
class AgentResponseStructure(BaseModel):
    """Response structure for the conversation agent."""
    agent_message: str
    memory: IWorkingMemory


class TestAgentWithMemory:
    """Integration tests for agent with memory integration."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with configured LLM and memory."""
        settings = MagicMock()
        
        # Configure LLM
        mock_llm = MockLLM(responses={
            "Hello": "Hello! I'm your assistant."
        })
        settings.create_llm.return_value = mock_llm
        
        # Configure memory
        user_memory = IWorkingMemory.initialize_memory()
        assistant_memory = IWorkingMemory.initialize_memory()
        
        # Configuration using the new IWorkingMemory format
        initial_memory = {
            f"{ConversationMemoryType.SHORT_TERM_MEMORY}:test-conversation": [
                user_memory,
                assistant_memory
            ]
        }
        mock_memory = MockMemoryManager(initial_memory=initial_memory)
        settings.create_memory_manager.return_value = mock_memory
        
        return settings
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return IAgentConfig(
            task_context="You are a helpful assistant with memory",
            tools=[],
            output_structure=AgentResponseStructure
        )
    
    def test_agent_uses_memory(self, mock_settings, agent_config):
        """Test that an agent uses memory correctly when processing messages."""
        # Configure the mock LLM to return a response with updated memory
        mock_llm = mock_settings.create_llm.return_value
        
        # Create a response with updated memory
        mock_memory = IWorkingMemory.initialize_memory()
        mock_memory.working_memory["last_interaction"] = "The user said Hello"
        mock_response = AgentResponseStructure(
            agent_message="Hello! I'm your assistant.",
            memory=mock_memory
        )
        
        # Set the mock LLM to return our prepared response
        mock_llm.chat_with_tools = MagicMock(return_value={
            "llm_response": mock_response,
            "usage": {"total_tokens": 100}
        })
        
        # Initialize the agent with the real implementation
        from arshai.factories.agent_factory import AgentFactory
        agent = AgentFactory.create("conversation", agent_config, settings=mock_settings)
        
        # Process a message
        input_data = IAgentInput(
            message="Hello",
            conversation_id="test-conversation",
            stream=False
        )
        agent.process_message(input_data)
        
        # Access the mock memory manager
        memory_manager = mock_settings.create_memory_manager.return_value
        
        # Verify memory was retrieved
        assert len(memory_manager.retrieve_calls) > 0, "Memory retrieve should have been called"
        
        # Verify memory was stored after processing
        assert len(memory_manager.store_calls) > 0, "Memory store should have been called"
        
        # Check that the conversation ID was used correctly
        if memory_manager.retrieve_calls:
            assert "test-conversation" in memory_manager.retrieve_calls[0].conversation_id, "Wrong conversation ID used for retrieval"
        
        if memory_manager.store_calls:
            assert "test-conversation" in memory_manager.store_calls[0].conversation_id, "Wrong conversation ID used for storage" 