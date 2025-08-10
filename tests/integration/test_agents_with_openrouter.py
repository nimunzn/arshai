"""
Integration tests for agents using real OpenRouter client.

These tests verify that agents work correctly with actual LLM providers.
"""

import os
import pytest
import asyncio
from pathlib import Path
from arshai.llms.openrouter import OpenRouterClient
from arshai.core.interfaces.illm import ILLMConfig
from arshai.core.interfaces.iagent import IAgentInput
from arshai.agents.working_memory import WorkingMemoryAgent
from tests.unit.agents.simple_agent import SimpleAgent
from arshai.utils.logging import get_logger

logger = get_logger(__name__)

# Load environment variables from .env.openrouter file
def load_openrouter_env():
    """Load OpenRouter API key from .env.openrouter file."""
    env_file = Path(__file__).parent.parent / "unit" / "llms" / ".env.openrouter"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load the environment variables
load_openrouter_env()


# Test configuration
TEST_MODEL = "openai/gpt-4o-mini"  # GPT-4o Mini for testing
TEST_SYSTEM_PROMPT = "You are a helpful AI assistant. Keep responses concise and clear."


@pytest.fixture
def openrouter_config():
    """Create OpenRouter configuration for testing."""
    return ILLMConfig(
        model=TEST_MODEL,
        temperature=0.1,
        max_tokens=100
    )


@pytest.fixture
def openrouter_client(openrouter_config):
    """Create OpenRouter client for testing."""
    # Skip test if no API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not found in environment")
    
    return OpenRouterClient(openrouter_config)


@pytest.fixture
def simple_agent(openrouter_client):
    """Create simple agent with OpenRouter client."""
    return SimpleAgent(
        llm_client=openrouter_client,
        system_prompt=TEST_SYSTEM_PROMPT
    )


@pytest.fixture
def working_memory_agent(openrouter_client):
    """Create working memory agent with OpenRouter client."""
    return WorkingMemoryAgent(
        llm_client=openrouter_client,
        system_prompt=TEST_SYSTEM_PROMPT
    )


class TestSimpleAgentWithOpenRouter:
    """Test SimpleAgent with real OpenRouter client."""
    
    @pytest.mark.asyncio
    async def test_simple_message_processing(self, simple_agent):
        """Test that simple agent can process basic messages."""
        # Test input
        test_input = IAgentInput(message="Hello, how are you?")
        
        # Process message
        response = await simple_agent.process(test_input)
        logger.info(f"simple_message_processing response: {response}")
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        assert response.strip() != ""
    
    @pytest.mark.asyncio
    async def test_question_response(self, simple_agent):
        """Test that agent can answer simple questions."""
        # Test input
        test_input = IAgentInput(message="What is 2 + 2?")
        
        # Process message
        response = await simple_agent.process(test_input)
        logger.info(f"question_response response: {response}")

        # Verify response contains answer
        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain "4" somewhere in the response
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_multiple_messages(self, simple_agent):
        """Test processing multiple messages in sequence."""
        messages = [
            "Hello!",
            "What is the capital of France?",
            "Thank you!"
        ]
        
        responses = []
        for msg in messages:
            test_input = IAgentInput(message=msg)
            response = await simple_agent.process(test_input)
            responses.append(response)
        
        # Verify all responses
        assert len(responses) == 3
        logger.info(f"multiple_messages responses: {responses}")

        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Check that Paris is mentioned in the France question response
        assert "Paris" in responses[1] or "paris" in responses[1].lower()


class TestWorkingMemoryAgentWithOpenRouter:
    """Test WorkingMemoryAgent with real OpenRouter client."""
    
    @pytest.mark.asyncio
    async def test_working_memory_agent_initialization(self, working_memory_agent):
        """Test that working memory agent initializes correctly."""
        assert working_memory_agent.llm_client is not None
        assert working_memory_agent.system_prompt is not None
        assert working_memory_agent.memory_manager is None  # No memory manager in this test
        assert working_memory_agent.chat_history is None  # No chat history client in this test
    
    @pytest.mark.asyncio
    async def test_memory_process_without_conversation_id(self, working_memory_agent):
        """Test memory agent handling when no conversation_id is provided."""
        # Test input without conversation_id
        test_input = IAgentInput(
            message="User is asking about pricing for premium plans",
            metadata={}  # No conversation_id
        )
        
        # Process should return error (nothing to do without conversation_id)
        result = await working_memory_agent.process(test_input)
        
        assert result == "error: no conversation_id provided"
    
    @pytest.mark.asyncio
    async def test_memory_process_with_conversation_id_no_manager(self, working_memory_agent):
        """Test memory agent when conversation_id is provided but no memory manager."""
        # Test input with conversation_id
        test_input = IAgentInput(
            message="User is asking about product features",
            metadata={"conversation_id": "test_123"}
        )
        
        # Process should complete but not store anything (no memory manager)
        result = await working_memory_agent.process(test_input)
        # Should return success (memory was generated, just warn about storage)
        assert result == "success"
    
    @pytest.mark.asyncio  
    async def test_memory_agent_llm_integration(self, working_memory_agent):
        """Test that memory agent can generate memory using LLM."""
        # Mock a simple memory manager for this test
        class MockMemoryManager:
            def __init__(self):
                self.stored_data = None
            
            async def retrieve(self, query):
                return None  # No existing memory
            
            async def store(self, data):
                self.stored_data = data
        
        # Add mock memory manager
        mock_memory = MockMemoryManager()
        working_memory_agent.memory_manager = mock_memory
        
        # Test input
        test_input = IAgentInput(
            message="User is interested in enterprise pricing and asked about bulk discounts",
            metadata={"conversation_id": "enterprise_inquiry_456"}
        )
        
        # Process memory update
        result = await working_memory_agent.process(test_input)
        
        # Verify results
        assert result == "success"  # Should return success status
        assert mock_memory.stored_data is not None
        assert mock_memory.stored_data["conversation_id"] == "enterprise_inquiry_456"
        assert "working_memory" in mock_memory.stored_data
        
        # The working memory should contain relevant information
        working_memory = mock_memory.stored_data["working_memory"]
        assert isinstance(working_memory, str)
        assert len(working_memory) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])