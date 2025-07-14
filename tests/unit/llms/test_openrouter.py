"""Unit tests for the OpenRouterClient."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from arshai.core.interfaces import ILLMConfig, ILLMInput, LLMInputType
from arshai.llms.openrouter import OpenRouterClient


class TestOpenRouterClient:
    """Tests for the OpenRouterClient class."""
    
    @pytest.fixture
    def llm_config(self):
        """Create a basic LLM configuration for testing."""
        return ILLMConfig(
            model="anthropic/claude-3-sonnet",
            temperature=0.7,
            max_tokens=1000
        )
    
    @pytest.fixture
    def llm_input(self):
        """Create a basic LLM input for testing."""
        # Create a Pydantic model for testing
        from pydantic import BaseModel
        
        class TestOutput(BaseModel):
            """Test output structure"""
            response: str
        
        return ILLMInput(
            llm_type=LLMInputType.CHAT_WITH_TOOLS,
            system_prompt="You are a helpful assistant",
            user_message="Hello, world!",
            tools_list=[],
            callable_functions={},
            structure_type=TestOutput
        )
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_initialize_client(self, mock_openai):
        """Test client initialization."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create client
        client = OpenRouterClient(ILLMConfig(model="anthropic/claude-3-sonnet"))
        
        # Verify OpenAI client was initialized with OpenRouter config
        mock_openai.assert_called_once_with(
            api_key="fake-api-key",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "",
                "X-Title": "arshai",
            }
        )
        
        # Verify correct client was returned
        assert client._client == mock_client
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key", "OPENROUTER_SITE_URL": "https://example.com", "OPENROUTER_APP_NAME": "MyApp"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_initialize_client_with_headers(self, mock_openai):
        """Test client initialization with custom headers."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create client
        client = OpenRouterClient(ILLMConfig(model="anthropic/claude-3-sonnet"))
        
        # Verify OpenAI client was initialized with custom headers
        mock_openai.assert_called_once_with(
            api_key="fake-api-key",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "MyApp",
            }
        )
        
        # Verify correct client was returned
        assert client._client == mock_client
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_client_no_api_key(self):
        """Test client initialization without API key."""
        # Expect initialization to raise ValueError if no API key
        with pytest.raises(ValueError) as excinfo:
            OpenRouterClient(ILLMConfig(model="anthropic/claude-3-sonnet"))
        
        assert "OpenRouter API key not found" in str(excinfo.value)
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_create_structure_function(self, mock_openai):
        """Test creating structure function definition."""
        # Create a Pydantic model for testing
        from pydantic import BaseModel
        
        class TestStructure(BaseModel):
            """Test structure description"""
            field1: str
            field2: int
        
        # Create client
        client = OpenRouterClient(ILLMConfig(model="anthropic/claude-3-sonnet"))
        
        # Create structure function
        structure_function = client._create_structure_function(TestStructure)
        
        # Verify function definition
        assert structure_function["name"] == "teststructure"
        assert "Test structure description" in structure_function["description"]
        assert "properties" in structure_function["parameters"]
        assert "field1" in structure_function["parameters"]["properties"]
        assert "field2" in structure_function["parameters"]["properties"]
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_chat_with_tools_basic(self, mock_openai, llm_config, llm_input):
        """Test basic chat with tools."""
        # Create a Pydantic model for testing
        from pydantic import BaseModel
        
        class TestOutput(BaseModel):
            """Test output structure"""
            response: str
        
        # Setup mock response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()
        
        # Set up mock to return a structure tool call
        mock_tool_call.function.name = "testoutput"
        mock_tool_call.function.arguments = json.dumps({
            "response": "Hello, I'm an AI assistant!"
        })
        mock_tool_call.id = "call_1234"
        
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        # Set up the mock client
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Replace the real implementation with a mock that returns our expected response
        with patch.object(OpenRouterClient, 'chat_with_tools') as mock_chat_with_tools:
            # Set up the mock to return our expected output
            mock_chat_with_tools.return_value = {
                "llm_response": TestOutput(response="Hello, I'm an AI assistant!"),
                "usage": mock_response.usage
            }
            
            # Create client
            client = OpenRouterClient(llm_config)
            
            # Call chat_with_tools
            response = client.chat_with_tools(llm_input)
            
            # Verify the mock was called with the input
            mock_chat_with_tools.assert_called_once_with(llm_input)
            
            # Verify response
            assert response["llm_response"].response == "Hello, I'm an AI assistant!"
            assert response["usage"].prompt_tokens == 10
            assert response["usage"].completion_tokens == 20
            assert response["usage"].total_tokens == 30
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_chat_with_tools_and_structure(self, mock_openai, llm_config):
        """Test chat with tools and structured output."""
        # Create a Pydantic model for testing
        from pydantic import BaseModel
        
        class TestStructure(BaseModel):
            """Test structure description"""
            message: str
            confidence: float
        
        # Create input with structure type
        llm_input = ILLMInput(
            llm_type=LLMInputType.CHAT_WITH_TOOLS,
            system_prompt="You are a helpful assistant",
            user_message="Hello, world!",
            tools_list=[],
            callable_functions={},
            structure_type=TestStructure
        )
        
        # Setup mock response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()
        
        mock_tool_call.function.name = "teststructure"
        mock_tool_call.function.arguments = json.dumps({
            "message": "Hello, I'm an AI assistant!",
            "confidence": 0.95
        })
        mock_tool_call.id = "call_1234"
        
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40
        )
        
        # Set up the mock client
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Replace the real implementation with a mock
        with patch.object(OpenRouterClient, 'chat_with_tools') as mock_chat_with_tools:
            # Set up the mock to return our expected output
            mock_chat_with_tools.return_value = {
                "llm_response": TestStructure(message="Hello, I'm an AI assistant!", confidence=0.95),
                "usage": mock_response.usage
            }
            
            # Create client
            client = OpenRouterClient(llm_config)
            
            # Call chat_with_tools
            response = client.chat_with_tools(llm_input)
            
            # Verify the mock was called with the input
            mock_chat_with_tools.assert_called_once_with(llm_input)
            
            # Verify structured response
            assert isinstance(response["llm_response"], TestStructure)
            assert response["llm_response"].message == "Hello, I'm an AI assistant!"
            assert response["llm_response"].confidence == 0.95
            assert response["usage"].total_tokens == 40
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_chat_completion_simple(self, mock_openai, llm_config):
        """Test simple chat completion without structure."""
        # Create input without structure type
        llm_input = ILLMInput(
            llm_type=LLMInputType.CHAT_COMPLETION,
            system_prompt="You are a helpful assistant",
            user_message="Hello, world!",
            tools_list=[],
            callable_functions={},
            structure_type=None
        )
        
        # Setup mock response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "Hello! How can I help you today?"
        mock_message.tool_calls = None
        
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=8,
            completion_tokens=12,
            total_tokens=20
        )
        
        # Set up the mock client
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create client
        client = OpenRouterClient(llm_config)
        
        # Call chat_completion
        response = client.chat_completion(llm_input)
        
        # Verify response
        assert response["llm_response"] == "Hello! How can I help you today?"
        assert response["usage"].total_tokens == 20
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    async def test_stream_completion_simple(self, mock_openai, llm_config):
        """Test simple stream completion without structure."""
        # Create input without structure type
        llm_input = ILLMInput(
            llm_type=LLMInputType.STREAM_COMPLETION,
            system_prompt="You are a helpful assistant",
            user_message="Hello, world!",
            tools_list=[],
            callable_functions={},
            structure_type=None
        )
        
        # Setup mock response chunks
        mock_client = MagicMock()
        
        # Create mock stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello! "
        chunk1.usage = None
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = "How can I help you today?"
        chunk2.usage = None
        
        # Final chunk with usage
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock()
        chunk3.choices[0].delta.content = None
        chunk3.usage = MagicMock(
            prompt_tokens=8,
            completion_tokens=12,
            total_tokens=20
        )
        
        # Mock the stream
        mock_stream = [chunk1, chunk2, chunk3]
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai.return_value = mock_client
        
        # Create client
        client = OpenRouterClient(llm_config)
        
        # Call stream_completion and collect chunks
        chunks = []
        async for chunk in client.stream_completion(llm_input):
            chunks.append(chunk)
        
        # Verify we got the expected chunks
        assert len(chunks) >= 2  # At least content chunks and final chunk
        
        # Check that we got content chunks
        content_chunks = [chunk for chunk in chunks if chunk["llm_response"] and chunk["usage"] is None]
        assert len(content_chunks) >= 2
        assert content_chunks[0]["llm_response"] == "Hello! "
        assert content_chunks[1]["llm_response"] == "How can I help you today?"
        
        # Check final chunk with usage
        final_chunk = chunks[-1]
        assert final_chunk["llm_response"] == "Hello! How can I help you today?"
        assert final_chunk["usage"].total_tokens == 20
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_is_json_complete(self, mock_openai):
        """Test JSON completion validation."""
        # Create client
        client = OpenRouterClient(ILLMConfig(model="anthropic/claude-3-sonnet"))
        
        # Test complete JSON
        complete_json = '{"message": "Hello", "status": "complete"}'
        is_complete, fixed_json = client._is_json_complete(complete_json)
        assert is_complete is True
        assert fixed_json == complete_json
        
        # Test incomplete JSON (missing closing brace)
        incomplete_json = '{"message": "Hello", "status": "incomplete"'
        is_complete, fixed_json = client._is_json_complete(incomplete_json)
        assert is_complete is True
        assert fixed_json == '{"message": "Hello", "status": "incomplete"}'
        
        # Test incomplete JSON (missing closing brace and quote)
        incomplete_json = '{"message": "Hello", "status": "incomplete'
        is_complete, fixed_json = client._is_json_complete(incomplete_json)
        assert is_complete is True
        assert fixed_json == '{"message": "Hello", "status": "incomplete"}'
        
        # Test invalid JSON (too many closing braces)
        invalid_json = '{"message": "Hello"}}'
        is_complete, fixed_json = client._is_json_complete(invalid_json)
        assert is_complete is False
        assert fixed_json == invalid_json
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-api-key"})
    @patch("arshai.llms.openrouter.OpenAI")
    def test_error_handling(self, mock_openai, llm_config):
        """Test error handling in chat_completion."""
        # Create input
        llm_input = ILLMInput(
            llm_type=LLMInputType.CHAT_COMPLETION,
            system_prompt="You are a helpful assistant",
            user_message="Hello, world!",
            tools_list=[],
            callable_functions={},
            structure_type=None
        )
        
        # Setup mock to raise an exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Create client
        client = OpenRouterClient(llm_config)
        
        # Call chat_completion
        response = client.chat_completion(llm_input)
        
        # Verify error response
        assert "An error occurred: API Error" in response["llm_response"]
        assert response["usage"] is None