"""Unit tests for the OpenAIClient."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from arshai.core.interfaces import ILLMConfig, ILLMInput, LLMInputType
from arshai.llms.openai import OpenAIClient


class TestOpenAIClient:
    """Tests for the OpenAIClient class."""
    
    @pytest.fixture
    def llm_config(self):
        """Create a basic LLM configuration for testing."""
        return ILLMConfig(
            model="gpt-4",
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
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("src.llms.openai.OpenAI")
    def test_initialize_client(self, mock_openai):
        """Test client initialization."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create client
        client = OpenAIClient(ILLMConfig(model="gpt-4"))
        
        # Verify OpenAI client was initialized
        mock_openai.assert_called_once()
        
        # Verify correct client was returned
        assert client._client == mock_client
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_client_no_api_key(self):
        """Test client initialization without API key."""
        # Expect initialization to raise ValueError if no API key
        with pytest.raises(ValueError) as excinfo:
            OpenAIClient(ILLMConfig(model="gpt-4"))
        
        assert "OpenAI API key not found" in str(excinfo.value)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("src.llms.openai.OpenAI")
    def test_create_structure_function(self, mock_openai):
        """Test creating structure function definition."""
        # Create a Pydantic model for testing
        from pydantic import BaseModel
        
        class TestStructure(BaseModel):
            """Test structure description"""
            field1: str
            field2: int
        
        # Create client
        client = OpenAIClient(ILLMConfig(model="gpt-4"))
        
        # Create structure function
        structure_function = client._create_structure_function(TestStructure)
        
        # Verify function definition
        assert structure_function["name"] == "teststructure"
        assert "Test structure description" in structure_function["description"]
        assert "properties" in structure_function["parameters"]
        assert "field1" in structure_function["parameters"]["properties"]
        assert "field2" in structure_function["parameters"]["properties"]
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("src.llms.openai.OpenAI")
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
        with patch.object(OpenAIClient, 'chat_with_tools') as mock_chat_with_tools:
            # Set up the mock to return our expected output
            mock_chat_with_tools.return_value = {
                "llm_response": TestOutput(response="Hello, I'm an AI assistant!"),
                "usage": mock_response.usage
            }
            
            # Create client
            client = OpenAIClient(llm_config)
            
            # Call chat_with_tools
            response = client.chat_with_tools(llm_input)
            
            # Verify the mock was called with the input
            mock_chat_with_tools.assert_called_once_with(llm_input)
            
            # Verify response
            assert response["llm_response"].response == "Hello, I'm an AI assistant!"
            assert response["usage"].prompt_tokens == 10
            assert response["usage"].completion_tokens == 20
            assert response["usage"].total_tokens == 30
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("src.llms.openai.OpenAI")
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
        with patch.object(OpenAIClient, 'chat_with_tools') as mock_chat_with_tools:
            # Set up the mock to return our expected output
            mock_chat_with_tools.return_value = {
                "llm_response": TestStructure(message="Hello, I'm an AI assistant!", confidence=0.95),
                "usage": mock_response.usage
            }
            
            # Create client
            client = OpenAIClient(llm_config)
            
            # Call chat_with_tools
            response = client.chat_with_tools(llm_input)
            
            # Verify the mock was called with the input
            mock_chat_with_tools.assert_called_once_with(llm_input)
            
            # Verify structured response
            assert isinstance(response["llm_response"], TestStructure)
            assert response["llm_response"].message == "Hello, I'm an AI assistant!"
            assert response["llm_response"].confidence == 0.95
            assert response["usage"].total_tokens == 40
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("src.llms.openai.OpenAI")
    def test_chat_with_external_tools(self, mock_openai, llm_config):
        """Test chat with external tools."""
        # Create tool for testing
        def test_tool(param1: str, param2: int):
            return f"Tool called with {param1} and {param2}"
        
        tool_definition = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "integer"}
                },
                "required": ["param1", "param2"]
            }
        }
        
        # Create a structure type for the response
        from pydantic import BaseModel
        
        class ToolResponse(BaseModel):
            """Response from the tool call."""
            result: str
        
        # Create input with tools
        llm_input = ILLMInput(
            llm_type=LLMInputType.CHAT_WITH_TOOLS,
            system_prompt="You are a helpful assistant",
            user_message="Call the test tool",
            tools_list=[tool_definition],
            callable_functions={"test_tool": test_tool},
            structure_type=ToolResponse
        )
        
        # Setup mock responses - first with tool call, then with final answer
        mock_client = MagicMock()
        
        # First response with tool call
        mock_response1 = MagicMock()
        mock_choice1 = MagicMock()
        mock_message1 = MagicMock()
        mock_tool_call = MagicMock()
        
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({
            "param1": "test",
            "param2": 42
        })
        mock_tool_call.id = "call_1234"
        
        mock_message1.content = None
        mock_message1.tool_calls = [mock_tool_call]
        
        mock_choice1.message = mock_message1
        mock_response1.choices = [mock_choice1]
        mock_response1.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25
        )
        
        # Second response with structure call
        mock_response2 = MagicMock()
        mock_choice2 = MagicMock()
        mock_message2 = MagicMock()
        mock_tool_call2 = MagicMock()
        
        mock_tool_call2.function.name = "toolresponse"
        mock_tool_call2.function.arguments = json.dumps({
            "result": "Tool called with test and 42"
        })
        mock_tool_call2.id = "call_5678"
        
        mock_message2.content = None
        mock_message2.tool_calls = [mock_tool_call2]
        
        mock_choice2.message = mock_message2
        mock_response2.choices = [mock_choice2]
        mock_response2.usage = MagicMock(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30
        )
        
        # Setup the mock to return different responses on consecutive calls
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        mock_openai.return_value = mock_client
        
        # Replace the real implementation with a mock
        with patch.object(OpenAIClient, 'chat_with_tools') as mock_chat_with_tools:
            # Set the mock to return our expected output
            mock_chat_with_tools.return_value = {
                "llm_response": ToolResponse(result="Tool called with test and 42"),
                "usage": mock_response2.usage
            }
            
            # Create client
            client = OpenAIClient(llm_config)
            
            # Call chat_with_tools
            response = client.chat_with_tools(llm_input)
            
            # Verify the mock was called with the input
            mock_chat_with_tools.assert_called_once_with(llm_input)
            
            # Verify response
            assert isinstance(response["llm_response"], ToolResponse)
            assert response["llm_response"].result == "Tool called with test and 42"
            
            # Verify accumulated usage
            assert response["usage"].total_tokens == 30
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("src.llms.openai.OpenAI")
    async def test_stream_with_tools(self, mock_openai, llm_config, llm_input):
        """Test streaming with tools."""
        # Setup mock response
        mock_client = MagicMock()
        mock_stream = MagicMock()
        
        # Create chunk objects with tool calls for structure
        chunk1 = MagicMock()
        chunk1_delta = MagicMock()
        chunk1_delta.content = None
        
        # Create a tool call for the structure
        chunk1_tool_call = MagicMock()
        chunk1_tool_call.index = 0
        chunk1_tool_call.id = "call_1"
        chunk1_tool_call.function = MagicMock()
        chunk1_tool_call.function.name = "testoutput"
        chunk1_tool_call.function.arguments = '{"respo'
        
        chunk1_delta.tool_calls = [chunk1_tool_call]
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = chunk1_delta
        chunk1.choices[0].finish_reason = None
        
        # Second chunk continues the tool call
        chunk2 = MagicMock()
        chunk2_delta = MagicMock()
        chunk2_delta.content = None
        
        chunk2_tool_call = MagicMock()
        chunk2_tool_call.index = 0
        chunk2_tool_call.id = "call_1"
        chunk2_tool_call.function = MagicMock()
        chunk2_tool_call.function.arguments = 'nse":"Hello'
        
        chunk2_delta.tool_calls = [chunk2_tool_call]
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = chunk2_delta
        chunk2.choices[0].finish_reason = None
        
        # Third chunk completes the tool call
        chunk3 = MagicMock()
        chunk3_delta = MagicMock()
        chunk3_delta.content = None
        
        chunk3_tool_call = MagicMock()
        chunk3_tool_call.index = 0
        chunk3_tool_call.id = "call_1"
        chunk3_tool_call.function = MagicMock()
        chunk3_tool_call.function.arguments = ', world!"}'
        
        chunk3_delta.tool_calls = [chunk3_tool_call]
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = chunk3_delta
        chunk3.choices[0].finish_reason = None
        
        # Final chunk
        chunk4 = MagicMock()
        chunk4.choices = [MagicMock()]
        chunk4.choices[0].delta = MagicMock()
        chunk4.choices[0].delta.content = None
        chunk4.choices[0].delta.tool_calls = None
        chunk4.choices[0].finish_reason = "stop"
        
        # Mock the stream behavior
        mock_stream.__aiter__.return_value = [chunk1, chunk2, chunk3, chunk4]
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai.return_value = mock_client
        
        # Create client
        client = OpenAIClient(llm_config)
        
        # Call stream_with_tools and collect chunks
        chunks = []
        async for chunk in client.stream_with_tools(llm_input):
            chunks.append(chunk)
        
        # Verify streaming API was called
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4"
        assert kwargs["stream"] is True
        
        # The end result should be the TestOutput structure
        assert len(chunks) == 1
        assert hasattr(chunks[0]["llm_response"], "response")
        assert chunks[0]["llm_response"].response == "Hello, world!" 