"""Unit tests for Azure OpenAI LLM client."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import os
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from arshai.core.interfaces import ILLMConfig, ILLMInput, LLMInputType
from arshai.llms.azure import AzureClient


class TestResponseModel(BaseModel):
    """Test response model for structured outputs"""
    result: str
    confidence: float
    details: Optional[Dict[str, Any]] = None


@pytest.fixture
def mock_openai_client():
    """Create a mock Azure OpenAI client."""
    mock_client = MagicMock()
    
    # Mock chat completions create method
    mock_chat_completions = MagicMock()
    mock_client.chat.completions.create = mock_chat_completions
    
    # Set up a default mock response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "This is a test response"
    mock_message.function_call = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    # Add usage information
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150
    mock_response.usage = mock_usage
    
    mock_chat_completions.return_value = mock_response
    
    return mock_client


@pytest.fixture
def azure_config():
    """Create a basic LLM configuration for Azure."""
    return ILLMConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def azure_client(mock_openai_client):
    """Create an Azure client with mocked OpenAI client."""
    with patch('src.llms.azure.AzureOpenAI', return_value=mock_openai_client):
        with patch.dict('os.environ', {
            'AZURE_DEPLOYMENT': 'test-deployment',
            'AZURE_API_VERSION': '2023-05-15'
        }):
            client = AzureClient(
                config=ILLMConfig(
                    model="gpt-4",
                    temperature=0.7,
                    max_tokens=1000
                )
            )
            return client


@pytest.fixture
def basic_input():
    """Create a basic LLM input for testing."""
    return ILLMInput(
        input_type=LLMInputType.CHAT_COMPLETION,
        system_prompt="You are a helpful assistant",
        user_message="Tell me about AI",
        tools_list=[],
        callable_functions={},
        structure_type=TestResponseModel
    )


def test_initialization():
    """Test initialization with different parameters."""
    # Test with explicit parameters
    with patch('src.llms.azure.AzureOpenAI') as mock_azure:
        client = AzureClient(
            config=ILLMConfig(model="gpt-4", temperature=0.7),
            azure_deployment="test-deployment",
            api_version="2023-05-15"
        )
        
        assert client.azure_deployment == "test-deployment"
        assert client.api_version == "2023-05-15"
        mock_azure.assert_called_once_with(
            azure_deployment="test-deployment",
            api_version="2023-05-15"
        )
    
    # Test with environment variables
    with patch('src.llms.azure.AzureOpenAI') as mock_azure:
        with patch.dict('os.environ', {
            'AZURE_DEPLOYMENT': 'env-deployment',
            'AZURE_API_VERSION': 'env-api-version'
        }):
            client = AzureClient(
                config=ILLMConfig(model="gpt-4", temperature=0.7)
            )
            
            assert client.azure_deployment == "env-deployment"
            assert client.api_version == "env-api-version"
            mock_azure.assert_called_once_with(
                azure_deployment="env-deployment",
                api_version="env-api-version"
            )
    
    # Test with missing deployment
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="Azure deployment is required"):
            AzureClient(config=ILLMConfig(model="gpt-4", temperature=0.7))
    
    # Test with missing API version
    with patch.dict('os.environ', {'AZURE_DEPLOYMENT': 'test-deployment'}):
        with pytest.raises(ValueError, match="Azure API version is required"):
            AzureClient(config=ILLMConfig(model="gpt-4", temperature=0.7))


def test_chat_completion(azure_client, basic_input, mock_openai_client):
    """Test basic chat completion without tools."""
    result = azure_client.chat_completion(basic_input)
    
    # Verify client was called with correct parameters
    mock_openai_client.chat.completions.create.assert_called_once()
    args = mock_openai_client.chat.completions.create.call_args[1]
    
    assert args["model"] == "gpt-4"
    assert args["temperature"] == 0.7
    assert len(args["messages"]) == 2
    assert args["messages"][0]["role"] == "system"
    assert args["messages"][0]["content"] == "You are a helpful assistant"
    assert args["messages"][1]["role"] == "user"
    assert args["messages"][1]["content"] == "Tell me about AI"
    
    # Verify response format
    assert result["llm_response"] == "This is a test response"
    assert result["usage"].prompt_tokens == 100
    assert result["usage"].completion_tokens == 50
    assert result["usage"].total_tokens == 150


def test_chat_with_tools_no_function_call(azure_client, basic_input, mock_openai_client):
    """Test chat with tools when no functions are called."""
    result = azure_client.chat_with_tools(basic_input)
    
    # Verify client was called with correct parameters
    mock_openai_client.chat.completions.create.assert_called_once()
    args = mock_openai_client.chat.completions.create.call_args[1]
    
    assert args["model"] == "gpt-4"
    assert args["temperature"] == 0.7
    assert len(args["messages"]) == 2
    assert args["functions"] == []
    
    # Verify response format
    assert result["llm_response"] == "This is a test response"
    assert result["usage"].prompt_tokens == 100
    assert result["usage"].total_tokens == 150


def test_chat_with_tools_function_call(azure_client, basic_input, mock_openai_client):
    """Test chat with tools with function calling."""
    # Define test tool
    test_tool = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "string"}
            },
            "required": ["param1"]
        }
    }
    
    # Add tool to input
    input_with_tools = ILLMInput(
        input_type=LLMInputType.CHAT_COMPLETION,
        system_prompt="You are a helpful assistant",
        user_message="Use the test tool",
        tools_list=[test_tool],
        callable_functions={"test_tool": lambda **kwargs: "Tool result"},
        structure_type=TestResponseModel
    )
    
    # Mock function call response
    mock_message = MagicMock()
    mock_message.content = None
    mock_function_call = MagicMock()
    mock_function_call.name = "test_tool"
    mock_function_call.arguments = json.dumps({"param1": "value1"})
    mock_message.function_call = mock_function_call
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    # For second call, return a content response
    mock_second_message = MagicMock()
    mock_second_message.content = "Final response after using tool"
    mock_second_message.function_call = None
    
    mock_second_choice = MagicMock()
    mock_second_choice.message = mock_second_message
    
    mock_second_response = MagicMock()
    mock_second_response.choices = [mock_second_choice]
    mock_second_response.usage.prompt_tokens = 150
    mock_second_response.usage.completion_tokens = 75
    mock_second_response.usage.total_tokens = 225
    
    mock_openai_client.chat.completions.create.side_effect = [mock_response, mock_second_response]
    
    result = azure_client.chat_with_tools(input_with_tools)
    
    # Verify client was called twice (initial + after function)
    assert mock_openai_client.chat.completions.create.call_count == 2
    
    # Verify first call parameters
    first_call_args = mock_openai_client.chat.completions.create.call_args_list[0][1]
    assert first_call_args["functions"] == [test_tool]
    assert first_call_args["function_call"] == "auto"
    
    # Verify second call parameters (should include function result)
    second_call_args = mock_openai_client.chat.completions.create.call_args_list[1][1]
    assert len(second_call_args["messages"]) == 4
    assert second_call_args["messages"][2]["role"] == "function"
    assert second_call_args["messages"][2]["name"] == "test_tool"
    assert second_call_args["messages"][2]["content"] == "Tool result"
    
    # Verify response
    assert result["llm_response"] == "Final response after using tool"
    # Verify accumulated usage
    assert result["usage"].prompt_tokens == 250
    assert result["usage"].completion_tokens == 125
    assert result["usage"].total_tokens == 375


def test_chat_with_structured_output(azure_client, mock_openai_client):
    """Test chat with structured output."""
    # Create input with structure type
    structured_input = ILLMInput(
        input_type=LLMInputType.CHAT_COMPLETION,
        system_prompt="You are a helpful assistant",
        user_message="Analyze this text",
        tools_list=[],
        callable_functions={},
        structure_type=TestResponseModel
    )
    
    # Mock structured response via function call
    mock_message = MagicMock()
    mock_message.content = None
    mock_function_call = MagicMock()
    mock_function_call.name = "testresponsemodel"
    mock_function_call.arguments = json.dumps({
        "result": "Analysis complete",
        "confidence": 0.95,
        "details": {"key": "value"}
    })
    mock_message.function_call = mock_function_call
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    result = azure_client.chat_with_tools(structured_input)
    
    # Verify client was called with structure function
    mock_openai_client.chat.completions.create.assert_called_once()
    args = mock_openai_client.chat.completions.create.call_args[1]
    
    # Verify structure function was added
    assert len(args["functions"]) == 1
    assert args["functions"][0]["name"] == "testresponsemodel"
    
    # Verify structured response
    assert result["llm_response"].result == "Analysis complete"
    assert result["llm_response"].confidence == 0.95
    assert result["llm_response"].details == {"key": "value"}
    assert result["usage"].total_tokens == 150


def test_chat_with_tools_error_handling(azure_client, basic_input, mock_openai_client):
    """Test error handling in chat with tools."""
    # Make API call raise an exception
    mock_openai_client.chat.completions.create.side_effect = Exception("API error")
    
    result = azure_client.chat_with_tools(basic_input)
    
    # Verify error is handled and returned
    assert "llm_response" in result
    assert "An error occurred: API error" in result["llm_response"]
    assert result["usage"] is None


@pytest.mark.asyncio
async def test_stream_completion(azure_client, basic_input, mock_openai_client):
    """Test streaming chat completion."""
    # Mock streaming response
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "This "
    mock_chunk1.choices[0].delta.role = "assistant"
    
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = "is "
    mock_chunk2.choices[0].delta.role = None
    
    mock_chunk3 = MagicMock()
    mock_chunk3.choices = [MagicMock()]
    mock_chunk3.choices[0].delta.content = "a test"
    mock_chunk3.choices[0].delta.role = None
    
    mock_final_chunk = MagicMock()
    mock_final_chunk.choices = [MagicMock()]
    mock_final_chunk.choices[0].delta.content = None
    mock_final_chunk.choices[0].delta.role = None
    mock_final_chunk.choices[0].finish_reason = "stop"
    
    # Setup usage information in the final chunk
    mock_final_chunk.usage = MagicMock()
    mock_final_chunk.usage.prompt_tokens = 100
    mock_final_chunk.usage.completion_tokens = 50
    mock_final_chunk.usage.total_tokens = 150
    
    # Make create return an async iterator of chunks
    async_iter = AsyncMock()
    async_iter.__aiter__.return_value = [mock_chunk1, mock_chunk2, mock_chunk3, mock_final_chunk]
    mock_openai_client.chat.completions.create.return_value = async_iter
    
    # Collect all chunks from the stream
    chunks = []
    async for chunk in azure_client.stream_completion(basic_input):
        chunks.append(chunk)
    
    # Verify client was called with correct parameters
    mock_openai_client.chat.completions.create.assert_called_once()
    args = mock_openai_client.chat.completions.create.call_args[1]
    
    assert args["model"] == "gpt-4"
    assert args["temperature"] == 0.7
    assert args["stream"] is True
    
    # Verify chunks content
    assert len(chunks) == 4
    assert chunks[0]["delta"] == "This "
    assert chunks[1]["delta"] == "is "
    assert chunks[2]["delta"] == "a test"
    assert chunks[3]["finish_reason"] == "stop"
    assert chunks[3]["usage"].total_tokens == 150


@pytest.mark.asyncio
async def test_stream_with_tools(azure_client, mock_openai_client):
    """Test streaming with tools."""
    # Define test tool
    test_tool = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {"param1": {"type": "string"}},
            "required": ["param1"]
        }
    }
    
    # Create input with tool
    input_with_tools = ILLMInput(
        input_type=LLMInputType.CHAT_COMPLETION,
        system_prompt="You are a helpful assistant",
        user_message="Use the test tool",
        tools_list=[test_tool],
        callable_functions={"test_tool": lambda **kwargs: "Tool result"},
        structure_type=TestResponseModel
    )
    
    # Mock first response with function call
    mock_message = MagicMock()
    mock_message.content = None
    mock_function_call = MagicMock()
    mock_function_call.name = "test_tool"
    mock_function_call.arguments = json.dumps({"param1": "value1"})
    mock_message.function_call = mock_function_call
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    
    # Mock streaming response after function call
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Response "
    mock_chunk1.choices[0].delta.role = "assistant"
    
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = "after "
    mock_chunk2.choices[0].delta.role = None
    
    mock_chunk3 = MagicMock()
    mock_chunk3.choices = [MagicMock()]
    mock_chunk3.choices[0].delta.content = "tool call"
    mock_chunk3.choices[0].delta.role = None
    
    mock_final_chunk = MagicMock()
    mock_final_chunk.choices = [MagicMock()]
    mock_final_chunk.choices[0].delta.content = None
    mock_final_chunk.choices[0].delta.role = None
    mock_final_chunk.choices[0].finish_reason = "stop"
    
    # Setup usage information in the final chunk
    mock_final_chunk.usage = MagicMock()
    mock_final_chunk.usage.prompt_tokens = 150
    mock_final_chunk.usage.completion_tokens = 75
    mock_final_chunk.usage.total_tokens = 225
    
    # Setup mocked responses
    mock_openai_client.chat.completions.create.side_effect = [
        mock_response,  # First call for function calling
        AsyncMock(__aiter__=AsyncMock(return_value=[mock_chunk1, mock_chunk2, mock_chunk3, mock_final_chunk]))  # Second call for streaming
    ]
    
    # Collect all chunks from the stream
    chunks = []
    async for chunk in azure_client.stream_with_tools(input_with_tools):
        chunks.append(chunk)
    
    # Verify client was called twice
    assert mock_openai_client.chat.completions.create.call_count == 2
    
    # Verify chunks content
    assert len(chunks) == 5  # Function call + 4 content chunks
    assert chunks[0]["type"] == "function_call"
    assert chunks[0]["name"] == "test_tool"
    assert chunks[1]["delta"] == "Response "
    assert chunks[2]["delta"] == "after "
    assert chunks[3]["delta"] == "tool call"
    assert chunks[4]["finish_reason"] == "stop"
    assert chunks[4]["usage"].total_tokens == 225 