"""Mock LLM implementation for testing."""

from typing import Dict, List, Optional, Union, TypeVar, Type, Any, AsyncGenerator

from arshai.core.interfaces import ILLM, ILLMConfig, ILLMInput, LLMInputType

T = TypeVar('T')

class MockLLM(ILLM):
    """A mock LLM implementation for testing."""

    def __init__(
        self, 
        config: ILLMConfig = None,
        responses: Optional[Dict[str, str]] = None,
        tool_calls: Optional[Dict[str, List[Dict]]] = None
    ):
        """
        Initialize the mock LLM.
        
        Args:
            config: LLM configuration
            responses: Dictionary mapping input messages to output responses
            tool_calls: Dictionary mapping input messages to tool calls
        """
        self.config = config or ILLMConfig(model="gpt-4", temperature=0.7)
        self.responses = responses or {}
        self.tool_calls = tool_calls or {}
        self.chat_history = []
        self.call_count = 0

    def _initialize_client(self) -> Any:
        """Initialize the LLM provider client"""
        return self  # Return self as the client for mocking

    def _create_structure_function(self, structure_type: Type[T]) -> Dict:
        """Create a function definition from the structure type"""
        return {
            "name": structure_type.__name__.lower(),
            "description": structure_type.__doc__ or f"Create a {structure_type.__name__} response",
            "parameters": {
                "type": "object",
                "properties": {
                    "field1": {"type": "string"},
                    "field2": {"type": "number"}
                }
            }
        }

    def _parse_to_structure(self, content: Union[str, dict], structure_type: Type[T]) -> T:
        """Parse response content into the specified structure type"""
        if isinstance(content, dict):
            return structure_type(**content)
        return structure_type(content=content)

    def chat_with_tools(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Mock implementation of chat_with_tools.
        
        Args:
            input: The input to the LLM
            
        Returns:
            A mocked output
        """
        self.call_count += 1
        self.chat_history.append(input)
        
        # Get the response based on the input or use a default
        response_text = self.responses.get(
            input.user_message,
            f"Mock response #{self.call_count} for: {input.user_message}"
        )
        
        # Get tool calls if any
        tool_calls_list = self.tool_calls.get(input.user_message, [])
        
        # Return as dict to match the observed interface
        return {
            "llm_response": response_text if not input.structure_type else {"agent_message": response_text},
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        
    def chat_completion(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Mock implementation of chat_completion.
        
        Args:
            input: The input to the LLM
            
        Returns:
            A mocked output
        """
        self.call_count += 1
        self.chat_history.append(input)
        
        # Get the response based on the input or use a default
        response_text = self.responses.get(
            input.user_message,
            f"Mock response #{self.call_count} for: {input.user_message}"
        )
        
        # Return as dict to match the observed interface
        return {
            "llm_response": response_text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        }
        
    async def stream_with_tools(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Mock implementation of stream_with_tools.
        
        Args:
            input: The input to the LLM
            
        Yields:
            Chunks of the response
        """
        self.call_count += 1
        self.chat_history.append(input)
        
        # Get the response based on the input or use a default
        response_text = self.responses.get(
            input.user_message,
            f"Mock response #{self.call_count} for: {input.user_message}"
        )
        
        # Split the response into chunks to simulate streaming
        words = response_text.split()
        chunks = [" ".join(words[i:i+2]) for i in range(0, len(words), 2)]
        
        for chunk in chunks:
            yield {
                "llm_response": chunk,
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
            }
        
        # Final chunk with the complete response
        yield {
            "llm_response": response_text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        } 