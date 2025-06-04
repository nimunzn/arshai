"""Mock tools implementation for testing."""

from typing import Any, Dict, List, Optional

from arshai.core.interfaces import ITool


class MockTool(ITool):
    """A mock tool implementation for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        parameters: Optional[Dict] = None,
        responses: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the mock tool.
        
        Args:
            name: The name of the tool
            description: The description of the tool
            parameters: The parameters for the tool
            responses: Dictionary mapping argument combinations to responses
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
        self.responses = responses or {}
        self.execution_history = []
    
    @property
    def function_definition(self) -> Dict:
        """
        Get the function definition for the LLM.
        
        Returns:
            Dict: Function definition in OpenAI format
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def execute(self, **kwargs) -> Any:
        """
        Mock implementation of execute.
        
        Args:
            **kwargs: The arguments for the tool
            
        Returns:
            Any: The result of the tool execution
        """
        self.execution_history.append(kwargs)
        
        # Convert kwargs to a sorted tuple of key-value pairs for dictionary key
        args_items = sorted(kwargs.items())
        args_key = tuple((k, str(v)) for k, v in args_items)
        
        # Look up the response or use a default
        if args_key in self.responses:
            result = self.responses[args_key]
        else:
            result = f"Mock execution of {self.name} with args: {kwargs}"
            
        return result
    
    async def aexecute(self, **kwargs) -> Any:
        """
        Async version of execute.
        
        Args:
            **kwargs: The arguments for the tool
            
        Returns:
            Any: The result of the tool execution
        """
        return self.execute(**kwargs)


class FailingMockTool(ITool):
    """A mock tool that fails when executed."""
    
    def __init__(self, name: str = "failing_tool", description: str = "A tool that fails"):
        """Initialize the failing mock tool."""
        self.name = name
        self.description = description
        self.execution_history = []
    
    @property
    def function_definition(self) -> Dict:
        """
        Get the function definition for the LLM.
        
        Returns:
            Dict: Function definition in OpenAI format
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        }
    
    def execute(self, **kwargs) -> Any:
        """
        Mock implementation that raises an exception.
        
        Args:
            **kwargs: The arguments for the tool
            
        Raises:
            Exception: Always raises an exception
        """
        self.execution_history.append(kwargs)
        raise Exception(f"Mock tool failure for {self.name}")
    
    async def aexecute(self, **kwargs) -> Any:
        """
        Async version of execute.
        
        Args:
            **kwargs: The arguments for the tool
            
        Raises:
            Exception: Always raises an exception
        """
        return self.execute(**kwargs) 