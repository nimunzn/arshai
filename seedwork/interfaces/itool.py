from typing import Protocol, Dict, Any

class ITool(Protocol):
    """
    Interface defining the contract for tools.
    Any tool implementation must conform to this interface.
    """
    
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool's functionality.

        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments

        Returns:
            Any: Result of the tool execution
        """
        ...

    async def aexecute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Asynchronous execution of the tool's functionality.

        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments

        Returns:
            Any: Result of the tool execution
        """
        ...
    
    @property
    def function_definition(self) -> Dict:
        """
        Get the function definition for the LLM.

        Returns:
            Dict: Function definition in OpenAI format
        """
        ... 

        