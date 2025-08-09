from typing import Protocol, Dict, Any, Optional
from pydantic import Field
from .idto import IDTO


class IAgentInput(IDTO):
    """
    Input for agents.
    
    Attributes:
        message: The user's message to process
        metadata: Optional metadata for agent-specific context (conversation_id, stream, etc.)
    """
    message: str = Field(description="The user's message to process")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional metadata for agent-specific context (conversation_id, stream, etc.)"
    )


class IAgent(Protocol):
    """
    Agent interface.
    
    All agents must implement this protocol.
    The interface gives maximum flexibility to developers for response format.
    """
    
    async def process(self, input: IAgentInput) -> Any:
        """
        Process the input and return a response.
        
        This is the only required method for agents.
        The return type is Any to give developers full authority over:
        - Response format (streaming, structured, simple string, etc.)
        - Data structure (custom DTOs, tuples, dicts, etc.)
        - Streaming vs non-streaming responses
        
        Agents can internally use tools, memory, or any other
        capabilities as needed.
        
        Args:
            input: The input containing message and optional metadata
            
        Returns:
            Any: Developer-defined response format
        """
        ...

