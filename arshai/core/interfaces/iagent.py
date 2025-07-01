from typing import Protocol, Tuple, Any, List, Dict, Callable, Optional, TypeVar, Type, Union
from pydantic import BaseModel, Field, model_validator
from .idto import IDTO, IStreamDTO
from .itool import ITool
from .imemorymanager import IWorkingMemory

T = TypeVar('T')


class IAgentConfig(IDTO):
    """
    Represents the configuration for the agent.

    Attributes:
        task_context: The conversation latest context
        tools: list of tool classes that could be used as llm tool
        response_example: Example structure for agent responses
    """
    task_context: str = Field(description="conversation latest context")
    tools: List[Any] = Field(description="""list of tool classes that could used as llm tool. 
        note that they must introduce in 'task_context' field""")
    # response_example: str = Field(description="Example structure for agent responses")

    @model_validator(mode='after')
    def validate_tools(self) -> 'IAgentConfig':
        """Validate that all tools implement ITool protocol"""
        for tool in self.tools:
            if not isinstance(tool, type) or not issubclass(tool, ITool):
                raise ValueError(f"Tool {tool} must be a class implementing ITool protocol")
        return self


class IAgentInput(IDTO):
    """
    Represents the input for the agent.
    """
    message: str = Field(description="The user input to be sent to the llm.")
    conversation_id: str = Field(description="Unique identifier for the conversation")
    stream: bool = Field(default=False, description="Whether to stream the response")

class IAgentOutput(IDTO):
    """
    Represents the final response from the agent to the user.

    Attributes:
        message: The actual response text to be sent to the user
    """
    ...

class IAgentStreamOutput(IStreamDTO):
    """
    Represents the stream output from the agent to the user.
    """
    message: str = Field(description="The message (response) of agent to the user message.")

class IAgent(Protocol):
    """
    Interface defining the contract for nodes.
    Any node implementation must conform to this interface.
    """

    def __init__(self, config: IAgentConfig) -> None: 
        """Initialize the agent with the given configuration"""
        ...
    
    async def aprocess_message(
            self,
            input: IAgentInput,
            stream: bool = False,
    ) -> Tuple[IAgentOutput, str]:
        """
        Process incoming message and generate response asynchronously.
        """
        ...

    async def process_message(
            self,
            input: IAgentInput,
    ) -> Tuple[IAgentOutput, str]:
        """
        Process incoming message and generate response asynchronously.

        Args:
            input: IAgentInput containing message, context and available functions
        Returns:
            Tuple[AgentResponse, str]: Response and message type
        """
        ...

    def _get_function_description(self) -> list:
        """Define available functions for the node"""
        ...

    def _get_callable_functions(self) -> dict:
        """Define function implementations"""
        ...

    def _prepare_system_prompt(self, working_memory: IWorkingMemory) -> str:
        """
        Prepare system prompt with context and response structure
        
        Args:
            working_memory: The working memory to use in the prompt
            
        Returns:
            str: The prepared system prompt
        """
        ...

    async def _get_llm_response(self, system_prompt: str, user_message: str) -> Any:
        """
        Get response from LLM with tools asynchronously
        
        Args:
            system_prompt: The prepared system prompt
            user_message: The user's input message
            
        Returns:
            Any: The LLM's response
        """
        ...

