from typing import List, Dict, Callable, Any, Optional, TypeVar, Type, Union, Protocol, AsyncGenerator
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from enum import Enum, StrEnum
from pydantic import BaseModel, Field, model_validator

from .idto import IDTO, IStreamDTO
from .iagent import IAgentOutput, IAgentStreamOutput

T = TypeVar('T')


class LLMInputType(StrEnum):
    CHAT_COMPLETION = 'CHAT_COMPLETION'
    CHAT_WITH_TOOLS = 'CHAT_WITH_TOOLS'


class ILLMInput(IDTO):
    """
    Represents the input for the llm
    """
    input_type: LLMInputType = Field(
        default=LLMInputType.CHAT_COMPLETION, exclude=True, description="the type of llm calling"
    )
    system_prompt: str = Field(description="""The system prompt can be thought of as the input or query that the model
        uses to generate its response. The quality and specificity of the system prompt can have a significant impact 
        on the relevance and accuracy of the model's response. Therefore, it is important to provide a clear and 
        concise system prompt that accurately conveys the user's intended message or question.""")
    user_message: str = Field(description="the message of the user prompt")
    tools_list: List[Dict] = Field(default=[], description="list of defined tools for llm")
    callable_functions: Dict[str, Callable] = Field(default={}, description="list of callable tools for this message")
    background_tasks: Dict[str, Callable] = Field(default={}, description="fire-and-forget background tasks that run independently")
    structure_type: Type[T] = Field(default=None, description="Output response")
    max_turns: int = Field(default=3, description="Times that llm can call tools")

    @model_validator(mode='before')
    @classmethod
    def validate_input_based_on_type(cls, data):
        if not isinstance(data, dict):  # Ensure data is a dictionary
            return data
        if not data.get('system_prompt'):
            raise ValueError("system_prompt is required for chat completion and chat with tools")
        if not data.get('user_message'):
            raise ValueError("user_message is required for chat completion and chat with tools")

        input_type = data.get('input_type', LLMInputType.CHAT_COMPLETION)
        if input_type == LLMInputType.CHAT_WITH_TOOLS:
            if not data.get('tools_list'):
                raise ValueError("tools_list is required for chat with tools")
            if not data.get('callable_functions'):
                raise ValueError("callable_functions is required for chat with tools")
            if not data.get('structure_type'):
                raise ValueError("structure_type is required for chat with tools")
            if not data.get('max_turns'):
                raise ValueError("max_turns is required for chat with tools")
        return data

    def __getattr__(self, item):
        if item == 'input_type':
            raise AttributeError("'input_type' is a dynamically generated attribute and cannot be directly accessed")
        return super().__getattr__(item)

    def __getattribute__(self, item):
        if item == 'input_type':
            raise AttributeError("'input_type' is a dynamically generated attribute and cannot be directly accessed")
        return super().__getattribute__(item)

    def model_dump(self, **kwargs):
        # Override model_dump to exclude input_type
        kwargs.setdefault('exclude', set()).add('input_type')
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        # Override model_dump_json to exclude input_type
        kwargs.setdefault('exclude', set()).add('input_type')
        return super().model_dump_json(**kwargs)


class ILLMOutput(IDTO):
    """
    Represents the complete response from the agent

    agent_message: The actual response message to the user
    """
    agent_message: IAgentOutput = Field(
        description="""The message (response) of llm to the user message."""
    )

class ILLMStreamOutput(IStreamDTO):
    """
    Represents the complete stream response from the agent, 

    agent_message: The actual response message to the user
    """
    agent_message: IAgentStreamOutput = Field(description="The message (response) of llm to the user message.")


class ILLMConfig(IDTO):
    """Configuration for LLM providers"""
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

class ILLM(Protocol):
    """Protocol class for LLM providers"""
    
    def __init__(self, config: ILLMConfig) -> None: ...
    
    def _initialize_client(self) -> Any:
        """Initialize the LLM provider client"""
        ...

    def _needs_function_calling(self, input: ILLMInput) -> bool:
        """Determine if function calling is needed based on input"""
        ...

    def _convert_functions_to_llm_format(
        self, 
        functions: Union[List[Dict], Dict[str, Any]], 
        function_type: str = "tool"
    ) -> List[Any]:
        """Convert functions to LLM provider-specific format"""
        ...

    
    async def chat(self, input: ILLMInput) -> Dict[str, Any]:
        """Process a chat message with optional tools and structured output"""
        ...
    
    async def stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a streaming chat message with optional tools and structured output"""
        ...
