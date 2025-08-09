from typing import List, Dict, Callable, Any, Optional, TypeVar, Type, Union, Protocol, AsyncGenerator
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

from .idto import IDTO, IStreamDTO

T = TypeVar('T')


class ILLMInput(IDTO):
    """
    Represents the input for the llm - Unified interface supporting all functionality
    """
    system_prompt: str = Field(description="""The system prompt can be thought of as the input or query that the model
        uses to generate its response. The quality and specificity of the system prompt can have a significant impact 
        on the relevance and accuracy of the model's response. Therefore, it is important to provide a clear and 
        concise system prompt that accurately conveys the user's intended message or question.""")
    user_message: str = Field(description="the message of the user prompt")
    regular_functions: Dict[str, Callable] = Field(default={}, description="list of regular callable functions for this message")
    background_tasks: Dict[str, Callable] = Field(default={}, description="list of background tasks for fire-and-forget execution")
    structure_type: Type[T] = Field(default=None, description="Output response")
    max_turns: int = Field(default=10, description="Times that llm can call tools")

    @model_validator(mode='before')
    @classmethod
    def validate_input(cls, data):
        """Simplified validation focusing on actual requirements"""
        if not isinstance(data, dict):
            return data
        
        # Core requirements
        if not data.get('system_prompt'):
            raise ValueError("system_prompt is required")
        if not data.get('user_message'):
            raise ValueError("user_message is required")
        
        # No additional validation needed - structured output works independently
        
        return data


class ILLMConfig(IDTO):
    """Configuration for LLM providers"""
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

class ILLM(Protocol):
    """Protocol class for LLM providers - Updated to match BaseLLMClient implementation"""
    
    def __init__(self, config: ILLMConfig) -> None: ...
    
    # Core interface methods (what agents call)
    async def chat(self, input: ILLMInput) -> Dict[str, Any]: 
        """Main chat interface supporting all functionality"""
        ...
    
    async def stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Main streaming interface supporting all functionality"""  
        ...
    
    # Required abstract methods (what providers implement)
    def _initialize_client(self) -> Any:
        """Initialize the LLM provider client"""
        ...
    
    def _convert_callables_to_provider_format(self, functions: Dict[str, Callable]) -> Any:
        """Convert python callables to provider-specific function declarations"""
        ...
    
