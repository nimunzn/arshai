"""
Base agent implementation for the Arshai framework.

This module provides the foundational BaseAgent class that all other agents
must extend. It provides common functionality while requiring developers
to implement their own process method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from arshai.core.interfaces.iagent import IAgent, IAgentInput
from arshai.core.interfaces.illm import ILLM


class BaseAgent(IAgent, ABC):
    """
    Abstract base agent implementation.
    
    This class provides the foundational structure for all agents.
    Developers must implement the process method to define their agent's behavior.
    
    The agent gives developers complete authority over:
    - Response format (streaming, structured, simple string, etc.)
    - Data structure returned
    - How to use the LLM client
    - Tool integration patterns
    - Memory management
    
    Example:
        class MyAgent(BaseAgent):
            async def process(self, input: IAgentInput) -> str:
                # Simple string response
                result = await self.llm_client.chat(ILLMInput(
                    system_prompt=self.system_prompt,
                    user_message=input.message
                ))
                return result['response']
    """
    
    def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
        """
        Initialize the base agent.
        
        Args:
            llm_client: The LLM client to use for processing
            system_prompt: The system prompt that defines the agent's behavior
            **kwargs: Additional configuration passed to the agent
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.config = kwargs
    
    @abstractmethod
    async def process(self, input: IAgentInput) -> Any:
        """
        Process the input and return a response.
        
        This method MUST be implemented by subclasses.
        The return type is Any to give developers complete flexibility over:
        - Response format (streaming, non-streaming, structured, etc.)
        - Data types (string, dict, custom DTOs, generators, etc.)
        - Error handling approach
        
        Args:
            input: The input containing message and optional metadata
            
        Returns:
            Any: Developer-defined response format
        """
        ...