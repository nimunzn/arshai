"""
Base LLM Client implementation.

Provides common functionality and standardized patterns that all LLM clients
should inherit from. Implements the routing logic and helper methods while
leaving provider-specific implementations as abstract methods.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Union, AsyncGenerator, List

from arshai.core.interfaces.illm import ILLM, ILLMConfig, ILLMInput
from arshai.llms.utils import FunctionOrchestrator

T = TypeVar("T")


class BaseLLMClient(ILLM, ABC):
    """
    Base implementation for all LLM clients.
    
    Provides standardized routing, error handling, and common functionality
    while requiring providers to implement their specific methods.
    """

    def __init__(self, config: ILLMConfig):
        """
        Initialize the base LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Shared infrastructure
        self._background_tasks: set = set()
        self._function_orchestrator = FunctionOrchestrator()

        self.logger.info(f"Initializing {self.__class__.__name__} with model: {self.config.model}")

        # Initialize the provider-specific client
        self._client = self._initialize_client()

    # Abstract methods that providers must implement
    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the LLM provider client."""
        pass

    @abstractmethod
    def _convert_functions_to_llm_format(
        self, 
        functions: Union[List[Dict], Dict[str, Any]], 
        function_type: str = "tool"
    ) -> List[Any]:
        """Convert functions to LLM provider-specific format."""
        pass

    @abstractmethod
    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle simple chat without tools or background tasks."""
        pass

    @abstractmethod
    async def _chat_with_tools(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle complex chat with tools and/or background tasks."""
        pass

    @abstractmethod
    async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle simple streaming without tools or background tasks."""
        pass

    @abstractmethod
    async def _stream_with_tools(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex streaming with tools and/or background tasks."""
        pass

    # Standard implementations (can be overridden if needed)
    def _needs_function_calling(self, input: ILLMInput) -> bool:
        """
        Determine if function calling is needed based on input.
        
        Standard implementation that checks for tools and background tasks.
        
        Args:
            input: The LLM input to evaluate
            
        Returns:
            True if function calling (tools or background tasks) is needed
        """
        has_tools = input.tools_list and len(input.tools_list) > 0
        has_background_tasks = input.background_tasks and len(input.background_tasks) > 0
        return has_tools or has_background_tasks

    def _prepare_base_context(self, input: ILLMInput) -> str:
        """
        Build base conversation context from system prompt and user message.
        
        Default implementation that can be overridden for provider-specific formats.
        
        Args:
            input: The LLM input
            
        Returns:
            Formatted conversation context string
        """
        return f"{input.system_prompt}\n\nUser: {input.user_message}"

    # Standard routing methods
    async def chat(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Process a chat message with optional tools and structured output.
        
        Standard routing implementation that delegates to provider-specific methods.

        Args:
            input: The LLM input containing system prompt, user message, tools, and options

        Returns:
            Dict containing the LLM response and usage information
        """
        try:
            # Route to appropriate handler based on function calling needs
            if self._needs_function_calling(input):
                return await self._chat_with_tools(input)
            else:
                return await self._chat_simple(input)
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__} chat: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"llm_response": f"An error occurred: {str(e)}", "usage": None}

    async def stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a streaming chat message with optional tools and structured output.
        
        Standard routing implementation that delegates to provider-specific methods.

        Args:
            input: The LLM input containing system prompt, user message, tools, and options

        Yields:
            Dict containing streaming response chunks and usage information
        """
        try:
            if self._needs_function_calling(input):
                async for chunk in self._stream_with_tools(input):
                    yield chunk
            else:
                async for chunk in self._stream_simple(input):
                    yield chunk
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__} stream: {str(e)}")
            self.logger.error(traceback.format_exc())
            yield {"llm_response": f"An error occurred: {str(e)}", "usage": None}

    # Helper methods for common operations
    def _log_provider_info(self, message: str):
        """Log provider-specific information."""
        self.logger.info(f"[{self.__class__.__name__}] {message}")

    def _log_provider_debug(self, message: str):
        """Log provider-specific debug information."""
        self.logger.debug(f"[{self.__class__.__name__}] {message}")

    def _log_provider_error(self, message: str):
        """Log provider-specific error information."""
        self.logger.error(f"[{self.__class__.__name__}] {message}")

    def _get_provider_name(self) -> str:
        """Get the provider name for logging and identification."""
        return self.__class__.__name__.replace("Client", "").lower()