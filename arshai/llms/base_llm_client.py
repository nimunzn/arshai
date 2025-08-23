"""
LLM-friendly Base LLM Client implementation.

Updated to use the new LLM-friendly observability system that:
1. Never creates OTEL providers  
2. Respects parent application's OTEL configuration
3. Uses get_tracer("arshai", version) pattern
4. Works with and without OTEL dependencies
5. Provides graceful fallbacks

This serves as the updated template for all LLM provider implementations.
"""

import asyncio
import logging
import traceback
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Union, AsyncGenerator, List, Type, Optional, Callable

from arshai.core.interfaces.illm import ILLM, ILLMConfig, ILLMInput
from arshai.llms.utils.function_execution import FunctionOrchestrator, FunctionExecutionInput, FunctionCall, StreamingExecutionState

# Import new observability system
from arshai.observability import get_llm_observability, PackageObservabilityConfig, TimingData
from arshai.llms.utils.usage_tracking import standardize_usage_metadata, accumulate_usage_safely

T = TypeVar("T")


class BaseLLMClient(ILLM, ABC):
    """
    LLM-friendly base class for all LLM clients.
    
    Updated to use the new observability system that properly respects
    parent OTEL configuration and never creates providers.
    
    Framework Features:
    - LLM-friendly observability (no provider creation)
    - Dual interface support with proper deprecation
    - Function calling orchestration
    - Structured output handling
    - Usage tracking standardization
    - Error handling and resilience
    """

    def __init__(
        self, 
        config: ILLMConfig, 
        observability_config: Optional[PackageObservabilityConfig] = None
    ):
        """
        Initialize the base LLM client with LLM-friendly observability.
        
        Args:
            config: LLM configuration
            observability_config: Optional package-specific observability configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Framework infrastructure
        self._function_orchestrator = FunctionOrchestrator()

        # Initialize LLM-friendly observability
        self.observability = get_llm_observability(observability_config)
        self._provider_name = self._get_provider_name()

        self.logger.info(f"Initializing {self.__class__.__name__} with model: {self.config.model}")
        
        if self.observability.is_enabled():
            self.logger.info(f"LLM-friendly observability enabled for {self._provider_name}")
        else:
            self.logger.info(f"Observability disabled for {self._provider_name}")

        # Initialize the provider-specific client
        self._client = self._initialize_client()

    # ========================================================================
    # ABSTRACT METHODS - What contributors must implement
    # ========================================================================

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the LLM provider client."""
        pass

    @abstractmethod
    def _convert_callables_to_provider_format(self, functions: Dict[str, Callable]) -> Any:
        """Convert python callables to provider-specific function declarations."""
        pass

    @abstractmethod
    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle simple chat without tools or background tasks."""
        pass

    @abstractmethod
    async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle complex chat with tools and/or background tasks."""
        pass

    @abstractmethod
    async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle simple streaming without tools or background tasks."""
        pass

    @abstractmethod
    async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex streaming with tools and/or background tasks."""
        pass

    def _get_provider_name(self) -> str:
        """Get the provider name for logging and usage tracking."""
        return self.__class__.__name__.replace("Client", "").lower()

    @abstractmethod
    def close(self):
        """Close any open connections or resources."""
        pass

    # ========================================================================
    # FRAMEWORK METHODS - Standard implementation for all providers
    # ========================================================================

    async def chat(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Framework-standardized chat method with LLM-friendly observability.
        
        This method handles:
        1. Routing between simple and complex cases
        2. LLM-friendly observability integration
        3. Error handling and logging
        4. Usage data recording
        """
        # Determine execution path
        has_tools = input.regular_functions and len(input.regular_functions) > 0
        has_background_tasks = input.background_tasks and len(input.background_tasks) > 0
        needs_function_calling = has_tools or has_background_tasks

        # Get model name for observability
        model_name = getattr(self.config, 'model', 'unknown')

        try:
            # Use LLM-friendly observability
            async with self.observability.observe_llm_call(
                provider=self._provider_name,
                model=model_name,
                method_name="chat",
                has_tools=has_tools,
                has_background_tasks=has_background_tasks,
                structure_requested=input.structure_type is not None
            ) as timing_data:
                
                # Add input content if logging is enabled and safe
                if (self.observability.config.should_log_content("prompts") and 
                    hasattr(timing_data, 'input_value')):
                    prompt_content = f"System: {str(input.system_prompt)}\nUser: {str(input.user_message)}"
                    max_length = self.observability.config.get_content_length_limit("prompts")
                    if len(prompt_content) > max_length:
                        prompt_content = prompt_content[:max_length] + "..."
                    timing_data.input_value = str(prompt_content)
                
                # Add invocation parameters from LLM config
                timing_data.invocation_parameters = {
                    "model": getattr(self.config, 'model', ''),
                    "temperature": getattr(self.config, 'temperature', 0.0),
                    "max_tokens": getattr(self.config, 'max_tokens', 0),
                }
                
                # Execute appropriate method based on complexity
                if not needs_function_calling:
                    result = await self._chat_simple(input)
                else:
                    result = await self._chat_with_functions(input)
                
                # Add response content if logging is enabled and safe
                if (self.observability.config.should_log_content("responses") and 
                    hasattr(timing_data, 'output_value') and
                    'llm_response' in result):
                    response_content = str(result['llm_response'])
                    max_length = self.observability.config.get_content_length_limit("responses")
                    if len(response_content) > max_length:
                        response_content = response_content[:max_length] + "..."
                    timing_data.output_value = str(response_content)
                
                # Record usage data if available
                if 'usage' in result:
                    await self.observability.record_usage_data(timing_data, result['usage'])
                
                return result

        except Exception as e:
            self.logger.error(f"Error in {self._provider_name} chat: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    async def stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Framework-standardized streaming method with LLM-friendly observability.
        """
        # Determine execution path
        has_tools = input.regular_functions and len(input.regular_functions) > 0
        has_background_tasks = input.background_tasks and len(input.background_tasks) > 0
        needs_function_calling = has_tools or has_background_tasks

        # Get model name for observability  
        model_name = getattr(self.config, 'model', 'unknown')

        try:
            # Use LLM-friendly observability for streaming
            async with self.observability.observe_streaming_llm_call(
                provider=self._provider_name,
                model=model_name,
                method_name="stream",
                has_tools=has_tools,
                has_background_tasks=has_background_tasks,
                structure_requested=input.structure_type is not None
            ) as timing_data:
                
                # Add input content if logging is enabled and safe
                if (self.observability.config.should_log_content("prompts") and
                    hasattr(timing_data, 'input_value')):
                    prompt_content = f"System: {str(input.system_prompt)}\nUser: {str(input.user_message)}"
                    max_length = self.observability.config.get_content_length_limit("prompts")
                    if len(prompt_content) > max_length:
                        prompt_content = prompt_content[:max_length] + "..."
                    timing_data.input_value = str(prompt_content)
                
                # Add invocation parameters from LLM config
                timing_data.invocation_parameters = {
                    "model": getattr(self.config, 'model', ''),
                    "temperature": getattr(self.config, 'temperature', 0.0),
                    "max_tokens": getattr(self.config, 'max_tokens', 0),
                }
                
                accumulated_response = []
                
                # Execute appropriate streaming method
                if not needs_function_calling:
                    stream_generator = self._stream_simple(input)
                else:
                    stream_generator = self._stream_with_functions(input)
                
                # Process stream chunks
                async for chunk in stream_generator:
                    # Record token timing for first chunk
                    if 'llm_response' in chunk and chunk['llm_response']:
                        if not accumulated_response:  # First token
                            timing_data.record_first_token()
                        timing_data.record_token()  # Update last token time
                        accumulated_response.append(chunk['llm_response'])
                    
                    # Record usage data if available
                    if 'usage' in chunk:
                        await self.observability.record_usage_data(timing_data, chunk['usage'])
                    
                    yield chunk
                
                # Add final response content if logging is enabled and safe
                if (accumulated_response and 
                    self.observability.config.should_log_content("responses") and
                    hasattr(timing_data, 'output_value')):
                    response_content = str(''.join(str(item) for item in accumulated_response))
                    max_length = self.observability.config.get_content_length_limit("responses") 
                    if len(response_content) > max_length:
                        response_content = response_content[:max_length] + "..."
                    timing_data.output_value = str(response_content)

        except Exception as e:
            self.logger.error(f"Error in {self._provider_name} stream: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    # ========================================================================
    # LEGACY METHOD SUPPORT (with deprecation warnings)
    # ========================================================================

    async def call_llm_async(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """DEPRECATED: Use chat() method instead."""
        warnings.warn(
            "call_llm_async is deprecated. Use chat() method instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert to new interface
        input_data = ILLMInput(
            system_prompt=system_prompt,
            user_message=user_message,
            **kwargs
        )
        
        result = await self.chat(input_data)
        return result.get('llm_response', '')

    async def call_llm_with_tools_async(
        self, 
        system_prompt: str, 
        user_message: str, 
        tools_list: List = None, 
        **kwargs
    ) -> str:
        """DEPRECATED: Use chat() method instead."""
        warnings.warn(
            "call_llm_with_tools_async is deprecated. Use chat() method instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert to new interface
        input_data = ILLMInput(
            system_prompt=system_prompt,
            user_message=user_message,
            regular_functions={f"tool_{i}": tool for i, tool in enumerate(tools_list or [])},
            **kwargs
        )
        
        result = await self.chat(input_data)
        return result.get('llm_response', '')

    # ========================================================================
    # UTILITY METHODS - Framework standardization helpers
    # ========================================================================

    def _standardize_usage_metadata(self, raw_usage_metadata: Any, provider: str = None, *args) -> Dict[str, Any]:
        """Standardize usage metadata from LLM providers.
        
        Args:
            raw_usage_metadata: Raw usage data from provider
            provider: Provider name (ignored, uses self._provider_name)
            *args: Additional arguments (ignored for compatibility)
        
        Returns:
            Standardized usage metadata dictionary
        """
        # Use the provider name from the instance
        provider_name = provider or self._provider_name or "generic"
        
        # Convert to standard format using utility function
        standardized = standardize_usage_metadata(raw_usage_metadata, provider_name)
        
        # Convert to observability format (input_tokens, output_tokens, total_tokens)
        return {
            'input_tokens': standardized.get('prompt_tokens', 0),
            'output_tokens': standardized.get('completion_tokens', 0),
            'total_tokens': standardized.get('total_tokens', 0),
        }
    
    def _accumulate_usage_safely(self, current_usage: Dict[str, Any], accumulated_usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Safely accumulate usage metadata.
        
        Args:
            current_usage: Current usage data
            accumulated_usage: Previously accumulated usage
        
        Returns:
            Updated accumulated usage
        """
        if accumulated_usage is None:
            return current_usage.copy() if current_usage else {}
        
        if current_usage is None:
            return accumulated_usage.copy()
        
        # Safely accumulate each field
        return {
            'input_tokens': accumulated_usage.get('input_tokens', 0) + current_usage.get('input_tokens', 0),
            'output_tokens': accumulated_usage.get('output_tokens', 0) + current_usage.get('output_tokens', 0), 
            'total_tokens': accumulated_usage.get('total_tokens', 0) + current_usage.get('total_tokens', 0),
        }

    # ========================================================================
    # CONTEXT MANAGER SUPPORT
    # ========================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        """Cleanup when client is garbage collected."""
        try:
            self.close()
        except Exception:
            # Ignore cleanup errors during garbage collection
            pass