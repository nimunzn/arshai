"""
Base LLM Client implementation - Version 2.

Provides a comprehensive framework-standardized base class that serves as the 
contributor's guide for implementing new LLM providers. Handles all framework 
requirements including dual interface support, function calling, background tasks,
structured output, and usage tracking.

This is the template that all LLM providers should inherit from.
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

T = TypeVar("T")


class BaseLLMClient(ILLM, ABC):
    """

    Framework-standardized base class for all LLM clients.
    
    This class serves as the contributor's guide and template for implementing
    new LLM providers. It handles all framework requirements while requiring
    providers to implement only their specific API integration methods.
    
    Framework Features Handled by Base Class:
    - Dual interface support (old + new methods with deprecation)
    - Function calling orchestration (regular + background tasks)  
    - Structured output handling
    - Usage tracking standardization
    - Error handling and resilience
    - Routing logic between simple and complex cases
    
    What Contributors Need to Implement:
    - Provider-specific API client initialization
    - Provider-specific chat/stream methods
    - Provider-specific format conversions

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

        """
        Handle complex streaming with tools and/or background tasks.
        
        Args:
            input: LLM input with regular_functions, background_tasks
            
        Yields:
            Dict with 'llm_response' and optional 'usage' keys
        """
        pass

    # ========================================================================
    # FRAMEWORK HELPER METHODS - Available to all providers
    # ========================================================================

    def _needs_function_calling(self, input: ILLMInput) -> bool:
        """
        Determine if function calling is needed based on input.
        
        Framework-standardized logic that all providers should use.
        
        Args:
            input: The LLM input to evaluate
            
        Returns:
            True if function calling (regular functions or background tasks) is needed
        """
        has_regular_functions = input.regular_functions and len(input.regular_functions) > 0
        has_background_tasks = input.background_tasks and len(input.background_tasks) > 0
        return has_regular_functions or has_background_tasks

    async def _execute_functions_with_orchestrator(
        self,
        execution_input_or_legacy: Union[FunctionExecutionInput, Dict[str, Any]],
        regular_args: Dict[str, Dict[str, Any]] = None,
        background_tasks: Dict[str, Any] = None,
        background_args: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute functions using the framework's standardized orchestrator.
        
        Supports both new object-based approach and legacy dictionary approach for backward compatibility.
        
        Args:
            execution_input_or_legacy: Either FunctionExecutionInput (new) or regular_functions dict (legacy)
            regular_args: Dict of arguments for regular functions (legacy only)
            background_tasks: Dict of background task functions (legacy only) 
            background_args: Dict of arguments for background tasks (legacy only)
            
        Returns:
            Orchestrator execution result in generic format
        """
        # Check if we're using the new object-based approach
        if isinstance(execution_input_or_legacy, FunctionExecutionInput):
            # New object-based approach
            execution_input = execution_input_or_legacy
        else:
            # Legacy dictionary approach - convert to new format
            regular_functions = execution_input_or_legacy
            execution_input = FunctionExecutionInput(
                function_calls=[],  # Will be populated below
                available_functions=regular_functions,
                available_background_tasks=background_tasks or {}
            )
            
            # Convert dictionaries to FunctionCall objects
            function_calls = []
            
            # Convert regular functions
            for name, func in regular_functions.items():
                args = regular_args.get(name, {}) if regular_args else {}
                function_calls.append(FunctionCall(
                    name=name,
                    args=args,
                    is_background=False
                ))
            
            # Convert background tasks
            if background_tasks:
                for name, func in background_tasks.items():
                    args = background_args.get(name, {}) if background_args else {}
                    function_calls.append(FunctionCall(
                        name=name,
                        args=args,
                        is_background=True
                    ))
            
            execution_input.function_calls = function_calls
        
        result = await self._function_orchestrator.execute_functions(execution_input)
        
        # Convert to dict format for easier handling by providers
        return {
            "regular_results": result.regular_results,
            "background_initiated": result.background_initiated,
            "failed_functions": result.failed_functions
        }

    # def _standardize_usage_metadata(self, raw_usage: Any, provider: str, model: str, request_id: str = None) -> Dict[str, Any]:
    #     """
    #     Standardize usage metadata to framework format.
        
    #     Framework-standardized usage format that all providers should return.
        
    #     Args:
    #         raw_usage: Provider-specific usage metadata
    #         provider: Provider name (e.g., "openai", "gemini")
    #         model: Model name used
    #         request_id: Optional request ID from provider
            
    #     Returns:
    #         Standardized usage metadata dict
    #     """
    #     if not raw_usage:
    #         return {
    #             "input_tokens": 0,
    #             "output_tokens": 0,
    #             "total_tokens": 0,
    #             "thinking_tokens": 0,
    #             "tool_calling_tokens": 0,
    #             "provider": provider,
    #             "model": model,
    #             "request_id": request_id
    #         }

    #     # Extract standard fields (providers should override this method if needed)
    #     input_tokens = getattr(raw_usage, 'prompt_tokens', 0) or getattr(raw_usage, 'input_tokens', 0)
    #     output_tokens = getattr(raw_usage, 'completion_tokens', 0) or getattr(raw_usage, 'output_tokens', 0)
    #     total_tokens = getattr(raw_usage, 'total_tokens', input_tokens + output_tokens)
        
    #     # Optional advanced fields
    #     thinking_tokens = getattr(raw_usage, 'reasoning_tokens', 0) or getattr(raw_usage, 'thinking_tokens', 0)
    #     tool_calling_tokens = getattr(raw_usage, 'tool_calling_tokens', 0) or getattr(raw_usage, 'function_call_tokens', 0)

    #     return {
    #         "input_tokens": input_tokens,
    #         "output_tokens": output_tokens,
    #         "total_tokens": total_tokens,
    #         "thinking_tokens": thinking_tokens,
    #         "tool_calling_tokens": tool_calling_tokens,
    #         "provider": provider,
    #         "model": model,
    #         "request_id": request_id
    #     }

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
        Framework-standardized chat method with optional LLM-friendly observability.
        
        This method handles:
        1. Routing between simple and complex cases
        2. Optional LLM-friendly observability integration
        3. Error handling and logging
        4. Usage data recording
        """
        # Get common values used by both observability and core logic
        model_name = getattr(self.config, 'model', 'unknown')
        has_tools = input.regular_functions and len(input.regular_functions) > 0
        has_background_tasks = input.background_tasks and len(input.background_tasks) > 0

        if self.observability.is_enabled():
            # WITH observability - wrap the core logic
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
                        "model": model_name,
                        "temperature": getattr(self.config, 'temperature', 0.0),
                        "max_tokens": getattr(self.config, 'max_tokens', 0),
                    }
                    
                    # Execute core chat logic
                    result = await self._execute_chat(input)
                    
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
                return {
                    "llm_response": f"An error occurred: {str(e)}", 
                    "usage": {
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "thinking_tokens": 0, "tool_calling_tokens": 0,
                        "provider": self._provider_name, "model": getattr(self.config, 'model', 'unknown'),
                        "request_id": None
                    }
                }
        else:
            # WITHOUT observability - direct execution
            try:
                return await self._execute_chat(input)
            except Exception as e:
                self.logger.error(f"Error in {self._provider_name} chat: {e}")
                self.logger.debug(traceback.format_exc())
                return {
                    "llm_response": f"An error occurred: {str(e)}", 
                    "usage": {
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "thinking_tokens": 0, "tool_calling_tokens": 0,
                        "provider": self._provider_name, "model": getattr(self.config, 'model', 'unknown'),
                        "request_id": None
                    }
                }

    async def _execute_chat(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Execute core chat logic without observability concerns.
        
        This method contains the actual LLM execution logic and works
        regardless of observability state.
        
        Args:
            input: The LLM input containing system prompt, user message, tools, and options

        Returns:
            Dict containing 'llm_response' and 'usage' keys
        """
        try:
            self.logger.info(f"Processing chat request - Regular Functions: {bool(input.regular_functions)}, "
                           f"Background: {bool(input.background_tasks)}, "
                           f"Structured: {bool(input.structure_type)}")

            # Use the helper method to determine execution path
            needs_function_calling = self._needs_function_calling(input)

            # Execute appropriate method based on complexity
            if not needs_function_calling:
                return await self._chat_simple(input)
            else:
                return await self._chat_with_functions(input)

        except Exception as e:
            self.logger.error(f"Error in {self._provider_name} chat: {e}")
            self.logger.debug(traceback.format_exc())
            return {
                "llm_response": f"An error occurred: {str(e)}", 
                "usage": {
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "thinking_tokens": 0, "tool_calling_tokens": 0,
                    "provider": self._provider_name, "model": getattr(self.config, 'model', 'unknown'),
                    "request_id": None
                }
            }

    async def stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Framework-standardized streaming method with optional LLM-friendly observability.
        """
        # Get common values used by both observability and core logic
        model_name = getattr(self.config, 'model', 'unknown')
        has_tools = input.regular_functions and len(input.regular_functions) > 0
        has_background_tasks = input.background_tasks and len(input.background_tasks) > 0

        if self.observability.is_enabled():
            # WITH observability - wrap the core streaming logic
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
                        "model": model_name,
                        "temperature": getattr(self.config, 'temperature', 0.0),
                        "max_tokens": getattr(self.config, 'max_tokens', 0),
                    }
                    
                    accumulated_response = []
                    
                    # Execute core streaming logic
                    async for chunk in self._execute_stream(input):
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
                yield {
                    "llm_response": f"An error occurred: {str(e)}", 
                    "usage": {
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "thinking_tokens": 0, "tool_calling_tokens": 0,
                        "provider": self._provider_name, "model": getattr(self.config, 'model', 'unknown'),
                        "request_id": None
                    }
                }
        else:
            # WITHOUT observability - direct streaming execution
            try:
                async for chunk in self._execute_stream(input):
                    yield chunk
            except Exception as e:
                self.logger.error(f"Error in {self._provider_name} stream: {e}")
                self.logger.debug(traceback.format_exc())
                yield {
                    "llm_response": f"An error occurred: {str(e)}", 
                    "usage": {
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "thinking_tokens": 0, "tool_calling_tokens": 0,
                        "provider": self._provider_name, "model": getattr(self.config, 'model', 'unknown'),
                        "request_id": None
                    }
                }

    async def _execute_stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute core streaming logic without observability concerns.
        
        This method contains the actual streaming LLM execution logic and works
        regardless of observability state.
        
        Args:
            input: The LLM input containing system prompt, user message, tools, and options

        Yields:
            Dict containing 'llm_response' and optional 'usage' keys
        """
        try:
            self.logger.info(f"Processing stream request - Regular Functions: {bool(input.regular_functions)}, "
                           f"Background: {bool(input.background_tasks)}, "
                           f"Structured: {bool(input.structure_type)}")

            # Use the helper method to determine execution path
            needs_function_calling = self._needs_function_calling(input)

            # Execute appropriate streaming method
            if not needs_function_calling:
                stream_generator = self._stream_simple(input)
            else:
                stream_generator = self._stream_with_functions(input)
            
            # Yield stream chunks
            async for chunk in stream_generator:
                yield chunk

        except Exception as e:
            self.logger.error(f"Error in {self._provider_name} stream: {e}")
            self.logger.debug(traceback.format_exc())
            yield {
                "llm_response": f"An error occurred: {str(e)}", 
                "usage": self._standardize_usage_metadata(None)
            }



    # ========================================================================
    # PUBLIC INTERFACE - DEPRECATED METHODS (Backward Compatibility)
    # ========================================================================

    async def chat_with_tools(self, input: ILLMInput) -> Union[Dict[str, Any], str]:
        """
        DEPRECATED: Use chat() instead.
        
        Process a chat message with tools. Maintained for backward compatibility.
        
        Args:
            input: The LLM input
            
        Returns:
            Chat response (same format as chat() method)
        """
        warnings.warn(
            "chat_with_tools() is deprecated and will be removed in 2026. Use chat() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        result = await self.chat(input)
        
        # For backward compatibility, some old code might expect string responses on error
        if isinstance(result.get("llm_response"), str) and "error occurred" in result.get("llm_response", "").lower():
            return result["llm_response"]
            
        return result

    async def stream_with_tools(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        DEPRECATED: Use stream() instead.
        
        Process a streaming chat message with tools. Maintained for backward compatibility.
        
        Args:
            input: The LLM input
            
        Yields:
            Stream response (same format as stream() method)
        """
        warnings.warn(
            "stream_with_tools() is deprecated and will be removed in 2026. Use stream() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        async for chunk in self.stream(input):
            yield chunk

    async def chat_completion(self, input: ILLMInput) -> Union[Dict[str, Any], str]:
        """
        DEPRECATED: Use chat() instead.
        
        Chat completion method. Maintained for backward compatibility.
        
        Args:
            input: The LLM input
            
        Returns:
            Chat response (same format as chat() method)
        """
        warnings.warn(
            "chat_completion() is deprecated and will be removed in 2026. Use chat() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        result = await self.chat(input)
        
        # For backward compatibility, some old code might expect string responses on error
        if isinstance(result.get("llm_response"), str) and "error occurred" in result.get("llm_response", "").lower():
            return result["llm_response"]
            
        return result

    async def stream_completion(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        DEPRECATED: Use stream() instead.
        
        Streaming completion method. Maintained for backward compatibility.
        
        Args:
            input: The LLM input
            
        Yields:
            Stream response (same format as stream() method)
        """
        warnings.warn(
            "stream_completion() is deprecated and will be removed in 2026. Use stream() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        async for chunk in self.stream(input):
            yield chunk

    # ========================================================================
    # UTILITY METHODS - Available to all providers
    # ========================================================================

    def _log_provider_info(self, message: str):
        """Log provider-specific information."""
        self.logger.info(f"[{self._get_provider_name()}] {message}")

    def _log_provider_debug(self, message: str):
        """Log provider-specific debug information."""
        self.logger.debug(f"[{self._get_provider_name()}] {message}")

    def _log_provider_error(self, message: str):
        """Log provider-specific error information."""
        self.logger.error(f"[{self._get_provider_name()}] {message}")

    def get_active_background_tasks_count(self) -> int:
        """
        Get the number of currently active background tasks.
        
        Returns:
            Number of active background tasks
        """
        return self._function_orchestrator.get_active_background_tasks_count()

    async def wait_for_background_tasks(self, timeout: float = None) -> None:
        """
        Wait for all background tasks to complete (useful for testing).
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        await self._function_orchestrator.wait_for_background_tasks(timeout)

    # ========================================================================
    # PROGRESSIVE STREAMING METHODS (NEW) - Available to all providers  
    # ========================================================================
    
    def _is_function_complete(self, function_data: Dict[str, Any]) -> bool:
        """
        Check if function is complete and ready for execution during streaming.
        
        Uses JSON validation to determine if function arguments are complete.
        This approach is universal across all providers.
        
        Args:
            function_data: Function data with name and arguments
            
        Returns:
            True if function is complete and ready for execution
        """
        # Basic requirements check (minimum safety)
        if not (function_data.get("name") and "arguments" in function_data):
            return False
        
        try:
            import json
            if isinstance(function_data["arguments"], str):
                 json.loads(function_data["arguments"])  # Valid JSON = complete
                 return True
            elif isinstance(function_data["arguments"], dict):
                  return True  # Already parsed = complete
        except json.JSONDecodeError:
              return False  # Still streaming arguments
        return False
    
    async def _execute_function_progressively(
        self,
        function_call: FunctionCall,
        input: ILLMInput
    ) -> asyncio.Task:
        """
        Execute a single function progressively and return the task.
        
        This method enables real-time function execution during streaming,
        providing better user experience and resource utilization.
        
        Args:
            function_call: The function call to execute
            input: The LLM input containing available functions
            
        Returns:
            asyncio.Task that can be awaited later for the result
        """
        return await self._function_orchestrator.execute_function_progressively(
            function_call,
            input.regular_functions or {},
            input.background_tasks or {}
        )
    
    async def _gather_progressive_results(self, function_tasks: List[asyncio.Task]) -> Dict[str, Any]:
        """
        Gather results from progressively executed functions.
        
        Args:
            function_tasks: List of tasks from progressive execution
            
        Returns:
            Dict containing consolidated results in standardized format
        """
        try:
            result = await self._function_orchestrator.gather_progressive_results(function_tasks)
            
            # Convert to dict format for provider compatibility
            return {
                "regular_results": result.regular_results,
                "background_initiated": result.background_initiated,
                "failed_functions": result.failed_functions
            }
        except Exception as e:
            self.logger.error(f"Progressive function gathering failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "regular_results": [],
                "background_initiated": [],
                "failed_functions": [{
                    "name": "gather_error",
                    "args": {},
                    "error": str(e),
                    "call_id": None
                }]
            }
    
    def _add_failed_functions_to_context(self, failed_functions: List[Dict[str, Any]], contents: List[str]):
        """
        Add failed function context messages for the model with safeguards.
        
        This helps the model understand what went wrong and provide
        appropriate fallback responses or error handling.
        
        Includes safeguards for:
        - Duplicate message prevention
        - Context length limits
        - Message truncation for large arguments
        
        Args:
            failed_functions: List of failed function results
            contents: Contents list to add context messages to
        """
        if not failed_functions:
            return
        
        # Track added messages to prevent duplicates
        added_messages = set()
        max_context_messages = 10  # Limit context bloat
        max_arg_length = 200  # Truncate long arguments
        
        messages_added = 0
        for failed in failed_functions:
            if messages_added >= max_context_messages:
                self.logger.info(f"Reached maximum context messages limit ({max_context_messages}), skipping remaining failures")
                break
            
            # Create unique key for deduplication
            failure_key = f"{failed['name']}_{failed.get('call_id', 'no_id')}"
            if failure_key in added_messages:
                continue  # Skip duplicate
            
            # Truncate arguments if too long
            args_str = str(failed['args'])
            if len(args_str) > max_arg_length:
                args_str = args_str[:max_arg_length] + "... (truncated)"
            
            # Truncate error message if too long
            error_str = str(failed['error'])
            if len(error_str) > max_arg_length:
                error_str = error_str[:max_arg_length] + "... (truncated)"
            
            context_msg = (
                f"Function '{failed['name']}' called with arguments {args_str} "
                f"failed with error: {error_str}. Please handle this gracefully "
                f"and provide an appropriate response or fallback."
            )
            
            contents.append(context_msg)
            added_messages.add(failure_key)
            messages_added += 1
            
            self.logger.warning(f"Added failure context for {failed['name']}: {error_str[:100]}{'...' if len(str(failed['error'])) > 100 else ''}")
        
        if messages_added > 0:
            self.logger.info(f"Added {messages_added} failure context messages to conversation")
    # ========================================================================
    # UTILITY METHODS - Framework standardization helpers
    # ========================================================================

    @abstractmethod
    def _extract_and_standardize_usage(self, response: Any) -> Dict[str, Any]:
        """
        Extract and standardize usage metadata from provider response.
        
        Each provider implements this to handle their specific response format
        and return standardized usage metadata.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            Standardized usage metadata dictionary with fields:
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens  
            - total_tokens: Total tokens used
            - thinking_tokens: Reasoning/thinking tokens (if available)
            - tool_calling_tokens: Function calling tokens (if available)
            - provider: Provider name
            - model: Model name used
            - request_id: Request ID from provider (if available)
        """
        pass
    
    def _accumulate_usage_safely(self, current_usage: Dict[str, Any], accumulated_usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Safely accumulate usage metadata without in-place mutations.
        
        Args:
            current_usage: Current usage metadata
            accumulated_usage: Previously accumulated usage (optional)
        
        Returns:
            New accumulated usage dictionary
        """
        if accumulated_usage is None:
            return current_usage.copy() if current_usage else {}
        
        if current_usage is None:
            return accumulated_usage.copy() if accumulated_usage else {}
        
        # Safely accumulate each field
        return {
            "input_tokens": accumulated_usage.get("input_tokens", 0) + current_usage.get("input_tokens", 0),
            "output_tokens": accumulated_usage.get("output_tokens", 0) + current_usage.get("output_tokens", 0),
            "total_tokens": accumulated_usage.get("total_tokens", 0) + current_usage.get("total_tokens", 0),
            "thinking_tokens": accumulated_usage.get("thinking_tokens", 0) + current_usage.get("thinking_tokens", 0),
            "tool_calling_tokens": accumulated_usage.get("tool_calling_tokens", 0) + current_usage.get("tool_calling_tokens", 0),
            "provider": current_usage.get("provider", accumulated_usage.get("provider", "unknown")),
            "model": current_usage.get("model", accumulated_usage.get("model", "unknown")),
            "request_id": current_usage.get("request_id", accumulated_usage.get("request_id"))
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
