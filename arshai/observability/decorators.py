"""
DEPRECATED: Non-intrusive decorators for LLM observability.

⚠️  DEPRECATION WARNING ⚠️

This decorator-based approach is DEPRECATED and will be removed in a future version.

The new constructor-based approach is much simpler and cleaner:

    from arshai.llms.openai import OpenAIClient
    from arshai.observability import ObservabilityManager, ObservabilityConfig
    
    # Create observability manager
    obs_config = ObservabilityConfig(service_name="my-app")
    obs_manager = ObservabilityManager(obs_config)
    
    # Use client constructor directly - no decorators needed!
    client = OpenAIClient(config, observability_manager=obs_manager)

This is cleaner, more direct, and eliminates the complexity of decorators.

MIGRATION PATH:
- Replace @with_observability decorators with constructor parameters
- Replace ObservabilityMixin inheritance with constructor parameters
- Use client constructors directly instead of decorated classes

This module will be removed in the next major version.
"""

import functools
import asyncio
import logging
from typing import Any, Callable, Dict, Union, AsyncGenerator, Optional

from arshai.core.interfaces.illm import ILLMInput
from .core import ObservabilityManager
from .config import ObservabilityConfig


def with_observability(provider: str, 
                      observability_manager: Optional[ObservabilityManager] = None,
                      config: Optional[ObservabilityConfig] = None,
                      system: Optional[str] = None):
    """Decorator to add observability to LLM methods without side effects.
    
    Args:
        provider: LLM provider name
        observability_manager: Optional observability manager instance
        config: Optional observability configuration
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(self, llm_input: ILLMInput, *args, **kwargs):
            # Get or create observability manager
            manager = observability_manager
            if manager is None:
                if hasattr(self, '_observability_manager'):
                    manager = self._observability_manager
                else:
                    obs_config = config or ObservabilityConfig.from_config_file_or_env()
                    manager = ObservabilityManager(obs_config)
            
            # Extract method name and model
            method_name = func.__name__
            model = getattr(self.config, 'model', 'unknown') if hasattr(self, 'config') else 'unknown'
            
            # Extract input attributes from ILLMInput
            input_attrs = {}
            if hasattr(llm_input, 'system_prompt'):
                input_attrs['system_prompt'] = llm_input.system_prompt
            if hasattr(llm_input, 'user_message'):
                input_attrs['user_message'] = llm_input.user_message
            
            # Use observability context manager with input attributes
            with manager.observe_llm_call(provider, model, method_name, system=system, **input_attrs) as timing_data:
                try:
                    # Capture input data
                    timing_data.input_messages = [
                        {"role": "system", "content": llm_input.system_prompt},
                        {"role": "user", "content": llm_input.user_message}
                    ]
                    timing_data.input_value = f"System: {llm_input.system_prompt}\nUser: {llm_input.user_message}"
                    
                    # Capture invocation parameters
                    if hasattr(self, 'config'):
                        timing_data.invocation_parameters = {
                            "model": getattr(self.config, 'model', 'unknown'),
                            "temperature": getattr(self.config, 'temperature', None),
                            "max_tokens": getattr(self.config, 'max_tokens', None),
                            "provider": provider
                        }
                    
                    # Record first token time before making the call
                    timing_data.record_first_token()
                    
                    # Call the original method
                    result = func(self, llm_input, *args, **kwargs)
                    
                    # Record completion timing (last token)
                    timing_data.record_token()
                    
                    # Capture output data
                    if isinstance(result, dict) and 'llm_response' in result:
                        timing_data.output_value = result['llm_response']
                        timing_data.output_messages = [
                            {"role": "assistant", "content": result['llm_response']}
                        ]
                    
                    # Extract usage information if available (non-intrusive)
                    if isinstance(result, dict) and 'usage' in result:
                        usage = result['usage']
                        if usage and (hasattr(usage, 'input_tokens') or hasattr(usage, 'prompt_tokens')):
                            # For sync methods, we need to run async method in event loop
                            usage_data = {
                                'input_tokens': getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0)),
                                'output_tokens': getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0)),
                                'total_tokens': getattr(usage, 'total_tokens', 0),
                                'thinking_tokens': getattr(usage, 'thinking_tokens', 0),
                                'tool_calling_tokens': getattr(usage, 'tool_calling_tokens', 0)
                            }
                            asyncio.run(manager.record_usage_data(timing_data, usage_data))
                            # For sync methods, we need to run async method in event loop
                            try:
                                # Check if we're already in an event loop
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # Schedule the coroutine to run in the existing loop
                                    asyncio.create_task(manager.record_usage_data(timing_data, usage_data))
                                else:
                                    # Create a new event loop
                                    asyncio.run(manager.record_usage_data(timing_data, usage_data))
                            except RuntimeError:
                                # If no event loop exists, create one
                                asyncio.run(manager.record_usage_data(timing_data, usage_data))
                            
                            # Also update timing_data directly
                            timing_data.prompt_tokens = usage_data['prompt_tokens']
                            timing_data.completion_tokens = usage_data['completion_tokens']
                            timing_data.total_tokens = usage_data['total_tokens']
                    
                    return result
                    
                except Exception as e:
                    logging.getLogger(__name__).error(f"LLM call failed: {e}")
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(self, llm_input: ILLMInput, *args, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
            # Get or create observability manager
            manager = observability_manager
            if manager is None:
                if hasattr(self, '_observability_manager'):
                    manager = self._observability_manager
                else:
                    obs_config = config or ObservabilityConfig.from_config_file_or_env()
                    manager = ObservabilityManager(obs_config)
            
            # Extract method name and model
            method_name = func.__name__
            model = getattr(self.config, 'model', 'unknown') if hasattr(self, 'config') else 'unknown'
            
            # Extract input attributes from ILLMInput
            input_attrs = {}
            if hasattr(llm_input, 'system_prompt'):
                input_attrs['system_prompt'] = llm_input.system_prompt
            if hasattr(llm_input, 'user_message'):
                input_attrs['user_message'] = llm_input.user_message
            
            # Use async observability context manager with input attributes
            async with manager.observe_streaming_llm_call(provider, model, method_name, system=system, **input_attrs) as timing_data:
                first_token_recorded = False
                final_usage = None
                accumulated_response = ""
                
                # Capture input data
                timing_data.input_messages = [
                    {"role": "system", "content": llm_input.system_prompt},
                    {"role": "user", "content": llm_input.user_message}
                ]
                timing_data.input_value = f"System: {llm_input.system_prompt}\nUser: {llm_input.user_message}"
                
                # Capture invocation parameters
                if hasattr(self, 'config'):
                    timing_data.invocation_parameters = {
                        "model": getattr(self.config, 'model', 'unknown'),
                        "temperature": getattr(self.config, 'temperature', None),
                        "max_tokens": getattr(self.config, 'max_tokens', None),
                        "provider": provider,
                        "streaming": True
                    }
                
                try:
                    async for chunk in func(self, llm_input, *args, **kwargs):
                        # Record first token timing
                        if not first_token_recorded:
                            timing_data.record_first_token()
                            first_token_recorded = True
                        
                        # Record each token
                        timing_data.record_token()
                        
                        # Accumulate response for output capture
                        if isinstance(chunk, dict) and 'llm_response' in chunk:
                            if isinstance(chunk['llm_response'], str):
                                accumulated_response += chunk['llm_response']
                        
                        # Check for usage data in the chunk (non-intrusive)
                        if isinstance(chunk, dict) and 'usage' in chunk and chunk['usage']:
                            usage = chunk['usage']

                            if hasattr(usage, 'input_tokens') or hasattr(usage, 'prompt_tokens'):
                                # Use async method for recording usage data
                                usage_data = {
                                    'input_tokens': getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0)),
                                    'output_tokens': getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0)),
                                    'total_tokens': getattr(usage, 'total_tokens', 0),
                                    'thinking_tokens': getattr(usage, 'thinking_tokens', 0),
                                    'tool_calling_tokens': getattr(usage, 'tool_calling_tokens', 0)
                                }
                                await manager.record_usage_data(timing_data, usage_data)
                                
                                # Also update timing_data directly
                                timing_data.prompt_tokens = usage_data['prompt_tokens']
                                timing_data.completion_tokens = usage_data['completion_tokens']
                                timing_data.total_tokens = usage_data['total_tokens']
                                final_usage = usage
                        
                        yield chunk
                    
                    # Capture final output data
                    if accumulated_response:
                        timing_data.output_value = accumulated_response
                        timing_data.output_messages = [
                            {"role": "assistant", "content": accumulated_response}
                        ]
                    
                    # Record final timing if we had tokens
                    if first_token_recorded:
                        timing_data.record_token()
                        
                except Exception as e:
                    logging.getLogger(__name__).error(f"Streaming LLM call failed: {e}")
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def observable_llm_method(provider: str, 
                         observability_manager: Optional[ObservabilityManager] = None,
                         system: Optional[str] = None):
    """Simple decorator for making LLM methods observable.
    
    Args:
        provider: LLM provider name
        observability_manager: Optional observability manager instance
        system: Optional AI system identifier
    """
    return with_observability(provider, observability_manager, system=system)


def create_observable_wrapper(original_method: Callable, 
                            provider: str,
                            observability_manager: Optional[ObservabilityManager] = None,
                            system: Optional[str] = None) -> Callable:
    """Create an observable wrapper for an existing LLM method.
    
    Args:
        original_method: The original method to wrap
        provider: LLM provider name
        observability_manager: Optional observability manager
        system: Optional AI system identifier
        
    Returns:
        Wrapped method with observability
    """
    return with_observability(provider, observability_manager, system=system)(original_method)


class ObservabilityMixin:
    """Mixin class to add observability capabilities to LLM clients."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize observability manager if config is available
        observability_config = kwargs.get('observability_config')
        if observability_config is None:
            observability_config = ObservabilityConfig.from_config_file_or_env()
        
        self._observability_manager = ObservabilityManager(observability_config)
    
    def _make_observable(self, provider: str):
        """Make all LLM methods observable.
        
        Args:
            provider: LLM provider name
        """
        # List of methods to make observable
        methods_to_observe = [
            'chat_completion', 
            'chat_with_tools', 
            'stream_completion', 
            'stream_with_tools'
        ]
        
        for method_name in methods_to_observe:
            if hasattr(self, method_name):
                original_method = getattr(self, method_name)
                observable_method = with_observability(
                    provider, 
                    self._observability_manager
                )(original_method)
                setattr(self, method_name, observable_method)
    
    def get_observability_manager(self) -> ObservabilityManager:
        """Get the observability manager instance."""
        return self._observability_manager