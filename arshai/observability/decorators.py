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
                      config: Optional[ObservabilityConfig] = None):
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
            
            # Use observability context manager
            with manager.observe_llm_call(provider, model, method_name) as timing_data:
                try:
                    # Call the original method
                    result = func(self, llm_input, *args, **kwargs)
                    
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
                    
                    # Record completion timing
                    timing_data.record_token()
                    
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
            
            # Use async observability context manager
            async with manager.observe_streaming_llm_call(provider, model, method_name) as timing_data:
                first_token_recorded = False
                final_usage = None
                
                try:
                    async for chunk in func(self, llm_input, *args, **kwargs):
                        # Record first token timing
                        if not first_token_recorded:
                            timing_data.record_first_token()
                            first_token_recorded = True
                        
                        # Record each token
                        timing_data.record_token()
                        
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
                                final_usage = usage
                        
                        yield chunk
                    
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
                         observability_manager: Optional[ObservabilityManager] = None):
    """Simple decorator for making LLM methods observable.
    
    Args:
        provider: LLM provider name
        observability_manager: Optional observability manager instance
    """
    return with_observability(provider, observability_manager)


def create_observable_wrapper(original_method: Callable, 
                            provider: str,
                            observability_manager: Optional[ObservabilityManager] = None) -> Callable:
    """Create an observable wrapper for an existing LLM method.
    
    Args:
        original_method: The original method to wrap
        provider: LLM provider name
        observability_manager: Optional observability manager
        
    Returns:
        Wrapped method with observability
    """
    return with_observability(provider, observability_manager)(original_method)


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