"""Utility functions for LLM-friendly observability.

This module provides convenience functions and decorators for
integrating observability into LLM applications.
"""

import functools
import asyncio
import logging
from typing import Callable, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

from .llm_observability import get_llm_observability
from .package_config import PackageObservabilityConfig
from .metrics import TimingData


logger = logging.getLogger(__name__)


def observe_llm_method(
    provider: str,
    model: Optional[str] = None,
    method_name: Optional[str] = None,
    **span_attributes
):
    """Decorator to observe LLM method calls.
    
    Args:
        provider: LLM provider name
        model: Model name (can be dynamic if not provided)
        method_name: Method name (defaults to function name)
        **span_attributes: Additional span attributes
    
    Example:
        @observe_llm_method("openai", "gpt-4")
        async def chat(self, messages):
            return await self.client.chat(messages)
    """
    def decorator(func: Callable):
        actual_method_name = method_name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            observability = get_llm_observability()
            
            # Try to extract model from args/kwargs if not provided
            actual_model = model
            if not actual_model:
                # Look for model in common places
                for arg_name in ['model', 'model_name', 'model_id']:
                    if arg_name in kwargs:
                        actual_model = kwargs[arg_name]
                        break
                
                # If still no model, try to get it from self (for class methods)
                if not actual_model and args and hasattr(args[0], 'model'):
                    actual_model = args[0].model
            
            actual_model = actual_model or "unknown"
            
            async with observability.observe_llm_call(
                provider=provider,
                model=actual_model,
                method_name=actual_method_name,
                **span_attributes
            ) as timing_data:
                result = await func(*args, **kwargs)
                
                # Try to extract usage data from result
                if hasattr(result, 'usage') and result.usage:
                    await observability.record_usage_data(timing_data, {
                        'input_tokens': getattr(result.usage, 'input_tokens', 0),
                        'output_tokens': getattr(result.usage, 'output_tokens', 0),
                        'total_tokens': getattr(result.usage, 'total_tokens', 0),
                    })
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, just call them directly
            # Observability is designed for async operations
            logger.warning(f"Sync function {func.__name__} decorated with observe_llm_method")
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def observe_agent_operation(
    operation_name: Optional[str] = None,
    **span_attributes
):
    """Decorator to observe agent operations.
    
    Args:
        operation_name: Operation name (defaults to function name)
        **span_attributes: Additional span attributes
    
    Example:
        @observe_agent_operation("process_message")
        async def process_message(self, input_data):
            return await self._llm_client.chat(input_data)
    """
    def decorator(func: Callable):
        actual_operation_name = operation_name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            observability = get_llm_observability()
            
            if not observability.config.is_feature_enabled("trace_agent_operations"):
                return await func(*args, **kwargs)
            
            # Create span for agent operation
            span_attributes_with_defaults = {
                "arshai.component": "agent",
                "arshai.operation": actual_operation_name,
                **span_attributes
            }
            
            async with observability.telemetry.create_async_span(
                f"agent.{actual_operation_name}",
                attributes=span_attributes_with_defaults
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    observability.telemetry.set_span_status_safely(span, True)
                    return result
                except Exception as e:
                    observability.telemetry.set_span_status_safely(span, False, str(e))
                    observability.telemetry.record_exception_safely(span, e)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@asynccontextmanager
async def observe_workflow_step(
    step_name: str,
    workflow_name: Optional[str] = None,
    **span_attributes
):
    """Context manager to observe workflow step execution.
    
    Args:
        step_name: Name of the workflow step
        workflow_name: Optional workflow name
        **span_attributes: Additional span attributes
    
    Example:
        async with observe_workflow_step("knowledge_retrieval", "support_workflow"):
            results = await knowledge_base.search(query)
    """
    observability = get_llm_observability()
    
    if not observability.config.is_feature_enabled("trace_workflow_execution"):
        yield
        return
    
    span_attributes_with_defaults = {
        "arshai.component": "workflow",
        "arshai.step": step_name,
        **span_attributes
    }
    
    if workflow_name:
        span_attributes_with_defaults["arshai.workflow"] = workflow_name
    
    span_name = f"workflow.{step_name}"
    if workflow_name:
        span_name = f"workflow.{workflow_name}.{step_name}"
    
    async with observability.telemetry.create_async_span(
        span_name,
        attributes=span_attributes_with_defaults
    ) as span:
        try:
            yield
            observability.telemetry.set_span_status_safely(span, True)
        except Exception as e:
            observability.telemetry.set_span_status_safely(span, False, str(e))
            observability.telemetry.record_exception_safely(span, e)
            raise


def configure_observability_from_env() -> PackageObservabilityConfig:
    """Configure observability from environment variables.
    
    Returns:
        PackageObservabilityConfig instance
    
    Example:
        # Set environment variables:
        # ARSHAI_TELEMETRY_ENABLED=true
        # ARSHAI_TELEMETRY_LEVEL=INFO
        # ARSHAI_TRACE_LLM_CALLS=true
        
        config = configure_observability_from_env()
    """
    return PackageObservabilityConfig.from_environment()


def disable_observability() -> PackageObservabilityConfig:
    """Create a configuration with all observability disabled.
    
    Returns:
        PackageObservabilityConfig with everything disabled
    
    Example:
        config = disable_observability()
        observability = get_llm_observability(config)
    """
    return PackageObservabilityConfig().disable_all()


def create_provider_config(
    provider: str,
    enabled: bool = True,
    track_token_timing: bool = True,
    log_prompts: bool = False,
    log_responses: bool = False
) -> Dict[str, Any]:
    """Create provider-specific observability configuration.
    
    Args:
        provider: Provider name
        enabled: Enable observability for this provider
        track_token_timing: Track token-level timing
        log_prompts: Log prompts (privacy sensitive)
        log_responses: Log responses (privacy sensitive)
    
    Returns:
        Provider configuration dictionary
    
    Example:
        config = PackageObservabilityConfig()
        openai_config = create_provider_config("openai", log_prompts=True)
        config = config.configure_provider("openai", **openai_config)
    """
    return {
        "enabled": enabled,
        "track_token_timing": track_token_timing,
        "log_prompts": log_prompts,
        "log_responses": log_responses,
    }


class ObservabilityMixin:
    """Mixin class to add observability to LLM clients.
    
    This mixin provides a standardized way to add observability
    to LLM client classes without modifying their core logic.
    
    Example:
        class MyLLMClient(ObservabilityMixin):
            def __init__(self, ...):
                super().__init__()
                self._setup_observability("my_provider")
            
            async def chat(self, messages):
                async with self._observe_llm_call("chat", "gpt-4") as timing_data:
                    response = await self._actual_chat(messages)
                    await self._record_usage(timing_data, response.usage)
                    return response
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observability = None
        self._provider_name = None
    
    def _setup_observability(
        self,
        provider_name: str,
        config: Optional[PackageObservabilityConfig] = None
    ):
        """Setup observability for this client.
        
        Args:
            provider_name: Name of the LLM provider
            config: Optional observability configuration
        """
        self._provider_name = provider_name
        self._observability = get_llm_observability(config)
    
    @asynccontextmanager
    async def _observe_llm_call(
        self,
        method_name: str,
        model: str,
        **extra_attributes
    ):
        """Observe an LLM call.
        
        Args:
            method_name: Name of the method being called
            model: Model name
            **extra_attributes: Additional span attributes
        """
        if not self._observability or not self._provider_name:
            yield TimingData()
            return
        
        async with self._observability.observe_llm_call(
            provider=self._provider_name,
            model=model,
            method_name=method_name,
            **extra_attributes
        ) as timing_data:
            yield timing_data
    
    async def _record_usage(self, timing_data: TimingData, usage_data: Dict[str, Any]):
        """Record usage data.
        
        Args:
            timing_data: Timing data from observe_llm_call
            usage_data: Usage data from LLM response
        """
        if self._observability:
            await self._observability.record_usage_data(timing_data, usage_data)
    
    def _is_observability_enabled(self) -> bool:
        """Check if observability is enabled for this client."""
        if not self._observability or not self._provider_name:
            return False
        return self._observability.is_provider_enabled(self._provider_name)