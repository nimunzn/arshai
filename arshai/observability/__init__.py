"""LLM-friendly observability module for Arshai framework.

This module provides OTEL-compliant observability that:
1. Never creates TracerProvider/MeterProvider instances
2. Uses get_tracer("arshai", version) pattern
3. Respects parent application's OTEL configuration
4. Works with and without OTEL dependencies
5. Provides package-specific configuration controls

Main Classes:
- LLMObservability: Main observability interface for LLM operations
- TelemetryManager: Low-level telemetry abstraction
- PackageObservabilityConfig: Package-specific configuration
- TimingData: Container for timing measurements

Example usage:
    from arshai.observability import get_llm_observability
    
    observability = get_llm_observability()
    
    async with observability.observe_llm_call(
        provider="openai",
        model="gpt-4",
        method_name="chat"
    ) as timing_data:
        # Your LLM call here
        response = await llm_client.chat(...)
        
        # Record token usage
        await observability.record_usage_data(timing_data, {
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens,
            'total_tokens': response.usage.total_tokens
        })
"""

# Comprehensive observability system supporting all span types
from .llm_observability import (
    ArshaiObservability, LLMObservability,  # LLMObservability is backwards compatibility alias
    get_llm_observability, get_observability, 
    reset_llm_observability, reset_observability
)
from .telemetry_manager import TelemetryManager, get_telemetry_manager, reset_telemetry_manager
from .package_config import PackageObservabilityConfig, ObservabilityLevel, SpanKind
from .metrics import TimingData
from .utils import (
    observe_llm_method,
    observe_agent_operation,
    observe_workflow_step,
    configure_observability_from_env,
    disable_observability,
    create_provider_config,
    ObservabilityMixin
)

# Main exports
__all__ = [
    # Core observability system
    "ArshaiObservability", 
    "LLMObservability",  # Backwards compatibility
    "get_observability",
    "get_llm_observability", 
    "reset_observability",
    "reset_llm_observability",
    "TelemetryManager",
    "get_telemetry_manager",
    "reset_telemetry_manager",
    
    # Configuration
    "PackageObservabilityConfig",
    "ObservabilityLevel", 
    "SpanKind",
    "configure_observability_from_env",
    "disable_observability",
    "create_provider_config",
    
    # Data containers
    "TimingData",
    
    # Utilities and decorators
    "observe_llm_method",
    "observe_agent_operation", 
    "observe_workflow_step",
    "ObservabilityMixin",
]