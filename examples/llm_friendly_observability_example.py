"""
Complete example of LLM-friendly observability integration.

This example demonstrates:
1. How parent applications should set up OTEL
2. How Arshai automatically detects and uses the setup
3. How to configure package-specific observability settings
4. How to use the observability system with LLM clients
5. How the system gracefully handles missing OTEL dependencies
"""

import asyncio
import os
from typing import Dict, Any, Optional

# Parent application sets up OTEL (this is YOUR responsibility)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.sdk.resources import Resource
    
    # This is what YOUR APPLICATION does (not Arshai)
    def setup_parent_otel():
        """Set up OTEL for the parent application."""
        print("🔧 Parent application setting up OTEL...")
        
        # Create resource with your app's information
        resource = Resource.create({
            "service.name": "my-ai-application",
            "service.version": "1.0.0",
            "environment": "production"
        })
        
        # Set up tracing
        tracer_provider = TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)
        
        # Set up metrics
        metric_reader = PeriodicExportingMetricReader(
            exporter=ConsoleMetricExporter(),
            export_interval_millis=5000
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        
        print("✅ Parent OTEL setup complete")
    
    OTEL_AVAILABLE = True
    
except ImportError:
    print("📝 OTEL not available - Arshai will use no-op implementations")
    OTEL_AVAILABLE = False
    
    def setup_parent_otel():
        """No-op setup when OTEL is not available."""
        pass


# Arshai imports - work with or without OTEL
from arshai.observability import (
    get_llm_observability, 
    PackageObservabilityConfig, 
    ObservabilityLevel,
    TimingData
)
from arshai.observability.utils import (
    observe_llm_method,
    observe_agent_operation, 
    ObservabilityMixin
)


class ExampleLLMClient(ObservabilityMixin):
    """Example LLM client showing proper observability integration."""
    
    def __init__(self, model: str, observability_config: Optional[PackageObservabilityConfig] = None):
        super().__init__()
        self.model = model
        
        # Set up Arshai observability (automatically detects parent OTEL)
        self._setup_observability("example_provider", observability_config)
        
        print(f"🤖 LLM Client initialized - model: {model}")
        if self._is_observability_enabled():
            print("📊 Observability enabled for example_provider")
        else:
            print("📊 Observability disabled")
    
    async def chat(self, messages: str) -> Dict[str, Any]:
        """Chat method with proper observability integration."""
        
        # Use the mixin's observability context manager
        async with self._observe_llm_call("chat", self.model) as timing_data:
            # Simulate LLM call
            print(f"💬 Making LLM call to {self.model}...")
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Simulate first token
            timing_data.record_first_token()
            await asyncio.sleep(0.05)  # Simulate processing
            
            # Simulate more tokens
            timing_data.record_token()
            timing_data.record_token()
            
            # Create mock response
            response = f"Response from {self.model} to: {messages}"
            
            # Record usage data
            usage_data = {
                'input_tokens': 25,
                'output_tokens': 15,
                'total_tokens': 40,
                'thinking_tokens': 5
            }
            
            await self._record_usage(timing_data, usage_data)
            
            return {
                'response': response,
                'usage': usage_data
            }

    @observe_llm_method("example_provider", "gpt-4")
    async def chat_with_decorator(self, messages: str) -> str:
        """Alternative approach using decorator."""
        print(f"🎯 Decorated LLM call to {self.model}...")
        await asyncio.sleep(0.1)
        return f"Decorated response from {self.model}"


class ExampleAgent:
    """Example agent showing observability integration."""
    
    def __init__(self, llm_client: ExampleLLMClient):
        self.llm_client = llm_client
        self.observability = get_llm_observability()
    
    @observe_agent_operation("process_user_request")
    async def process_user_request(self, user_input: str) -> str:
        """Agent method with observability."""
        print(f"🧠 Agent processing: {user_input}")
        
        # This LLM call will be traced as a child of the agent operation
        response = await self.llm_client.chat(user_input)
        
        # Additional agent processing
        processed_response = f"Agent processed: {response['response']}"
        
        return processed_response


def configure_arshai_observability() -> PackageObservabilityConfig:
    """Configure Arshai-specific observability settings."""
    
    # Option 1: From environment variables
    config = PackageObservabilityConfig.from_environment()
    
    # Option 2: Programmatic configuration
    if not os.getenv("ARSHAI_TELEMETRY_ENABLED"):
        config = PackageObservabilityConfig(
            enabled=True,
            level=ObservabilityLevel.INFO,
            trace_llm_calls=True,
            trace_agent_operations=True,
            collect_metrics=True,
            track_token_timing=True,
            log_prompts=False,  # Privacy setting
            log_responses=False,  # Privacy setting
            track_cost_metrics=True
        )
    
    # Option 3: Provider-specific configuration
    config = config.configure_provider(
        provider="example_provider",
        enabled=True,
        track_token_timing=True,
        log_prompts=False
    )
    
    return config


async def demonstrate_observability():
    """Demonstrate the LLM-friendly observability system."""
    
    print("🚀 Starting LLM-Friendly Observability Demo")
    print("=" * 50)
    
    # 1. Parent application sets up OTEL (if available)
    setup_parent_otel()
    print()
    
    # 2. Configure Arshai observability
    print("⚙️ Configuring Arshai observability...")
    arshai_config = configure_arshai_observability()
    print(f"📋 Configuration: enabled={arshai_config.enabled}, level={arshai_config.level}")
    print()
    
    # 3. Create LLM client with observability
    llm_client = ExampleLLMClient("gpt-4", arshai_config)
    print()
    
    # 4. Create agent
    agent = ExampleAgent(llm_client)
    print()
    
    # 5. Demonstrate various observability features
    print("🔍 Demonstrating observability features:")
    print("-" * 40)
    
    # Basic LLM call with observability
    print("1. Basic LLM call with observability:")
    result = await llm_client.chat("What is the capital of France?")
    print(f"   Result: {result['response'][:50]}...")
    print()
    
    # Decorated LLM call
    print("2. Decorated LLM call:")
    decorated_result = await llm_client.chat_with_decorator("Tell me a joke")
    print(f"   Result: {decorated_result[:50]}...")
    print()
    
    # Agent operation with nested LLM call
    print("3. Agent operation with nested LLM call:")
    agent_result = await agent.process_user_request("Explain quantum computing")
    print(f"   Result: {agent_result[:50]}...")
    print()
    
    # 6. Show configuration flexibility
    print("4. Configuration flexibility:")
    
    # Disable observability for testing
    disabled_config = arshai_config.disable_all()
    disabled_client = ExampleLLMClient("gpt-3.5-turbo", disabled_config)
    await disabled_client.chat("This won't be observed")
    print("   ✅ Successfully ran with observability disabled")
    print()
    
    print("✨ Demo complete! Check the console output for traces and metrics.")


def demonstrate_environment_config():
    """Show how to use environment variables for configuration."""
    
    print("🌍 Environment Variable Configuration:")
    print("-" * 40)
    
    # Set some example environment variables
    example_env_vars = {
        "ARSHAI_TELEMETRY_ENABLED": "true",
        "ARSHAI_TELEMETRY_LEVEL": "INFO", 
        "ARSHAI_TRACE_LLM_CALLS": "true",
        "ARSHAI_COLLECT_METRICS": "true",
        "ARSHAI_TRACK_TOKEN_TIMING": "true",
        "ARSHAI_LOG_PROMPTS": "false",
        "ARSHAI_LOG_RESPONSES": "false",
        "ARSHAI_ATTR_deployment": "staging",
        "ARSHAI_ATTR_team": "ai-platform"
    }
    
    print("Example environment variables:")
    for key, value in example_env_vars.items():
        print(f"  {key}={value}")
    
    # Temporarily set environment variables
    for key, value in example_env_vars.items():
        os.environ[key] = value
    
    # Load configuration from environment
    config = PackageObservabilityConfig.from_environment()
    
    print(f"\nLoaded configuration:")
    print(f"  Enabled: {config.enabled}")
    print(f"  Level: {config.level}")
    print(f"  Trace LLM calls: {config.trace_llm_calls}")
    print(f"  Custom attributes: {config.custom_attributes}")
    
    # Cleanup
    for key in example_env_vars:
        os.environ.pop(key, None)


async def main():
    """Main demo function."""
    
    print("🎯 LLM-Friendly Observability System Demo")
    print("=" * 60)
    print()
    
    # Show environment configuration
    demonstrate_environment_config()
    print()
    
    # Run main observability demo
    await demonstrate_observability()
    
    print("\n" + "=" * 60)
    print("🎉 All demos completed successfully!")
    print()
    print("Key Benefits Demonstrated:")
    print("✅ No OTEL provider creation by Arshai")
    print("✅ Automatic detection of parent OTEL setup")
    print("✅ Graceful fallback when OTEL is unavailable") 
    print("✅ Package-specific configuration control")
    print("✅ LLM-optimized tracing and metrics")
    print("✅ Privacy-aware content logging")


if __name__ == "__main__":
    asyncio.run(main())