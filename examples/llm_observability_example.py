"""
Example demonstrating comprehensive LLM observability with the new constructor-based approach.

This example shows how to:
1. Configure observability for LLM providers using constructor injection
2. Use any LLM client with automatic observability integration
3. Collect and export metrics and traces
4. Monitor token-level timing and performance
"""

import asyncio
import os
import logging
from typing import Dict, Any

# Arshai imports
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.observability import ObservabilityConfig, ObservabilityManager
from arshai.llms.openai import OpenAIClient
from arshai.llms.azure import AzureClient
from arshai.llms.google_genai import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating LLM observability."""
    
    # 1. Configure observability
    observability_config = ObservabilityConfig(
        service_name="arshai-llm-example",
        service_version="1.0.0",
        environment="development",
        
        # Enable tracing and metrics
        trace_enabled=True,
        metrics_enabled=True,
        
        # Token timing configuration (KEY METRICS)
        track_token_timing=True,
        
        # Privacy settings (be careful with these in production)
        log_prompts=True,  # Set to False in production for privacy
        log_responses=True,  # Set to False in production for privacy
        max_prompt_length=500,
        max_response_length=500,
        
        # OTLP configuration (optional - remove if not using OTLP)
        # otlp_endpoint="http://localhost:4317",  # Uncomment if using Jaeger/OTLP
        # otlp_headers={"Authorization": "Bearer your-token"},
        
        # Custom attributes for all traces
        custom_attributes={
            "team": "ai-platform",
            "component": "llm-client",
        }
    )
    
    # 2. Create observability manager
    obs_manager = ObservabilityManager(observability_config)
    
    # 3. Configure LLM client
    llm_config = ILLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=150,
    )
    
    # 4. Create LLM client with observability using constructor
    client = OpenAIClient(llm_config, observability_manager=obs_manager)
    
    logger.info("LLM client with observability created successfully")
    
    # 5. Example 1: Simple chat completion with observability
    logger.info("=== Example 1: Simple Chat Completion ===")
    
    simple_input = ILLMInput(
        system_prompt="You are a helpful AI assistant.",
        user_message="What are the benefits of observability in AI systems?"
    )
    
    try:
        response = await client.chat(simple_input)
        logger.info(f"Response: {response['llm_response'][:100]}...")
        
        # Usage information is automatically tracked
        if response.get('usage'):
            usage = response['usage']
            logger.info(f"Token usage - Input: {usage.get('input_tokens')}, "
                       f"Output: {usage.get('output_tokens')}, "
                       f"Total: {usage.get('total_tokens')}")
    
    except Exception as e:
        logger.error(f"Error in simple completion: {e}")
    
    # 6. Example 2: Async streaming completion with enhanced performance
    logger.info("=== Example 2: Async Streaming Chat Completion ===")
    
    streaming_input = ILLMInput(
        system_prompt="You are a creative writing assistant.",
        user_message="Write a short story about a robot learning to paint."
    )
    
    try:
        token_count = 0
        full_response = ""
        
        # Streaming with automatic async observability for better performance
        async for chunk in client.stream(streaming_input):
            if chunk.get('llm_response'):
                content = chunk['llm_response']
                full_response += content
                token_count += 1
                # Each chunk is automatically timed with async methods for better performance
                
            # Final chunk contains usage information
            if chunk.get('usage'):
                usage = chunk['usage']
                logger.info(f"Async streaming completed - Tokens: {usage.get('total_tokens')}, "
                           f"Chunks: {token_count}")
                logger.info(f"Response: {full_response[:100]}...")
                break
    
    except Exception as e:
        logger.error(f"Error in async streaming completion: {e}")
    
    # 7. Example 3: Demonstrate error tracking
    logger.info("=== Example 3: Error Tracking ===")
    
    # This will likely fail and demonstrate error tracking
    error_config = ILLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=10000  # Excessive token limit
    )
    
    error_client = OpenAIClient(error_config, observability_manager=obs_manager)
    
    error_input = ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="Generate a response with exactly 10000 tokens"
    )
    
    try:
        response = await error_client.chat(error_input)
        logger.info("Unexpected success!")
    except Exception as e:
        logger.info(f"Expected error tracked: {type(e).__name__}")
    
    # 8. Example 4: Concurrent async requests for enhanced performance
    logger.info("=== Example 4: Concurrent Async Requests ===")
    
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing briefly.",
        "What are the benefits of renewable energy?",
        "How does natural language processing work?",
        "What is the future of artificial intelligence?"
    ]
    
    async def make_llm_request(prompt: str, request_id: int):
        """Make a single LLM request with observability."""
        test_input = ILLMInput(
            system_prompt="You are a knowledgeable assistant. Be concise.",
            user_message=prompt
        )
        
        try:
            response = await client.chat(test_input)
            logger.info(f"Async request {request_id}/5 completed")
            return response
        except Exception as e:
            logger.error(f"Async request {request_id} failed: {e}")
            return None
    
    # Execute all requests concurrently for much better performance
    import time
    start_time = time.time()
    
    tasks = [make_llm_request(prompt, i+1) for i, prompt in enumerate(test_prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    concurrent_duration = end_time - start_time
    
    successful_requests = sum(1 for result in results if result is not None and not isinstance(result, Exception))
    logger.info(f"Completed {successful_requests}/5 concurrent requests in {concurrent_duration:.2f}s")
    logger.info(f"Average time per request: {concurrent_duration/len(test_prompts):.2f}s")
    
    # 9. Display observability configuration
    logger.info("=== Observability Configuration ===")
    logger.info(f"Service: {observability_config.service_name}")
    logger.info(f"Environment: {observability_config.environment}")
    logger.info(f"Token timing enabled: {observability_config.track_token_timing}")
    logger.info(f"Tracing enabled: {observability_config.trace_enabled}")
    
    # 10. Example with different providers
    logger.info("=== Example 5: Multiple Providers with Same Observability ===")
    
    # All providers support the same observability pattern
    providers_to_test = []
    
    # OpenAI (already created above)
    providers_to_test.append(("OpenAI", client))
    
    # Azure (if configured)
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        azure_config = ILLMConfig(model="gpt-4", temperature=0.7, max_tokens=150)
        azure_client = AzureClient(
            azure_config, 
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT"),
            api_version="2024-02-01",
            observability_manager=obs_manager
        )
        providers_to_test.append(("Azure", azure_client))
    
    # Google Gemini (if configured)
    if os.environ.get("GOOGLE_API_KEY"):
        gemini_config = ILLMConfig(model="gemini-2.0-flash-exp", temperature=0.7, max_tokens=150)
        gemini_client = GeminiClient(gemini_config, observability_manager=obs_manager)
        providers_to_test.append(("Gemini", gemini_client))
    
    test_input = ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="What is 2+2?"
    )
    
    for provider_name, provider_client in providers_to_test:
        try:
            logger.info(f"Testing {provider_name}...")
            response = await provider_client.chat(test_input)
            logger.info(f"{provider_name} response: {response['llm_response']}")
            logger.info(f"{provider_name} usage: {response.get('usage', {})}")
        except Exception as e:
            logger.error(f"{provider_name} error: {e}")
    
    # 11. Graceful shutdown
    logger.info("=== Shutting Down ===")
    obs_manager.shutdown()
    logger.info("Example completed successfully!")


def run_production_example():
    """Example of production-ready observability setup."""
    
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    
    # Configure OpenTelemetry for production
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    
    # Add OTLP exporter for production monitoring
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"),
        insecure=True  # Set to False in production with proper TLS
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Production observability configuration
    config = ObservabilityConfig(
        service_name=os.environ.get("SERVICE_NAME", "production-ai-service"),
        service_version=os.environ.get("SERVICE_VERSION", "1.0.0"),
        environment=os.environ.get("ENVIRONMENT", "production"),
        
        # Production settings
        trace_enabled=True,
        metrics_enabled=True,
        track_token_timing=True,
        
        # Privacy-conscious production settings
        log_prompts=False,  # Never log prompts in production
        log_responses=False,  # Never log responses in production
        
        # Custom attributes for production monitoring
        custom_attributes={
            "region": os.environ.get("AWS_REGION", "us-east-1"),
            "instance_id": os.environ.get("INSTANCE_ID", "unknown"),
            "deployment": os.environ.get("DEPLOYMENT_ID", "unknown"),
        }
    )
    
    obs_manager = ObservabilityManager(config)
    logger.info("Production observability configured")
    
    # Create LLM client with production observability
    llm_config = ILLMConfig(
        model=os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
        temperature=0.7
    )
    
    client = OpenAIClient(llm_config, observability_manager=obs_manager)
    logger.info("Production LLM client ready")
    
    # The rest would be the same as the main example...
    return client, obs_manager


if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Run the main example
    logger.info("Starting LLM observability example...")
    asyncio.run(main())
    
    # Also demonstrate production configuration
    logger.info("\n" + "="*50)
    logger.info("Production configuration example...")
    client, obs_manager = run_production_example()
    logger.info("Production example completed")
    obs_manager.shutdown()