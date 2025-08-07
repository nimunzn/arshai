"""
Example demonstrating comprehensive LLM observability with OpenTelemetry.

This example shows how to:
1. Configure observability for LLM providers using the new factory-based approach
2. Use LLMFactory with automatic observability integration
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
from src.factories.llm_factory import LLMFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating LLM observability."""
    
    # 1. Configure observability (always enabled)
    observability_config = ObservabilityConfig(
        service_name="arshai-llm-example",
        service_version="1.0.0",
        environment="development",
        
        # Enable tracing and metrics
        trace_requests=True,
        collect_metrics=True,
        
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
    
    # 2. Configure LLM client
    llm_config = ILLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=150,
    )
    
    # 3. Create LLM client with automatic observability using factory
    client = LLMFactory.create_with_observability(
        provider="openai",
        config=llm_config,
        observability_config=observability_config
    )
    
    logger.info("LLM client with observability created successfully")
    
    # 5. Example 1: Simple chat completion with observability
    logger.info("=== Example 1: Simple Chat Completion ===")
    
    simple_input = ILLMInput(
        system_prompt="You are a helpful AI assistant.",
        user_message="What are the benefits of observability in AI systems?"
    )
    
    try:
        response = client.chat_completion(simple_input)
        logger.info(f"Response: {response['llm_response'][:100]}...")
        
        # Usage information is automatically tracked
        if response.get('usage'):
            usage = response['usage']
            logger.info(f"Token usage - Prompt: {usage.prompt_tokens}, "
                       f"Completion: {usage.completion_tokens}, "
                       f"Total: {usage.total_tokens}")
    
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
        async for chunk in client.stream_completion(streaming_input):
            if chunk.get('llm_response'):
                content = chunk['llm_response']
                full_response += content
                token_count += 1
                # Each chunk is automatically timed with async methods for better performance
                
            # Final chunk contains usage information
            if chunk.get('usage'):
                usage = chunk['usage']
                logger.info(f"Async streaming completed - Tokens: {usage.total_tokens}, "
                           f"Chunks: {token_count}")
                logger.info(f"Response: {full_response[:100]}...")
                break
    
    except Exception as e:
        logger.error(f"Error in async streaming completion: {e}")
    
    # 7. Example 3: Demonstrate error tracking
    logger.info("=== Example 3: Error Tracking ===")
    
    # This will likely fail and demonstrate error tracking
    error_input = ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="Generate a response with exactly 10000 tokens"  # Likely to exceed limits
    )
    
    try:
        response = client.chat_completion(error_input)
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
            response = client.chat_completion(test_input)
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
    logger.info(f"Tracing enabled: {observability_config.trace_requests}")
    
    # 10. Example of manual async observability manager usage
    logger.info("=== Manual Async Observability Manager Usage ===")
    obs_manager = ObservabilityManager(observability_config)
    
    # Example: Create a manual async timing context for better performance
    async with obs_manager.observe_streaming_llm_call("openai", "gpt-3.5-turbo", "async_manual_test") as timing:
        # Simulate async work (much faster than synchronous)
        await asyncio.sleep(0.1)
        timing.record_first_token()
        await asyncio.sleep(0.2)
        timing.record_token()
        
        # Simulate usage data from LLM response
        usage_data = {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40
        }
        await obs_manager.record_usage_data(timing, usage_data)
        
        logger.info(f"Async manual timing - Time to first token: {timing.time_to_first_token:.3f}s")
        logger.info(f"Total tokens: {timing.total_tokens}")
        logger.info("✨ Async observability provides better performance for concurrent operations")
    
    # 11. Graceful shutdown
    logger.info("=== Shutting Down ===")
    obs_manager.shutdown()
    logger.info("Example completed successfully!")


def run_environment_config_example():
    """Example of setting up observability using environment variables."""
    
    # Set environment variables (in practice, these would be set externally)
    os.environ.update({
        "ARSHAI_SERVICE_NAME": "arshai-llm-env-example",
        "ARSHAI_ENVIRONMENT": "staging",
        "ARSHAI_LOG_PROMPTS": "false",  # Privacy-conscious default
        "ARSHAI_LOG_RESPONSES": "false",
        "ARSHAI_TRACK_TOKEN_TIMING": "true",
        # "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",  # Uncomment for OTLP
    })
    
    # Setup from environment
    config = ObservabilityConfig.from_environment()
    logger.info("Environment-based observability config loaded")
    
    # Create LLM with observability
    llm_config = ILLMConfig(model="gpt-3.5-turbo", temperature=0.7)
    client = LLMFactory.create_with_observability(
        provider="openai",
        config=llm_config,
        observability_config=config
    )
    
    # The rest would be the same as the main example...
    logger.info("Environment configuration example completed")


if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Run the main example
    logger.info("Starting LLM observability example...")
    asyncio.run(main())
    
    # Also demonstrate environment configuration
    logger.info("\n" + "="*50)
    logger.info("Environment configuration example...")
    run_environment_config_example()