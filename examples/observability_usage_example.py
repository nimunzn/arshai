"""
Example demonstrating how to use the restructured observability system with LLM factory.

This example shows:
1. Loading configuration from YAML file
2. Using the factory with automatic observability
3. Making LLM calls with comprehensive metrics collection
4. Token counting and timing measurements
"""

import asyncio
import logging
import os
from pathlib import Path

# Arshai imports
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.observability import ObservabilityConfig, ObservabilityManager
from src.factories.llm_factory import LLMFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example demonstrating observability integration."""
    
    # 1. Load observability configuration from YAML
    config_path = Path(__file__).parent / "observability_config.yaml"
    
    try:
        observability_config = ObservabilityConfig.from_yaml(config_path)
        logger.info(f"Loaded observability config from {config_path}")
    except FileNotFoundError:
        logger.warning("Config file not found, using environment variables")
        observability_config = ObservabilityConfig.from_environment()
    
    # 2. Configure LLM
    llm_config = ILLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=150,
    )
    
    # 3. Method 1: Create LLM with automatic observability using factory
    logger.info("=== Method 1: Factory with Automatic Observability ===")
    
    try:
        llm_client = LLMFactory.create_with_observability(
            provider="openai",
            config=llm_config,
            observability_config=observability_config
        )
        
        # Simple chat completion - automatically instrumented
        simple_input = ILLMInput(
            system_prompt="You are a helpful AI assistant.",
            user_message="Explain observability in AI systems in 2 sentences."
        )
        
        response = llm_client.chat_completion(simple_input)
        logger.info(f"Response: {response['llm_response']}")
        
        # Check usage information
        if response.get('usage'):
            usage = response['usage']
            logger.info(f"Token usage - Prompt: {usage.prompt_tokens}, "
                       f"Completion: {usage.completion_tokens}, "
                       f"Total: {usage.total_tokens}")
        
    except ImportError as e:
        logger.error(f"Observability not available: {e}")
        logger.info("Using regular factory instead...")
        
        # Fallback to regular factory
        llm_client = LLMFactory.create(
            provider="openai", 
            config=llm_config
        )
        
        response = llm_client.chat_completion(simple_input)
        logger.info(f"Response (no observability): {response['llm_response']}")
    
    # 4. Method 2: Using Observable Factory directly
    logger.info("=== Method 2: Observable Factory ===")
    
    try:
        observable_factory = LLMFactory.get_observable_factory(
            observability_config=observability_config
        )
        
        # Create client through observable factory
        observable_client = observable_factory.create(
            provider="openai",
            config=llm_config
        )
        
        # Make multiple requests to see metrics aggregation
        test_messages = [
            "What is machine learning?",
            "How does natural language processing work?",
            "Explain the benefits of AI observability."
        ]
        
        for i, message in enumerate(test_messages, 1):
            test_input = ILLMInput(
                system_prompt="You are a technical expert. Be concise.",
                user_message=message
            )
            
            response = observable_client.chat_completion(test_input)
            logger.info(f"Request {i}/3 completed")
            
            # Small delay to see timing differences
            await asyncio.sleep(0.5)
        
        # Check if observability is working
        obs_manager = observable_factory.get_observability_manager()
        logger.info(f"Observability enabled: {obs_manager.is_enabled()}")
        
    except ImportError:
        logger.warning("Observable factory not available")
    
    # 5. Method 3: Streaming with observability
    logger.info("=== Method 3: Streaming with Observability ===")
    
    try:
        streaming_input = ILLMInput(
            system_prompt="You are a creative writing assistant.",
            user_message="Write a short paragraph about the future of AI."
        )
        
        token_count = 0
        full_response = ""
        
        # Manual streaming observability example
        obs_manager = observable_factory.get_observability_manager()
        async with obs_manager.observe_streaming_llm_call("openai", "gpt-3.5-turbo", "stream_example") as timing:
            async for chunk in observable_client.stream_completion(streaming_input):
                # Process each chunk for token counting and timing
                usage_data = obs_manager.process_streaming_chunk("openai", "gpt-3.5-turbo", chunk, timing)
                
                if chunk.get('llm_response'):
                    content = chunk['llm_response']
                    full_response += content
                    token_count += 1
                    # Each chunk is automatically timed for metrics
                
                # Final chunk contains usage information
                if usage_data:
                    logger.info(f"Streaming completed - Tokens: {usage_data.get('total_tokens', 0)}, "
                               f"Chunks: {token_count}")
                    logger.info(f"Time to first token: {timing.time_to_first_token:.3f}s")
                    logger.info(f"Time to last token: {timing.time_to_last_token:.3f}s")
                    logger.info(f"Generation duration: {timing.duration_first_to_last_token:.3f}s")
                    break
        
        logger.info(f"Generated response: {full_response[:100]}...")
        
    except (NameError, ImportError):
        logger.warning("Streaming example skipped - observable client not available")
    
    # 6. Method 4: Manual observability manager usage
    logger.info("=== Method 4: Manual Observability Manager ===")
    
    obs_manager = ObservabilityManager(observability_config)
    
    # Example of manual token counting
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather like today?"}
    ]
    
    token_info = obs_manager.pre_call_token_count("openai", "gpt-3.5-turbo", messages)
    if token_info:
        logger.info(f"Pre-call token count: {token_info}")
    
    # Example of manual timing
    with obs_manager.observe_llm_call("openai", "gpt-3.5-turbo", "manual_call") as timing:
        # Simulate LLM call
        await asyncio.sleep(0.1)  # Simulate time to first token
        timing.record_first_token()
        
        await asyncio.sleep(0.2)  # Simulate generation time
        timing.record_token()
        
        # Simulate token counts
        timing.update_token_counts(prompt_tokens=15, completion_tokens=25, total_tokens=40)
    
    logger.info("Manual observability example completed")
    
    # 7. Configuration validation
    logger.info("=== Configuration Information ===")
    logger.info(f"Service: {observability_config.service_name}")
    logger.info(f"Environment: {observability_config.environment}")
    logger.info(f"Token timing enabled: {observability_config.track_token_timing}")
    logger.info(f"Non-intrusive mode: {observability_config.non_intrusive}")
    
    if observability_config.otlp_endpoint:
        logger.info(f"OTLP endpoint: {observability_config.otlp_endpoint}")
    else:
        logger.info("OTLP endpoint not configured (using console exporters)")
    
    # Cleanup
    obs_manager.shutdown()
    logger.info("Example completed successfully!")


def simple_synchronous_example():
    """Simple synchronous example for basic usage."""
    
    logger.info("=== Simple Synchronous Example ===")
    
    # Load config from environment variables
    config = ObservabilityConfig.from_environment()
    
    # Configure LLM
    llm_config = ILLMConfig(model="gpt-3.5-turbo", temperature=0.7)
    
    try:
        # Create with observability
        client = LLMFactory.create_with_observability(
            provider="openai",
            config=llm_config,
            observability_config=config
        )
        
        # Make a simple call
        input_data = ILLMInput(
            system_prompt="You are helpful.",
            user_message="Hello!"
        )
        
        response = client.chat_completion(input_data)
        logger.info(f"Simple response: {response['llm_response']}")
        
    except ImportError:
        logger.warning("Observability not available, using regular client")
        client = LLMFactory.create(provider="openai", config=llm_config)
        response = client.chat_completion(input_data)
        logger.info(f"Response without observability: {response['llm_response']}")


if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY environment variable")
        logger.info("You can still run the configuration examples...")
        
        # Show configuration loading examples
        try:
            config = ObservabilityConfig.from_environment()
            logger.info(f"Config from environment: enabled={config.enabled}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        exit(1)
    
    # Run the main async example
    logger.info("Starting comprehensive observability example...")
    asyncio.run(main())
    
    # Run simple synchronous example
    simple_synchronous_example()
    
    logger.info("All examples completed!")