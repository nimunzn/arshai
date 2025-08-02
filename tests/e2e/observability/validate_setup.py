#!/usr/bin/env python3
"""
Quick validation script to check if the observability setup works.
This runs a minimal test to verify basic functionality.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Arshai imports
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.observability import ObservabilityConfig, ObservabilityManager
from src.factories.llm_factory import LLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def validate_observability():
    """Run a minimal validation test."""
    logger.info("🔍 Running observability validation...")
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set")
        return False
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "test_config.yaml"
        observability_config = ObservabilityConfig.from_yaml(str(config_path))
        logger.info("✅ Configuration loaded")
        
        # Create LLM with observability
        llm_config = ILLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50
        )
        
        client = LLMFactory.create_with_observability(
            provider="openai",
            config=llm_config,
            observability_config=observability_config
        )
        logger.info("✅ LLM client created with observability")
        
        # Test simple completion
        test_input = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Say hello in exactly 5 words."
        )
        
        response = client.chat_completion(test_input)
        
        # Verify response structure
        assert 'llm_response' in response, "Missing llm_response"
        assert 'usage' in response, "Missing usage data"
        
        usage = response['usage']
        assert hasattr(usage, 'total_tokens'), "Missing token counts"
        
        logger.info(f"✅ Response received: {len(response['llm_response'])} chars, {usage.total_tokens} tokens")
        
        # Test streaming
        logger.info("🔍 Testing streaming...")
        streaming_input = ILLMInput(
            system_prompt="You are helpful.",
            user_message="Count from 1 to 3."
        )
        
        full_response = ""
        chunk_count = 0
        
        async for chunk in client.stream_completion(streaming_input):
            chunk_count += 1
            if chunk.get('llm_response'):
                full_response += chunk['llm_response']
            
            if chunk.get('usage'):
                logger.info(f"✅ Streaming completed: {chunk_count} chunks, {len(full_response)} chars")
                break
        
        logger.info("🎉 Validation successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(validate_observability())
    sys.exit(0 if success else 1)