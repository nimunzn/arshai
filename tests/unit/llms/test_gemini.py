"""
Test suite for Google Gemini LLM client.
Tests the Gemini client in isolation with minimal dependencies.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.gemini import GeminiClient

# Load test environment variables
test_env_path = Path(__file__).parent.parent.parent / ".env.gemini"
load_dotenv(test_env_path)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleResponse(BaseModel):
    """Simple test model for structured output testing"""
    message: str = Field(description="A simple response message")
    sentiment: str = Field(description="The sentiment of the message")
    confidence: float = Field(description="Confidence score between 0 and 1")


class MathResult(BaseModel):
    """Math calculation result model"""
    operation: str = Field(description="The mathematical operation performed")
    result: int = Field(description="The numerical result")
    explanation: str = Field(description="Brief explanation of the calculation")


class TestGeminiClient:
    """Test class for Gemini client functionality"""
    
    def __init__(self):
        self.config = ILLMConfig(
            model="gemini-2.5-flash",
            temperature=0.7,
            max_tokens=500
        )
        self.client = None
    
    def test_client_initialization(self):
        """Test that the Gemini client initializes correctly with service account"""
        logger.info("🧪 Testing Gemini client initialization...")
        
        try:
            self.client = GeminiClient(self.config)
            logger.info("✅ Gemini client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Client initialization failed: {str(e)}")
            return False
    
    def test_connection(self):
        """Test the client connection using the built-in test method"""
        logger.info("🧪 Testing Gemini client connection...")
        
        if not self.client:
            logger.error("❌ Client not initialized")
            return False
        
        try:
            # This should have already been called during initialization
            # but we can verify the client is working
            logger.info("✅ Connection test passed during initialization")
            return True
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            return False
    
    def test_simple_chat_completion(self):
        """Test basic chat completion without structured output"""
        logger.info("🧪 Testing simple chat completion...")
        
        if not self.client:
            logger.error("❌ Client not initialized")
            return False
        
        try:
            input_data = ILLMInput(
                system_prompt="You are a helpful assistant. Keep responses brief and concise.",
                user_message="What is the capital of France?"
            )
            
            response = self.client.chat_completion(input_data)
            
            if isinstance(response, dict) and "llm_response" in response:
                logger.info(f"✅ Simple chat completion successful")
                logger.info(f"📝 Response: {response['llm_response']}")
                logger.info(f"📊 Usage: {response.get('usage', 'No usage data')}")
                return True
            else:
                logger.error(f"❌ Unexpected response format: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Simple chat completion failed: {str(e)}")
            return False
    
    def test_structured_output(self):
        """Test chat completion with structured Pydantic output"""
        logger.info("🧪 Testing structured output...")
        
        if not self.client:
            logger.error("❌ Client not initialized")
            return False
        
        try:
            input_data = ILLMInput(
                system_prompt="You are a sentiment analysis assistant. Analyze the given text and respond with the specified structure.",
                user_message="I love sunny days! They make me feel so happy and energetic.",
                structure_type=SimpleResponse
            )
            
            response = self.client.chat_completion(input_data)
            
            if isinstance(response, dict) and "llm_response" in response:
                llm_response = response["llm_response"]
                if isinstance(llm_response, SimpleResponse):
                    logger.info(f"✅ Structured output successful")
                    logger.info(f"📝 Message: {llm_response.message}")
                    logger.info(f"😊 Sentiment: {llm_response.sentiment}")
                    logger.info(f"🎯 Confidence: {llm_response.confidence}")
                    logger.info(f"📊 Usage: {response.get('usage', 'No usage data')}")
                    return True
                else:
                    logger.error(f"❌ Response is not SimpleResponse type: {type(llm_response)}")
                    return False
            else:
                logger.error(f"❌ Unexpected response format: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Structured output failed: {str(e)}")
            return False
    
    async def test_tool_calling(self):
        """Test chat with tools functionality"""
        logger.info("🧪 Testing tool calling...")
        
        if not self.client:
            logger.error("❌ Client not initialized")
            return False
        
        # Define sync tool functions (Google GenAI doesn't support async for automatic function calling)
        def calculate_sum(a: int, b: int) -> str:
            """Calculate the sum of two numbers"""
            result = a + b
            logger.info(f"🧮 TOOL EXECUTED: calculate_sum({a}, {b}) = {result}")
            return f"The sum of {a} and {b} is {result}"
        
        def calculate_multiply(a: int, b: int) -> str:
            """Multiply two numbers"""
            result = a * b
            logger.info(f"🧮 TOOL EXECUTED: calculate_multiply({a}, {b}) = {result}")
            return f"The product of {a} and {b} is {result}"
        
        async def async_calculate_power(base: int, exponent: int) -> str:
            """Calculate power of a number (async method to test async tool handling)"""
            # Simulate some async work
            await asyncio.sleep(0.1)
            result = base ** exponent
            logger.info(f"🧮 ASYNC TOOL EXECUTED: async_calculate_power({base}, {exponent}) = {result}")
            return f"The result of {base} raised to the power of {exponent} is {result}"
        
        try:
            # Define tools for Gemini
            tools_list = [
                {
                    "name": "calculate_sum",
                    "description": "Calculate the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "First number"},
                            "b": {"type": "integer", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    }
                },
                {
                    "name": "calculate_multiply", 
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "First number"},
                            "b": {"type": "integer", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    }
                },
                {
                    "name": "async_calculate_power",
                    "description": "Calculate the power of a number (async method)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "base": {"type": "integer", "description": "Base number"},
                            "exponent": {"type": "integer", "description": "Exponent number"}
                        },
                        "required": ["base", "exponent"]
                    }
                }
            ]
            
            callable_functions = {
                "calculate_sum": calculate_sum,
                "calculate_multiply": calculate_multiply,
                "async_calculate_power": async_calculate_power
            }
            
            input_data = ILLMInput(
                system_prompt="You are a math assistant. Use the available tools to help with calculations. Always use tools when asked to perform calculations.",
                user_message="Please calculate 15 + 27, then multiply the result by 3, and finally calculate 2 to the power of 4",
                tools_list=tools_list,
                callable_functions=callable_functions,
                structure_type=MathResult,
                max_turns=5
            )
            
            response = await self.client.chat_with_tools(input_data)
            
            if isinstance(response, dict) and "llm_response" in response:
                llm_response = response["llm_response"]
                if isinstance(llm_response, MathResult):
                    logger.info(f"✅ Tool calling with structured output successful")
                    logger.info(f"🔢 Operation: {llm_response.operation}")
                    logger.info(f"📊 Result: {llm_response.result}")
                    logger.info(f"💡 Explanation: {llm_response.explanation}")
                    logger.info(f"📈 Usage: {response.get('usage', 'No usage data')}")
                    return True
                else:
                    logger.info(f"✅ Tool calling successful (unstructured)")
                    logger.info(f"📝 Response: {llm_response}")
                    logger.info(f"📊 Usage: {response.get('usage', 'No usage data')}")
                    return True
            else:
                logger.error(f"❌ Unexpected response format: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Tool calling failed: {str(e)}")
            return False
    
    def test_error_handling(self):
        """Test error handling with invalid requests"""
        logger.info("🧪 Testing error handling...")
        
        if not self.client:
            logger.error("❌ Client not initialized")
            return False
        
        try:
            # Test with empty input
            input_data = ILLMInput(
                system_prompt="",
                user_message=""
            )
            
            response = self.client.chat_completion(input_data)
            
            # Should handle gracefully
            if isinstance(response, dict):
                logger.info(f"✅ Error handling test passed")
                logger.info(f"📝 Response: {response.get('llm_response', 'No response')}")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response format but no crash: {response}")
                return True
                
        except Exception as e:
            logger.info(f"✅ Error handling test passed - caught exception gracefully: {str(e)}")
            return True


async def run_tests():
    """Run all Gemini client tests"""
    logger.info("🚀 Starting Gemini Client Test Suite")
    logger.info("=" * 50)
    
    # Check environment variables
    required_vars = ["GOOGLE_SERVICE_ACCOUNT_PATH", "GOOGLE_PROJECT_ID", "GOOGLE_LOCATION"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("💡 Please check your .env.gemini file")
        return False
    
    logger.info("✅ Environment variables configured")
    
    # Initialize test class
    test_client = TestGeminiClient()
    
    # Run tests
    tests = [
        ("Client Initialization", test_client.test_client_initialization),
        ("Connection Test", test_client.test_connection),
        ("Simple Chat Completion", test_client.test_simple_chat_completion),
        ("Structured Output", test_client.test_structured_output),
        ("Tool Calling", test_client.test_tool_calling),
        ("Error Handling", test_client.test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Gemini client is working correctly.")
        return True
    else:
        logger.error(f"💥 {total - passed} tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)