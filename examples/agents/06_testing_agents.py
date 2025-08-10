"""
Example 6: Testing Agents
==========================

This example demonstrates comprehensive testing strategies for agents,
including unit tests, integration tests, and performance testing.

Prerequisites:
- Set OPENROUTER_API_KEY environment variable (for integration tests)
- Install arshai package
- pip install pytest (for running tests)
"""

import os
import json
import asyncio
import time
from unittest.mock import AsyncMock, Mock
from typing import Dict, Any, List
from arshai.agents.base import BaseAgent
from arshai.agents.working_memory import WorkingMemoryAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput, ILLMConfig, ILLM
from arshai.llms.openrouter import OpenRouterClient


class TestableAgent(BaseAgent):
    """Example agent designed with testability in mind."""
    
    def __init__(self, llm_client: ILLM, **kwargs):
        system_prompt = "You are a helpful assistant that provides structured responses."
        super().__init__(llm_client, system_prompt, **kwargs)
        self.call_count = 0
        self.last_input = None
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Process input with tracking for testing."""
        self.call_count += 1
        self.last_input = input
        
        # Simple validation
        if not input.message or len(input.message.strip()) == 0:
            return {
                "error": "Empty message not allowed",
                "status": "failed",
                "call_count": self.call_count
            }
        
        # Process with LLM
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        try:
            result = await self.llm_client.chat(llm_input)
            
            return {
                "response": result.get('llm_response', ''),
                "status": "success",
                "call_count": self.call_count,
                "input_length": len(input.message),
                "metadata": input.metadata
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error",
                "call_count": self.call_count
            }


class MockLLMClient:
    """Mock LLM client for unit testing."""
    
    def __init__(self, config: ILLMConfig = None):
        self.config = config
        self.call_count = 0
        self.calls = []
        self.responses = {}
        self.should_fail = False
        self.delay = 0
    
    def set_response(self, input_text: str, response: str, usage: Dict = None):
        """Set a mock response for a specific input."""
        self.responses[input_text.lower()] = {
            'llm_response': response,
            'usage': usage or {'tokens': len(response.split())}
        }
    
    def set_failure(self, should_fail: bool = True):
        """Configure the mock to simulate failures."""
        self.should_fail = should_fail
    
    def set_delay(self, delay: float):
        """Add artificial delay to simulate slow responses."""
        self.delay = delay
    
    async def chat(self, input: ILLMInput) -> Dict[str, Any]:
        """Mock chat method."""
        self.call_count += 1
        self.calls.append({
            'system_prompt': input.system_prompt,
            'user_message': input.user_message,
            'timestamp': time.time()
        })
        
        # Simulate delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        # Simulate failure
        if self.should_fail:
            raise Exception("Mock LLM failure")
        
        # Return mock response
        key = input.user_message.lower()
        if key in self.responses:
            return self.responses[key]
        else:
            return {
                'llm_response': f"Mock response to: {input.user_message}",
                'usage': {'tokens': 10}
            }
    
    async def stream(self, input: ILLMInput):
        """Mock streaming method."""
        response = await self.chat(input)
        words = response['llm_response'].split()
        
        for word in words:
            yield {'llm_response': word + ' '}
        
        yield {'llm_response': None, 'usage': response.get('usage', {})}


# Unit Test Examples
class TestAgentUnit:
    """Unit test examples for agents."""
    
    async def test_basic_functionality(self):
        """Test basic agent functionality with mock."""
        # Create mock LLM
        mock_llm = MockLLMClient()
        mock_llm.set_response("hello", "Hello! How can I help you?")
        
        # Create agent
        agent = TestableAgent(mock_llm)
        
        # Test processing
        result = await agent.process(IAgentInput(message="hello"))
        
        # Assertions
        assert result['status'] == 'success'
        assert result['response'] == "Hello! How can I help you?"
        assert result['call_count'] == 1
        assert mock_llm.call_count == 1
        
        print("✅ Basic functionality test passed")
    
    async def test_empty_input_handling(self):
        """Test handling of empty input."""
        mock_llm = MockLLMClient()
        agent = TestableAgent(mock_llm)
        
        # Test empty message
        result = await agent.process(IAgentInput(message=""))
        
        assert result['status'] == 'failed'
        assert 'error' in result
        assert mock_llm.call_count == 0  # Should not call LLM
        
        print("✅ Empty input handling test passed")
    
    async def test_error_handling(self):
        """Test agent error handling."""
        mock_llm = MockLLMClient()
        mock_llm.set_failure(True)
        
        agent = TestableAgent(mock_llm)
        
        result = await agent.process(IAgentInput(message="test"))
        
        assert result['status'] == 'error'
        assert 'Mock LLM failure' in result['error']
        
        print("✅ Error handling test passed")
    
    async def test_metadata_passing(self):
        """Test metadata handling."""
        mock_llm = MockLLMClient()
        agent = TestableAgent(mock_llm)
        
        metadata = {"user_id": "test123", "session": "session456"}
        result = await agent.process(IAgentInput(
            message="test", 
            metadata=metadata
        ))
        
        assert result['metadata'] == metadata
        
        print("✅ Metadata passing test passed")
    
    async def test_call_tracking(self):
        """Test agent call tracking."""
        mock_llm = MockLLMClient()
        agent = TestableAgent(mock_llm)
        
        # Make multiple calls
        for i in range(3):
            await agent.process(IAgentInput(message=f"message {i}"))
        
        assert agent.call_count == 3
        assert mock_llm.call_count == 3
        
        print("✅ Call tracking test passed")


# Integration Test Examples
class TestAgentIntegration:
    """Integration test examples with real LLM client."""
    
    async def test_with_real_llm(self):
        """Test agent with real LLM client."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("⚠️ Skipping real LLM test - no API key")
            return
        
        # Create real LLM client
        config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.1, max_tokens=50)
        llm_client = OpenRouterClient(config)
        
        # Create agent
        agent = TestableAgent(llm_client)
        
        # Test with real API
        result = await agent.process(IAgentInput(message="What is 2+2?"))
        
        assert result['status'] == 'success'
        assert len(result['response']) > 0
        assert '4' in result['response']  # Should contain the answer
        
        print("✅ Real LLM integration test passed")
    
    async def test_working_memory_agent(self):
        """Test WorkingMemoryAgent with mock manager."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("⚠️ Skipping WorkingMemoryAgent test - no API key")
            return
        
        # Mock memory manager
        class MockMemoryManager:
            def __init__(self):
                self.stored_data = None
            
            async def store(self, data):
                self.stored_data = data
            
            async def retrieve(self, query):
                return None  # No existing memory
        
        mock_memory = MockMemoryManager()
        
        # Create real LLM client
        config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.5)
        llm_client = OpenRouterClient(config)
        
        # Create agent
        memory_agent = WorkingMemoryAgent(
            llm_client=llm_client,
            memory_manager=mock_memory
        )
        
        # Test memory update
        result = await memory_agent.process(IAgentInput(
            message="User discussed their interest in machine learning",
            metadata={"conversation_id": "test_conversation"}
        ))
        
        assert result == "success"
        assert mock_memory.stored_data is not None
        assert mock_memory.stored_data["conversation_id"] == "test_conversation"
        
        print("✅ WorkingMemoryAgent integration test passed")


# Performance Test Examples
class TestAgentPerformance:
    """Performance testing examples."""
    
    async def test_response_time(self):
        """Test agent response time."""
        mock_llm = MockLLMClient()
        mock_llm.set_delay(0.1)  # 100ms delay
        
        agent = TestableAgent(mock_llm)
        
        start_time = time.time()
        result = await agent.process(IAgentInput(message="test"))
        elapsed_time = time.time() - start_time
        
        assert result['status'] == 'success'
        assert 0.1 <= elapsed_time <= 0.2  # Should be around 100ms
        
        print(f"✅ Response time test passed: {elapsed_time:.3f}s")
    
    async def test_concurrent_processing(self):
        """Test concurrent agent processing."""
        mock_llm = MockLLMClient()
        agent = TestableAgent(mock_llm)
        
        # Create concurrent tasks
        tasks = [
            agent.process(IAgentInput(message=f"message {i}"))
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        
        # All should succeed
        assert len(results) == 10
        assert all(r['status'] == 'success' for r in results)
        assert agent.call_count == 10
        
        print(f"✅ Concurrent processing test passed: 10 requests in {elapsed_time:.3f}s")
    
    async def test_memory_usage(self):
        """Test agent memory usage patterns."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        mock_llm = MockLLMClient()
        agent = TestableAgent(mock_llm)
        
        # Process many requests
        for i in range(100):
            await agent.process(IAgentInput(message=f"test message {i}"))
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10MB for this test)
        assert memory_increase < 10 * 1024 * 1024  # 10MB
        
        print(f"✅ Memory usage test passed: +{memory_increase / 1024 / 1024:.2f}MB")


# Load Testing Example
class TestAgentLoad:
    """Load testing examples."""
    
    async def test_sustained_load(self):
        """Test agent under sustained load."""
        mock_llm = MockLLMClient()
        mock_llm.set_delay(0.01)  # 10ms delay per request
        
        agent = TestableAgent(mock_llm)
        
        # Test parameters
        concurrent_users = 5
        requests_per_user = 20
        total_requests = concurrent_users * requests_per_user
        
        async def simulate_user(user_id: int, num_requests: int):
            """Simulate a user making multiple requests."""
            results = []
            for i in range(num_requests):
                result = await agent.process(IAgentInput(
                    message=f"User {user_id} request {i}",
                    metadata={"user_id": user_id}
                ))
                results.append(result)
                await asyncio.sleep(0.001)  # Small delay between requests
            return results
        
        # Start load test
        start_time = time.time()
        
        user_tasks = [
            simulate_user(user_id, requests_per_user)
            for user_id in range(concurrent_users)
        ]
        
        user_results = await asyncio.gather(*user_tasks)
        elapsed_time = time.time() - start_time
        
        # Flatten results
        all_results = [result for user_result in user_results for result in user_result]
        
        # Calculate metrics
        successful_requests = sum(1 for r in all_results if r['status'] == 'success')
        throughput = total_requests / elapsed_time
        
        # Assertions
        assert len(all_results) == total_requests
        assert successful_requests == total_requests
        assert throughput > 50  # Should handle at least 50 requests/second
        
        print(f"✅ Load test passed:")
        print(f"   Total requests: {total_requests}")
        print(f"   Success rate: {successful_requests/total_requests*100:.1f}%")
        print(f"   Duration: {elapsed_time:.2f}s")
        print(f"   Throughput: {throughput:.1f} req/s")


async def run_all_tests():
    """Run all test suites."""
    
    print("=" * 60)
    print("AGENT TESTING DEMONSTRATION")
    print("=" * 60)
    
    # Unit Tests
    print("\n" + "=" * 40)
    print("UNIT TESTS")
    print("=" * 40)
    
    unit_tests = TestAgentUnit()
    await unit_tests.test_basic_functionality()
    await unit_tests.test_empty_input_handling()
    await unit_tests.test_error_handling()
    await unit_tests.test_metadata_passing()
    await unit_tests.test_call_tracking()
    
    # Integration Tests
    print("\n" + "=" * 40)
    print("INTEGRATION TESTS")
    print("=" * 40)
    
    integration_tests = TestAgentIntegration()
    await integration_tests.test_with_real_llm()
    await integration_tests.test_working_memory_agent()
    
    # Performance Tests
    print("\n" + "=" * 40)
    print("PERFORMANCE TESTS")
    print("=" * 40)
    
    performance_tests = TestAgentPerformance()
    await performance_tests.test_response_time()
    await performance_tests.test_concurrent_processing()
    await performance_tests.test_memory_usage()
    
    # Load Tests
    print("\n" + "=" * 40)
    print("LOAD TESTS")
    print("=" * 40)
    
    load_tests = TestAgentLoad()
    await load_tests.test_sustained_load()
    
    print("\n" + "=" * 60)
    print("TESTING BEST PRACTICES")
    print("=" * 60)
    
    print("✅ Testing Strategies Demonstrated:")
    print("   • Unit Tests: Mock dependencies, test logic")
    print("   • Integration Tests: Real components, end-to-end")
    print("   • Performance Tests: Response time, concurrency")
    print("   • Load Tests: Sustained throughput, stability")
    
    print("\n📋 Key Testing Principles:")
    print("   • Test both success and failure paths")
    print("   • Use mocks for external dependencies")
    print("   • Test with real components when possible")
    print("   • Measure performance and resource usage")
    print("   • Simulate realistic load patterns")
    print("   • Include edge cases and error conditions")


def create_pytest_examples():
    """Generate pytest-compatible test files."""
    
    pytest_content = '''"""
Pytest-compatible tests for agents.
Run with: pytest test_agents_pytest.py -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput


class SimpleTestAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result.get('llm_response', '')


@pytest.fixture
def mock_llm_client():
    """Fixture for mock LLM client."""
    mock = AsyncMock()
    mock.chat.return_value = {'llm_response': 'Mock response'}
    return mock


@pytest.fixture
def test_agent(mock_llm_client):
    """Fixture for test agent."""
    return SimpleTestAgent(mock_llm_client, "Test prompt")


@pytest.mark.asyncio
async def test_agent_basic_processing(test_agent, mock_llm_client):
    """Test basic agent processing."""
    result = await test_agent.process(IAgentInput(message="test"))
    
    assert result == "Mock response"
    mock_llm_client.chat.assert_called_once()


@pytest.mark.asyncio
async def test_agent_with_metadata(test_agent):
    """Test agent with metadata."""
    metadata = {"user_id": "123"}
    input_data = IAgentInput(message="test", metadata=metadata)
    
    result = await test_agent.process(input_data)
    assert isinstance(result, str)


@pytest.mark.parametrize("message,expected", [
    ("hello", "Mock response"),
    ("test", "Mock response"),
    ("", "Mock response"),
])
@pytest.mark.asyncio
async def test_agent_various_inputs(test_agent, message, expected):
    """Test agent with various inputs."""
    result = await test_agent.process(IAgentInput(message=message))
    assert result == expected
'''
    
    with open('/tmp/test_agents_pytest.py', 'w') as f:
        f.write(pytest_content)
    
    print(f"\n📝 Created pytest example: /tmp/test_agents_pytest.py")
    print("   Run with: cd /tmp && pytest test_agents_pytest.py -v")


async def main():
    """Main testing demonstration."""
    
    await run_all_tests()
    create_pytest_examples()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    
    print("\n🎯 Next Steps for Testing Your Agents:")
    print("1. Create unit tests with mocks for fast feedback")
    print("2. Add integration tests with real components")
    print("3. Include performance benchmarks")
    print("4. Set up continuous integration")
    print("5. Monitor production metrics")


if __name__ == "__main__":
    asyncio.run(main())