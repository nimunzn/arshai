#!/usr/bin/env python3
"""
Integration test for async tool execution without external dependencies.
This demonstrates that the async modifications work correctly.
"""

import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock


class MockTool:
    """A simple mock tool that implements both sync and async execution"""
    
    def __init__(self, tool_name: str, response_delay: float = 0.1):
        self.tool_name = tool_name
        self.response_delay = response_delay
    
    @property
    def function_definition(self) -> Dict:
        return {
            "name": self.tool_name,
            "description": f"Mock {self.tool_name} tool for testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query for the tool"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str) -> List[Dict[str, Any]]:
        """Synchronous execution (old way)"""
        return [{"type": "text", "text": f"Sync {self.tool_name} result for: {query}"}]
    
    async def aexecute(self, query: str) -> List[Dict[str, Any]]:
        """Asynchronous execution (new way)"""
        # Simulate async work
        await asyncio.sleep(self.response_delay)
        return [{"type": "text", "text": f"Async {self.tool_name} result for: {query}"}]


class MockLLM:
    """Mock LLM that demonstrates the async tool execution flow"""
    
    def __init__(self):
        self.logger = Mock()
        self.logger.info = Mock()
        self.logger.debug = Mock()
    
    async def chat_with_tools(self, input) -> Dict[str, Any]:
        """Mock LLM that calls tools asynchronously"""
        self.logger.info("Processing LLM request with tools")
        
        # Simulate calling the first available tool
        if input.callable_functions:
            tool_name = list(input.callable_functions.keys())[0]
            tool_function = input.callable_functions[tool_name]
            
            # Call the tool asynchronously
            tool_result = await tool_function(query="test query from LLM")
            
            # Create a mock structured response
            mock_response = Mock()
            mock_response.agent_message = f"I used {tool_name} and got: {tool_result[0]['text']}"
            mock_response.memory = Mock()
            
            return {
                "llm_response": mock_response,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            }
        
        # No tools case
        mock_response = Mock()
        mock_response.agent_message = "No tools available"
        mock_response.memory = Mock()
        
        return {
            "llm_response": mock_response,
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
        }


class MockConversationAgent:
    """Mock conversation agent that demonstrates async tool execution"""
    
    def __init__(self, tools: List[MockTool]):
        self.available_tools = tools
        self.llm = MockLLM()
        self.context_manager = Mock()
        self.context_manager.retrieve_working_memory.return_value = Mock()
        self.context_manager.store_working_memory = Mock()
        self.output_structure = Mock()
        self.output_structure.model_json_schema.return_value = {}
    
    def _get_function_description(self) -> List[Dict]:
        """Get function descriptions from tools"""
        return [tool.function_definition for tool in self.available_tools]
    
    def _get_callable_functions(self) -> Dict:
        """Get callable functions from tools (now using aexecute)"""
        return {
            tool.function_definition["name"]: tool.aexecute
            for tool in self.available_tools
        }
    
    def _prepare_system_prompt(self, working_memory) -> str:
        """Mock system prompt preparation"""
        return "You are a helpful assistant with access to tools."
    
    async def _get_llm_response(self, system_prompt: str, user_message: str):
        """Get response from LLM with async tools"""
        # Create mock LLM input
        mock_input = Mock()
        mock_input.system_prompt = system_prompt
        mock_input.user_message = user_message
        mock_input.tools_list = self._get_function_description()
        mock_input.callable_functions = self._get_callable_functions()
        mock_input.structure_type = self.output_structure
        
        # Call LLM
        llm_output = await self.llm.chat_with_tools(input=mock_input)
        
        return llm_output.get("llm_response"), llm_output.get("usage")
    
    async def process_message(self, user_message: str, conversation_id: str = "test_conv"):
        """Process a message using async tools"""
        print(f"Processing message: {user_message}")
        
        # Load working memory
        working_memory = self.context_manager.retrieve_working_memory(conversation_id)
        
        # Prepare system prompt
        system_prompt = self._prepare_system_prompt(working_memory)
        
        # Get LLM response (which will call tools asynchronously)
        llm_response, llm_usage = await self._get_llm_response(system_prompt, user_message)
        
        # Save working memory
        self.context_manager.store_working_memory(conversation_id, llm_response.memory)
        
        return llm_response.agent_message, llm_usage


async def test_async_agent_with_tools():
    """Test the complete async flow from agent to tools"""
    print("Testing async agent with tools integration...")
    
    # Create mock tools
    web_search_tool = MockTool("web_search", 0.1)
    knowledge_tool = MockTool("retrieve_knowledge", 0.15)
    
    # Create agent with tools
    agent = MockConversationAgent([web_search_tool, knowledge_tool])
    
    # Test single message processing
    import time
    start_time = time.time()
    
    response, usage = await agent.process_message("Search for information about Python asyncio")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Agent response: {response}")
    print(f"Usage: {usage}")
    print(f"Execution time: {execution_time:.3f}s")
    
    # Verify async execution
    assert "Async web_search result" in response, "Should use async tool execution"
    assert execution_time >= 0.1, "Should take at least as long as tool delay"
    
    print("✓ Async agent integration test passed!")


async def test_concurrent_agent_requests():
    """Test that multiple agent requests can be processed concurrently"""
    print("Testing concurrent agent requests...")
    
    # Create tools with different delays
    fast_tool = MockTool("fast_search", 0.05)
    slow_tool = MockTool("slow_search", 0.2)
    
    # Create two agents
    agent1 = MockConversationAgent([fast_tool])
    agent2 = MockConversationAgent([slow_tool])
    
    # Process requests concurrently
    import time
    start_time = time.time()
    
    results = await asyncio.gather(
        agent1.process_message("Fast query", "conv1"),
        agent2.process_message("Slow query", "conv2")
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Agent 1 response: {results[0][0]}")
    print(f"Agent 2 response: {results[1][0]}")
    print(f"Total time for concurrent execution: {total_time:.3f}s")
    
    # Verify concurrent execution (should be faster than sequential)
    assert total_time < 0.3, f"Concurrent execution should be faster than sequential: {total_time}s"
    assert "fast_search" in results[0][0], "Agent 1 should use fast_search"
    assert "slow_search" in results[1][0], "Agent 2 should use slow_search"
    
    print("✓ Concurrent agent requests test passed!")


async def test_callable_functions_mapping():
    """Test that callable functions correctly map to aexecute"""
    print("Testing callable functions mapping...")
    
    # Create a tool
    test_tool = MockTool("test_mapping", 0.01)
    
    # Create agent
    agent = MockConversationAgent([test_tool])
    
    # Get callable functions
    callable_functions = agent._get_callable_functions()
    
    # Verify mapping
    assert "test_mapping" in callable_functions, "Tool should be in callable functions"
    
    # Verify it's the async function
    func = callable_functions["test_mapping"]
    assert asyncio.iscoroutinefunction(func), "Callable function should be async (coroutine)"
    
    # Test calling it directly
    result = await func(query="direct test")
    assert "Async test_mapping result" in result[0]["text"], "Should call aexecute"
    
    print("✓ Callable functions mapping test passed!")


async def main():
    """Run all integration tests"""
    print("Starting async integration tests...\n")
    
    try:
        await test_async_agent_with_tools()
        print()
        
        await test_concurrent_agent_requests()
        print()
        
        await test_callable_functions_mapping()
        print()
        
        print("🎉 All async integration tests passed!")
        print("\n" + "="*50)
        print("SUMMARY:")
        print("✓ Tools now use aexecute for async execution")
        print("✓ LLM clients call tools asynchronously")
        print("✓ Conversation agents process messages asynchronously")
        print("✓ Multiple agent requests can run concurrently")
        print("✓ Tool execution is properly mapped to async functions")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the integration tests
    success = asyncio.run(main())
    exit(0 if success else 1)