"""
Comprehensive Guide to Creating and Using Agents in Arshai Framework
=====================================================================

This guide demonstrates how to create, customize, and use agents in the Arshai framework.
It covers everything from basic agent creation to advanced custom implementations.

NOTE: This is a single-file comprehensive tutorial. For more focused examples, see:
- 01_basic_usage.py - Simple agent patterns and getting started
- 02_custom_agents.py - Specialized agents with structured output  
- 03_memory_patterns.py - WorkingMemoryAgent usage patterns
- 04_tool_integration.py - Function calling and tool integration
- 05_agent_composition.py - Multi-agent orchestration patterns
- 06_testing_agents.py - Comprehensive testing strategies
- agent_quickstart.py - 5-minute interactive getting started

Target Audience:
- Framework users who want to use existing agents
- Developers who want to create custom agents
- Contributors who want to understand the agent architecture
- Maintainers who need reference implementations

Table of Contents:
1. Basic Agent Usage
2. Creating Custom Agents
3. Working with Memory
4. Agent with Tools
5. Advanced Patterns
6. Testing Your Agents
"""

import asyncio
from typing import Any, Dict, Optional, List
from arshai.core.interfaces.iagent import IAgent, IAgentInput
from arshai.core.interfaces.illm import ILLM, ILLMInput, ILLMConfig
from arshai.agents.base import BaseAgent
from arshai.agents.working_memory import WorkingMemoryAgent
from arshai.llms.openrouter import OpenRouterClient


# ============================================================================
# SECTION 1: BASIC AGENT USAGE
# ============================================================================

async def basic_agent_usage_example():
    """
    Example 1: Using a pre-built agent with minimal configuration.
    
    This is the simplest way to get started with agents in Arshai.
    Perfect for users who want to quickly integrate AI capabilities.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Agent Usage")
    print("="*60)
    
    # Step 1: Configure your LLM client
    llm_config = ILLMConfig(
        model="openai/gpt-4o-mini",  # Choose your model
        temperature=0.7,               # Control randomness (0.0 = deterministic, 1.0 = creative)
        max_tokens=150                 # Maximum response length
    )
    
    # Step 2: Initialize the LLM client (using OpenRouter as example)
    llm_client = OpenRouterClient(llm_config)
    
    # Step 3: Create a simple agent
    class SimpleResponseAgent(BaseAgent):
        """A basic agent that processes messages and returns responses."""
        
        async def process(self, input: IAgentInput) -> str:
            """Process user input and return a string response."""
            # Create LLM input with system prompt and user message
            llm_input = ILLMInput(
                system_prompt=self.system_prompt,
                user_message=input.message
            )
            
            # Get response from LLM
            result = await self.llm_client.chat(llm_input)
            
            # Extract and return the response
            return result.get('llm_response', 'No response generated')
    
    # Step 4: Initialize your agent
    agent = SimpleResponseAgent(
        llm_client=llm_client,
        system_prompt="You are a helpful AI assistant. Be concise and friendly."
    )
    
    # Step 5: Use the agent
    user_input = IAgentInput(message="What is the capital of France?")
    response = await agent.process(user_input)
    
    print(f"User: {user_input.message}")
    print(f"Agent: {response}")
    
    return agent


# ============================================================================
# SECTION 2: CREATING CUSTOM AGENTS
# ============================================================================

class CustomAnalysisAgent(BaseAgent):
    """
    Example 2: A custom agent that performs sentiment analysis.
    
    This demonstrates how to create specialized agents for specific tasks.
    Great for developers who need domain-specific functionality.
    """
    
    def __init__(self, llm_client: ILLM, **kwargs):
        """Initialize with a specialized system prompt for sentiment analysis."""
        system_prompt = """You are a sentiment analysis expert. 
        Analyze the emotional tone of messages and provide:
        1. Overall sentiment (positive/negative/neutral)
        2. Confidence score (0-100%)
        3. Key emotional indicators
        
        Format your response as JSON."""
        
        super().__init__(llm_client, system_prompt, **kwargs)
        self.analysis_history = []  # Track analyses for reporting
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """
        Process message and return structured sentiment analysis.
        
        Args:
            input: User input containing the text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Prepare the analysis request
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=f"Analyze the sentiment of: {input.message}"
        )
        
        # Get analysis from LLM
        result = await self.llm_client.chat(llm_input)
        response_text = result.get('llm_response', '{}')
        
        # Parse the response (in production, use proper JSON parsing)
        import json
        try:
            analysis = json.loads(response_text)
        except:
            analysis = {
                "sentiment": "unknown",
                "confidence": 0,
                "indicators": ["unable to parse response"]
            }
        
        # Store in history
        self.analysis_history.append({
            "input": input.message,
            "analysis": analysis,
            "timestamp": input.metadata.get("timestamp") if input.metadata else None
        })
        
        return analysis
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed."""
        if not self.analysis_history:
            return {"message": "No analyses performed yet"}
        
        sentiments = [a["analysis"]["sentiment"] for a in self.analysis_history]
        return {
            "total_analyses": len(self.analysis_history),
            "sentiment_distribution": {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral"),
                "unknown": sentiments.count("unknown")
            }
        }


async def custom_agent_example():
    """Demonstrate custom agent usage."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Sentiment Analysis Agent")
    print("="*60)
    
    # Initialize LLM client
    llm_config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.3)
    llm_client = OpenRouterClient(llm_config)
    
    # Create custom agent
    sentiment_agent = CustomAnalysisAgent(llm_client)
    
    # Analyze some messages
    messages = [
        "I absolutely love this new feature! It's amazing!",
        "This is terrible, nothing works as expected.",
        "The weather is okay today, nothing special."
    ]
    
    for msg in messages:
        input_data = IAgentInput(message=msg)
        analysis = await sentiment_agent.process(input_data)
        print(f"\nText: {msg}")
        print(f"Analysis: {analysis}")
    
    # Get summary
    summary = sentiment_agent.get_analysis_summary()
    print(f"\nSummary: {summary}")
    
    return sentiment_agent


# ============================================================================
# SECTION 3: WORKING WITH MEMORY
# ============================================================================

async def memory_agent_example():
    """
    Example 3: Using the WorkingMemoryAgent for conversation context.
    
    This shows how to maintain conversation state across interactions.
    Essential for building conversational AI applications.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Working Memory Agent")
    print("="*60)
    
    # Initialize LLM client
    llm_config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.5)
    llm_client = OpenRouterClient(llm_config)
    
    # Create a simple in-memory storage for demo
    class InMemoryManager:
        """Simple memory manager for demonstration."""
        def __init__(self):
            self.memories = {}
        
        async def store(self, data: Dict[str, Any]):
            """Store memory for a conversation."""
            conv_id = data.get("conversation_id")
            if conv_id:
                self.memories[conv_id] = data.get("working_memory", "")
                print(f"  [Memory Stored] for conversation {conv_id}")
        
        async def retrieve(self, query: Dict[str, Any]):
            """Retrieve memory for a conversation."""
            conv_id = query.get("conversation_id")
            if conv_id and conv_id in self.memories:
                print(f"  [Memory Retrieved] for conversation {conv_id}")
                return [type('obj', (), {'working_memory': self.memories[conv_id]})()]
            return None
    
    # Initialize memory components
    memory_manager = InMemoryManager()
    
    # Create memory agent
    memory_agent = WorkingMemoryAgent(
        llm_client=llm_client,
        memory_manager=memory_manager
    )
    
    # Simulate a conversation
    conversation_id = "user_123_session_456"
    
    interactions = [
        "My name is Alice and I'm interested in learning Python",
        "I prefer hands-on projects over theory",
        "What would you recommend for someone at my level?"
    ]
    
    for interaction in interactions:
        print(f"\nUser: {interaction}")
        
        # Update memory with each interaction
        input_data = IAgentInput(
            message=interaction,
            metadata={"conversation_id": conversation_id}
        )
        
        result = await memory_agent.process(input_data)
        print(f"  Memory Update Status: {result}")
    
    # Show final memory state
    if conversation_id in memory_manager.memories:
        print(f"\nFinal Memory State:")
        print(f"{memory_manager.memories[conversation_id]}")
    
    return memory_agent, memory_manager


# ============================================================================
# SECTION 4: AGENT WITH TOOLS
# ============================================================================

class ToolEnabledAgent(BaseAgent):
    """
    Example 4: An agent that can use external tools.
    
    This demonstrates how to create agents that can perform actions
    beyond just generating text responses.
    """
    
    def __init__(self, llm_client: ILLM, tools: List[Dict[str, Any]] = None, **kwargs):
        """Initialize with tools configuration."""
        system_prompt = """You are an AI assistant with access to various tools.
        Use tools when appropriate to help users with their requests.
        Always explain what tools you're using and why."""
        
        super().__init__(llm_client, system_prompt, **kwargs)
        self.tools = tools or []
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """
        Process input and potentially use tools.
        
        Returns:
            Dictionary with response and any tool results
        """
        # Check if we need to use tools
        tool_needed = await self._determine_tool_need(input.message)
        
        if tool_needed:
            # Prepare tool-enabled LLM input
            tool_functions = self._prepare_tool_functions()
            
            llm_input = ILLMInput(
                system_prompt=self.system_prompt,
                user_message=input.message,
                regular_functions=tool_functions  # Add tools as functions
            )
            
            # Get response with potential tool calls
            result = await self.llm_client.chat(llm_input)
            
            return {
                "response": result.get('llm_response', ''),
                "tools_used": list(tool_functions.keys()),
                "metadata": input.metadata
            }
        else:
            # Regular response without tools
            llm_input = ILLMInput(
                system_prompt=self.system_prompt,
                user_message=input.message
            )
            
            result = await self.llm_client.chat(llm_input)
            
            return {
                "response": result.get('llm_response', ''),
                "tools_used": [],
                "metadata": input.metadata
            }
    
    async def _determine_tool_need(self, message: str) -> bool:
        """Determine if tools are needed for this request."""
        # Simple heuristic - in production, use LLM to decide
        tool_keywords = ["calculate", "search", "find", "look up", "check"]
        return any(keyword in message.lower() for keyword in tool_keywords)
    
    def _prepare_tool_functions(self) -> Dict[str, Any]:
        """Prepare tools as callable functions."""
        # Example calculator tool
        def calculate(expression: str) -> float:
            """Calculate a mathematical expression."""
            try:
                # In production, use safe evaluation
                return eval(expression)
            except:
                return 0.0
        
        # Example search tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Search results for '{query}': [Mock results would appear here]"
        
        return {
            "calculate": calculate,
            "search": search
        }


async def tool_agent_example():
    """Demonstrate agent with tools."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Tool-Enabled Agent")
    print("="*60)
    
    # Initialize LLM client
    llm_config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.3)
    llm_client = OpenRouterClient(llm_config)
    
    # Create tool-enabled agent
    tool_agent = ToolEnabledAgent(llm_client)
    
    # Test with different requests
    requests = [
        "What is the weather like?",  # No tool needed
        "Calculate 15 * 23 + 47",      # Calculator tool needed
        "Search for Python tutorials"   # Search tool needed
    ]
    
    for request in requests:
        print(f"\nUser: {request}")
        input_data = IAgentInput(message=request)
        result = await tool_agent.process(input_data)
        print(f"Agent Response: {result['response']}")
        print(f"Tools Used: {result['tools_used']}")
    
    return tool_agent


# ============================================================================
# SECTION 5: ADVANCED PATTERNS
# ============================================================================

class StreamingAgent(BaseAgent):
    """
    Example 5: An agent that supports streaming responses.
    
    This is useful for real-time applications where you want to
    show responses as they're being generated.
    """
    
    async def process(self, input: IAgentInput) -> Any:
        """Process input with streaming support."""
        # Check if streaming is requested
        is_streaming = input.metadata.get("stream", False) if input.metadata else False
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        if is_streaming:
            # Return async generator for streaming
            return self._stream_response(llm_input)
        else:
            # Regular non-streaming response
            result = await self.llm_client.chat(llm_input)
            return result.get('llm_response', '')
    
    async def _stream_response(self, llm_input: ILLMInput):
        """Generate streaming response."""
        async for chunk in self.llm_client.stream(llm_input):
            if chunk.get('llm_response'):
                yield chunk['llm_response']


class ChainedAgent(BaseAgent):
    """
    Example 6: An agent that chains multiple processing steps.
    
    Useful for complex workflows that require multiple stages of processing.
    """
    
    def __init__(self, llm_client: ILLM, preprocessor_agent: IAgent = None, **kwargs):
        """Initialize with optional preprocessor agent."""
        super().__init__(llm_client, "You are a helpful assistant.", **kwargs)
        self.preprocessor = preprocessor_agent
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Process input through multiple stages."""
        stages = []
        
        # Stage 1: Preprocessing (if available)
        if self.preprocessor:
            preprocessed = await self.preprocessor.process(input)
            stages.append({"stage": "preprocessing", "result": preprocessed})
            
            # Update input with preprocessed content
            input = IAgentInput(
                message=str(preprocessed),
                metadata=input.metadata
            )
        
        # Stage 2: Main processing
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        main_result = await self.llm_client.chat(llm_input)
        stages.append({
            "stage": "main_processing",
            "result": main_result.get('llm_response', '')
        })
        
        # Stage 3: Post-processing (example: format check)
        final_result = stages[-1]["result"]
        if len(final_result) > 500:
            # Summarize if too long
            summary_input = ILLMInput(
                system_prompt="Summarize the following text concisely.",
                user_message=final_result
            )
            summary = await self.llm_client.chat(summary_input)
            stages.append({
                "stage": "post_processing",
                "result": summary.get('llm_response', final_result)
            })
            final_result = stages[-1]["result"]
        
        return {
            "final_response": final_result,
            "processing_stages": stages,
            "metadata": input.metadata
        }


# ============================================================================
# SECTION 6: TESTING YOUR AGENTS
# ============================================================================

async def test_agent_example():
    """
    Example 7: How to test your custom agents.
    
    Shows best practices for testing agent implementations.
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Testing Your Agents")
    print("="*60)
    
    # Create a mock LLM client for testing
    class MockLLMClient:
        """Mock LLM client for testing without API calls."""
        
        def __init__(self, config: ILLMConfig):
            self.config = config
            self.call_count = 0
        
        async def chat(self, input: ILLMInput) -> Dict[str, Any]:
            """Return mock responses for testing."""
            self.call_count += 1
            
            # Return different responses based on input
            if "sentiment" in input.system_prompt.lower():
                return {
                    "llm_response": '{"sentiment": "positive", "confidence": 95, "indicators": ["test"]}',
                    "usage": {"tokens": 10}
                }
            elif "calculate" in input.user_message.lower():
                return {
                    "llm_response": "The result is 42",
                    "usage": {"tokens": 5}
                }
            else:
                return {
                    "llm_response": f"Mock response #{self.call_count}",
                    "usage": {"tokens": 3}
                }
        
        async def stream(self, input: ILLMInput):
            """Mock streaming response."""
            response = await self.chat(input)
            for word in response["llm_response"].split():
                yield {"llm_response": word + " "}
    
    # Test configuration
    test_config = ILLMConfig(model="mock-model", temperature=0.5)
    mock_client = MockLLMClient(test_config)
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic Agent Functionality")
    test_agent = SimpleResponseAgent(
        llm_client=mock_client,
        system_prompt="Test prompt"
    )
    
    test_input = IAgentInput(message="Test message")
    result = await test_agent.process(test_input)
    assert isinstance(result, str), "Agent should return a string"
    assert len(result) > 0, "Response should not be empty"
    print(f"✓ Basic test passed: {result}")
    
    # Test 2: Custom agent with specific behavior
    print("\nTest 2: Custom Agent Behavior")
    sentiment_agent = CustomAnalysisAgent(llm_client=mock_client)
    
    test_input = IAgentInput(message="Test sentiment analysis")
    analysis = await sentiment_agent.process(test_input)
    assert isinstance(analysis, dict), "Should return dictionary"
    assert "sentiment" in analysis, "Should contain sentiment field"
    print(f"✓ Custom agent test passed: {analysis}")
    
    # Test 3: Error handling
    print("\nTest 3: Error Handling")
    
    class ErrorTestAgent(BaseAgent):
        async def process(self, input: IAgentInput) -> str:
            if not input.message:
                return "Error: Empty message"
            return "Success"
    
    error_agent = ErrorTestAgent(mock_client, "Test")
    
    # Test with empty message
    empty_input = IAgentInput(message="")
    error_result = await error_agent.process(empty_input)
    assert "Error" in error_result, "Should handle empty input"
    print(f"✓ Error handling test passed: {error_result}")
    
    # Test 4: Performance check
    print("\nTest 4: Performance Check")
    import time
    
    start_time = time.time()
    for i in range(10):
        await test_agent.process(IAgentInput(message=f"Test {i}"))
    
    elapsed = time.time() - start_time
    print(f"✓ Performance test: 10 calls in {elapsed:.2f} seconds")
    print(f"  Average: {elapsed/10:.3f} seconds per call")
    
    return mock_client


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Run all examples to demonstrate agent capabilities.
    
    Note: Set OPENROUTER_API_KEY environment variable to run with real LLM.
    """
    import os
    
    print("\n" + "="*80)
    print(" ARSHAI AGENT FRAMEWORK - COMPREHENSIVE GUIDE")
    print("="*80)
    
    # Check for API key
    has_api_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    
    if not has_api_key:
        print("\n⚠️  No OPENROUTER_API_KEY found in environment.")
        print("   Running in demo mode with mock responses.")
        print("   Set OPENROUTER_API_KEY to use real LLM responses.")
    
    try:
        # Always run test example (doesn't need API key)
        await test_agent_example()
        
        if has_api_key:
            # Run real examples with API
            await basic_agent_usage_example()
            await custom_agent_example()
            await memory_agent_example()
            await tool_agent_example()
        else:
            print("\n" + "="*60)
            print("Additional examples require OPENROUTER_API_KEY")
            print("Export your key to run all examples:")
            print("  export OPENROUTER_API_KEY=your_key_here")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ Error in examples: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print(" GUIDE COMPLETED")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review the agent implementations above")
    print("2. Create your own custom agent by extending BaseAgent")
    print("3. Test your agent with the patterns shown")
    print("4. Integrate with your application")
    print("\nFor more information, see the documentation at:")
    print("https://github.com/MobileTechLab/ArsHai")


if __name__ == "__main__":
    asyncio.run(main())