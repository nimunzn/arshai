# Build Your First Custom Agent

This guide walks you through creating a custom agent from scratch, demonstrating the power of direct instantiation and the three-layer architecture.

## What We'll Build

We'll create a **Smart Assistant Agent** that:
- Analyzes user requests for complexity
- Routes simple requests to fast processing
- Uses tools for complex requests
- Returns structured responses with metadata

## Prerequisites

- Arshai installed: `pip install arshai[openai]`
- OpenAI API key: `export OPENAI_API_KEY="sk-..."`
- Basic understanding of [Core Concepts](core-concepts.md)

## Step 1: Set Up the Environment

First, let's create our project structure and imports:

```python
# smart_agent.py
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
from enum import Enum

# Arshai imports - Layer 1 (LLM Clients)
from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput

# Arshai imports - Layer 2 (Agents)
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput

print("✅ Imports successful")
```

## Step 2: Design Your Agent

Before coding, let's design our agent's behavior:

```python
class RequestComplexity(Enum):
    """Complexity levels for user requests."""
    SIMPLE = "simple"      # Basic questions, greetings
    MODERATE = "moderate"  # Requires some analysis
    COMPLEX = "complex"    # Needs tools or deep reasoning

class ResponseMetadata:
    """Metadata about the agent's response."""
    def __init__(self, complexity: RequestComplexity, processing_time: float, tools_used: List[str]):
        self.complexity = complexity.value
        self.processing_time = processing_time
        self.tools_used = tools_used
        self.timestamp = datetime.utcnow().isoformat()
```

## Step 3: Implement the Agent Core

Now let's build our custom agent:

```python
class SmartAssistantAgent(BaseAgent):
    """
    A smart assistant that adapts its processing based on request complexity.
    
    Features:
    - Complexity analysis
    - Adaptive processing strategies
    - Tool integration
    - Structured metadata responses
    """
    
    def __init__(self, llm_client, system_prompt: str, complexity_threshold: float = 0.5):
        """Initialize the smart assistant.
        
        Args:
            llm_client: LLM client for processing
            system_prompt: Base system prompt
            complexity_threshold: Threshold for complexity routing (0.0-1.0)
        """
        super().__init__(llm_client, system_prompt)
        self.complexity_threshold = complexity_threshold
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Process user input with adaptive complexity handling."""
        start_time = asyncio.get_event_loop().time()
        
        # Step 1: Analyze complexity
        complexity = await self._analyze_complexity(input.message)
        
        # Step 2: Route based on complexity
        if complexity == RequestComplexity.SIMPLE:
            response = await self._process_simple(input)
            tools_used = []
        elif complexity == RequestComplexity.MODERATE:
            response = await self._process_moderate(input)
            tools_used = []
        else:  # COMPLEX
            response, tools_used = await self._process_complex(input)
        
        # Step 3: Calculate metadata
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Step 4: Return structured response
        return {
            "response": response,
            "metadata": {
                "complexity": complexity.value,
                "processing_time": round(processing_time, 3),
                "tools_used": tools_used,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

## Step 4: Implement Complexity Analysis

Add intelligence to classify request complexity:

```python
    async def _analyze_complexity(self, message: str) -> RequestComplexity:
        """Analyze the complexity of a user request."""
        
        # Simple heuristics (you can make this more sophisticated)
        simple_patterns = [
            "hello", "hi", "thanks", "thank you", "bye", "goodbye",
            "what is your name", "who are you", "how are you"
        ]
        
        complex_patterns = [
            "calculate", "compute", "search", "find", "lookup",
            "analyze", "compare", "research", "explain in detail"
        ]
        
        message_lower = message.lower()
        
        # Check for simple patterns
        if any(pattern in message_lower for pattern in simple_patterns):
            return RequestComplexity.SIMPLE
        
        # Check for complex patterns
        if any(pattern in message_lower for pattern in complex_patterns):
            return RequestComplexity.COMPLEX
        
        # Use LLM to analyze ambiguous cases
        analysis_prompt = ILLMInput(
            system_prompt="Analyze request complexity. Reply only with 'simple', 'moderate', or 'complex'.",
            user_message=f"Classify this request: {message}"
        )
        
        result = await self.llm_client.chat(analysis_prompt)
        complexity_str = result.get("llm_response", "moderate").lower().strip()
        
        if "simple" in complexity_str:
            return RequestComplexity.SIMPLE
        elif "complex" in complexity_str:
            return RequestComplexity.COMPLEX
        else:
            return RequestComplexity.MODERATE
```

## Step 5: Implement Processing Strategies

Add different processing approaches for each complexity level:

```python
    async def _process_simple(self, input: IAgentInput) -> str:
        """Fast processing for simple requests."""
        llm_input = ILLMInput(
            system_prompt=f"{self.system_prompt} Be brief and friendly.",
            user_message=input.message
        )
        
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "I'm here to help!")
    
    async def _process_moderate(self, input: IAgentInput) -> str:
        """Standard processing for moderate requests."""
        llm_input = ILLMInput(
            system_prompt=f"{self.system_prompt} Provide a thoughtful response.",
            user_message=input.message
        )
        
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "Let me think about that...")
    
    async def _process_complex(self, input: IAgentInput) -> tuple[str, List[str]]:
        """Advanced processing with tools for complex requests."""
        
        # Define tools
        def search_web(query: str) -> str:
            """Search the web for information."""
            return f"Found information about: {query}"
        
        def calculate(expression: str) -> str:
            """Calculate mathematical expression."""
            try:
                # Simple calculator (use safe eval in production!)
                result = eval(expression.replace("^", "**"))
                return f"Result: {result}"
            except:
                return "Invalid calculation"
        
        def get_current_time() -> str:
            """Get the current time."""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Process with tools
        llm_input = ILLMInput(
            system_prompt=f"{self.system_prompt} You have access to tools. Use them when helpful.",
            user_message=input.message,
            regular_functions={
                "search_web": search_web,
                "calculate": calculate,
                "get_current_time": get_current_time
            }
        )
        
        result = await self.llm_client.chat(llm_input)
        
        # Extract which tools were used
        function_calls = result.get("function_calls", {})
        tools_used = list(function_calls.keys()) if function_calls else []
        
        response = result.get("llm_response", "I processed your complex request.")
        
        return response, tools_used
```

## Step 6: Create the Agent Instance

Now let's put it all together:

```python
async def create_smart_agent():
    """Create and configure the smart assistant agent."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Configure LLM client (Layer 1)
    llm_config = ILLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=200
    )
    
    llm_client = OpenAIClient(llm_config)
    print("✅ LLM client created")
    
    # Create agent (Layer 2)
    agent = SmartAssistantAgent(
        llm_client=llm_client,
        system_prompt="You are a helpful AI assistant that adapts to user needs.",
        complexity_threshold=0.5
    )
    print("✅ Smart agent created")
    
    return agent
```

## Step 7: Test Your Agent

Let's test our agent with different complexity levels:

```python
async def test_agent():
    """Test the smart agent with various inputs."""
    
    agent = await create_smart_agent()
    
    test_cases = [
        # Simple requests
        "Hello!",
        "Thanks for your help",
        
        # Moderate requests  
        "Explain what artificial intelligence is",
        "What are the benefits of exercise?",
        
        # Complex requests
        "Calculate 15 * 23 + 45",
        "Search for information about Python programming",
        "What time is it now?"
    ]
    
    print("\n" + "=" * 60)
    print("TESTING SMART ASSISTANT AGENT")
    print("=" * 60)
    
    for message in test_cases:
        print(f"\n👤 User: {message}")
        
        # Process with agent
        result = await agent.process(IAgentInput(message=message))
        
        # Display results
        print(f"🤖 Agent: {result['response']}")
        print(f"📊 Metadata:")
        print(f"   Complexity: {result['metadata']['complexity']}")
        print(f"   Processing time: {result['metadata']['processing_time']}s")
        if result['metadata']['tools_used']:
            print(f"   Tools used: {', '.join(result['metadata']['tools_used'])}")
        
        print("-" * 40)

# Run the test
if __name__ == "__main__":
    asyncio.run(test_agent())
```

## Step 8: Make It Interactive

Add an interactive chat loop:

```python
async def interactive_chat():
    """Interactive chat with the smart agent."""
    
    agent = await create_smart_agent()
    
    print("\n🤖 Smart Assistant Ready!")
    print("Features: Adaptive complexity, tool usage, detailed metadata")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("""
Available commands:
- 'quit' - Exit the chat
- 'help' - Show this help
- Any other text - Process with the agent

Try different types of requests:
- Simple: "Hello", "Thanks"
- Moderate: "Explain AI", "What is Python?"
- Complex: "Calculate 25*4", "What time is it?"
                """)
                continue
            
            if not user_input:
                continue
            
            # Process with agent
            result = await agent.process(IAgentInput(message=user_input))
            
            # Display response
            print(f"\n🤖 Agent: {result['response']}")
            
            # Display metadata (optional)
            metadata = result['metadata']
            print(f"\n📊 [Complexity: {metadata['complexity']} | "
                  f"Time: {metadata['processing_time']}s")
            if metadata['tools_used']:
                print(f" | Tools: {', '.join(metadata['tools_used'])}", end="")
            print("]")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Run interactive mode
if __name__ == "__main__":
    asyncio.run(interactive_chat())
```

## Complete Example

Here's the full working agent in one file:

<details>
<summary>Click to expand complete smart_agent.py</summary>

```python
# smart_agent.py - Complete Smart Assistant Agent
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
from enum import Enum

from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput

class RequestComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class SmartAssistantAgent(BaseAgent):
    def __init__(self, llm_client, system_prompt: str, complexity_threshold: float = 0.5):
        super().__init__(llm_client, system_prompt)
        self.complexity_threshold = complexity_threshold
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        
        complexity = await self._analyze_complexity(input.message)
        
        if complexity == RequestComplexity.SIMPLE:
            response = await self._process_simple(input)
            tools_used = []
        elif complexity == RequestComplexity.MODERATE:
            response = await self._process_moderate(input)
            tools_used = []
        else:
            response, tools_used = await self._process_complex(input)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        return {
            "response": response,
            "metadata": {
                "complexity": complexity.value,
                "processing_time": round(processing_time, 3),
                "tools_used": tools_used,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def _analyze_complexity(self, message: str) -> RequestComplexity:
        simple_patterns = ["hello", "hi", "thanks", "bye"]
        complex_patterns = ["calculate", "search", "analyze"]
        
        message_lower = message.lower()
        
        if any(pattern in message_lower for pattern in simple_patterns):
            return RequestComplexity.SIMPLE
        if any(pattern in message_lower for pattern in complex_patterns):
            return RequestComplexity.COMPLEX
        
        return RequestComplexity.MODERATE
    
    async def _process_simple(self, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt=f"{self.system_prompt} Be brief and friendly.",
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "Hello!")
    
    async def _process_moderate(self, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt=f"{self.system_prompt} Provide a thoughtful response.",
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "Let me help you with that.")
    
    async def _process_complex(self, input: IAgentInput) -> tuple[str, List[str]]:
        def calculate(expression: str) -> str:
            try:
                result = eval(expression.replace("^", "**"))
                return f"Result: {result}"
            except:
                return "Invalid calculation"
        
        def get_current_time() -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        llm_input = ILLMInput(
            system_prompt=f"{self.system_prompt} Use tools when helpful.",
            user_message=input.message,
            regular_functions={
                "calculate": calculate,
                "get_current_time": get_current_time
            }
        )
        
        result = await self.llm_client.chat(llm_input)
        function_calls = result.get("function_calls", {})
        tools_used = list(function_calls.keys()) if function_calls else []
        
        return result.get("llm_response", "Processed"), tools_used

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    config = ILLMConfig(model="gpt-3.5-turbo", temperature=0.7)
    llm_client = OpenAIClient(config)
    
    agent = SmartAssistantAgent(
        llm_client=llm_client,
        system_prompt="You are a helpful AI assistant."
    )
    
    print("🤖 Smart Assistant Ready! (Type 'quit' to exit)")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() == 'quit':
            break
        
        result = await agent.process(IAgentInput(message=user_input))
        print(f"🤖 Agent: {result['response']}")
        print(f"📊 [{result['metadata']['complexity']} | {result['metadata']['processing_time']}s]")

if __name__ == "__main__":
    asyncio.run(main())
```

</details>

## Key Concepts Demonstrated

### 1. **Direct Instantiation**
```python
# You create everything explicitly
llm_client = OpenAIClient(config)
agent = SmartAssistantAgent(llm_client, prompt)
```

### 2. **Single Responsibility**
```python
# Agent has ONE clear purpose: smart assistance with adaptive complexity
class SmartAssistantAgent(BaseAgent):
    """Smart assistant that adapts to request complexity."""
```

### 3. **Dependency Injection**
```python
# Dependencies passed in constructor
def __init__(self, llm_client, system_prompt: str, complexity_threshold: float):
    super().__init__(llm_client, system_prompt)  # Explicit
    self.complexity_threshold = complexity_threshold
```

### 4. **Stateless Design**
```python
# No internal state - same input = same output
async def process(self, input: IAgentInput) -> Dict[str, Any]:
    # All state comes from input parameters
```

### 5. **Three-Layer Architecture**
- **Layer 1**: `OpenAIClient` - LLM access
- **Layer 2**: `SmartAssistantAgent` - Business logic
- **Layer 3**: Your application - Orchestration

## Next Steps

Now that you've built your first custom agent:

1. **Experiment** - Try different complexity analysis strategies
2. **Add Tools** - Integrate more external capabilities  
3. **Compose Systems** - Combine multiple agents
4. **Explore Layers** - Learn [Layer 3: Systems](../02-layer-guides/layer3-systems/)
5. **Test Thoroughly** - Add comprehensive unit tests

## Testing Your Agent

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_smart_agent():
    # Mock LLM client
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = {"llm_response": "Test response"}
    
    # Create agent with mock
    agent = SmartAssistantAgent(mock_llm, "Test prompt")
    
    # Test processing
    result = await agent.process(IAgentInput(message="Hello"))
    
    # Verify structure
    assert "response" in result
    assert "metadata" in result
    assert result["metadata"]["complexity"] == "simple"
```

---

*Congratulations! You've built a sophisticated agent that demonstrates all the key principles of Arshai's architecture. You now have complete control over an AI component that can adapt, use tools, and provide rich metadata.* 🎉

*Ready for more advanced patterns? Check out [Layer 3: Systems](../02-layer-guides/layer3-systems/) →*