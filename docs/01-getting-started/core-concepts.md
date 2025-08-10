# Core Concepts

Understanding these core concepts will help you build powerful AI applications with Arshai. Let's explore the fundamental building blocks.

## The Three Layers

Arshai organizes components into three layers with increasing developer authority:

```
Layer 3: Agentic Systems (You orchestrate)
    ↓
Layer 2: Agents (You customize)
    ↓  
Layer 1: LLM Clients (You configure)
```

## Layer 1: LLM Clients

### What They Are
LLM clients are the foundation - they provide access to language models like GPT-4, Gemini, and Claude.

### Key Characteristics
- **Standardized interface** - All clients implement `ILLM` protocol
- **Direct instantiation** - You create them explicitly
- **Environment configuration** - API keys from environment variables
- **Provider abstraction** - Switch providers without changing code

### Example
```python
from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput

# You create the client
config = ILLMConfig(model="gpt-4", temperature=0.7)
client = OpenAIClient(config)

# Standardized usage
input_data = ILLMInput(
    system_prompt="You are helpful",
    user_message="Hello!"
)
response = await client.chat(input_data)
```

### Available Clients
- `OpenAIClient` - OpenAI GPT models
- `GeminiClient` - Google Gemini models
- `AzureClient` - Azure OpenAI service
- `OpenRouterClient` - OpenRouter proxy service

## Layer 2: Agents

### What They Are
Agents wrap LLM clients with purpose-driven logic and business rules. They're the "brains" of your application.

### Key Characteristics
- **Stateless design** - No internal state, pure functions
- **Inherit from BaseAgent** - Consistent structure
- **Implement process()** - Your custom logic
- **Flexible returns** - Return any type you need

### Example
```python
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput

class AnalysisAgent(BaseAgent):
    """Analyzes text sentiment."""
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # Your custom logic
        llm_input = ILLMInput(
            system_prompt="Analyze sentiment",
            user_message=input.message
        )
        
        result = await self.llm_client.chat(llm_input)
        
        # Return structured data
        return {
            "sentiment": self._extract_sentiment(result),
            "confidence": 0.95
        }
```

### Agent Lifecycle
```python
# 1. Create with dependencies
agent = AnalysisAgent(llm_client, system_prompt)

# 2. Process inputs
result = await agent.process(IAgentInput(message="text"))

# 3. Agents are stateless - same input = same output
```

## Layer 3: Agentic Systems

### What They Are
Systems orchestrate multiple agents and components into complex applications.

### Key Characteristics
- **Maximum control** - You design the architecture
- **Component composition** - Combine agents, memory, tools
- **Workflow management** - Define execution paths
- **State management** - Handle complex state

### Example
```python
class DocumentProcessingSystem:
    """Complex system with multiple agents."""
    
    def __init__(self):
        # You compose the system
        llm = OpenAIClient(config)
        self.extractor = ExtractorAgent(llm)
        self.analyzer = AnalyzerAgent(llm)
        self.summarizer = SummarizerAgent(llm)
        
    async def process_document(self, doc: str):
        # You control the flow
        extracted = await self.extractor.process(doc)
        analysis = await self.analyzer.process(extracted)
        summary = await self.summarizer.process(analysis)
        return summary
```

## Core Interfaces

### ILLMInput
Input for LLM clients:
```python
llm_input = ILLMInput(
    system_prompt="System instructions",
    user_message="User's message",
    regular_functions={"tool": func},  # Optional tools
    background_tasks={"task": task},   # Optional background
    structure_type=ResponseModel,      # Optional structure
    max_turns=3                        # Multi-turn limit
)
```

### IAgentInput
Input for agents:
```python
agent_input = IAgentInput(
    message="User message",
    conversation_id="optional-id",
    metadata={                      # Optional context
        "user_id": "123",
        "context": {...}
    }
)
```

### ILLMConfig
Configuration for LLM clients:
```python
config = ILLMConfig(
    model="gpt-4",              # Model name
    temperature=0.7,            # Creativity (0-1)
    max_tokens=150,            # Response length
    top_p=0.9,                 # Nucleus sampling
    frequency_penalty=0.0,      # Reduce repetition
    presence_penalty=0.0,       # Encourage variety
    seed=42                    # Reproducibility
)
```

## Tools and Function Calling

### What Are Tools?
Tools extend agent capabilities with external functions.

### How They Work
```python
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

def calculate(expr: str) -> float:
    """Calculate mathematical expression."""
    return eval(expr)  # Use safe eval in production

# Add tools to LLM input
llm_input = ILLMInput(
    system_prompt="You can search and calculate",
    user_message="What's the weather in Paris?",
    regular_functions={
        "search_web": search_web,
        "calculate": calculate
    }
)

# LLM automatically calls tools as needed
result = await llm_client.chat(llm_input)
```

### Background Tasks
Fire-and-forget functions that don't return results:
```python
def log_interaction(action: str, details: str = ""):
    """BACKGROUND TASK: Log user interaction."""
    print(f"Logged: {action} - {details}")

llm_input = ILLMInput(
    system_prompt="Assistant with logging",
    user_message="Hello",
    background_tasks={
        "log_interaction": log_interaction
    }
)
```

## Memory Management

### Types of Memory

#### Working Memory
Short-term conversation context:
```python
from arshai.memory.working_memory import InMemoryManager

memory = InMemoryManager(ttl=3600)  # 1 hour TTL
await memory.store(conversation_id, context)
context = await memory.retrieve(conversation_id)
```

#### Long-term Memory
Persistent storage (Redis, databases):
```python
from arshai.memory.working_memory import RedisMemoryManager

memory = RedisMemoryManager(
    url="redis://localhost:6379",
    ttl=86400  # 24 hours
)
```

### Memory in Agents
```python
class MemoryAgent(BaseAgent):
    def __init__(self, llm_client, memory_manager):
        super().__init__(llm_client, "prompt")
        self.memory = memory_manager
    
    async def process(self, input: IAgentInput):
        # Retrieve context
        context = await self.memory.retrieve(
            input.conversation_id
        )
        
        # Process with context
        enhanced_input = f"Context: {context}\n{input.message}"
        
        # Store new context
        await self.memory.store(
            input.conversation_id,
            new_context
        )
```

## Workflows

### What Are Workflows?
Workflows orchestrate multiple agents in a defined execution path.

### Basic Workflow
```python
from arshai.workflows import Workflow

workflow = Workflow()

# Add nodes (agents)
workflow.add_node("analyze", analyzer_agent)
workflow.add_node("summarize", summarizer_agent)

# Define flow
workflow.add_edge("analyze", "summarize")

# Execute
result = await workflow.run(input_data)
```

## Error Handling

### Best Practices
```python
class RobustAgent(BaseAgent):
    async def process(self, input: IAgentInput):
        try:
            # Main logic
            result = await self._process_internal(input)
            return result
            
        except ValueError as e:
            # Handle specific errors
            return f"Invalid input: {e}"
            
        except Exception as e:
            # Log and handle gracefully
            self.logger.error(f"Processing failed: {e}")
            return "An error occurred. Please try again."
```

## Testing

### Testing Agents
```python
from unittest.mock import Mock, AsyncMock

def test_agent():
    # Mock dependencies
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = {
        "llm_response": "Test response"
    }
    
    # Create agent with mocks
    agent = MyAgent(mock_llm, "test prompt")
    
    # Test behavior
    result = await agent.process(test_input)
    assert result == expected
    mock_llm.chat.assert_called_once()
```

## Key Principles

### 1. Direct Instantiation
```python
# You create everything explicitly
client = OpenAIClient(config)
agent = MyAgent(client, prompt)
```

### 2. Dependency Injection
```python
# Dependencies passed in, not created internally
def __init__(self, llm: ILLM, memory: IMemory):
    self.llm = llm
    self.memory = memory
```

### 3. Stateless Design
```python
# No internal state
async def process(self, input):
    # Pure function - same input = same output
    return transform(input)
```

### 4. Composition Over Inheritance
```python
# Compose complex behavior from simple parts
system = System(
    agents=[agent1, agent2],
    memory=memory,
    tools=tools
)
```

## Common Patterns

### Pattern: Multi-Agent System
```python
class MultiAgentSystem:
    def __init__(self, agents: List[IAgent]):
        self.agents = agents
    
    async def process_all(self, input_data):
        results = await asyncio.gather(*[
            agent.process(input_data) 
            for agent in self.agents
        ])
        return self.combine_results(results)
```

### Pattern: Retry Logic
```python
async def process_with_retry(agent, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await agent.process(input_data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

### Pattern: Response Caching
```python
from functools import lru_cache

class CachedAgent(BaseAgent):
    @lru_cache(maxsize=100)
    async def process(self, input: IAgentInput):
        # Expensive operation cached
        return await super().process(input)
```

## Summary

These core concepts form the foundation of Arshai:

- **LLM Clients** - Standardized access to language models
- **Agents** - Purpose-driven logic wrappers
- **Systems** - Complex orchestrations
- **Direct Instantiation** - You create components
- **Dependency Injection** - Explicit dependencies
- **Stateless Design** - Pure, testable functions
- **Composition** - Build complex from simple

## Next Steps

Now that you understand the concepts:

1. **[First Agent](first-agent.md)** - Build a custom agent
2. **[Layer Guides](../02-layer-guides/)** - Deep dive into each layer
3. **[Patterns](../03-patterns/)** - Learn best practices
4. **[Tutorials](../05-tutorials/)** - Build complete applications

---

*Ready to build your first custom agent? Continue to [First Agent](first-agent.md) →*