# Arshai Agent Architecture

This document provides a comprehensive overview of the Arshai Agent architecture, designed for building intelligent, composable AI components with maximum flexibility and minimal complexity.

## Architecture Overview

The Arshai Agent system is built on **clean architecture principles** with a focus on simplicity, developer empowerment, and focused responsibility. Agents are lightweight wrappers over LLM clients that add specific purpose and behavior.

### **Architectural Philosophy & Core Decisions**

The agent architecture represents carefully considered design decisions focused on creating an **intelligent component system** with these foundational requirements:

- **Minimal Interface**: Single method to implement, maximum flexibility in return types
- **Stateless Design**: All state managed externally for scalability and simplicity
- **Composition Over Inheritance**: Build complex behaviors by combining simple agents
- **Developer Freedom**: No framework constraints on implementation patterns

#### **Why This Architecture Was Chosen**

**1. Single Abstract Method Approach**
- **Decision Rationale**: After evaluating multiple patterns, chose minimal interface with just `process()` method
- **Developer Benefit**: Complete freedom in implementation while maintaining consistent interface
- **Flexibility**: Return any type - strings, dicts, streams, custom objects

**2. Stateless Agent Design**
- **Problem Solved**: State management complexity, synchronization issues, scaling challenges
- **Solution**: All state managed externally through metadata and storage systems
- **Benefits**: Horizontal scaling, pure functions, simplified testing

**3. Tool Integration Without Framework**
- **Philosophy**: Agents know best how to use their tools
- **Implementation**: No forced patterns - agents manage tools internally
- **Result**: Maximum flexibility for different use cases

### System Architecture Diagram

```mermaid
graph TB
    subgraph "Application Layer"
        A[User Request] --> B[IAgentInput]
        B --> C[message: str]
        B --> D[metadata: Dict]
    end
    
    subgraph "Agent Layer"
        E[BaseAgent] --> F{process()}
        F --> G[Simple Response]
        F --> H[Structured Data]
        F --> I[Streaming]
        F --> J[Custom Objects]
    end
    
    subgraph "Integration Layer"
        K[LLM Client]
        L[Memory Manager]
        M[Tools/Functions]
        N[Other Agents]
    end
    
    subgraph "Infrastructure"
        O[Redis]
        P[Vector DB]
        Q[External APIs]
    end
    
    B --> E
    E --> K
    E --> L
    E --> M
    E --> N
    L --> O
    M --> P
    M --> Q
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style K fill:#e8f5e8
    style O fill:#ffebee
```

### Core Design Philosophy

- **Simplicity Over Complexity**: Minimal interface, maximum capability
- **Developer Empowerment**: Give developers complete control over implementation
- **Focused Responsibility**: Each agent does one thing well
- **Composition-First**: Complex behaviors through agent composition
- **Framework Agnostic**: No forced patterns or heavy abstractions

## Architecture Layers

### 1. Interface Layer

The unified interface provides consistency with complete flexibility:

```python
class IAgent(Protocol):
    async def process(self, input: IAgentInput) -> Any:
        """Process input and return response in any format."""
        ...

class IAgentInput(BaseModel):
    message: str                              # User message
    metadata: Optional[Dict[str, Any]] = None # Flexible context
```

**Key Design Decisions**:
- `Any` return type for complete flexibility
- Metadata field for passing context without API changes
- Single method to implement - no complexity

### 2. Base Implementation Layer

```python
class BaseAgent(IAgent, ABC):
    def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
        self.llm_client = llm_client      # Reference to LLM client
        self.system_prompt = system_prompt # Agent's behavior definition
        self.config = kwargs              # Additional configuration
    
    @abstractmethod
    async def process(self, input: IAgentInput) -> Any:
        """Must be implemented by all agents."""
        ...
```

**Framework Provides**:
- LLM client reference management
- System prompt storage
- Configuration handling

**Developer Implements**:
- The `process()` method with chosen logic
- Return type decision
- Tool integration approach

### 3. Specialized Agents Layer

Built-in agents demonstrate patterns without enforcing them:

#### WorkingMemoryAgent
- **Purpose**: Manage conversation memory
- **Pattern**: Background task for memory updates
- **Returns**: Status strings ("success"/"error: description")

#### ConversationAgent
- **Purpose**: Interactive conversations
- **Pattern**: Integrates memory, tools, streaming
- **Returns**: Flexible based on configuration

### 4. Composition Layer

Agents naturally compose through the LLM function calling system:

```python
# Agent as Tool
async def search(query: str) -> str:
    result = await rag_agent.process(IAgentInput(message=query))
    return str(result)

# Agent as Background Task
async def update_memory(content: str) -> None:
    await memory_agent.process(IAgentInput(
        message=content,
        metadata={"conversation_id": "123"}
    ))

# Use in LLM
llm_input = ILLMInput(
    system_prompt="...",
    user_message="...",
    regular_functions={"search": search},
    background_tasks={"update_memory": update_memory}
)
```

## Key Architectural Patterns

### Pattern 1: Stateless Agents

**Principle**: No internal state storage

```python
# ✅ CORRECT - External state
async def process(self, input: IAgentInput) -> str:
    context = input.metadata.get("context", {})
    # Use context, don't store it
    
# ❌ WRONG - Internal state
class StatefulAgent(BaseAgent):
    def __init__(self, ...):
        self.history = []  # ❌ NO!
```

**Benefits**:
- Horizontal scaling
- Simplified testing
- No synchronization issues
- Pure function behavior

### Pattern 2: Flexible Return Types

**Principle**: Agents choose their return format

```python
# String for simple responses
async def process(self, input: IAgentInput) -> str:
    return "response"

# Dictionary for structured data
async def process(self, input: IAgentInput) -> Dict[str, Any]:
    return {"response": "...", "metadata": {...}}

# Generator for streaming
async def process(self, input: IAgentInput):
    async for chunk in source:
        yield chunk

# Custom object for domain needs
async def process(self, input: IAgentInput) -> DomainObject:
    return DomainObject(...)
```

### Pattern 3: Tool Integration

**Principle**: Agents manage tools internally

```python
async def process(self, input: IAgentInput) -> Any:
    # Agent decides tool usage
    if self._needs_tools(input):
        tools = self._prepare_tools()
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions=tools
        )
    else:
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
    
    return await self.llm_client.chat(llm_input)
```

### Pattern 4: Agent Composition

**Principle**: Agents as building blocks

```python
class OrchestratorAgent(BaseAgent):
    def __init__(self, llm_client, prompt, agents: Dict[str, IAgent]):
        super().__init__(llm_client, prompt)
        self.agents = agents
    
    async def process(self, input: IAgentInput) -> Any:
        # Compose agent capabilities
        functions = {}
        for name, agent in self.agents.items():
            async def agent_func(msg: str, agent=agent) -> str:
                result = await agent.process(IAgentInput(message=msg))
                return str(result)
            functions[name] = agent_func
        
        # Use composed functions
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions=functions
        )
        
        return await self.llm_client.chat(llm_input)
```

## Design Decisions & Rationale

### 1. Why `Any` Return Type?

**Decision**: The `process()` method returns `Any`

**Rationale**:
- **Flexibility**: Different agents need different return types
- **Evolution**: Requirements change without breaking interface
- **Simplicity**: No complex generic type system
- **Pragmatism**: Python's dynamic nature suits this approach

**Alternative Considered**: Generic types `IAgent[T]`
- **Rejected Because**: Added complexity without real benefit
- **Trade-off**: Less type safety for more flexibility

### 2. Why Stateless Design?

**Decision**: Agents cannot store state internally

**Rationale**:
- **Scalability**: Any agent instance can handle any request
- **Testing**: Pure functions are easier to test
- **Simplicity**: No state synchronization complexity
- **Reliability**: No state corruption issues

**Alternative Considered**: Stateful agents with session management
- **Rejected Because**: Complexity outweighs benefits
- **Trade-off**: External state management for simplicity

### 3. Why No Tool Framework?

**Decision**: No enforced tool integration pattern

**Rationale**:
- **Flexibility**: Different agents need different tool patterns
- **Simplicity**: No complex tool registration system
- **Natural Integration**: LLM function calling is sufficient
- **Developer Control**: Agents know best how to use tools

**Alternative Considered**: Formal tool registration system
- **Rejected Because**: Adds complexity without clear benefit
- **Trade-off**: Less standardization for more flexibility

### 4. Why Metadata Field?

**Decision**: `IAgentInput` includes optional metadata dictionary

**Rationale**:
- **Extensibility**: Add context without API changes
- **Flexibility**: Different agents need different context
- **Backward Compatibility**: Optional field doesn't break existing code
- **Simplicity**: Dictionary is universally understood

**Alternative Considered**: Strongly typed context objects
- **Rejected Because**: Too rigid for diverse use cases
- **Trade-off**: Less type safety for more flexibility

## Integration with Framework

### LLM System Integration

Agents leverage the full LLM system capabilities:

```python
# Function Calling
llm_input = ILLMInput(
    system_prompt=self.system_prompt,
    user_message=input.message,
    regular_functions={"tool": tool_func},
    background_tasks={"task": task_func}
)

# Structured Output
llm_input = ILLMInput(
    system_prompt=self.system_prompt,
    user_message=input.message,
    structure_type=ResponseModel
)

# Streaming
async for chunk in self.llm_client.stream(llm_input):
    yield process_chunk(chunk)
```

### Memory System Integration

Agents work with memory managers:

```python
# Store memory
await self.memory_manager.store({
    "conversation_id": conv_id,
    "working_memory": memory_content
})

# Retrieve memory
memory = await self.memory_manager.retrieve({
    "conversation_id": conv_id
})
```

### Observability Integration

Agents inherit observability from LLM clients:

```python
# LLM client tracks metrics
result = await self.llm_client.chat(llm_input)
# Metrics automatically collected:
# - Token usage
# - Latency
# - Function calls
# - Errors

# Agent decides what to expose
return {
    "response": result['llm_response'],
    "usage": result['usage']  # Optional
}
```

## Performance Considerations

### Scalability

**Horizontal Scaling**: Stateless design enables unlimited scaling
```python
# Any instance can handle any request
agent_pool = [AgentInstance() for _ in range(N)]
await random.choice(agent_pool).process(input)
```

**Vertical Scaling**: Async design maximizes resource usage
```python
# Process multiple requests concurrently
results = await asyncio.gather(*[
    agent.process(input) for input in inputs
])
```

### Latency Optimization

**Streaming Support**: Progressive response generation
```python
async def process(self, input: IAgentInput):
    # Start yielding immediately
    async for chunk in self.llm_client.stream(...):
        yield chunk  # No buffering
```

**Parallel Processing**: Concurrent tool execution
```python
# Tools execute in parallel via LLM client
regular_functions={
    "search": search_func,
    "analyze": analyze_func,
    "summarize": summarize_func
}
# All execute concurrently when called
```

### Resource Management

**Memory Efficiency**: No state accumulation
```python
# Each request is independent
# No memory leaks from state accumulation
# Garbage collection works effectively
```

**Connection Pooling**: Shared resources through clients
```python
# LLM clients manage connection pools
# Agents share client instances
# Efficient resource utilization
```

## Security Considerations

### Input Validation

Agents should validate input:
```python
async def process(self, input: IAgentInput) -> str:
    # Validate message
    if not input.message or len(input.message) > 10000:
        return "Error: Invalid input"
    
    # Sanitize if needed
    sanitized = self._sanitize(input.message)
```

### Secret Management

Never store secrets in agents:
```python
# ✅ CORRECT - Secrets in environment
api_key = os.environ.get("API_KEY")

# ❌ WRONG - Secrets in code
class BadAgent(BaseAgent):
    API_KEY = "secret123"  # ❌ NO!
```

### Access Control

Use metadata for authorization:
```python
async def process(self, input: IAgentInput) -> Any:
    # Check authorization
    user_role = input.metadata.get("user_role")
    if not self._is_authorized(user_role):
        return "Error: Unauthorized"
```

## Testing Strategies

### Unit Testing

Test agents in isolation:
```python
# Mock dependencies
mock_llm = AsyncMock()
mock_llm.chat.return_value = {"llm_response": "test"}

# Test agent
agent = YourAgent(mock_llm, "prompt")
result = await agent.process(IAgentInput(message="test"))

# Assert behavior
assert result == expected
```

### Integration Testing

Test with real components:
```python
# Use real LLM client
llm_client = OpenRouterClient(config)
agent = YourAgent(llm_client, "prompt")

# Test actual behavior
result = await agent.process(IAgentInput(message="test"))
assert validate_result(result)
```

### Performance Testing

Measure agent performance:
```python
# Concurrent load test
async def load_test(agent, num_requests=100):
    start = time.time()
    tasks = [
        agent.process(IAgentInput(message=f"test {i}"))
        for i in range(num_requests)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    return {
        "total_time": elapsed,
        "avg_time": elapsed / num_requests,
        "throughput": num_requests / elapsed
    }
```

## Future Considerations

### Potential Enhancements

1. **Agent Middleware**: Optional processing pipeline
2. **Agent Versioning**: Support for multiple versions
3. **Agent Discovery**: Runtime agent discovery mechanism
4. **Agent Metrics**: Dedicated agent-level metrics

### Maintaining Simplicity

Any future enhancements must:
- Maintain the simple interface
- Preserve stateless design
- Keep developer freedom
- Avoid framework complexity

## Examples and Implementation References

The agent system includes comprehensive examples covering all aspects of agent development:

### Quick Start Resources
- **`agent_quickstart.py`** - 5-minute interactive demo with real-time conversation
- **`agents_comprehensive_guide.py`** - Single-file tutorial covering all major patterns

### Focused Learning Examples
- **`01_basic_usage.py`** - Simple agent patterns and core concepts
- **`02_custom_agents.py`** - Specialized agents with structured output (sentiment analysis, translation, code review)
- **`03_memory_patterns.py`** - WorkingMemoryAgent usage with conversation context management
- **`04_tool_integration.py`** - Function calling patterns (regular functions + background tasks)
- **`05_agent_composition.py`** - Multi-agent orchestration (orchestrator, pipeline, mesh patterns)  
- **`06_testing_agents.py`** - Comprehensive testing strategies (unit, integration, performance, load)

### Example Coverage Matrix
| Pattern | Quick Start | Comprehensive | Focused Example |
|---------|-------------|---------------|-----------------|
| Basic Agent | ✅ | ✅ | 01_basic_usage.py |
| Custom Return Types | - | ✅ | 02_custom_agents.py |
| Memory Management | - | ✅ | 03_memory_patterns.py |
| Function Calling | - | ✅ | 04_tool_integration.py |
| Agent Orchestration | - | - | 05_agent_composition.py |
| Testing Strategies | - | ✅ | 06_testing_agents.py |
| Production Patterns | - | - | All examples |

All examples use OpenRouter client and demonstrate production-ready patterns with error handling, observability, and best practices.

## Summary

The Arshai Agent architecture prioritizes:

- **Simplicity**: Minimal interface, maximum capability
- **Flexibility**: Any return type, any implementation pattern
- **Composition**: Build complex from simple
- **Developer Freedom**: No framework constraints
- **Scalability**: Stateless design for horizontal scaling

Agents are lightweight, focused components that wrap LLM clients with purpose, not heavy frameworks that constrain implementation.