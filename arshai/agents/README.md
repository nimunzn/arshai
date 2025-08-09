# Arshai Agent Implementation Guide

This document provides critical implementation notes and step-by-step guidance for contributors and maintainers working with agents in the Arshai framework.

## ⚠️ CRITICAL IMPLEMENTATION NOTES

**READ THESE CAREFULLY BEFORE CREATING OR MODIFYING ANY AGENT**

### 1. BaseAgent Framework Compliance

**CRITICAL**: All agents MUST inherit from `BaseAgent` and implement exactly 1 abstract method:

#### Framework Implementation Requirements

```mermaid
graph LR
    A[Your Agent] --> B[Inherits BaseAgent]
    B --> C[Framework provides:<br/>• LLM client reference<br/>• System prompt management<br/>• Configuration storage]
    
    B --> D[You implement:<br/>• process() method]
    
    C --> E[✅ Consistent Interface]
    D --> F[✅ Custom Logic]
    
    E --> G[Production Ready]
    F --> G
    
    style A fill:#e1f5fe
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style G fill:#ccffcc
```

```python
@abstractmethod
async def process(self, input: IAgentInput) -> Any
```

**DO NOT**:
- Store state internally - agents are stateless by design
- Override `__init__` without calling `super().__init__()`
- Duplicate LLM client functionality - leverage existing capabilities

### 2. Return Type Flexibility

**CRITICAL PATTERN**: The `process()` method returns `Any` - this is intentional!

```python
# ✅ CORRECT - Simple string
async def process(self, input: IAgentInput) -> str:
    return "response"

# ✅ CORRECT - Structured data
async def process(self, input: IAgentInput) -> Dict[str, Any]:
    return {"response": "text", "metadata": {...}}

# ✅ CORRECT - Streaming
async def process(self, input: IAgentInput):
    async for chunk in self.llm_client.stream(llm_input):
        yield chunk

# ✅ CORRECT - Custom object
async def process(self, input: IAgentInput) -> CustomResponse:
    return CustomResponse(...)
```

### 3. Stateless Design

**MANDATORY**: All agents MUST be stateless. State is managed externally through:

```python
# ✅ CORRECT - Context through metadata
input = IAgentInput(
    message="Process this",
    metadata={"conversation_id": "123", "context": {...}}
)

# ❌ WRONG - Internal state
class BadAgent(BaseAgent):
    def __init__(self, ...):
        self.conversation_history = []  # ❌ NO INTERNAL STATE
```

### 4. Tool Integration Pattern

**CRITICAL**: Agents manage tools internally without framework enforcement:

```python
# ✅ CORRECT - Agent decides tool usage
async def process(self, input: IAgentInput) -> Any:
    # Agent internally decides when/how to use tools
    if self._needs_tools(input.message):
        tools = {"search": self.search_func, "calculate": self.calc_func}
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
    
    result = await self.llm_client.chat(llm_input)
    return result.get('llm_response', '')
```

## Step-by-Step Implementation Guide

### Step 1: Create Your Agent Class

```python
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput, ILLM
from typing import Any, Optional, Dict

class YourAgent(BaseAgent):
    """
    Brief description of what your agent does.
    
    Capabilities:
    - List key capabilities
    - Be specific about functionality
    
    Requirements:
    - List any dependencies
    - External services needed
    
    Returns:
    - Describe return format (string, dict, stream, etc.)
    
    Example:
        agent = YourAgent(llm_client, "System prompt")
        result = await agent.process(IAgentInput(message="Test"))
    """
    
    def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
        """Initialize your agent."""
        super().__init__(llm_client, system_prompt, **kwargs)
        # Add any additional initialization here
        # BUT REMEMBER: NO INTERNAL STATE!
```

### Step 2: Implement the Process Method

```python
async def process(self, input: IAgentInput) -> Any:
    """
    Process the input and return response.
    
    Args:
        input: IAgentInput containing message and optional metadata
        
    Returns:
        Any: Your chosen return format
    """
    # Extract metadata if needed
    metadata = input.metadata or {}
    context = metadata.get("context", {})
    
    # Prepare LLM input
    llm_input = ILLMInput(
        system_prompt=self.system_prompt,
        user_message=input.message
    )
    
    # Call LLM
    result = await self.llm_client.chat(llm_input)
    
    # Return in your chosen format
    return result.get('llm_response', '')
```

### Step 3: Add Tool Support (Optional)

```python
async def process(self, input: IAgentInput) -> Dict[str, Any]:
    # Define tools as needed
    def search_knowledge(query: str) -> str:
        """Search the knowledge base."""
        # Implementation
        return "search results"
    
    def calculate(expression: str) -> float:
        """Calculate mathematical expression."""
        return eval(expression)  # Use safe eval in production
    
    # Add tools to LLM input
    llm_input = ILLMInput(
        system_prompt=self.system_prompt,
        user_message=input.message,
        regular_functions={
            "search_knowledge": search_knowledge,
            "calculate": calculate
        }
    )
    
    result = await self.llm_client.chat(llm_input)
    
    return {
        "response": result.get('llm_response', ''),
        "tools_used": list(result.get('function_calls', {}).keys())
    }
```

### Step 4: Support Streaming (Optional)

```python
async def process(self, input: IAgentInput):
    """Process with streaming support."""
    # Check if streaming is requested
    stream = input.metadata.get("stream", False) if input.metadata else False
    
    llm_input = ILLMInput(
        system_prompt=self.system_prompt,
        user_message=input.message
    )
    
    if stream:
        # Return async generator for streaming
        async for chunk in self.llm_client.stream(llm_input):
            if chunk.get('llm_response'):
                yield chunk['llm_response']
    else:
        # Regular response
        result = await self.llm_client.chat(llm_input)
        yield result.get('llm_response', '')
```

## Built-in Agents Reference

### BaseAgent

**Purpose**: Abstract base class for all agents

```python
class BaseAgent(IAgent, ABC):
    def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.config = kwargs
    
    @abstractmethod
    async def process(self, input: IAgentInput) -> Any:
        """Must be implemented by subclasses."""
        ...
```

### WorkingMemoryAgent

**Purpose**: Manages conversation working memory

**Key Features**:
- Fetches conversation history
- Generates memory summaries
- Stores in persistent storage
- Returns status: "success" or "error: <description>"

```python
# Usage
memory_agent = WorkingMemoryAgent(
    llm_client=llm_client,
    memory_manager=memory_manager,  # Optional
    chat_history_client=history_client  # Optional
)

result = await memory_agent.process(IAgentInput(
    message="User asked about pricing",
    metadata={"conversation_id": "123"}
))
# Returns: "success" or "error: no conversation_id provided"
```

**Implementation Notes**:
- Returns `"success"` when memory is successfully generated and stored
- Returns `"error: <description>"` for various failure conditions
- Requires `conversation_id` in metadata
- Works without memory_manager (generates but doesn't store)

## Agent Composition Patterns

### Agents as Tools

```python
class OrchestratorAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, system_prompt: str, 
                 rag_agent: IAgent, memory_agent: IAgent):
        super().__init__(llm_client, system_prompt)
        self.rag_agent = rag_agent
        self.memory_agent = memory_agent
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # Use other agents as tools
        async def search(query: str) -> str:
            result = await self.rag_agent.process(
                IAgentInput(message=query)
            )
            return str(result)  # Handle any return type
        
        # Use as background task
        async def update_memory(content: str) -> None:
            await self.memory_agent.process(
                IAgentInput(
                    message=content,
                    metadata={"conversation_id": "123"}
                )
            )
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions={"search": search},
            background_tasks={"update_memory": update_memory}
        )
        
        result = await self.llm_client.chat(llm_input)
        return {"response": result['llm_response'], "usage": result['usage']}
```

## Testing Your Agent

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock

class TestYourAgent:
    @pytest.mark.asyncio
    async def test_process(self):
        # Mock LLM client
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = {
            "llm_response": "Test response",
            "usage": {"tokens": 10}
        }
        
        # Create agent
        agent = YourAgent(mock_llm, "Test prompt")
        
        # Test processing
        result = await agent.process(IAgentInput(message="Test"))
        
        # Assertions
        assert result == "Test response"
        mock_llm.chat.assert_called_once()
```

### Integration Testing

```python
from arshai.llms.openrouter import OpenRouterClient
from arshai.core.interfaces.illm import ILLMConfig

@pytest.mark.asyncio
async def test_with_real_llm():
    # Use real LLM client
    config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.7)
    llm_client = OpenRouterClient(config)
    
    # Create and test agent
    agent = YourAgent(llm_client, "System prompt")
    result = await agent.process(IAgentInput(message="Test"))
    
    assert isinstance(result, str)
    assert len(result) > 0
```

## Best Practices

### 1. Single Responsibility
Each agent should have ONE clear purpose:
```python
# ✅ GOOD - Focused agent
class SentimentAnalysisAgent(BaseAgent):
    """Analyzes sentiment of text."""

# ❌ BAD - Kitchen sink agent
class DoEverythingAgent(BaseAgent):
    """Analyzes sentiment, manages memory, searches web, etc."""
```

### 2. Clear Documentation
Always document capabilities, requirements, and return types:
```python
class YourAgent(BaseAgent):
    """
    What it does.
    
    Capabilities:
    - Specific capability 1
    - Specific capability 2
    
    Returns:
        Dict[str, Any]: {"response": str, "metadata": dict}
    """
```

### 3. Error Handling
Handle errors gracefully:
```python
async def process(self, input: IAgentInput) -> str:
    try:
        # Your logic
        result = await self.llm_client.chat(llm_input)
        return result.get('llm_response', '')
    except Exception as e:
        self.logger.error(f"Processing failed: {e}")
        return f"Error: {str(e)}"
```

### 4. Metadata Usage
Use metadata for flexible context:
```python
async def process(self, input: IAgentInput) -> Any:
    metadata = input.metadata or {}
    
    # Extract various context
    conversation_id = metadata.get("conversation_id")
    stream = metadata.get("stream", False)
    max_tokens = metadata.get("max_tokens", 150)
    
    # Use context to control behavior
    if stream:
        return self._stream_response(input)
    else:
        return self._regular_response(input)
```

## Common Patterns

### Pattern 1: Simple Response Agent
```python
class SimpleAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result.get('llm_response', '')
```

### Pattern 2: Structured Output Agent
```python
class StructuredAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        from pydantic import BaseModel
        
        class Analysis(BaseModel):
            sentiment: str
            confidence: float
            keywords: list[str]
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            structure_type=Analysis
        )
        
        result = await self.llm_client.chat(llm_input)
        return result.get('llm_response', {}).dict()
```

### Pattern 3: Streaming Agent
```python
class StreamingAgent(BaseAgent):
    async def process(self, input: IAgentInput):
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        async for chunk in self.llm_client.stream(llm_input):
            if chunk.get('llm_response'):
                yield chunk['llm_response']
```

## Troubleshooting

### Issue: Agent not returning expected format
**Solution**: Remember that `process()` can return ANY type. Document your return type clearly.

### Issue: State not persisting between calls
**Solution**: Agents are stateless. Use external storage (Redis, database) and pass context via metadata.

### Issue: Tools not being called
**Solution**: Ensure tools are properly formatted for the LLM client. Check the LLM documentation for function format.

### Issue: Memory agent returning errors
**Solution**: Ensure `conversation_id` is provided in metadata. Check memory_manager is properly initialized.

## Contributing Guidelines

When contributing a new agent:

1. **Extend BaseAgent** - All agents must inherit from BaseAgent
2. **Implement process()** - This is the only required method
3. **Document thoroughly** - Include docstrings with capabilities and examples
4. **Add tests** - Both unit and integration tests
5. **Follow patterns** - Use established patterns from existing agents
6. **Keep it focused** - Single responsibility principle

## Examples

Complete working examples are available in `/examples/agents/`:

### Quick Start Examples
- `agent_quickstart.py` - 5-minute interactive getting started guide
- `agents_comprehensive_guide.py` - Single-file comprehensive tutorial

### Focused Topic Examples  
- `01_basic_usage.py` - Getting started with agents
- `02_custom_agents.py` - Creating custom agents with structured output
- `03_memory_patterns.py` - Working with WorkingMemoryAgent and memory patterns
- `04_tool_integration.py` - Function calling and tool integration
- `05_agent_composition.py` - Multi-agent orchestration and composition patterns
- `06_testing_agents.py` - Comprehensive testing strategies (unit, integration, performance)

## Support

For questions or issues:
- GitHub Issues: https://github.com/MobileTechLab/ArsHai/issues
- Documentation: See `/docs/technical/agent_architecture.md` for detailed architecture