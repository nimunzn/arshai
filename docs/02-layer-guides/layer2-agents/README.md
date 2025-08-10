# Layer 2: Agents

Agents are the second layer of the Arshai architecture. They wrap LLM clients with purpose-driven logic and business rules, providing moderate developer authority for customizing behavior while maintaining structure.

## What Are Agents?

Agents transform raw LLM capabilities into focused, task-specific functionality:

```python
# Layer 1: Raw LLM client
llm_client = OpenAIClient(config)
response = await llm_client.chat(input)

# Layer 2: Purpose-driven agent
agent = SentimentAnalyzer(llm_client, "Analyze sentiment")
analysis = await agent.process(text)  # Returns structured sentiment data
```

## Core Principles

### 1. Stateless Design
Agents don't maintain internal state - they're pure functions:

```python
class StatelessAgent(BaseAgent):
    # No internal state variables!
    
    async def process(self, input: IAgentInput) -> Any:
        # Same input always produces same output
        # State comes from external sources
        context = input.metadata.get("context", {})
        return await self._process_with_context(input.message, context)
```

### 2. Single Responsibility  
Each agent has one clear purpose:

```python
class SentimentAnalyzer(BaseAgent):
    """Analyzes text sentiment - does ONE thing well."""
    
class TextSummarizer(BaseAgent): 
    """Summarizes text - different responsibility."""
    
class MultiPurposeAgent(BaseAgent):
    """BAD: Does everything - violates single responsibility."""
```

### 3. Explicit Dependencies
All dependencies are visible in the constructor:

```python
class DocumentAgent(BaseAgent):
    def __init__(self, 
                 llm_client: ILLM,           # Required
                 system_prompt: str,         # Required  
                 document_store: IStorage,   # Explicit dependency
                 max_tokens: int = 150):     # Optional parameter
        super().__init__(llm_client, system_prompt)
        self.document_store = document_store
        self.max_tokens = max_tokens
```

## Creating Agents

### Step 1: Inherit from BaseAgent

```python
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput

class MyAgent(BaseAgent):
    """Custom agent for specific task."""
    
    async def process(self, input: IAgentInput) -> Any:
        """Implement your logic here."""
        # This is the only method you MUST implement
        pass
```

### Step 2: Implement the process() Method

```python
class AnalysisAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # 1. Extract any context from metadata
        metadata = input.metadata or {}
        user_context = metadata.get("user_context", {})
        
        # 2. Prepare LLM input
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=f"Context: {user_context}\nAnalyze: {input.message}"
        )
        
        # 3. Call LLM
        result = await self.llm_client.chat(llm_input)
        
        # 4. Return your chosen format
        return {
            "analysis": result.get("llm_response"),
            "confidence": self._calculate_confidence(result),
            "metadata": {"processed_at": datetime.utcnow().isoformat()}
        }
```

### Step 3: Add Custom Logic

```python
class SmartAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, system_prompt: str, 
                 complexity_threshold: float = 0.5):
        super().__init__(llm_client, system_prompt)
        self.complexity_threshold = complexity_threshold
    
    async def process(self, input: IAgentInput) -> str:
        # Your custom pre-processing
        complexity = self._assess_complexity(input.message)
        
        if complexity > self.complexity_threshold:
            # Use more sophisticated prompting
            enhanced_prompt = self._create_complex_prompt(input.message)
        else:
            # Simple processing
            enhanced_prompt = input.message
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=enhanced_prompt
        )
        
        result = await self.llm_client.chat(llm_input)
        
        # Your custom post-processing
        return self._post_process(result.get("llm_response"))
```

## Agent Patterns

### Pattern 1: Simple Response Agent

```python
class SimpleAgent(BaseAgent):
    """Returns a simple string response."""
    
    async def process(self, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "")
```

### Pattern 2: Structured Output Agent

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    keywords: List[str]

class StructuredAgent(BaseAgent):
    """Returns structured data using Pydantic models."""
    
    async def process(self, input: IAgentInput) -> AnalysisResult:
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            structure_type=AnalysisResult  # LLM returns structured data
        )
        
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response")  # Already an AnalysisResult
```

### Pattern 3: Streaming Agent

```python
class StreamingAgent(BaseAgent):
    """Streams responses in real-time."""
    
    async def process(self, input: IAgentInput):
        """Returns async generator for streaming."""
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        async for chunk in self.llm_client.stream(llm_input):
            if chunk.get("llm_response"):
                yield chunk["llm_response"]
```

### Pattern 4: Tool-Using Agent

```python
class ToolAgent(BaseAgent):
    """Agent with tool capabilities."""
    
    def __init__(self, llm_client: ILLM, system_prompt: str, tools: List[Callable]):
        super().__init__(llm_client, system_prompt)
        self.tools = {tool.__name__: tool for tool in tools}
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # Define tools for this request
        def search_web(query: str) -> str:
            """Search the web for information."""
            return self._search_implementation(query)
        
        def calculate(expression: str) -> float:
            """Calculate mathematical expression."""
            return self._safe_eval(expression)
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions={
                "search_web": search_web,
                "calculate": calculate
            }
        )
        
        result = await self.llm_client.chat(llm_input)
        
        return {
            "response": result.get("llm_response"),
            "tools_used": list(result.get("function_calls", {}).keys())
        }
```

### Pattern 5: Context-Aware Agent

```python
class ContextAgent(BaseAgent):
    """Agent that uses conversation context."""
    
    async def process(self, input: IAgentInput) -> str:
        # Extract context from metadata
        metadata = input.metadata or {}
        conversation_history = metadata.get("history", [])
        user_profile = metadata.get("user_profile", {})
        
        # Build context-aware prompt
        context_prompt = self._build_context_prompt(
            conversation_history, 
            user_profile
        )
        
        llm_input = ILLMInput(
            system_prompt=f"{self.system_prompt}\n\nContext: {context_prompt}",
            user_message=input.message
        )
        
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "")
```

## Agent Composition

### Combining Multiple Agents

```python
class MultiAgentProcessor:
    """Orchestrates multiple specialized agents."""
    
    def __init__(self, llm_client: ILLM):
        self.analyzer = SentimentAgent(llm_client)
        self.summarizer = SummarizerAgent(llm_client)
        self.categorizer = CategoryAgent(llm_client)
    
    async def process_document(self, text: str) -> Dict[str, Any]:
        # Run agents in parallel
        analysis, summary, category = await asyncio.gather(
            self.analyzer.process(IAgentInput(message=text)),
            self.summarizer.process(IAgentInput(message=text)),
            self.categorizer.process(IAgentInput(message=text))
        )
        
        return {
            "sentiment": analysis,
            "summary": summary,
            "category": category
        }
```

### Sequential Agent Chain

```python
class AgentChain:
    """Chains agents in sequence."""
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
    
    async def process(self, initial_input: str) -> Any:
        current_input = initial_input
        
        for agent in self.agents:
            result = await agent.process(IAgentInput(message=current_input))
            current_input = str(result)  # Pass output to next agent
        
        return current_input
```

## Testing Agents

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock

class TestMyAgent:
    @pytest.fixture
    def mock_llm(self):
        mock = AsyncMock()
        mock.chat.return_value = {
            "llm_response": "Test response",
            "usage": {"tokens": 10}
        }
        return mock
    
    @pytest.fixture
    def agent(self, mock_llm):
        return MyAgent(mock_llm, "Test prompt")
    
    @pytest.mark.asyncio
    async def test_process(self, agent, mock_llm):
        # Test input
        input_data = IAgentInput(message="Test message")
        
        # Process
        result = await agent.process(input_data)
        
        # Assertions
        assert result == "Test response"
        mock_llm.chat.assert_called_once()
        
        # Verify LLM input
        call_args = mock_llm.chat.call_args[0][0]
        assert call_args.user_message == "Test message"
        assert call_args.system_prompt == "Test prompt"
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_agent_with_real_llm():
    # Use real LLM for integration test
    config = ILLMConfig(model="gpt-3.5-turbo", temperature=0)
    llm_client = OpenAIClient(config)
    
    agent = MyAgent(llm_client, "You are a test assistant")
    
    result = await agent.process(
        IAgentInput(message="Say exactly: Integration test successful")
    )
    
    assert "Integration test successful" in result
```

### Testing with Different Scenarios

```python
@pytest.mark.parametrize("input_message,expected_type", [
    ("Analyze sentiment", str),
    ("Generate report", dict),
    ("Stream response", AsyncGenerator)
])
@pytest.mark.asyncio
async def test_different_inputs(agent, input_message, expected_type):
    result = await agent.process(IAgentInput(message=input_message))
    assert isinstance(result, expected_type)
```

## Best Practices

### 1. Keep It Focused
```python
# Good: Single, clear purpose
class SentimentAnalyzer(BaseAgent):
    """Analyzes text sentiment."""

# Bad: Multiple purposes  
class MultiTool(BaseAgent):
    """Analyzes sentiment, summarizes, translates, and more."""
```

### 2. Make Dependencies Explicit
```python
# Good: Clear dependencies
def __init__(self, llm_client: ILLM, database: IDatabase, cache: ICache):
    
# Bad: Hidden dependencies
def __init__(self, llm_client: ILLM):
    self.db = get_database()  # Hidden!
```

### 3. Handle Errors Gracefully
```python
async def process(self, input: IAgentInput) -> str:
    try:
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response")
    except Exception as e:
        self.logger.error(f"Processing failed: {e}")
        return "I apologize, but I encountered an error processing your request."
```

### 4. Document Your Agent
```python
class DocumentProcessor(BaseAgent):
    """
    Processes documents for analysis and extraction.
    
    Capabilities:
    - Extracts key information from documents
    - Summarizes content
    - Categorizes documents by type
    
    Returns:
        Dict with 'summary', 'category', and 'key_points' fields
    
    Example:
        agent = DocumentProcessor(llm_client, "Document analysis")
        result = await agent.process(IAgentInput(message=document_text))
        print(result['summary'])
    """
```

### 5. Use Type Hints
```python
from typing import Dict, List, Optional, Any

async def process(self, input: IAgentInput) -> Dict[str, Any]:
    # Type hints help with IDE support and debugging
```

## Common Patterns

### Validation Agent
```python
class ValidationAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> Dict[str, bool]:
        llm_input = ILLMInput(
            system_prompt="Validate the input according to rules",
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        
        return {
            "is_valid": "valid" in result.get("llm_response", "").lower(),
            "reason": result.get("llm_response")
        }
```

### Translation Agent
```python
class TranslationAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, target_language: str):
        super().__init__(
            llm_client, 
            f"Translate text to {target_language}"
        )
        self.target_language = target_language
    
    async def process(self, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt=f"Translate to {self.target_language}",
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "")
```

### Retry Agent Wrapper
```python
class RetryAgent:
    """Wrapper that adds retry logic to any agent."""
    
    def __init__(self, agent: BaseAgent, max_retries: int = 3):
        self.agent = agent
        self.max_retries = max_retries
    
    async def process(self, input: IAgentInput, attempt: int = 0) -> Any:
        try:
            return await self.agent.process(input)
        except Exception as e:
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                return await self.process(input, attempt + 1)
            raise
```

## Next Steps

- **[Layer 3: Systems](../layer3-systems/)** - Orchestrate multiple agents
- **[Built-in Agents](../../04-components/agents.md)** - Explore available agents
- **[Testing Patterns](../../03-patterns/)** - Advanced testing strategies
- **[Tutorials](../../05-tutorials/)** - Build complete applications

---

*Agents give you the power to wrap raw AI capabilities with focused, testable business logic. You control the behavior while the framework handles the infrastructure.*