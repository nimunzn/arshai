# Three-Layer Architecture

The three-layer architecture is the foundational design pattern of Arshai. It organizes components into three distinct layers, each with increasing levels of developer authority and control.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 3: Agentic Systems                      │
│                  (Maximum Developer Authority)                   │
│                                                                   │
│   • Workflows      • Orchestration     • State Management        │
│   • Memory         • Tool Integration  • Complex Patterns        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Composes
┌───────────────────────────▼─────────────────────────────────────┐
│                      Layer 2: Agents                             │
│                  (Moderate Developer Authority)                  │
│                                                                   │
│   • BaseAgent      • Custom Logic      • Process Methods         │
│   • Stateless      • Tool Usage        • Response Formatting     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Uses
┌───────────────────────────▼─────────────────────────────────────┐
│                    Layer 1: LLM Clients                          │
│                  (Minimal Developer Authority)                   │
│                                                                   │
│   • OpenAI         • Gemini           • Azure                    │
│   • Standardized   • Environment Vars  • Streaming               │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1: LLM Clients (Foundation)

### Purpose
Provide standardized access to various LLM providers with consistent interfaces.

### Characteristics
- **Minimal Developer Authority**: Standard patterns, consistent behavior
- **Environment Configuration**: API keys from environment variables
- **Unified Interface**: All clients implement `ILLM` protocol
- **Provider Abstraction**: Switch providers without changing code

### Example
```python
from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput

# Direct instantiation - you create it
config = ILLMConfig(model="gpt-4", temperature=0.7)
llm_client = OpenAIClient(config)

# Standardized usage
llm_input = ILLMInput(
    system_prompt="You are a helpful assistant",
    user_message="Hello, how are you?"
)
response = await llm_client.chat(llm_input)
```

### What You Control
- When to create the client
- Which provider to use
- Configuration parameters
- Request timing and error handling

### What's Standardized
- Interface methods (`chat`, `stream`)
- Response format
- Environment variable patterns
- Usage tracking

## Layer 2: Agents (Logic)

### Purpose
Wrap LLM clients with purpose-driven logic and business rules.

### Characteristics
- **Moderate Developer Authority**: Custom logic within framework
- **Stateless Design**: No internal state, pure functions
- **Dependency Injection**: Explicit dependencies in constructor
- **Flexible Return Types**: Return anything (string, dict, generator)

### Example
```python
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput

class AnalysisAgent(BaseAgent):
    """Custom agent for data analysis."""
    
    def __init__(self, llm_client: ILLM, system_prompt: str):
        # Explicit dependencies
        super().__init__(llm_client, system_prompt)
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # Your custom logic here
        analysis = await self._analyze_data(input.message)
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=f"Analyze: {analysis}"
        )
        
        result = await self.llm_client.chat(llm_input)
        
        # Return structured data
        return {
            "analysis": analysis,
            "insights": result["llm_response"],
            "confidence": self._calculate_confidence(result)
        }
```

### What You Control
- Agent logic and behavior
- Input processing
- Output formatting
- Tool integration
- Error handling strategies

### What's Standardized
- Base class structure (`BaseAgent`)
- `process()` method signature
- Stateless design pattern

## Layer 3: Agentic Systems (Orchestration)

### Purpose
Compose agents and components into complex, multi-step systems.

### Characteristics
- **Maximum Developer Authority**: Complete control over flow
- **Component Composition**: Combine multiple agents
- **State Management**: Handle complex state across components
- **Workflow Orchestration**: Define execution paths

### Example
```python
from arshai.workflows import Workflow, WorkflowConfig
from arshai.memory import MemoryManager

class DataPipelineSystem:
    """Complex system orchestrating multiple agents."""
    
    def __init__(self):
        # You create all components
        llm_config = ILLMConfig(model="gpt-4")
        llm_client = OpenAIClient(llm_config)
        
        # You compose the system
        self.extractor = DataExtractorAgent(llm_client, "Extract data")
        self.analyzer = AnalysisAgent(llm_client, "Analyze patterns")
        self.reporter = ReportAgent(llm_client, "Generate reports")
        
        # You manage state
        self.memory = MemoryManager()
        
        # You define the workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Define how components work together."""
        workflow = Workflow()
        
        # You control the flow
        workflow.add_node("extract", self.extractor)
        workflow.add_node("analyze", self.analyzer)
        workflow.add_node("report", self.reporter)
        
        workflow.add_edge("extract", "analyze")
        workflow.add_edge("analyze", "report")
        
        return workflow
    
    async def process_data(self, data: str) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        # You control execution
        context = {"input": data}
        
        # Store in memory
        await self.memory.store(context)
        
        # Run workflow
        result = await self.workflow.run(context)
        
        # You control the output
        return {
            "report": result["report"],
            "metadata": self._generate_metadata(result),
            "memory_id": context.get("memory_id")
        }
```

### What You Control
- Everything! Complete orchestration control
- Component creation and lifecycle
- Execution flow and branching
- State management strategies
- Error recovery and retries
- Performance optimization

### What's Standardized
- Basic interfaces (optional to use)
- Common patterns (you can ignore)

## Developer Authority Progression

### Layer 1 → Layer 2
```python
# Layer 1: Use standard client
llm_client = OpenAIClient(config)

# Layer 2: Wrap with custom logic
agent = CustomAgent(llm_client, custom_prompt)
```

### Layer 2 → Layer 3
```python
# Layer 2: Individual agents
agent1 = AnalysisAgent(llm_client)
agent2 = SummaryAgent(llm_client)

# Layer 3: Orchestrate into system
system = MultiAgentSystem([agent1, agent2], memory, tools)
```

## Key Benefits

### 1. **Clear Separation of Concerns**
- Layer 1: Provider integration
- Layer 2: Business logic
- Layer 3: System orchestration

### 2. **Progressive Complexity**
- Start simple with Layer 1
- Add logic with Layer 2
- Build systems with Layer 3

### 3. **Maximum Flexibility**
- Use only what you need
- Skip layers if not required
- Mix and match components

### 4. **Testability**
```python
# Each layer is independently testable
def test_llm_client():
    client = MockLLMClient()  # Test Layer 1
    
def test_agent():
    mock_client = Mock()
    agent = MyAgent(mock_client)  # Test Layer 2
    
def test_system():
    mock_agents = [Mock(), Mock()]
    system = System(mock_agents)  # Test Layer 3
```

## Anti-Patterns to Avoid

### ❌ Don't Skip Dependency Injection
```python
# Bad: Hidden dependencies
class BadAgent(BaseAgent):
    def __init__(self):
        self.llm_client = OpenAIClient()  # Hidden creation
```

### ✅ Do Use Explicit Dependencies
```python
# Good: Explicit dependencies
class GoodAgent(BaseAgent):
    def __init__(self, llm_client: ILLM):
        self.llm_client = llm_client  # Injected
```

### ❌ Don't Mix Layer Responsibilities
```python
# Bad: LLM client doing agent logic
class BadLLMClient(BaseLLMClient):
    def analyze_sentiment(self, text):  # Wrong layer!
        pass
```

### ✅ Do Keep Layers Focused
```python
# Good: Each layer has its role
class LLMClient(BaseLLMClient):  # Layer 1: Just LLM access
    pass

class SentimentAgent(BaseAgent):  # Layer 2: Analysis logic
    def analyze_sentiment(self, text):
        pass
```

## Summary

The three-layer architecture provides:
- **Structure** without rigidity
- **Standards** without constraints  
- **Patterns** without enforcement
- **Power** with simplicity

You're always in control, with the framework providing building blocks rather than prescriptive patterns.

## Next Steps

- Read [Developer Authority](developer-authority.md) to understand the philosophy
- Check [Design Decisions](design-decisions.md) for architectural choices
- Start building with [Getting Started](../01-getting-started/)