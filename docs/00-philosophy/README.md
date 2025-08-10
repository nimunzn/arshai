# Philosophy & Architecture

This section explains the core philosophy and architectural principles behind Arshai. Understanding these concepts will help you make the most of the framework and understand why it's designed the way it is.

## Core Documents

### 📐 [Three-Layer Architecture](three-layer-architecture.md)
The foundational architecture pattern that organizes components into three layers with increasing developer authority:
- **Layer 1**: LLM Clients - Core AI components
- **Layer 2**: Agents - Purpose-driven wrappers
- **Layer 3**: Agentic Systems - Orchestration layer

### 🎯 [Developer Authority](developer-authority.md)
Why developers should have complete control over their components:
- Direct instantiation over factory patterns
- Explicit dependencies over hidden configuration
- Developer empowerment over framework control

### 🔧 [Design Decisions](design-decisions.md)
The reasoning behind key architectural choices:
- Why we removed the Settings pattern
- Why we chose direct instantiation
- Why agents are stateless
- Why we use interface-first design

## Key Principles

### 1. **Developer Control**
You explicitly create and configure every component. No hidden factories, no magic configuration, no forced patterns.

```python
# You control everything
llm_client = OpenAIClient(config)  # You create it
agent = MyAgent(llm_client, prompt)  # You inject dependencies
system = Orchestrator(agents)  # You compose systems
```

### 2. **Progressive Authority**
As you move up layers, you gain more control:
- **Layer 1**: Standard interfaces, environment variables
- **Layer 2**: Custom logic, flexible patterns
- **Layer 3**: Complete orchestration control

### 3. **Explicit Over Implicit**
Everything is visible and controllable:
```python
# Dependencies are explicit
class MyAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, memory: IMemory, tools: List[ITool]):
        # You see exactly what this agent needs
```

### 4. **Composability**
Components are designed to work together:
```python
# Compose complex systems from simple parts
agent1 = AnalysisAgent(llm_client)
agent2 = SummaryAgent(llm_client)
workflow = Workflow([agent1, agent2])
```

## Why This Matters

Traditional frameworks often hide complexity behind abstractions:
- Settings classes that create components for you
- Factory patterns that obscure dependencies
- Configuration files that hide runtime behavior

Arshai takes a different approach:
- **You** create components when you need them
- **You** configure them how you want
- **You** control their lifecycle
- **You** understand exactly what's happening

## Getting Started

1. Read [Three-Layer Architecture](three-layer-architecture.md) to understand the structure
2. Review [Developer Authority](developer-authority.md) to understand the philosophy
3. Check [Design Decisions](design-decisions.md) to understand the choices

Then head to [Getting Started](../01-getting-started/) to begin building!

---

*"The best framework is one that empowers developers, not constrains them."*