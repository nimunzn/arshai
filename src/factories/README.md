# Factories Module

## Overview
The Factories module provides a centralized approach to creating core components of the Arshai framework. It abstracts away implementation details, handles dependency injection, and ensures consistent configuration across the system.

## Architecture
```
factories/
├── agent_factory.py         # Creates agent instances
├── llm_factory.py           # Creates LLM provider instances
├── memory_factory.py        # Creates memory manager instances
├── workflow_factory.py      # Creates workflow instances
├── tool_factory.py          # Creates tool instances
└── registry.py              # Central registry for component types
```

## Implementation Guide

### Creating a Component Factory

```python
from seedwork.interfaces.registry import IComponentRegistry

class ToolFactory:
    def __init__(self, registry: IComponentRegistry = None):
        self.registry = registry or default_registry
        
    def create_tool(self, tool_type: str, config: dict = None):
        """Create a tool instance of the specified type."""
        if tool_type not in self.registry.tools:
            raise ValueError(f"Unknown tool type: {tool_type}")
            
        tool_class = self.registry.tools[tool_type]
        return tool_class(config)
        
    def register_tool(self, tool_type: str, tool_class):
        """Register a new tool type."""
        self.registry.register_tool(tool_type, tool_class)
```

### Using Factories

```python
from src.factories import agent_factory, llm_factory, memory_factory

# Create an LLM instance
llm = llm_factory.create_llm(
    provider="openai",
    model="gpt-4",
    api_key="your_api_key"
)

# Create a memory manager
memory = memory_factory.create_memory_manager(
    provider="redis",
    connection_string="redis://localhost:6379"
)

# Create an agent that uses the LLM and memory
agent = agent_factory.create_agent(
    agent_type="conversation",
    llm=llm,
    memory_manager=memory
)
```

## Integration Points

### With Configuration System
Factories use configuration to create properly configured components:

```python
from src.config import Settings

settings = Settings()

# Create components from configuration
llm = llm_factory.create_llm_from_settings(settings)
memory = memory_factory.create_memory_from_settings(settings)
```

### With Dependency Injection
Factories handle dependency resolution and injection:

```python
def create_agent_with_dependencies(agent_type: str, settings: Settings = None):
    """Create an agent with all required dependencies."""
    settings = settings or Settings()
    
    # Create dependencies
    llm = llm_factory.create_llm_from_settings(settings)
    memory = memory_factory.create_memory_from_settings(settings)
    tools = tool_factory.create_tools_from_settings(settings)
    
    # Create and return agent with dependencies injected
    return agent_factory.create_agent(
        agent_type=agent_type,
        llm=llm,
        memory_manager=memory,
        tools=tools
    )
```

## Configuration
Configure factories through the registry system:

```python
from src.factories.registry import ComponentRegistry
from src.agents.custom_agents.my_agent import MyCustomAgent

# Create and configure a custom registry
registry = ComponentRegistry()
registry.register_agent("my_custom", MyCustomAgent)

# Create a factory using the custom registry
custom_agent_factory = AgentFactory(registry=registry)

# Use the factory
agent = custom_agent_factory.create_agent("my_custom", config=agent_config)
``` 