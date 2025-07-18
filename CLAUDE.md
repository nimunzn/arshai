# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/unit/        # Run unit tests only
pytest tests/integration/ # Run integration tests only
pytest --cov=arshai      # Run tests with coverage
```

### Code Quality
```bash
black .                  # Format code
isort .                  # Sort imports
mypy arshai/             # Type checking
bandit -r arshai/        # Security analysis
safety check             # Check dependencies for vulnerabilities
```

### Documentation
```bash
cd docs_sphinx && make html  # Build Sphinx documentation
```

### Package Management
```bash
poetry install           # Install all dependencies
poetry install -E all    # Install with all optional dependencies (redis, milvus, flashrank)
poetry install -E docs   # Install documentation dependencies
```

## Architecture Overview

Arshai is an AI application framework built on **clean architecture principles** with **interface-driven design**. The system follows a layered approach with clear separation of concerns:

### Core Layers

1. **Application Layer**: Workflows and Agents that orchestrate business logic
2. **Domain Layer**: Memory, Tools, and LLM integrations that handle core functionality  
3. **Infrastructure Layer**: Document processing, vector storage, and external integrations

### Key Components

- **Agents** (`arshai/agents/`): Intelligence units that process user interactions
- **Workflows** (`arshai/workflows/`): Orchestrate complex multi-step processes with state management
- **Memory** (`arshai/memory/`): Handle conversation context and state persistence
- **Tools** (`arshai/tools/`): External capabilities that extend agent functionality
- **LLMs** (`arshai/llms/`): Language model integrations with unified interface
- **Factories** (`arshai/factories/`): Component creation and dependency injection

### Design Patterns

- **Interface-First**: All major components implement well-defined protocols in `arshai/core/interfaces/`
- **Factory Pattern**: Component creation abstracted through factory classes
- **DTO Pattern**: Structured data transfer using Pydantic models
- **Provider Pattern**: Multiple implementations for external services (LLMs, memory, vector DBs)
- **Async-First**: Most operations support asynchronous execution

## Development Guidelines

### Working with the Codebase

1. **Start with Interfaces**: Examine contracts in `arshai/core/interfaces/` before implementations
2. **Use Factories**: Leverage existing factory classes in `arshai/factories/` for component creation
3. **Follow DTO Pattern**: Use structured Pydantic models for all data interactions
4. **Prefer Async**: Use async methods for better performance and concurrency
5. **Maintain Immutability**: Especially important for workflow state management

### Project Structure

- **Main Package**: `arshai/` - New unified structure (use this for new development)
- **Legacy Code**: `src/` - Being migrated (avoid for new features)
- **Core Interfaces**: `arshai/core/interfaces/` - System contracts and protocols
- **Configuration**: `arshai/config/` - Settings and configuration management
- **Examples**: `examples/` - Working code samples and usage patterns

### Key Interface Locations

- **IAgent**: Core agent contract for user interactions
- **IWorkflowOrchestrator/IWorkflowRunner**: Workflow system contracts
- **IMemoryManager**: Memory management interface
- **ITool**: Tool integration protocol
- **ILLM**: Language model provider interface
- **IVectorDBClient/IEmbedding**: Vector storage and embeddings

### Extension Points

- **Custom Agents**: Implement `IAgent` interface
- **New Tools**: Implement `ITool` interface  
- **LLM Providers**: Implement `ILLM` interface
- **Memory Backends**: Implement `IMemoryManager` interface
- **Workflow Nodes**: Extend base node classes for business logic

### Common Patterns

```python
# Component creation via factories
settings = Settings()
agent = settings.create_agent("conversation", agent_config)

# Structured input/output
input_data = IAgentInput(message="...", conversation_id="...")
response, usage = await agent.process_message(input_data)

# Tool integration
tools = [WebSearchTool(settings), KnowledgeBaseTool(settings)]
agent_config = IAgentConfig(task_context="...", tools=tools)

# Workflow orchestration
workflow_runner = WorkflowRunner(workflow_config)
result = await workflow_runner.run({"message": "...", "state": initial_state})
```