# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
poetry run pytest                    # Run all tests
poetry run pytest tests/unit/        # Run unit tests only
poetry run pytest tests/integration/ # Run integration tests only
poetry run pytest --cov=arshai      # Run tests with coverage
```

### Code Quality
```bash
poetry run black .                  # Format code
poetry run isort .                  # Sort imports
poetry run mypy arshai/             # Type checking
poetry run bandit -r arshai/        # Security analysis
poetry run safety check             # Check dependencies for vulnerabilities
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

# Background tasks (fire-and-forget execution)
def notify_admin(event: str, details: str = ""):
    # Background task that runs independently
    print(f"Admin notification: {event} - {details}")

background_tasks = {"send_admin_notification": notify_admin}
llm_input = ILLMInput(
    system_prompt="...",
    user_message="...",
    tools_list=tools,
    background_tasks=background_tasks
)

# Workflow orchestration
workflow_runner = WorkflowRunner(workflow_config)
result = await workflow_runner.run({"message": "...", "state": initial_state})
```

## LLM Background Tasks

### Overview
The LLM system supports **background tasks** - functions that execute in fire-and-forget mode without returning results to the conversation. These are ideal for logging, notifications, analytics, or other side effects.

### Key Features
- **Fire-and-forget execution**: Tasks run independently using `asyncio.create_task()`
- **No conversation impact**: Results don't return to the LLM conversation flow
- **Reference management**: Tasks are tracked to prevent garbage collection
- **Auto-declaration**: Uses Gemini SDK's `FunctionDeclaration.from_callable()` for type safety
- **Parallel execution**: Background tasks run concurrently with regular tools

### Implementation Details

#### Interface Extension
```python
# arshai/core/interfaces/illm.py
class ILLMInput(BaseModel):
    background_tasks: Dict[str, Callable] = Field(
        default={}, 
        description="Functions for fire-and-forget execution"
    )
```

#### Usage Example
```python
def send_notification(event: str, user_id: str, priority: str = "normal"):
    """Send notification to admin system"""
    # This runs in background, no return value needed
    print(f"Notification: {event} for user {user_id} (priority: {priority})")

def log_interaction(action: str, metadata: dict = None):
    """Log user interaction for analytics"""
    # Background logging, doesn't affect conversation
    print(f"Logged: {action} with metadata: {metadata}")

# Configure LLM with background tasks
llm_input = ILLMInput(
    system_prompt="You are a helpful assistant...",
    user_message="What's the weather like?",
    background_tasks={
        "send_notification": send_notification,
        "log_interaction": log_interaction
    }
)

# LLM can call these functions independently
response = await llm_client.chat(llm_input)
```

### Architecture Notes
- **Task Lifecycle**: Background tasks are tracked in `self._background_tasks` set
- **Cleanup**: Tasks auto-remove from tracking set on completion via `add_done_callback()`
- **Error Handling**: Background task failures don't affect main conversation
- **Performance**: Uses `asyncio.create_task()` for true parallel execution

## LLM Client Architecture Standards

### Reference Implementation: Google Gemini Client
The Google Gemini client (`arshai/llms/google_genai.py`) serves as the **canonical reference implementation** for all LLM clients in the Arshai framework. All other LLM providers should follow these established patterns and standards.

### Core LLM Client Patterns

#### 1. Unified Input Processing
**Standard Pattern for All LLM Clients:**
```python
# Decision logic that all LLM clients should implement
has_tools = input.tools_list and len(input.tools_list) > 0
has_background_tasks = input.background_tasks and len(input.background_tasks) > 0
has_structure = input.structure_type is not None
needs_function_calling = has_tools or has_background_tasks

if not needs_function_calling:
    # Simple case: direct response
    return simple_llm_call()
else:
    # Complex case: multi-turn function calling
    return multi_turn_function_calling_loop()
```

#### 2. Function Calling Architecture
**All LLM clients must implement:**
- **Parallel Execution**: Regular tools execute concurrently via `asyncio.gather()`
- **Fire-and-Forget**: Background tasks run independently via `asyncio.create_task()`
- **Enhanced Context**: Function calls include arguments and results in conversation
- **Reference Management**: Background tasks tracked to prevent garbage collection

```python
# Standard function execution pattern
for function_call in function_calls:
    if function_name in input.background_tasks:
        # Background task (fire-and-forget)
        task = asyncio.create_task(func(**args))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    elif function_name in input.callable_functions:
        # Regular tool (parallel execution)
        function_tasks.append(func(**args))

# Execute regular tools in parallel
if function_tasks:
    results = await asyncio.gather(*function_tasks)
```

#### 3. Streaming Implementation Standards
**Required streaming capabilities:**
- **Real-time Processing**: Functions execute immediately when detected
- **Progressive Responses**: Text streams while maintaining function call capability
- **Safe Usage Accumulation**: Handle None values in usage metadata
- **Completion Logic**: Proper finish_reason + function_call status handling

```python
# Standard streaming pattern
for chunk in stream:
    # 1. Safe usage processing
    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
        safely_accumulate_usage(chunk.usage_metadata)
    
    # 2. Progressive text streaming
    if hasattr(chunk, "text") and chunk.text:
        yield_text_progressively()
    
    # 3. Immediate function execution
    if hasattr(chunk, "function_calls") and chunk.function_calls:
        execute_functions_immediately()
    
    # 4. Completion detection
    if finish_reason_detected():
        handle_completion_logic()
```

#### 4. Error Handling Standards
**Defensive programming patterns:**
- **Graceful Degradation**: Partial failures don't break conversation flow
- **Usage Preservation**: Always return usage data, even during errors
- **Comprehensive Logging**: Debug info for troubleshooting
- **Safe Null Handling**: Protect against None values in provider responses

#### 5. Interface Compliance
**All LLM clients must:**
- Implement `ILLM` interface completely
- Support both `chat()` and `stream()` methods
- Handle `ILLMInput` with all fields (tools, background_tasks, structure_type)
- Return standardized response format with usage information
- Maintain consistent error handling patterns

### Provider-Specific Adaptations

#### Function Declaration Generation
Each LLM provider should implement auto-generation using their SDK's capabilities:
```python
# Gemini example (reference implementation)
def _convert_background_tasks_to_gemini_format(self, background_tasks):
    return [
        FunctionDeclaration.from_callable(
            func, 
            name=name,
            description=f"BACKGROUND TASK: {original_description}"
        )
        for name, func in background_tasks.items()
    ]

# Other providers should follow similar patterns:
# - OpenAI: Convert to OpenAI function calling format
# - Anthropic: Adapt to Claude function calling
# - Azure: Use Azure OpenAI function specifications
```

#### Multi-Turn Conversation Management
**Standard loop structure:**
```python
current_turn = 0
while current_turn < input.max_turns:
    # 1. Generate response with tools
    response = provider_specific_generate_with_tools()
    
    # 2. Process usage metadata
    accumulate_usage_safely()
    
    # 3. Handle function calls (parallel + background)
    execute_functions_with_standard_pattern()
    
    # 4. Check completion conditions
    if should_finish_conversation():
        break
    
    current_turn += 1
```

### Development Guidelines for New LLM Clients

1. **Start with Gemini**: Use the Gemini client as template and reference
2. **Maintain Patterns**: Follow the established architectural patterns exactly
3. **Provider Adaptation**: Only adapt provider-specific API calls and formats
4. **Test Compatibility**: Ensure new clients pass the same test scenarios
5. **Performance Standards**: Match or exceed the reference implementation's performance

### Quality Assurance
- **Cross-Client Testing**: All LLM clients should pass identical test suites
- **Behavioral Consistency**: Same inputs should produce equivalent outputs across providers
- **Performance Benchmarks**: Maintain performance standards set by reference implementation
- **Interface Compliance**: Strict adherence to `ILLM` interface contracts