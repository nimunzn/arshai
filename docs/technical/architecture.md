# Arshai Framework Architecture

## System Overview

The Arshai framework provides a comprehensive architecture for building AI-powered applications, with a focus on modularity, extensibility, and clean interfaces. The system follows a layered approach with clear separation of concerns between components.

```mermaid
graph TB
    subgraph "Client Applications"
        APP[Applications]
    end
    
    subgraph "Workflow Layer"
        WR[Workflow Runner] --> WC[Workflow Config]
        WR --> WO[Workflow Orchestrator]
        WO --> ND[Nodes]
        WO --> ED[Edges]
        WO --> EP[Entry Points]
        ND --> AG[Agents]
    end
    
    subgraph "Agent Layer"
        AG --> LLM[LLMs]
        AG --> MEM[Memory]
        AG --> TL[Tools]
    end
    
    subgraph "Infrastructure Layer"
        LLM --> LLMPROV[LLM Providers]
        MEM --> MEMPROV[Memory Providers]
        TL --> EXTINT[External Integrations]
        
        subgraph "Document Processing"
            DOC[Document Management]
            EXT[Content Extraction]
            CHUNK[Content Chunking]
            PROC[Content Processing]
            VEC[Vector Storage]
        end
    end
    
    APP --> WR
    MEM --> DOC
```

## Core Architectural Principles

The Arshai framework is built on several foundational principles:

### Interface-First Design
- All components implement well-defined interfaces
- Interfaces are separated from implementations
- Multiple implementations can be provided for each interface
- Enables easy testing, mocking, and extensibility

### Clean Architecture
- Clear separation between domain, application, and infrastructure layers
- Domain logic is independent of infrastructure concerns
- Components depend on abstractions, not concrete implementations
- Business rules are isolated from external dependencies

### Factory Pattern
- Component creation is delegated to factory classes
- Factories handle dependency injection and configuration
- Enables centralized creation logic and service discovery
- Simplifies component instantiation and management

### Provider Pattern
- External services are accessed through provider interfaces
- Multiple providers can implement the same interface
- Configuration determines which provider is used
- Enables easy switching between service implementations

## Component Architecture

### Workflow System

The Workflow System serves as the orchestration layer, managing the flow of information between agents and external systems.

```mermaid
classDiagram
    class IWorkflowRunner {
        <<interface>>
        +run(workflow_id, input, context) Any
        +start(workflow_id, input, context) str
        +resume(execution_id, input) Any
        +get_status(execution_id) str
    }
    
    class WorkflowRunner {
        -orchestrator: IWorkflowOrchestrator
        -storage: IWorkflowStorage
        -config: IWorkflowConfig
        +run(workflow_id, input, context) Any
        +start(workflow_id, input, context) str
        +resume(execution_id, input) Any
        +get_status(execution_id) str
    }
    
    class IWorkflowOrchestrator {
        <<interface>>
        +execute_node(node_id, input, state) Any
        +evaluate_condition(condition, state) bool
        +get_next_nodes(node_id, state) List[str]
        +update_workflow_state(state, result) Any
    }
    
    class IWorkflowNode {
        <<interface>>
        +process(input, context) Any
        +get_id() str
        +get_type() str
        +get_config() dict
    }
    
    IWorkflowRunner <|.. WorkflowRunner
    WorkflowRunner --> IWorkflowOrchestrator
    IWorkflowOrchestrator --> IWorkflowNode
```

| Component | Responsibility |
|-----------|----------------|
| Workflow Runner | Executes workflow instances, manages state and handles errors |
| Workflow Config | Defines workflow structure, nodes, edges, and entry points |
| Workflow Orchestrator | Coordinates execution flow between nodes based on edge conditions |
| Nodes | Encapsulate agents with specific business logic adaptations |
| Edges | Define transition paths and conditions between nodes |
| Entry Points | Provide interfaces for external systems to interact with workflows |

### Agent System

The Agent System encapsulates language models with contextual awareness, tool usage capabilities, and memory management.

```mermaid
classDiagram
    class IAgent {
        <<interface>>
        +process_message(message, conversation_id) Response
        +process_messages(messages, conversation_id) Response
        +get_tools() List[Tool]
        +get_memory() IMemoryManager
    }
    
    class BaseAgent {
        -llm_provider: ILLMProvider
        -memory_manager: IMemoryManager
        -tools: List[Tool]
        -config: AgentConfig
        +process_message(message, conversation_id) Response
        +process_messages(messages, conversation_id) Response
        #_prepare_prompt(message, context) str
        #_post_process_response(response) Response
    }
    
    class ConversationAgent {
        +process_message(message, conversation_id) Response
        +process_messages(messages, conversation_id) Response
        #_prepare_system_prompt() str
        #_handle_tool_calls(response) Response
    }
    
    class RAGAgent {
        -vector_store: IVectorStore
        -document_processor: IDocumentProcessor
        +process_message(message, conversation_id) Response
        #_retrieve_context(message) List[Document]
        #_prepare_rag_prompt(message, context) str
    }
    
    IAgent <|.. BaseAgent
    BaseAgent <|-- ConversationAgent
    BaseAgent <|-- RAGAgent
```

| Component | Responsibility |
|-----------|----------------|
| Agent Interfaces | Define contracts for agent behavior and integration |
| Agent Implementations | Provide concrete agent behaviors with specific capabilities |
| Tool Integration | Connect agents to external functionalities and systems |
| Structured Output | Define and enforce schemas for agent responses |

### Memory Management

The Memory System handles conversation history, context management, and data persistence across interactions.

```mermaid
classDiagram
    class IMemoryManager {
        <<interface>>
        +store(input) str
        +retrieve(input) List[IWorkingMemory]
        +update(input) None
        +delete(input) None
    }
    
    class MemoryManagerService {
        -working_memory_manager: IMemoryManager
        -short_term_memory_manager: IMemoryManager
        -long_term_memory_manager: IMemoryManager
        +store_working_memory(conversation_id, memory_data, metadata) str
        +retrieve_working_memory(conversation_id) IWorkingMemory
        +update_working_memory(conversation_id, memory_data) None
        +delete_working_memory(conversation_id) None
    }
    
    class InMemoryManager {
        -memory_store: Dict
        -ttl: int
        +store(input) str
        +retrieve(input) List[IWorkingMemory]
        +update(input) None
        +delete(input) None
        -_clean_expired_memories() None
    }
    
    class RedisWorkingMemoryManager {
        -redis_url: str
        -ttl: int
        -key_prefix: str
        +store(input) str
        +retrieve(input) List[IWorkingMemory]
        +update(input) None
        +delete(input) None
        -_get_redis_client() Redis
        -_get_key(memory_id) str
    }
    
    IMemoryManager <|.. InMemoryManager
    IMemoryManager <|.. RedisWorkingMemoryManager
    MemoryManagerService --> IMemoryManager
```

| Component | Responsibility |
|-----------|----------------|
| Memory Manager | Coordinates memory operations and provider selection |
| Memory Providers | Implement specific storage mechanisms (in-memory, Redis, etc.) |
| Context Window Management | Handle limitations of LLM context sizes |
| Summarization | Create compressed representations of conversation history |

### Document Processing

The Document Processing system handles loading, processing, embedding, and retrieval of documents.

```mermaid
classDiagram
    class IFileLoader {
        <<interface>>
        +load(file_path) List[Document]
        +load_directory(directory_path) List[Document]
        +file_type_supported(file_path) bool
    }
    
    class ITextSplitter {
        <<interface>>
        +split(text) List[str]
        +split_documents(documents) List[Document]
    }
    
    class ITextProcessor {
        <<interface>>
        +process(documents) List[Document]
    }
    
    class BaseFileLoader {
        -config: IFileLoaderConfig
        -text_splitter: ITextSplitter
        +load(file_path) List[Document]
        +load_directory(directory_path) List[Document]
        +file_type_supported(file_path) bool
        #_extract_content(file_path) str
    }
    
    class RecursiveTextSplitter {
        -config: ITextSplitterConfig
        +split(text) List[str]
        +split_documents(documents) List[Document]
        -_split_text(text) List[str]
    }
    
    class TextCleaner {
        -config: ITextProcessorConfig
        +process(documents) List[Document]
        -_clean_text(text) str
    }
    
    IFileLoader <|.. BaseFileLoader
    BaseFileLoader <|-- UnstructuredLoader
    BaseFileLoader <|-- PDFLoader
    BaseFileLoader <|-- AudioLoader
    
    ITextSplitter <|.. RecursiveTextSplitter
    
    ITextProcessor <|.. TextCleaner
    ITextProcessor <|.. ContextEnricher
```

| Component | Responsibility |
|-----------|----------------|
| Document Loaders | Extract content from various file formats |
| Text Processors | Transform and normalize text content |
| Text Splitters | Chunk documents into appropriate segments |
| Embedding Service | Generate vector representations of text |
| Vector Store | Index and retrieve document chunks based on semantic similarity |

## Data Flow Patterns

### Basic Agent Interaction

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Agent
    participant L as LLM
    participant M as Memory
    participant T as Tools
    
    C->>A: process_message(input)
    A->>M: retrieve_context(conversation_id)
    M-->>A: conversation_history
    A->>L: generate_response(input, history, tools)
    L-->>A: response_with_tool_calls
    
    alt Tool Usage Required
        A->>T: execute_tool(tool_name, parameters)
        T-->>A: tool_result
        A->>L: generate_final_response(tool_result)
        L-->>A: final_response
    end
    
    A->>M: save_interaction(conversation_id, input, response)
    A-->>C: agent_response
```

### Workflow Execution Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant R as WorkflowRunner
    participant O as Orchestrator
    participant N1 as Node_1
    participant N2 as Node_2
    participant S as State
    
    C->>R: run_workflow(input)
    R->>O: start_workflow(input)
    O->>S: initialize_state(input)
    O->>N1: process(state)
    N1->>S: update_state(output)
    O->>O: evaluate_edges(state)
    O->>N2: process(state)
    N2->>S: update_state(output)
    O->>O: check_completion(state)
    O-->>R: workflow_result
    R-->>C: final_response
```

### Document Processing Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant DO as DocumentOrchestrator
    participant DL as DocumentLoader
    participant TS as TextSplitter
    participant ES as EmbeddingService
    participant VS as VectorStore
    
    C->>DO: index_documents(request)
    DO->>DL: load_documents(file_paths)
    DL-->>DO: documents
    DO->>TS: split_documents(documents)
    TS-->>DO: chunks
    DO->>ES: create_embeddings(chunks)
    ES-->>DO: embeddings
    DO->>VS: store_embeddings(collection, embeddings)
    VS-->>DO: storage_result
    DO-->>C: indexing_result
```

## Implementation Details

### Document Processing Implementation

The document processing pipeline follows a three-stage architecture:

1. **Content Extraction** (File Loaders):
   - Extract raw content from various file formats
   - Preserve document metadata
   - Handle different document structures

2. **Content Chunking** (Text Splitters):
   - Break down large documents into manageable chunks
   - Preserve semantic coherence across chunks
   - Apply appropriate chunking strategies for different content types

3. **Content Processing** (Text Processors):
   - Clean and normalize text
   - Enrich document metadata
   - Prepare content for embedding and indexing

This modular design allows each component to be used independently or combined in processing pipelines:

```python
from src.document_loaders import PDFLoader, RecursiveTextSplitter, TextCleaner

# Create components
loader = PDFLoader()
splitter = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=200)
cleaner = TextCleaner()

# Process document
raw_documents = loader.load("document.pdf")
chunked_documents = splitter.split_documents(raw_documents)
processed_documents = cleaner.process(chunked_documents)
```

### Memory Management Implementation

The memory management system provides a unified interface for different types of memory:

1. **Working Memory**:
   - Short-term conversation context
   - Current interaction state
   - Recent message history

2. **Short-Term Memory**:
   - Intermediate information storage
   - Session-specific data
   - Temporary caching

3. **Long-Term Memory**:
   - Persistent knowledge base
   - User preferences and settings
   - Learned information

Implementation example:

```python
from src.memory import MemoryManagerService
from src.memory.working_memory import InMemoryManager, RedisWorkingMemoryManager

# Configure memory manager with different providers
memory_manager = MemoryManagerService(
    config={
        "working_memory": {
            "provider": "redis",  # or "in_memory"
            "ttl": 3600,
            "redis_url": "redis://localhost:6379/0"
        }
    }
)

# Store and retrieve working memory
memory_id = memory_manager.store_working_memory(
    conversation_id="conversation123",
    memory_data="Working memory content",
    metadata={"user_id": "user123"}
)

memory = memory_manager.retrieve_working_memory(
    conversation_id="conversation123"
)
```

## Design Patterns and Architectural Styles

### Clean Architecture

Arshai follows clean architecture principles with clear separation of concerns:

- **Domain Layer**: Core business logic and interfaces (seedwork/interfaces)
- **Application Layer**: Use case implementations and orchestration (src/workflows)
- **Infrastructure Layer**: External integrations and technical implementations (src/llms, src/memory, etc.)

### Interface-First Design

All components implement interfaces defined in the seedwork module, allowing for:
- Multiple implementations of the same interface
- Easy testing through mocking
- Clear contracts between components
- Pluggable architecture for extensions

### Factory Pattern

Component creation is managed through factory classes that:
- Encapsulate initialization logic
- Handle configuration injection
- Provide consistent access patterns
- Enable component registration and discovery

### Repository Pattern

Data access is abstracted through repository interfaces that:
- Hide implementation details of storage
- Provide consistent data access methods
- Enable switching between storage backends
- Encapsulate query construction

### Strategy Pattern

Interchangeable algorithms are implemented using the strategy pattern:
- Text splitting strategies
- Embedding strategies
- Memory management strategies
- Tool selection strategies

## Resilience Strategies

### LLM Service Failures
- Automatic retries with exponential backoff
- Circuit breaker pattern to prevent cascading failures
- Fallback to alternative providers when primary is unavailable
- Graceful degradation with simpler models when needed

### Memory Storage Failures
- Local caching for temporary resilience
- Automatic recovery when storage becomes available
- Graceful degradation to in-memory storage when persistent storage fails
- Periodic state snapshots for recovery

### Tool Execution Failures
- Timeouts for external tool calls
- Error reporting to agents for handling in conversation
- Fallback options for critical tools
- Monitoring and alerting for recurring failures

### Workflow Execution Failures
- State persistence for long-running workflows
- Resumable execution from checkpoints
- Error handling nodes for managing failures
- Compensation logic for rolling back partial changes

## Security Model

### Authentication
- API key authentication for service access
- OAuth integration for user-context operations
- Role-based access control for administrative functions
- JWT token validation for session management

### Authorization
- Granular permissions for workflow and agent operations
- Resource-based access control for document collections
- Scoped access tokens with limited privileges
- User context isolation for multi-tenant deployments

### Data Protection
- Encryption of sensitive data at rest
- Secure transmission with TLS
- Configurable data retention policies
- Content filtering for prohibited information

### Audit and Compliance
- Comprehensive logging of all operations
- Audit trails for security-relevant events
- Configurable logging levels for different environments
- Compliance modes for regulated industries

## Integration Patterns

### External System Integration
- Adapter pattern for system adaptation
- Gateway pattern for unified access
- Facade pattern for simplification
- Event-driven integration for loose coupling

### API Integration
- RESTful API for synchronous operations
- WebSocket for real-time communication
- GraphQL for flexible data retrieval
- Webhooks for event notification

### Message Queue Integration
- Producer/Consumer pattern for asynchronous processing
- Command pattern for operation encapsulation
- Event sourcing for state reconstruction
- CQRS for read/write separation

### Database Integration
- Repository pattern for data access abstraction
- Unit of work pattern for transaction management
- Query object pattern for query composition
- Data mapper pattern for object-relational mapping

## Architecture Evolution

### Current State
The architecture currently emphasizes:
- Flexible workflow orchestration
- Modular component design
- Multiple integration points for LLMs and tools
- Comprehensive document processing capabilities

### Observability System

The Observability System provides comprehensive monitoring and instrumentation for production AI applications, with a focus on LLM performance tracking and OpenTelemetry integration.

```mermaid
classDiagram
    class IObservabilityManager {
        <<interface>>
        +observe_llm_call(provider, model, method) ContextManager
        +observe_streaming_llm_call(provider, model, method) AsyncContextManager
        +pre_call_token_count(provider, model, messages) TokenCountResult
        +process_streaming_chunk(provider, model, chunk, timing) UsageData
    }
    
    class ObservabilityManager {
        -config: ObservabilityConfig
        -metrics_collector: MetricsCollector
        -tracer: Tracer
        -token_counter_factory: TokenCounterFactory
        +observe_llm_call(provider, model, method) ContextManager
        +observe_streaming_llm_call(provider, model, method) AsyncContextManager
        +pre_call_token_count(provider, model, messages) TokenCountResult
        +process_streaming_chunk(provider, model, chunk, timing) UsageData
    }
    
    class MetricsCollector {
        -meter: Meter
        -config: ObservabilityConfig
        +record_llm_request(provider, model, timing) None
        +record_token_metrics(timing_data) None
        +record_error(provider, model, error) None
    }
    
    class TokenCounterFactory {
        +create_counter(provider, model) BaseTokenCounter
        +register_counter(provider, counter_class) None
        +get_supported_providers() List[str]
    }
    
    class BaseTokenCounter {
        <<abstract>>
        +count_tokens(messages) TokenCountResult
        +count_streaming_tokens(content) int
        +extract_usage_from_stream_chunk(chunk) UsageData
        +supports_usage_api() bool
    }
    
    IObservabilityManager <|.. ObservabilityManager
    ObservabilityManager --> MetricsCollector
    ObservabilityManager --> TokenCounterFactory
    TokenCounterFactory --> BaseTokenCounter
```

| Component | Responsibility |
|-----------|----------------|
| Observability Manager | Coordinates monitoring activities and provides context managers |
| Metrics Collector | Collects and exports metrics using OpenTelemetry |
| Token Counter Factory | Creates provider-specific token counting implementations |
| Token Counters | Handle accurate token counting for different LLM providers |
| Timing Data | Tracks token-level timing metrics for streaming responses |

#### Key Metrics

The system provides four core metrics for LLM performance monitoring:

- **`llm_time_to_first_token_seconds`**: Time from request start to first token
- **`llm_time_to_last_token_seconds`**: Time from request start to last token  
- **`llm_duration_first_to_last_token_seconds`**: Duration from first token to last token
- **`llm_completion_tokens`**: Count of completion tokens generated

#### Provider Support

- **OpenAI/Azure**: Uses `tiktoken` for accurate token counting + API usage data
- **Anthropic**: Uses native token counting API with event-based streaming
- **Google Gemini**: Uses native token counting API with content-based streaming
- **OpenRouter**: Uses OpenAI-compatible API with estimation fallbacks

#### Integration Patterns

The observability system integrates seamlessly with the existing architecture:

```python
# Factory Integration
client = LLMFactory.create_with_observability(
    provider="openai",
    config=llm_config,
    observability_config=obs_config
)

# Manual Integration
with obs_manager.observe_llm_call("openai", "gpt-4") as timing:
    response = llm_client.chat_completion(input_data)
    # Metrics automatically collected
```

### Future Directions
Planned architectural enhancements include:
- Enhanced distributed processing capabilities
- Improved scaling for high-volume scenarios
- Advanced caching strategies for performance optimization
- Extended observability backends and custom metric exporters 