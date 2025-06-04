# Building with Arshai

This document provides comprehensive guidance for building AI solutions using the Arshai framework. It covers architecture principles, design patterns, and implementation approaches to help you create powerful, modular, and maintainable AI applications.

## Table of Contents

- [Core Architecture](#core-architecture)
- [Design Principles](#design-principles)
- [Common Solution Patterns](#common-solution-patterns)
- [Component Integration](#component-integration)
- [Application Structure](#application-structure)
- [Performance Considerations](#performance-considerations)
- [Testing and Quality Assurance](#testing-and-quality-assurance)
- [Deployment Patterns](#deployment-patterns)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)

## Core Architecture

Arshai follows a clean, modular architecture with interface-first design principles.

### Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Application Layer                    │
│       (User-Facing Applications, API Services)           │
└───────────────────────────┬───────────────────────────┘
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     Workflow Layer                       │
│         (Workflow Definitions, Orchestration)            │
└───────────────────────────┬───────────────────────────┘
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     Component Layer                      │
│    (Agents, Memory, Indexers, Speech, WebSearch)         │
└───────────────────────────┬───────────────────────────┘
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     Infrastructure Layer                 │
│       (LLMs, Vector DBs, Storage, External APIs)         │
└─────────────────────────────────────────────────────────┘
```

### Interface-First Design

The foundation of Arshai is the interface-first design approach:

1. All components are defined by interfaces in the `seedwork` package
2. Implementations are provided in the `src` package
3. Components depend on interfaces, not concrete implementations

This design provides:
- **Loose coupling**: Components can be replaced without affecting others
- **Testability**: Easy mocking of dependencies for testing
- **Extensibility**: New implementations can be added without modifying existing code

## Design Principles

### 1. Separation of Concerns

Each component should have a single, well-defined responsibility:
- **Agents**: Handle conversation and tool usage
- **Memory**: Manage conversation context
- **Workflows**: Orchestrate component interactions
- **Indexers**: Process and retrieve documents
- **Settings**: Configure the application

### 2. Configuration vs. Code

Separate configuration from code:
- Use the `Settings` class to manage configuration
- Configure components through settings files
- Inject dependencies rather than creating them directly

### 3. Factory Pattern

Use factories for creating predefined components:
- `LLMFactory`: Creates language model instances
- `MemoryFactory`: Creates memory components
- `AgentFactory`: Creates predefined agent types
- `VectorDBFactory`: Creates vector database clients

### 4. Direct Instantiation

Use direct instantiation for custom components:
- Create custom agents by implementing the `IAgent` protocol
- Create custom tools by implementing the `ITool` protocol
- Create custom nodes by extending `BaseNode`

### 5. Composition Over Inheritance

Build complex components by composing simpler ones:
- Add capabilities to agents by providing tools
- Create workflows by connecting nodes
- Build RAG systems by combining indexers, retrievers, and agents

## Common Solution Patterns

### Conversational Agents

```python
from arshai import Settings
from seedwork.interfaces.iagent import IAgentConfig, IAgentInput

# Initialize settings
settings = Settings()

# Configure agent
agent_config = IAgentConfig(
    task_context="You are a helpful customer service agent for Acme Inc.",
    tools=[]
)

# Create agent
agent = settings.create_agent("operator", agent_config)

# Process messages
response = agent.process_message(
    IAgentInput(
        message="What are your business hours?",
        conversation_id="cs-123",
        stream=False
    )
)

print(response.agent_message)
```

### Retrieval-Augmented Generation (RAG)

```python
from arshai import Settings
from src.indexers import DocumentOrchestrator, IndexingRequest, RetrievalRequest
from seedwork.interfaces.iagent import IAgentConfig, IAgentInput

# Initialize settings
settings = Settings()

# Index documents
orchestrator = DocumentOrchestrator(settings)
index_request = IndexingRequest(
    file_paths=["policies.pdf", "handbook.docx"],
    collection_name="company_docs",
    chunking_strategy="paragraph"
)
orchestrator.index_documents(index_request)

# Create RAG agent
rag_config = IAgentConfig(
    task_context="You are a company knowledge base assistant. Answer questions based on company documents.",
    tools=[],
    rag_config={
        "collection_name": "company_docs",
        "top_k": 5,
        "rerank": True
    }
)
rag_agent = settings.create_agent("rag", rag_config)

# Process query
response = rag_agent.process_message(
    IAgentInput(
        message="What is our vacation policy?",
        conversation_id="rag-123"
    )
)

print(response.agent_message)
```

### Multi-Agent Workflows

```python
from arshai import Settings
from src.workflows import WorkflowConfig, WorkflowRunner
from src.workflows.nodes import AgentNode, ConditionalNode
from seedwork.interfaces.iagent import IAgentConfig
from seedwork.interfaces.iworkflow import IWorkflowOrchestrator

# Initialize settings
settings = Settings()

# Define workflow
class CustomerSupportWorkflow(WorkflowConfig):
    def _configure_workflow(self, orchestrator: IWorkflowOrchestrator) -> None:
        # Create intake agent
        intake_config = IAgentConfig(
            task_context="You are an intake agent who categorizes customer issues",
            tools=[]
        )
        intake_agent = settings.create_agent("operator", intake_config)
        
        # Create technical support agent
        tech_config = IAgentConfig(
            task_context="You are a technical support specialist",
            tools=[]
        )
        tech_agent = settings.create_agent("operator", tech_config)
        
        # Create billing support agent
        billing_config = IAgentConfig(
            task_context="You are a billing support specialist",
            tools=[]
        )
        billing_agent = settings.create_agent("operator", billing_config)
        
        # Add nodes
        orchestrator.add_node("intake", AgentNode(
            node_id="intake",
            name="Intake Agent",
            agent=intake_agent,
            settings=settings
        ))
        
        orchestrator.add_node("router", ConditionalNode(
            node_id="router",
            name="Issue Router",
            condition=self._route_issue,
            settings=settings
        ))
        
        orchestrator.add_node("tech", AgentNode(
            node_id="tech",
            name="Technical Support",
            agent=tech_agent,
            settings=settings
        ))
        
        orchestrator.add_node("billing", AgentNode(
            node_id="billing",
            name="Billing Support",
            agent=billing_agent,
            settings=settings
        ))
        
        # Add edges
        orchestrator.add_edge("intake", "router")
        orchestrator.add_edge("router", "tech")
        orchestrator.add_edge("router", "billing")
        
        # Set entry points
        orchestrator.set_entry_points(
            lambda _: "default",
            {"default": "intake"}
        )
    
    def _route_issue(self, input_data):
        # Route based on the message from the intake agent
        message = input_data.get("message", "").lower()
        
        if any(word in message for word in ["technical", "software", "hardware", "not working"]):
            return "tech"
        else:
            return "billing"

# Create and run workflow
workflow_config = CustomerSupportWorkflow(settings)
runner = WorkflowRunner(workflow_config)

result = runner.process({
    "message": "I'm having trouble logging into my account",
    "conversation_id": "cs-workflow-123"
})

print(result["message"])
```

## Component Integration

### Integrating with LLMs

```python
from arshai import Settings
from seedwork.interfaces.illm import ILLMConfig

# Initialize settings
settings = Settings()

# Create LLM with default settings
llm = settings.create_llm()

# Or create with specific configuration
llm_config = ILLMConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)
custom_llm = settings.create_llm()

# Use the LLM
response = llm.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Arshai."}
    ]
)

print(response.message)
```

### Integrating with Memory

```python
from arshai import Settings

# Initialize settings
settings = Settings()

# Create memory manager
memory_manager = settings.create_memory_manager()

# Work with memory
conversation_id = "user123-session456"
working_memory = memory_manager.retrieve_working_memory(conversation_id)

# Add messages
working_memory.add_user_message("Hello, how can you help me?")
working_memory.add_agent_message("I can answer questions and provide assistance.")

# Store memory updates
memory_manager.store_memory(
    conversation_id, 
    {"history": working_memory.get_raw_history()}
)
```

### Integrating with Vector Databases

```python
from arshai import Settings

# Initialize settings
settings = Settings()

# Create vector database client
vector_db = settings.create_vector_db()

# Create a collection
vector_db.create_collection("my_documents", dimension=1536)

# Insert embeddings
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # Simplified for example
metadata = [{"source": "doc1.txt"}, {"source": "doc2.txt"}]
vector_db.insert("my_documents", embeddings, metadata)

# Search
query_embedding = [0.2, 0.3, ...]  # Simplified for example
results = vector_db.search("my_documents", query_embedding, top_k=5)

for result in results:
    print(f"Score: {result.score}, Text: {result.text}")
```

### Integrating with Speech

```python
from arshai import Settings
from seedwork.interfaces.ispeech import AudioInput

# Initialize settings
settings = Settings()

# Create speech processor
speech_processor = settings.create_speech_model()

# Speech-to-text
with open("audio.mp3", "rb") as audio_file:
    audio_data = audio_file.read()
    
audio_input = AudioInput(content=audio_data, format="mp3")
transcription = speech_processor.speech_to_text(audio_input)

print(f"Transcription: {transcription}")

# Text-to-speech
speech_data = speech_processor.text_to_speech("Hello, this is a test", voice="alloy")
with open("output.mp3", "wb") as output_file:
    output_file.write(speech_data.content)
```

## Application Structure

### Basic Structure

```
my_app/
├── config/
│   ├── default.yaml      # Default configuration
│   └── production.yaml   # Production overrides
├── src/
│   ├── agents/           # Custom agent implementations
│   │   └── custom_agent.py
│   ├── tools/            # Custom tool implementations
│   │   └── custom_tool.py
│   ├── workflows/        # Custom workflow definitions
│   │   └── main_workflow.py
│   ├── settings.py       # Application settings
│   └── main.py           # Application entry point
├── data/
│   └── documents/        # Documents for indexing
├── tests/
│   ├── test_agents.py
│   └── test_workflows.py
├── requirements.txt
└── README.md
```

### Settings Structure

```python
from src.config.settings import Settings

class AppSettings(Settings):
    """Application-specific settings."""
    
    def __init__(self, config_path=None):
        super().__init__(config_path)
        
        # Initialize application-specific resources
        self._custom_service = self._create_custom_service()
    
    def _create_custom_service(self):
        """Create application-specific service."""
        service_config = self.get("custom_service", {})
        return CustomService(service_config)
    
    def get_custom_service(self):
        """Get the custom service."""
        return self._custom_service
```

### Main Entry Point

```python
from src.settings import AppSettings
from src.workflows.main_workflow import MainWorkflowConfig
from src.workflows import WorkflowRunner

def main():
    # Initialize settings
    settings = AppSettings("config/default.yaml")
    
    # Create workflow
    workflow_config = MainWorkflowConfig(settings)
    runner = WorkflowRunner(workflow_config)
    
    # Run workflow
    result = runner.process({
        "message": "Hello, how can you help me?",
        "conversation_id": "app-123"
    })
    
    # Process result
    print(result["message"])

if __name__ == "__main__":
    main()
```

## Performance Considerations

### Memory Usage

- **Working Memory Size**: Keep working memory size reasonable
- **Embedding Batching**: Batch embed documents for better performance
- **Caching**: Cache embeddings and frequent LLM prompts

### LLM Costs

- **Token Usage**: Be mindful of token usage
- **Prompt Design**: Optimize prompts for token efficiency
- **Model Selection**: Use simpler models for simpler tasks

### Concurrency

- **Async Processing**: Use async methods for better concurrency
- **Worker Pools**: Implement worker pools for parallel processing
- **Batch Processing**: Process documents in batches

## Testing and Quality Assurance

### Unit Testing

```python
import pytest
from src.agents.custom_agent import CustomAgent
from seedwork.interfaces.iagent import IAgentConfig, IAgentInput

def test_custom_agent():
    # Create mock settings
    mock_settings = MockSettings()
    
    # Create agent
    agent_config = IAgentConfig(
        task_context="Test context",
        tools=[]
    )
    agent = CustomAgent(agent_config, mock_settings)
    
    # Test message processing
    response = agent.process_message(
        IAgentInput(
            message="Test message",
            conversation_id="test-123"
        )
    )
    
    # Assert on response
    assert response.agent_message is not None
    assert "expected content" in response.agent_message
```

### Integration Testing

```python
import pytest
from src.settings import AppSettings
from src.workflows.main_workflow import MainWorkflowConfig
from src.workflows import WorkflowRunner

def test_workflow_integration():
    # Initialize settings with test configuration
    settings = AppSettings("tests/config/test.yaml")
    
    # Create workflow
    workflow_config = MainWorkflowConfig(settings)
    runner = WorkflowRunner(workflow_config)
    
    # Run workflow with test input
    result = runner.process({
        "message": "Test input",
        "conversation_id": "test-integration-123"
    })
    
    # Assert on result
    assert "expected response" in result["message"]
```

## Deployment Patterns

### API Service

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from src.settings import AppSettings
from src.workflows.main_workflow import MainWorkflowConfig
from src.workflows import WorkflowRunner

app = FastAPI()
settings = AppSettings("config/production.yaml")

class Message(BaseModel):
    content: str
    user_id: str

@app.post("/chat")
async def chat(message: Message):
    # Create workflow
    workflow_config = MainWorkflowConfig(settings)
    runner = WorkflowRunner(workflow_config)
    
    # Process message
    result = await runner.aprocess({
        "message": message.content,
        "conversation_id": f"user-{message.user_id}"
    })
    
    # Return response
    return {"response": result["message"]}
```

### Batch Processing

```python
import asyncio
from src.settings import AppSettings
from src.indexers import DocumentOrchestrator, IndexingRequest

async def index_documents(file_paths, collection_name):
    # Initialize settings
    settings = AppSettings("config/production.yaml")
    
    # Create orchestrator
    orchestrator = DocumentOrchestrator(settings)
    
    # Create request
    request = IndexingRequest(
        file_paths=file_paths,
        collection_name=collection_name,
        chunking_strategy="paragraph"
    )
    
    # Execute indexing
    result = await orchestrator.aindex_documents(request)
    
    return result

if __name__ == "__main__":
    file_paths = ["data/doc1.pdf", "data/doc2.docx", "data/doc3.txt"]
    result = asyncio.run(index_documents(file_paths, "production_docs"))
    print(f"Indexed {result.indexed_count} documents")
```

## Common Pitfalls

### 1. Component Coupling

**Problem**: Tight coupling between components makes the system hard to change.

**Solution**: Use interfaces and dependency injection:

```python
# BAD: Tight coupling
class MyAgent:
    def __init__(self):
        self.llm = OpenAIClient(...)  # Direct instantiation

# GOOD: Loose coupling
class MyAgent:
    def __init__(self, settings):
        self.llm = settings.create_llm()  # Dependency injection
```

### 2. Configuration Hardcoding

**Problem**: Hardcoded configuration values make the system inflexible.

**Solution**: Use settings and configuration files:

```python
# BAD: Hardcoded values
embeddings = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=texts
)

# GOOD: Configuration from settings
embedding_model = settings.get("embedding.model", "text-embedding-ada-002")
embeddings = openai.Embedding.create(
    model=embedding_model,
    input=texts
)
```

### 3. Error Handling

**Problem**: Insufficient error handling leads to cascade failures.

**Solution**: Implement comprehensive error handling:

```python
# BAD: No error handling
def process_document(file_path):
    loader = PDFLoader()
    document = loader.load(file_path)
    chunks = chunker.split(document.page_content)
    return chunks

# GOOD: Proper error handling
def process_document(file_path):
    try:
        loader = PDFLoader()
        document = loader.load(file_path)
        chunks = chunker.split(document.page_content)
        return chunks
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []
```

### 4. Memory Management

**Problem**: Poor memory management leads to context loss.

**Solution**: Implement proper memory handling:

```python
# BAD: Not storing memory updates
def process_message(message, conversation_id):
    memory = memory_manager.retrieve_working_memory(conversation_id)
    memory.add_user_message(message)
    # Process message...
    # Missing: storing memory updates

# GOOD: Proper memory handling
def process_message(message, conversation_id):
    memory = memory_manager.retrieve_working_memory(conversation_id)
    memory.add_user_message(message)
    # Process message...
    memory_manager.store_memory(
        conversation_id, 
        {"history": memory.get_raw_history()}
    )
```

## Best Practices

### 1. Follow the Interface-First Approach

- Define interfaces before implementations
- Depend on abstractions, not concrete classes
- Use protocols from the `seedwork` package

### 2. Use the Settings System

- Configure components through settings
- Extend the `Settings` class for application-specific needs
- Use factory methods for predefined components

### 3. Design for Testability

- Write unit tests for custom components
- Use dependency injection to make testing easier
- Mock external dependencies in tests

### 4. Implement Proper Error Handling

- Handle LLM API errors gracefully
- Provide fallbacks for component failures
- Log errors with appropriate context

### 5. Document Your Code

- Write clear docstrings for all classes and methods
- Provide examples in documentation
- Explain the rationale behind complex design decisions

### 6. Follow Clean Code Principles

- Keep methods small and focused
- Use clear, descriptive names
- Follow consistent code style

### 7. Monitor and Log

- Implement logging for important events
- Monitor LLM usage and costs
- Track performance metrics

### 8. Start Simple and Iterate

- Begin with simple implementations
- Add complexity incrementally
- Test thoroughly at each stage

By following these guidelines, you can build robust, maintainable AI applications with the Arshai framework. 