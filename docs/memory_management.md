# Memory Management System

The memory management system in Arshai provides a flexible framework for storing and retrieving conversation context and knowledge. This document explains the architecture, components, and usage of the memory system.

## Architecture Overview

The memory management system follows a clean, modular architecture:

```
┌─────────────────────────────────────────────────────────┐
│                  Memory Manager                          │
└─────────────────┬─────────────────────┬─────────────────┘
                  │                     │
          ┌───────▼────────┐    ┌───────▼────────┐
          │ Working Memory │    │  Long-Term     │
          │ (Conversations)│    │  Knowledge     │
          └───────┬────────┘    └───────┬────────┘
                  │                     │
          ┌───────▼────────┐    ┌───────▼────────┐
          │  Storage       │    │   Storage      │
          │  Providers     │    │   Providers    │
          └───────┬────────┘    └───────┬────────┘
                  │                     │
                  ▼                     ▼
          ┌───────────────┐    ┌───────────────┐
          │  In-Memory    │    │  Persistent   │
          │  Storage      │    │  Storage      │
          └───────────────┘    └───────────────┘
```

## Core Components

### Memory Manager

The `IMemoryManager` interface defines the contract for memory management:

```python
class IMemoryManager(Protocol):
    """Interface for memory management."""
    
    def retrieve_working_memory(self, conversation_id: str) -> IWorkingMemory:
        """
        Retrieve working memory for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Working memory for the conversation
        """
        ...
    
    def store_memory(self, conversation_id: str, memory_update: Dict[str, Any]) -> None:
        """
        Store memory for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            memory_update: Memory data to store
        """
        ...
```

The `MemoryManager` implementation provides a unified interface for accessing different types of memory:

```python
class MemoryManager(IMemoryManager):
    """
    Implementation of the memory manager.
    
    This class provides a unified interface for accessing
    working memory and long-term knowledge.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory manager.
        
        Args:
            config: Configuration for memory components
        """
        self.working_memory_provider = self._create_working_memory_provider(
            config.get("working_memory", {})
        )
        self.knowledge_store = self._create_knowledge_store(
            config.get("knowledge_store", {})
        )
        
    def retrieve_working_memory(self, conversation_id: str) -> IWorkingMemory:
        """Retrieve working memory for a conversation."""
        return self.working_memory_provider.retrieve(conversation_id)
    
    def store_memory(self, conversation_id: str, memory_update: Dict[str, Any]) -> None:
        """Store memory for a conversation."""
        self.working_memory_provider.store(conversation_id, memory_update)
```

### Working Memory

Working memory represents short-term conversation context. It's implemented through the `IWorkingMemory` interface:

```python
class IWorkingMemory(Protocol):
    """Interface for working memory."""
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to memory."""
        ...
    
    def add_agent_message(self, message: str) -> None:
        """Add an agent message to memory."""
        ...
    
    def add_system_note(self, note: str) -> None:
        """Add a system note to memory."""
        ...
    
    def get_formatted_memory(self) -> str:
        """Get formatted memory for prompt inclusion."""
        ...
    
    def get_raw_history(self) -> List[Dict[str, str]]:
        """Get raw memory entries."""
        ...
```

Implementations include:
- `InMemoryWorkingMemory`: Stores conversation history in memory
- `RedisWorkingMemory`: Stores conversation history in Redis
- `DatabaseWorkingMemory`: Stores conversation history in a database

### Storage Providers

Storage providers handle the actual storage of memory data:

```python
class IMemoryStorage(Protocol):
    """Interface for memory storage providers."""
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get data from storage.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored data, or None if not found
        """
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set data in storage.
        
        Args:
            key: The key to store under
            value: The data to store
            ttl: Time-to-live in seconds (optional)
        """
        ...
    
    def delete(self, key: str) -> None:
        """Delete data from storage."""
        ...
```

Implementations include:
- `InMemoryStorage`: Simple in-memory storage with optional TTL
- `RedisStorage`: Persistent storage using Redis
- `SQLStorage`: Persistent storage using SQL databases

## Memory Types

### Working Memory

Working memory holds recent conversation history and is typically structured as a sequence of entries:

```python
class WorkingMemoryEntry:
    """Entry in working memory."""
    
    role: str              # "user", "assistant", or "system"
    content: str           # Message content
    timestamp: datetime    # When the message was added
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
```

Working memory is specifically designed for:
- Maintaining conversation context for LLMs
- Tracking the flow of conversation
- Supporting multi-turn interactions

### Knowledge Store

The knowledge store manages long-term, persistent knowledge:

```python
class IKnowledgeStore(Protocol):
    """Interface for knowledge stores."""
    
    def add_knowledge(self, key: str, data: Dict[str, Any]) -> None:
        """
        Add knowledge to the store.
        
        Args:
            key: Knowledge identifier
            data: Knowledge data
        """
        ...
    
    def retrieve_knowledge(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge from the store.
        
        Args:
            key: Knowledge identifier
            
        Returns:
            Knowledge data, or None if not found
        """
        ...
    
    def update_knowledge(self, key: str, data: Dict[str, Any]) -> None:
        """
        Update existing knowledge.
        
        Args:
            key: Knowledge identifier
            data: Updated knowledge data
        """
        ...
    
    def delete_knowledge(self, key: str) -> None:
        """
        Delete knowledge from the store.
        
        Args:
            key: Knowledge identifier
        """
        ...
```

The knowledge store is specifically designed for:
- Storing facts and information learned through conversations
- Managing user preferences and settings
- Maintaining persistent data across conversations

## Using the Memory System

### Creating a Memory Manager

To create a memory manager, use the `MemoryFactory`:

```python
from arshai import Settings
from src.factories.memory_factory import MemoryFactory

# Use settings
settings = Settings()
memory_manager = settings.create_memory_manager()

# Or create directly with configuration
memory_config = {
    "working_memory": {
        "provider": "in_memory",
        "ttl": 3600  # 1 hour in seconds
    },
    "knowledge_store": {
        "provider": "redis",
        "url": "redis://localhost:6379/0"
    }
}
memory_manager = MemoryFactory.create_memory_manager_service(memory_config)
```

### Working with Conversation Memory

Retrieve and update working memory:

```python
# Retrieve working memory for a conversation
conversation_id = "user123-session456"
working_memory = memory_manager.retrieve_working_memory(conversation_id)

# Add messages
working_memory.add_user_message("Hello, what can you help me with today?")
working_memory.add_agent_message("I can help you with information, tasks, and more. What do you need?")
working_memory.add_system_note("User authenticated as premium member")

# Get formatted memory for LLM prompt
formatted_memory = working_memory.get_formatted_memory()

# Store updated memory
memory_manager.store_memory(conversation_id, {"history": working_memory.get_raw_history()})
```

### Memory in Agent Implementation

Integrate memory into an agent:

```python
class MyAgent(IAgent):
    """Simple agent implementation with memory."""
    
    def __init__(self, config: IAgentConfig, settings):
        """Initialize the agent."""
        self.config = config
        self.settings = settings
        self.llm = settings.create_llm()
        self.memory_manager = settings.create_memory_manager()
    
    def process_message(self, input: IAgentInput) -> IAgentOutput:
        """Process a message with memory integration."""
        
        # Get conversation memory
        working_memory = self.memory_manager.retrieve_working_memory(input.conversation_id)
        
        # Add user message to memory
        working_memory.add_user_message(input.message)
        
        # Prepare system prompt with memory context
        system_prompt = self._prepare_system_prompt(working_memory)
        
        # Get LLM response
        llm_response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input.message}
            ]
        )
        
        # Add agent response to memory
        working_memory.add_agent_message(llm_response.message)
        
        # Store updated memory
        self.memory_manager.store_memory(
            input.conversation_id, 
            {"history": working_memory.get_raw_history()}
        )
        
        # Return response
        return IAgentOutput(
            agent_message=llm_response.message,
            memory={"history": working_memory.get_raw_history()}
        )
    
    def _prepare_system_prompt(self, working_memory: IWorkingMemory) -> str:
        """Prepare system prompt with memory integration."""
        prompt = f"You are a helpful assistant. {self.config.task_context}\n\n"
        prompt += "Conversation history:\n"
        prompt += working_memory.get_formatted_memory()
        return prompt
```

## Memory Storage Options

### In-Memory Storage

Simplest storage option, ideal for development and testing:

```python
# Create in-memory working memory
memory_config = {
    "working_memory": {
        "provider": "in_memory",
        "ttl": 3600  # 1 hour in seconds
    }
}
memory_manager = MemoryFactory.create_memory_manager_service(memory_config)
```

### Redis Storage

Redis provides persistent storage with optional expiration:

```python
# Create Redis working memory
memory_config = {
    "working_memory": {
        "provider": "redis",
        "url": "redis://localhost:6379/0",
        "ttl": 86400  # 24 hours in seconds
    }
}
memory_manager = MemoryFactory.create_memory_manager_service(memory_config)
```

### Database Storage

SQL databases offer fully persistent storage:

```python
# Create database working memory
memory_config = {
    "working_memory": {
        "provider": "database",
        "connection_string": "sqlite:///conversations.db",
        "ttl": None  # No expiration
    }
}
memory_manager = MemoryFactory.create_memory_manager_service(memory_config)
```

## Advanced Memory Features

### Memory Summarization

For long conversations, you can summarize memory:

```python
from arshai import Settings
from src.memory.summarization import MemorySummarizer

# Create settings and memory manager
settings = Settings()
memory_manager = settings.create_memory_manager()
llm = settings.create_llm()

# Create summarizer
summarizer = MemorySummarizer(llm)

# Retrieve working memory
conversation_id = "user123-session456"
working_memory = memory_manager.retrieve_working_memory(conversation_id)

# Summarize if too long
if len(working_memory.get_raw_history()) > 20:
    summarized_memory = summarizer.summarize(working_memory)
    
    # Replace working memory with summary
    new_memory = memory_manager.retrieve_working_memory(conversation_id + "_new")
    new_memory.add_system_note(f"Summary of previous conversation: {summarized_memory}")
    
    # Continue with the new memory
    working_memory = new_memory
```

### Memory Filtering

You can filter memory to include only relevant messages:

```python
# Filter memory to include only user questions
def filter_questions(memory_entries):
    filtered = []
    for entry in memory_entries:
        if entry["role"] == "user" and "?" in entry["content"]:
            filtered.append(entry)
    return filtered

# Get working memory
working_memory = memory_manager.retrieve_working_memory(conversation_id)
raw_history = working_memory.get_raw_history()

# Apply filter
questions_only = filter_questions(raw_history)

# Use filtered history
for entry in questions_only:
    print(f"Question: {entry['content']}")
```

### Memory Persistence Across Sessions

You can use the knowledge store to maintain information across sessions:

```python
# Store user preferences
knowledge_store = memory_manager.knowledge_store
user_id = "user123"

# Add or update preferences
knowledge_store.add_knowledge(
    f"preferences:{user_id}", 
    {
        "language": "en",
        "theme": "dark",
        "notifications": True
    }
)

# Retrieve in a later session
preferences = knowledge_store.retrieve_knowledge(f"preferences:{user_id}")
if preferences:
    user_language = preferences.get("language", "en")
    print(f"User language: {user_language}")
```

## Best Practices

### Memory Management

1. **Use Appropriate TTL**: Set reasonable time-to-live values for different use cases
   - Short for casual conversations (1-24 hours)
   - Longer for ongoing support sessions (days to weeks)
   - Permanent for important information

2. **Handle Multi-User Scenarios**: Use unique conversation IDs that include user identification
   - Example: `{user_id}-{session_id}` format

3. **Implement Memory Cleanup**: Periodically clean up expired memory
   - For Redis and databases, use built-in expiration mechanisms
   - For in-memory storage, implement periodic cleanup

### Memory Structure

1. **Keep History Structured**: Maintain clear role separation (user/assistant/system)

2. **Include Metadata**: Add timestamps and additional context to memory entries

3. **Use System Notes**: Add system notes for important context that isn't part of the conversation

### Performance Considerations

1. **Batch Operations**: For high-throughput applications, batch memory operations

2. **Cache Heavily Used Memory**: Cache frequently accessed memory in application memory

3. **Monitor Memory Usage**: Set up monitoring for memory size and access patterns

## Configuration

Configure memory through settings:

```yaml
# config.yaml
memory:
  working_memory:
    provider: redis
    url: redis://localhost:6379/0
    ttl: 86400
    prefix: conversation:
  knowledge_store:
    provider: database
    connection_string: postgresql://user:password@localhost/knowledge
    table_name: knowledge_items
```

```python
from arshai import Settings

# Load settings
settings = Settings("config.yaml")

# Create memory manager
memory_manager = settings.create_memory_manager()
```

## Examples

### Basic Memory Usage

```python
from arshai import Settings
from seedwork.interfaces.iagent import IAgentInput, IAgentOutput

# Initialize settings and memory manager
settings = Settings()
memory_manager = settings.create_memory_manager()

# Function to process a message
def process_message(message: str, conversation_id: str) -> str:
    # Retrieve memory
    working_memory = memory_manager.retrieve_working_memory(conversation_id)
    
    # Add user message
    working_memory.add_user_message(message)
    
    # Get formatted memory for prompt
    formatted_memory = working_memory.get_formatted_memory()
    print(f"Memory:\n{formatted_memory}")
    
    # Generate response (using a mock for this example)
    response = f"Echo: {message}"
    
    # Add agent response to memory
    working_memory.add_agent_message(response)
    
    # Store updated memory
    memory_manager.store_memory(
        conversation_id, 
        {"history": working_memory.get_raw_history()}
    )
    
    return response

# Simulate a conversation
conversation_id = "demo-123"
responses = []

responses.append(process_message("Hello, who are you?", conversation_id))
responses.append(process_message("What can you help me with?", conversation_id))
responses.append(process_message("Tell me about memory management.", conversation_id))

for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
```

### Memory with Redis Storage

```python
from arshai import Settings
from src.factories.memory_factory import MemoryFactory
from seedwork.interfaces.imemorymanager import IMemoryManager

# Create memory manager with Redis storage
memory_config = {
    "working_memory": {
        "provider": "redis",
        "url": "redis://localhost:6379/0",
        "ttl": 3600
    }
}
memory_manager = MemoryFactory.create_memory_manager_service(memory_config)

# Use the memory manager
conversation_id = "redis-demo-123"
working_memory = memory_manager.retrieve_working_memory(conversation_id)

# Add messages
working_memory.add_user_message("This message is stored in Redis")
working_memory.add_agent_message("Your message has been stored")

# Store memory
memory_manager.store_memory(
    conversation_id, 
    {"history": working_memory.get_raw_history()}
)

# Later, retrieve the memory
retrieved_memory = memory_manager.retrieve_working_memory(conversation_id)
print(retrieved_memory.get_formatted_memory())
```

### Multi-User Memory Management

```python
from arshai import Settings
from src.memory.conversation_manager import ConversationManager

# Initialize settings and conversation manager
settings = Settings()
conversation_manager = ConversationManager(settings)

# Process messages for multiple users
def process_user_message(user_id: str, message: str) -> str:
    # Generate conversation ID
    conversation_id = f"user-{user_id}"
    
    # Process the message
    response = conversation_manager.process_message(conversation_id, message)
    
    return response

# Simulate multiple users
user1_responses = []
user2_responses = []

user1_responses.append(process_user_message("alice", "Hello, I'm Alice"))
user2_responses.append(process_user_message("bob", "Hi there, I'm Bob"))
user1_responses.append(process_user_message("alice", "What were we talking about?"))
user2_responses.append(process_user_message("bob", "Do you remember my name?"))

# Print responses
print("Alice's conversation:")
for resp in user1_responses:
    print(f"- {resp}")

print("\nBob's conversation:")
for resp in user2_responses:
    print(f"- {resp}")
``` 