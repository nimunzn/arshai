# Migration Guide: From Settings to Three-Layer Architecture

This guide provides a comprehensive migration path from the old Settings-based patterns to Arshai's new three-layer, developer-empowered architecture.

## Overview of Changes

The major change is moving from **framework-controlled component creation** to **developer-controlled direct instantiation**. This gives you complete authority over how components are created, configured, and composed.

### Before (Settings-Based)
```python
# Framework decides everything
settings = Settings("config.yaml")
llm = settings.create_llm()
memory = settings.create_memory_manager()
agent = settings.create_agent("conversation", config)
```

### After (Direct Instantiation)
```python
# Developer decides everything
from arshai.llms.openai import OpenAIClient
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
from arshai.agents.working_memory import WorkingMemoryAgent

llm_config = ILLMConfig(model="gpt-4o", temperature=0.7)
llm = OpenAIClient(llm_config)

memory = RedisMemoryManager(redis_url=os.getenv("REDIS_URL"))
agent = WorkingMemoryAgent(llm, memory, "You are helpful")
```

## Migration Benefits

- **Explicit Dependencies**: No hidden Settings dependencies
- **Better Testing**: Easy to mock individual components  
- **More Flexible**: Compose components exactly as needed
- **Clearer Code**: Obvious what components are being used
- **Developer Control**: Complete authority over configuration

## Step-by-Step Migration

### Step 1: Identify Settings Usage

Find all places where Settings is used:

```python
# Common Settings patterns to replace:
settings = Settings()
settings = Settings("config.yaml")
llm = settings.create_llm()
memory = settings.create_memory_manager()
agent = settings.create_agent("type", config)
vector_db = settings.create_vector_db()
embedding = settings.create_embedding()
```

### Step 2: Replace LLM Creation

**Before:**
```python
from src.config.settings import Settings

settings = Settings()
llm = settings.create_llm()
```

**After:**
```python
from arshai.llms.openai import OpenAIClient  # or Azure, Gemini, OpenRouter
from arshai.core.interfaces.illm import ILLMConfig

# Direct instantiation
config = ILLMConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2000  # optional
)
llm = OpenAIClient(config)
```

**Environment Variables:**
Each LLM client reads its own environment variables:
- OpenAI: `OPENAI_API_KEY`
- Azure: `OPENAI_API_KEY`, `AZURE_DEPLOYMENT`, `AZURE_API_VERSION`
- Gemini: `GOOGLE_API_KEY` or Vertex AI service account
- OpenRouter: `OPENROUTER_API_KEY`

### Step 3: Replace Memory Manager Creation

**Before:**
```python
memory = settings.create_memory_manager()
```

**After:**
```python
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
from arshai.memory.working_memory.in_memory_manager import InMemoryManager

# Choose implementation based on your needs
if production:
    memory = RedisMemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        ttl=3600,
        key_prefix="my_app"  # optional
    )
else:
    memory = InMemoryManager(ttl=1800)
```

### Step 4: Replace Agent Creation

**Before:**
```python
agent_config = IAgentConfig(task_context="...", tools=[])
agent = settings.create_agent("conversation", agent_config)
```

**After:**
```python
from arshai.agents.working_memory import WorkingMemoryAgent
from arshai.agents.base import BaseAgent

# Option 1: Use existing agent
agent = WorkingMemoryAgent(
    llm_client=llm,
    memory_manager=memory,  # optional
    system_prompt="You are a helpful assistant"
)

# Option 2: Create custom agent
class MyAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> str:
        # Your custom logic
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result['llm_response']

# Instantiate custom agent
agent = MyAgent(llm, "Custom system prompt")
```

### Step 5: Replace Vector Database Creation

**Before:**
```python
vector_db, collection_config, embedding = settings.create_vector_db()
```

**After:**
```python
from arshai.embeddings.openai_embeddings import OpenAIEmbeddings
from arshai.vector_db.milvus_client import MilvusClient

# Create components independently
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
    # Reads OPENAI_API_KEY automatically
)

vector_db = MilvusClient(
    host=os.getenv("MILVUS_HOST", "localhost"),
    port=int(os.getenv("MILVUS_PORT", "19530"))
)

# Create collection config as needed
collection_config = {
    "collection_name": "my_documents",
    "dimension": embedding.dimension,
    "metric_type": "L2"
}
```

### Step 6: Replace Other Components

**Web Search:**
```python
# Before
web_search = settings.create_web_search()

# After
from arshai.tools import WebSearchTool
web_search = WebSearchTool()
```

**Reranker:**
```python
# Before
reranker = settings.create_reranker()

# After
from arshai.rerankers.flashrank_reranker import FlashRankReranker
reranker = FlashRankReranker(
    model_name="ms-marco-MiniLM-L-12-v2"
)
```

**Speech Processing:**
```python
# Before
speech = settings.create_speech_model()

# After
from arshai.speech.openai import OpenAISpeechProcessor
from arshai.core.interfaces.ispeech import ISpeechConfig

speech_config = ISpeechConfig(
    stt_model="whisper-1",
    tts_model="tts-1",
    tts_voice="alloy"
)
speech = OpenAISpeechProcessor(speech_config)
```

## Configuration Migration

If you were using configuration files with Settings, you have several options:

### Option 1: Optional Config Loader (Recommended)

```python
from arshai.config import load_config

# Load configuration (returns {} if file doesn't exist)
config = load_config("app.yaml")

# Use config data to create components
llm_settings = config.get("llm", {})
llm_config = ILLMConfig(
    model=llm_settings.get("model", "gpt-4o"),
    temperature=llm_settings.get("temperature", 0.7)
)
llm = OpenAIClient(llm_config)
```

### Option 2: Environment Variables Only

```python
import os

# Read directly from environment
llm_config = ILLMConfig(
    model=os.getenv("LLM_MODEL", "gpt-4o"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
)
llm = OpenAIClient(llm_config)
```

### Option 3: Custom Configuration System

```python
# Create your own configuration management
class AppConfig:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_llm_config(self) -> ILLMConfig:
        llm_config = self.config.get("llm", {})
        return ILLMConfig(
            model=llm_config.get("model", "gpt-4o"),
            temperature=llm_config.get("temperature", 0.7)
        )

app_config = AppConfig("config.yaml")
llm = OpenAIClient(app_config.get_llm_config())
```

## Complete Migration Example

Here's a complete before/after example:

### Before (Settings-Based)
```python
#!/usr/bin/env python3
from src.config.settings import Settings
from arshai.core.interfaces.iagent import IAgentConfig, IAgentInput

# Settings-controlled creation
settings = Settings("app.yaml")

# Framework decides component creation
llm = settings.create_llm()
memory = settings.create_memory_manager()
vector_db, collection_config, embedding = settings.create_vector_db()
web_search = settings.create_web_search()

# Agent creation through settings
agent_config = IAgentConfig(
    task_context="You are a helpful assistant",
    tools=[]
)
agent = settings.create_agent("conversation", agent_config)

# Usage
response = await agent.process(IAgentInput(message="Hello!"))
```

### After (Direct Instantiation)
```python
#!/usr/bin/env python3
import os
from arshai.llms.openai import OpenAIClient
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
from arshai.embeddings.openai_embeddings import OpenAIEmbeddings
from arshai.vector_db.milvus_client import MilvusClient
from arshai.tools import WebSearchTool
from arshai.agents.working_memory import WorkingMemoryAgent
from arshai.core.interfaces.illm import ILLMConfig
from arshai.core.interfaces.iagent import IAgentInput
from arshai.config import load_config

# Optional: Load configuration
config = load_config("app.yaml")  # Returns {} if file doesn't exist

# Developer-controlled creation
llm_config = ILLMConfig(
    model=config.get("llm", {}).get("model", "gpt-4o"),
    temperature=config.get("llm", {}).get("temperature", 0.7)
)
llm = OpenAIClient(llm_config)

memory = RedisMemoryManager(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    ttl=3600
)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = MilvusClient(
    host=os.getenv("MILVUS_HOST", "localhost"),
    port=int(os.getenv("MILVUS_PORT", "19530"))
)

web_search = WebSearchTool()

# Direct agent creation
agent = WorkingMemoryAgent(
    llm_client=llm,
    memory_manager=memory,
    system_prompt="You are a helpful assistant"
)

# Usage (same as before)
response = await agent.process(IAgentInput(message="Hello!"))
```

## Optional Utilities

If you want some convenience without losing control, optional utilities are available:

```python
# Optional utilities for common patterns
from arshai.utils.llm_utils import create_llm_client
from arshai.utils.memory_utils import create_memory_manager

# Still explicit, just more convenient
llm = create_llm_client("openai", {"model": "gpt-4o", "temperature": 0.7})
memory = create_memory_manager("redis", {"redis_url": "redis://localhost:6379"})

# You still control composition
agent = WorkingMemoryAgent(llm, memory, system_prompt)
```

## Testing Migration

### Before (Hard to Test)
```python
# Difficult to mock Settings dependencies
def test_agent():
    settings = Settings()  # Creates real components
    agent = settings.create_agent("conversation", config)  # Hard to mock
    # Testing is complex
```

### After (Easy to Test)
```python
from unittest.mock import Mock

def test_agent():
    # Easy to mock individual components
    mock_llm = Mock()
    mock_memory = Mock()
    
    # Direct instantiation with mocks
    agent = WorkingMemoryAgent(mock_llm, mock_memory, "test prompt")
    
    # Clear testing of specific behavior
    mock_llm.chat.return_value = {"llm_response": "test response"}
    result = await agent.process(IAgentInput(message="test"))
    
    assert result == "success"
    mock_llm.chat.assert_called_once()
```

## Common Migration Issues

### Issue 1: Missing Environment Variables

**Problem:** Components fail because they can't find environment variables that Settings used to provide.

**Solution:** Ensure each component has access to its required environment variables:

```python
# Check what each component needs:
# OpenAI: OPENAI_API_KEY
# Azure: OPENAI_API_KEY, AZURE_DEPLOYMENT, AZURE_API_VERSION  
# Redis: REDIS_URL
# Milvus: MILVUS_HOST, MILVUS_PORT

# Set them explicitly if needed:
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["REDIS_URL"] = "redis://localhost:6379"
```

### Issue 2: Configuration File Structure Changes

**Problem:** Old YAML structure doesn't work with new config loader.

**Solution:** Either restructure your YAML or create a config adapter:

```python
# Old structure in Settings
# llm:
#   provider: openai
#   model: gpt-4o

# New structure (simpler)
# llm:
#   model: gpt-4o
#   temperature: 0.7

# Or create an adapter
def adapt_old_config(old_config):
    llm_config = old_config.get("llm", {})
    return {
        "llm": {
            "model": llm_config.get("model", "gpt-4o"),
            "temperature": llm_config.get("temperature", 0.7)
        }
    }
```

### Issue 3: Component Interdependencies

**Problem:** Some components were automatically linked through Settings.

**Solution:** Make dependencies explicit:

```python
# Before: Settings automatically connected embedding to vector DB
vector_db, collection_config, embedding = settings.create_vector_db()

# After: Make connection explicit
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = MilvusClient(host="localhost", port=19530)

# If they need to work together, connect them explicitly
collection_config = {
    "dimension": embedding.dimension,  # Explicit connection
    "metric_type": "L2"
}
```

### Issue 4: Factory Pattern Dependencies

**Problem:** Code relied on factory patterns for component discovery.

**Solution:** Use direct imports or create your own registry:

```python
# Instead of factory discovery
# provider_class = LLMFactory.get_provider("openai")

# Use direct imports
from arshai.llms.openai import OpenAIClient

# Or create your own registry if needed
LLM_PROVIDERS = {
    "openai": OpenAIClient,
    "azure": AzureClient,
    # ...
}

provider_class = LLM_PROVIDERS["openai"]
llm = provider_class(config)
```

## Migration Checklist

- [ ] **Identify all Settings usage** in your codebase
- [ ] **Replace Settings imports** with direct component imports
- [ ] **Convert create_llm()** to direct LLM client instantiation
- [ ] **Convert create_memory_manager()** to direct memory manager creation
- [ ] **Convert create_agent()** to direct agent instantiation
- [ ] **Convert create_vector_db()** to separate embedding and vector DB creation
- [ ] **Replace other Settings methods** (web search, reranker, etc.)
- [ ] **Update configuration loading** to use optional config loader
- [ ] **Set environment variables** for all components
- [ ] **Update tests** to use direct instantiation and mocking
- [ ] **Verify all imports** are explicit and clear
- [ ] **Test the migrated code** end-to-end

## Post-Migration Benefits

After migration, you'll have:

1. **Complete Control**: You decide exactly how components are created and configured
2. **Clear Dependencies**: All component requirements are explicit in your code
3. **Better Testing**: Easy to mock individual components for unit tests
4. **Flexible Architecture**: Compose components exactly as your application needs
5. **No Hidden Magic**: Everything is visible and under your control
6. **Easier Debugging**: Clear code path from creation to usage

The migration transforms your code from framework-controlled to developer-controlled, giving you the power to build exactly what you need.