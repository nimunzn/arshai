# Component Usage Guide

This guide provides direct usage examples for all Arshai framework components using the three-layer architecture approach.

## Component Overview by Layer

### Layer 1: Core AI Components (Use As Provided)
- **LLM Clients**: OpenAI, Azure, Gemini, OpenRouter
- **Core Interfaces**: ILLM, IAgent, IMemoryManager, etc.

### Layer 2: Customizable Components (Extend or Use)
- **Agent Base Classes**: BaseAgent, WorkingMemoryAgent
- **Memory Managers**: Redis, In-Memory
- **Embedding Services**: OpenAI, Voyage AI
- **Vector Databases**: Milvus

### Layer 3: Application Components (Complete Control)
- **Tools**: WebSearch, Custom tools
- **Workflows**: Optional base classes and interfaces
- **Configuration**: Optional config loader

## LLM Client Usage (Layer 1)

### OpenAI Client

```python
from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput

# Environment variable required: OPENAI_API_KEY

# Create configuration
config = ILLMConfig(
    model="gpt-4o",  # or gpt-4, gpt-3.5-turbo, gpt-4o-mini
    temperature=0.7,
    max_tokens=1000  # optional
)

# Create client
client = OpenAIClient(config)

# Simple usage
input_data = ILLMInput(
    system_prompt="You are a helpful assistant.",
    user_message="Hello, world!"
)

response = await client.chat(input_data)
print(response["llm_response"])
print(f"Tokens used: {response['usage']['total_tokens']}")
```

### Azure OpenAI Client

```python
from arshai.llms.azure import AzureClient

# Environment variables required:
# - OPENAI_API_KEY (your Azure API key)
# - AZURE_DEPLOYMENT (your deployment name)
# - AZURE_API_VERSION (e.g., "2024-02-01")

config = ILLMConfig(
    model="gpt-4",  # Must match your Azure deployment
    temperature=0.7
)

client = AzureClient(config)

# Usage is identical to OpenAI
response = await client.chat(input_data)
```

### Google Gemini Client

```python
from arshai.llms.google_genai import GeminiClient

# Option 1: API Key authentication
# Environment variable: GOOGLE_API_KEY

# Option 2: Vertex AI authentication
# Environment variables:
# - VERTEX_AI_SERVICE_ACCOUNT_PATH
# - VERTEX_AI_PROJECT_ID  
# - VERTEX_AI_LOCATION

config = ILLMConfig(
    model="gemini-1.5-pro",  # or gemini-1.5-flash
    temperature=0.7
)

client = GeminiClient(config)
response = await client.chat(input_data)
```

### OpenRouter Client

```python
from arshai.llms.openrouter import OpenRouterClient

# Environment variables:
# - OPENROUTER_API_KEY (required)
# - OPENROUTER_SITE_URL (optional)
# - OPENROUTER_APP_NAME (optional)

config = ILLMConfig(
    model="anthropic/claude-3.5-sonnet",  # Any OpenRouter model
    temperature=0.7
)

client = OpenRouterClient(config)
response = await client.chat(input_data)
```

### Function Calling with LLM Clients

```python
# Define tools
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 22°C, sunny"

def calculate(expression: str) -> float:
    """Calculate mathematical expression."""
    return eval(expression)  # Note: Use safely in production

async def log_query(query: str, timestamp: str) -> None:
    """BACKGROUND TASK: Log user query."""
    print(f"Logged query at {timestamp}: {query}")

# Use with any LLM client
input_data = ILLMInput(
    system_prompt="You are a helpful assistant with access to tools.",
    user_message="What's the weather in Tokyo and what's 15 * 7?",
    regular_functions={
        "get_weather": get_weather,
        "calculate": calculate
    },
    background_tasks={
        "log_query": log_query
    }
)

# Works with any LLM client
response = await client.chat(input_data)
```

### Streaming with LLM Clients

```python
# Streaming works with all LLM clients
async for chunk in client.stream(input_data):
    if chunk.get("llm_response"):
        print(chunk["llm_response"], end="", flush=True)
    
    # Functions execute in real-time during streaming!
```

## Agent Usage (Layer 2)

### Working Memory Agent

```python
from arshai.agents.working_memory import WorkingMemoryAgent
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
from arshai.core.interfaces.iagent import IAgentInput

# Create dependencies
llm_client = OpenAIClient(config)
memory_manager = RedisMemoryManager(
    redis_url="redis://localhost:6379",
    ttl=3600
)

# Create agent
agent = WorkingMemoryAgent(
    llm_client=llm_client,
    memory_manager=memory_manager,
    system_prompt="You are a helpful assistant with memory."
)

# Use agent
input_data = IAgentInput(
    message="Remember that my favorite color is blue",
    metadata={"conversation_id": "conv_123"}
)

result = await agent.process(input_data)
print(result)  # Returns "success" or error message
```

### Custom Agent with BaseAgent

```python
from arshai.agents.base import BaseAgent
from typing import Dict, Any

class AnalysisAgent(BaseAgent):
    """Agent specialized for data analysis tasks."""
    
    def __init__(self, llm_client: ILLM, data_source: DataSource):
        super().__init__(
            llm_client=llm_client, 
            system_prompt="You are a data analysis expert."
        )
        self.data_source = data_source
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # Define analysis tools
        def query_data(sql_query: str) -> str:
            return self.data_source.execute_query(sql_query)
        
        def create_visualization(data: str, chart_type: str) -> str:
            return f"Created {chart_type} chart with data: {data[:50]}..."
        
        async def save_analysis(analysis: str) -> None:
            """BACKGROUND TASK: Save analysis results."""
            await self.data_source.save_analysis(analysis)
        
        # Prepare LLM input with tools
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions={
                "query_data": query_data,
                "create_visualization": create_visualization
            },
            background_tasks={
                "save_analysis": save_analysis
            }
        )
        
        result = await self.llm_client.chat(llm_input)
        
        return {
            "analysis": result["llm_response"],
            "usage": result["usage"],
            "confidence": self._assess_confidence(result)
        }
    
    def _assess_confidence(self, result: Dict[str, Any]) -> float:
        # Custom confidence assessment logic
        return 0.85

# Usage
data_source = DatabaseDataSource(connection_string)
agent = AnalysisAgent(llm_client, data_source)

result = await agent.process(IAgentInput(
    message="Analyze sales trends for Q1 2024"
))
```

### Direct Interface Implementation

```python
from arshai.core.interfaces.iagent import IAgent

class MinimalAgent(IAgent):
    """Minimal agent implementing IAgent directly."""
    
    def __init__(self, llm_client: ILLM, custom_behavior: str):
        self.llm_client = llm_client
        self.custom_behavior = custom_behavior
    
    async def process(self, input: IAgentInput) -> str:
        # Complete control over processing
        enhanced_prompt = f"{self.custom_behavior}\n\nUser message: {input.message}"
        
        llm_input = ILLMInput(
            system_prompt="You are a custom assistant.",
            user_message=enhanced_prompt
        )
        
        result = await self.llm_client.chat(llm_input)
        return result["llm_response"].upper()  # Custom post-processing

# Usage
agent = MinimalAgent(llm_client, "Always be enthusiastic")
result = await agent.process(IAgentInput(message="Hello"))
```

## Memory Manager Usage (Layer 2)

### Redis Memory Manager

```python
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager

# Create Redis memory manager
memory_manager = RedisMemoryManager(
    redis_url="redis://localhost:6379/0",  # Can be from env var
    ttl=3600,  # 1 hour
    key_prefix="my_app"  # Optional prefix
)

# Store memory
memory_id = await memory_manager.store({
    "conversation_id": "conv_123",
    "working_memory": "User is asking about product pricing",
    "metadata": {"user_id": "user_456", "session_id": "session_789"}
})

# Retrieve memory
memories = await memory_manager.retrieve({
    "conversation_id": "conv_123"
})

if memories:
    memory_data = memories[0].data
    print(f"Retrieved: {memory_data}")

# Update memory
await memory_manager.update({
    "memory_id": memory_id,
    "working_memory": "Updated conversation context"
})

# Delete memory
await memory_manager.delete({
    "conversation_id": "conv_123"
})
```

### In-Memory Manager (for development/testing)

```python
from arshai.memory.working_memory.in_memory_manager import InMemoryManager

# Create in-memory manager (no persistence)
memory_manager = InMemoryManager(
    ttl=1800  # 30 minutes
)

# Usage is identical to Redis manager
memory_id = await memory_manager.store({
    "conversation_id": "conv_123",
    "working_memory": "Temporary memory for testing"
})
```

## Embedding Services Usage (Layer 2)

### OpenAI Embeddings

```python
from arshai.embeddings.openai_embeddings import OpenAIEmbeddings

# Environment variable required: OPENAI_API_KEY

# Create embedding service
embedding_service = OpenAIEmbeddings(
    model="text-embedding-3-small"  # or text-embedding-3-large
)

# Embed single text
text = "This is a sample document"
embedding = await embedding_service.embed_text(text)
print(f"Embedding dimension: {len(embedding)}")

# Embed multiple documents
documents = [
    "First document content",
    "Second document content",
    "Third document content"
]

embeddings = await embedding_service.embed_documents(documents)
print(f"Generated {len(embeddings)} embeddings")

# Get embedding dimension
print(f"Service dimension: {embedding_service.dimension}")
```

### Voyage AI Embeddings

```python
from arshai.embeddings.voyageai_embedding import VoyageAIEmbeddings

# Environment variable required: VOYAGEAI_API_KEY

embedding_service = VoyageAIEmbeddings(
    model="voyage-2"  # or other Voyage AI models
)

# Usage is identical to OpenAI embeddings
embedding = await embedding_service.embed_text("Sample text")
embeddings = await embedding_service.embed_documents(documents)
```

## Vector Database Usage (Layer 2)

### Milvus Client

```python
from arshai.vector_db.milvus_client import MilvusClient
from arshai.embeddings.openai_embeddings import OpenAIEmbeddings

# Environment variables for Milvus (optional):
# - MILVUS_HOST (default: localhost)
# - MILVUS_PORT (default: 19530)

# Create components
embedding_service = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = MilvusClient(
    host="localhost",
    port=19530
)

# Create collection
collection_name = "my_documents"
await vector_db.create_collection(
    collection_name=collection_name,
    dimension=embedding_service.dimension,
    index_type="IVF_FLAT",
    metric_type="L2"
)

# Store vectors
documents = ["Document 1 content", "Document 2 content"]
embeddings = await embedding_service.embed_documents(documents)

# Store in vector database
await vector_db.store_vectors(
    collection_name=collection_name,
    vectors=embeddings,
    documents=documents,
    metadata={"source": "user_upload", "timestamp": "2024-01-01"}
)

# Search for similar vectors
query = "Find similar documents"
query_embedding = await embedding_service.embed_text(query)

results = await vector_db.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5
)

for result in results:
    print(f"Score: {result.score}, Document: {result.document}")
```

## Tool Usage (Layer 3)

### Web Search Tool (Optional Framework Tool)

```python
from arshai.tools import WebSearchTool

# Environment variable: SEARX_INSTANCE (your SearxNG instance URL)

# Create web search tool
search_tool = WebSearchTool()

# Use directly
results = await search_tool.search("Python programming best practices", num_results=5)
print(f"Found {len(results)} results")

# Use with agents
class ResearchAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, search_tool: WebSearchTool):
        super().__init__(llm_client, "You are a research assistant")
        self.search_tool = search_tool
    
    async def process(self, input: IAgentInput) -> str:
        async def web_search(query: str, num_results: int = 3) -> str:
            results = await self.search_tool.search(query, num_results)
            return "\n".join([f"- {result.title}: {result.content}" for result in results])
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions={"web_search": web_search}
        )
        
        result = await self.llm_client.chat(llm_input)
        return result["llm_response"]

# Usage
agent = ResearchAgent(llm_client, search_tool)
result = await agent.process(IAgentInput(
    message="Research the latest developments in AI safety"
))
```

### Custom Tool Development

```python
class DatabaseTool:
    """Custom tool for database operations."""
    
    def __init__(self, connection_string: str):
        self.db = self._connect(connection_string)
    
    def search_customers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for customers in database."""
        # Database query logic
        return [
            {"id": "1", "name": "John Doe", "email": "john@example.com"},
            {"id": "2", "name": "Jane Smith", "email": "jane@example.com"}
        ]
    
    def get_order_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get customer's order history."""
        # Database query logic
        return [
            {"order_id": "order_123", "total": 99.99, "date": "2024-01-01"}
        ]
    
    async def update_customer_notes(self, customer_id: str, notes: str) -> None:
        """BACKGROUND TASK: Update customer notes."""
        # Database update logic
        print(f"Updated notes for customer {customer_id}: {notes}")
    
    def _connect(self, connection_string: str):
        # Database connection logic
        return None

# Use custom tool with agent
class CustomerServiceAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, db_tool: DatabaseTool):
        super().__init__(llm_client, "You are a customer service representative")
        self.db_tool = db_tool
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        regular_functions = {
            "search_customers": self.db_tool.search_customers,
            "get_order_history": self.db_tool.get_order_history,
        }
        
        background_tasks = {
            "update_customer_notes": self.db_tool.update_customer_notes
        }
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions=regular_functions,
            background_tasks=background_tasks
        )
        
        result = await self.llm_client.chat(llm_input)
        return {"response": result["llm_response"]}
```

## Configuration Usage (Layer 3)

### Configuration Loader (Optional Utility)

```python
from arshai.config import load_config, ConfigLoader

# Method 1: Function approach
config = load_config("app.yaml")  # Returns {} if file doesn't exist
print(f"Config loaded: {config}")

# Method 2: Class approach
config_loader = ConfigLoader("app.yaml")
database_config = config_loader.get("database", {})
llm_config = config_loader.get("llm", {})

# Use config to create components
if llm_config:
    client_config = ILLMConfig(
        model=llm_config.get("model", "gpt-4o"),
        temperature=llm_config.get("temperature", 0.7)
    )
    llm_client = OpenAIClient(client_config)
else:
    # Fallback to defaults
    llm_client = OpenAIClient(ILLMConfig(model="gpt-4o"))
```

**Example YAML Configuration:**
```yaml
# app.yaml
llm:
  model: "gpt-4o"
  temperature: 0.7
  max_tokens: 2000

database:
  redis_url: "redis://localhost:6379"
  ttl: 3600

tools:
  web_search:
    enabled: true
    max_results: 5
  
debug: false
```

## Speech Processing Usage (Layer 2)

### OpenAI Speech Processor

```python
from arshai.speech.openai import OpenAISpeechProcessor
from arshai.core.interfaces.ispeech import ISpeechConfig

# Environment variable required: OPENAI_API_KEY

# Create speech configuration
speech_config = ISpeechConfig(
    stt_model="whisper-1",
    tts_model="tts-1",
    tts_voice="alloy"  # or nova, echo, fable, onyx, shimmer
)

# Create speech processor
speech_processor = OpenAISpeechProcessor(speech_config)

# Speech to text
with open("audio.mp3", "rb") as audio_file:
    transcript = await speech_processor.transcribe(audio_file)
    print(f"Transcript: {transcript}")

# Text to speech
text = "Hello, this is a test of text to speech"
audio_data = await speech_processor.synthesize(text)

# Save audio
with open("output.mp3", "wb") as audio_file:
    audio_file.write(audio_data)
```

### Azure Speech Processor

```python
from arshai.speech.azure import AzureSpeechProcessor

# Environment variables required:
# - AZURE_SPEECH_KEY
# - AZURE_SPEECH_REGION

speech_config = ISpeechConfig(
    region="eastus",
    tts_voice="en-US-JennyNeural"
)

speech_processor = AzureSpeechProcessor(speech_config)

# Usage is similar to OpenAI speech processor
transcript = await speech_processor.transcribe(audio_file)
audio_data = await speech_processor.synthesize(text)
```

## Reranker Usage (Layer 2)

### FlashRank Reranker

```python
from arshai.rerankers.flashrank_reranker import FlashRankReranker

# Create reranker
reranker = FlashRankReranker(
    model_name="ms-marco-MiniLM-L-12-v2"
)

# Rerank search results
query = "How to use machine learning"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a programming language",
    "Machine learning algorithms can learn from data",
    "Web development uses HTML and CSS"
]

# Rerank documents by relevance to query
reranked_results = await reranker.rerank(
    query=query,
    documents=documents,
    top_k=3
)

for result in reranked_results:
    print(f"Score: {result.score}, Text: {result.document}")
```

### Voyage Reranker

```python
from arshai.rerankers.voyage_reranker import VoyageReranker

# Environment variable required: VOYAGEAI_API_KEY

reranker = VoyageReranker(
    model="rerank-lite-1"
)

# Usage is identical to FlashRank
reranked_results = await reranker.rerank(query, documents, top_k=3)
```

## Workflow Usage (Layer 3) - Optional Framework Components

### Using Workflow Base Classes (Optional)

```python
from arshai.workflows.base import BaseWorkflowRunner
from arshai.core.interfaces.iworkflow import IWorkflowConfig

class CustomerSupportWorkflow(BaseWorkflowRunner):
    """Custom workflow extending base class."""
    
    def __init__(self, agents: Dict[str, IAgent]):
        workflow_config = IWorkflowConfig(
            workflow_id="customer_support",
            max_steps=10
        )
        super().__init__(workflow_config)
        self.agents = agents
    
    async def configure_nodes(self):
        # Define workflow steps
        self.add_node("classify", self._classify_request)
        self.add_node("technical_support", self._handle_technical)
        self.add_node("billing_support", self._handle_billing)
        self.add_node("escalate", self._escalate_to_human)
    
    async def _classify_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        classifier_agent = self.agents["classifier"]
        result = await classifier_agent.process(IAgentInput(
            message=input_data["message"],
            metadata={"step": "classification"}
        ))
        return {"classification": result, "original_message": input_data["message"]}
    
    async def _handle_technical(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        tech_agent = self.agents["technical"]
        result = await tech_agent.process(IAgentInput(
            message=input_data["original_message"],
            metadata={"classification": "technical"}
        ))
        return {"response": result, "handled_by": "technical"}

# Usage
workflow = CustomerSupportWorkflow({
    "classifier": classifier_agent,
    "technical": technical_agent,
    "billing": billing_agent
})

result = await workflow.run({"message": "My app crashed when exporting data"})
```

### Direct Workflow Implementation (Maximum Control)

```python
from arshai.core.interfaces.iworkflow import IWorkflowRunner

class CustomWorkflowRunner(IWorkflowRunner):
    """Complete custom workflow implementation."""
    
    def __init__(self, agents: Dict[str, IAgent], config: Dict[str, Any]):
        self.agents = agents
        self.config = config
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Custom orchestration logic."""
        # Step 1: Initial processing
        initial_agent = self.agents["initial_processor"]
        initial_result = await initial_agent.process(IAgentInput(
            message=input_data["message"]
        ))
        
        # Step 2: Decide next step based on result
        if "error" in initial_result.lower():
            error_agent = self.agents["error_handler"]
            final_result = await error_agent.process(IAgentInput(
                message=f"Handle this error: {initial_result}"
            ))
        else:
            success_agent = self.agents["success_processor"]
            final_result = await success_agent.process(IAgentInput(
                message=f"Process this success: {initial_result}"
            ))
        
        return {
            "initial_result": initial_result,
            "final_result": final_result,
            "workflow_path": "custom_logic"
        }

# Usage - complete developer control
workflow = CustomWorkflowRunner(
    agents={
        "initial_processor": initial_agent,
        "error_handler": error_agent,
        "success_processor": success_agent
    },
    config={"timeout": 30, "retries": 3}
)

result = await workflow.run({"message": "Process this request"})
```

## Complete Application Example

```python
#!/usr/bin/env python3
"""Complete application demonstrating all component types."""

import os
import asyncio
from typing import Dict, Any

# Layer 1: LLM Clients
from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig

# Layer 2: Supporting Components
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
from arshai.embeddings.openai_embeddings import OpenAIEmbeddings
from arshai.vector_db.milvus_client import MilvusClient
from arshai.agents.base import BaseAgent

# Layer 3: Tools and Configuration
from arshai.tools import WebSearchTool
from arshai.config import load_config

class SmartAssistantApp:
    """Complete AI application using all component types."""
    
    def __init__(self):
        # Layer 3: Optional configuration
        self.config = load_config("assistant_config.yaml")
        
        # Layer 1: Create LLM client
        self.llm_client = OpenAIClient(ILLMConfig(
            model=self.config.get("llm", {}).get("model", "gpt-4o"),
            temperature=0.7
        ))
        
        # Layer 2: Supporting components
        self.memory_manager = RedisMemoryManager(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        
        self.embedding_service = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        self.vector_db = MilvusClient(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=int(os.getenv("MILVUS_PORT", "19530"))
        )
        
        # Layer 3: Tools
        self.web_search = WebSearchTool()
        
        # Create specialized agent
        self.assistant_agent = self._create_assistant_agent()
    
    def _create_assistant_agent(self) -> BaseAgent:
        """Create the main assistant agent with all capabilities."""
        
        class SmartAssistant(BaseAgent):
            def __init__(self, llm_client, memory_manager, web_search, vector_db):
                super().__init__(llm_client, "You are a smart assistant with memory, search, and knowledge capabilities.")
                self.memory_manager = memory_manager
                self.web_search = web_search
                self.vector_db = vector_db
            
            async def process(self, input_data) -> Dict[str, Any]:
                # Define capabilities
                async def search_web(query: str, num_results: int = 3) -> str:
                    results = await self.web_search.search(query, num_results)
                    return "\n".join([f"- {r.title}: {r.content}" for r in results])
                
                async def search_knowledge(query: str) -> str:
                    # Embed query and search vector DB
                    query_embedding = await app.embedding_service.embed_text(query)
                    results = await self.vector_db.search("knowledge", query_embedding, limit=3)
                    return "\n".join([f"- {r.document}" for r in results])
                
                async def update_memory(content: str) -> None:
                    """BACKGROUND TASK: Update conversation memory."""
                    conv_id = input_data.metadata.get("conversation_id")
                    if conv_id:
                        await self.memory_manager.store({
                            "conversation_id": conv_id,
                            "working_memory": content
                        })
                
                # Prepare LLM input with all tools
                llm_input = ILLMInput(
                    system_prompt=self.system_prompt,
                    user_message=input_data.message,
                    regular_functions={
                        "search_web": search_web,
                        "search_knowledge": search_knowledge
                    },
                    background_tasks={
                        "update_memory": update_memory
                    }
                )
                
                result = await self.llm_client.chat(llm_input)
                
                return {
                    "response": result["llm_response"],
                    "usage": result["usage"]
                }
        
        return SmartAssistant(
            self.llm_client,
            self.memory_manager, 
            self.web_search,
            self.vector_db
        )
    
    async def process_query(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process user query with full assistant capabilities."""
        
        input_data = IAgentInput(
            message=message,
            metadata={"conversation_id": conversation_id} if conversation_id else {}
        )
        
        return await self.assistant_agent.process(input_data)

async def main():
    """Demo the complete application."""
    
    app = SmartAssistantApp()
    
    # Process a complex query that uses multiple capabilities
    result = await app.process_query(
        message="Search for recent AI developments and remember my interest in machine learning safety",
        conversation_id="user_123_session_456"
    )
    
    print(f"Response: {result['response']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

1. **Direct Instantiation**: All components are created explicitly by you
2. **Layer Separation**: Clear boundaries between component types
3. **Interface Consistency**: All similar components follow the same interfaces
4. **Environment Variables**: Components read their own configuration
5. **Developer Control**: You decide how components work together
6. **Optional Utilities**: Use framework helpers only when they add value

This approach gives you complete authority over your AI application while providing powerful, well-designed building blocks that work consistently together.