# Quickstart

Build a working AI agent in 5 minutes! This guide shows you the essentials of Arshai through a practical example.

## Prerequisites

- Arshai installed (`pip install arshai`)
- OpenAI API key set (`export OPENAI_API_KEY="sk-..."`)

## Your First Agent in 5 Minutes

### Step 1: Import and Configure (30 seconds)

```python
import asyncio
import os
from arshai.llms.openai import OpenAIClient
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.core.interfaces.iagent import IAgentInput

# Configure LLM client
config = ILLMConfig(
    model="gpt-3.5-turbo",  # Fast and affordable
    temperature=0.7,         # Balanced creativity
    max_tokens=150          # Reasonable response length
)

# Create LLM client (direct instantiation!)
llm_client = OpenAIClient(config)
print("✅ LLM client created")
```

### Step 2: Create a Simple Agent (1 minute)

```python
class AssistantAgent(BaseAgent):
    """A helpful assistant agent."""
    
    async def process(self, input: IAgentInput) -> str:
        """Process user input and return response."""
        
        # Create LLM input with system prompt and user message
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        # Get response from LLM
        result = await self.llm_client.chat(llm_input)
        
        # Return the response
        return result.get("llm_response", "I couldn't process that.")

# Create your agent
agent = AssistantAgent(
    llm_client=llm_client,
    system_prompt="You are a helpful AI assistant. Be concise and friendly."
)
print("✅ Agent created")
```

### Step 3: Use Your Agent (30 seconds)

```python
async def chat_with_agent():
    """Have a conversation with your agent."""
    
    # Test messages
    messages = [
        "Hello! Who are you?",
        "What can you help me with?",
        "Tell me a short joke"
    ]
    
    for message in messages:
        print(f"\n👤 You: {message}")
        
        # Create input
        agent_input = IAgentInput(message=message)
        
        # Get response
        response = await agent.process(agent_input)
        
        print(f"🤖 Agent: {response}")

# Run the conversation
asyncio.run(chat_with_agent())
```

### Complete Working Example

Here's everything together in one file:

```python
# quickstart.py
import asyncio
import os
from arshai.llms.openai import OpenAIClient
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.core.interfaces.iagent import IAgentInput

class AssistantAgent(BaseAgent):
    """A helpful assistant agent."""
    
    async def process(self, input: IAgentInput) -> str:
        """Process user input and return response."""
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "I couldn't process that.")

async def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Configure and create LLM client
    config = ILLMConfig(model="gpt-3.5-turbo", temperature=0.7)
    llm_client = OpenAIClient(config)
    
    # Create agent
    agent = AssistantAgent(
        llm_client=llm_client,
        system_prompt="You are a helpful AI assistant. Be concise and friendly."
    )
    
    # Interactive chat
    print("🤖 Assistant Agent Ready! (Type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        # Get user input
        user_message = input("\n👤 You: ")
        
        if user_message.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        # Process with agent
        agent_input = IAgentInput(message=user_message)
        response = await agent.process(agent_input)
        
        print(f"🤖 Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python quickstart.py
```

## Adding Tools (2 minutes)

Let's enhance our agent with tool capabilities:

```python
class ToolAgent(BaseAgent):
    """Agent with tool capabilities."""
    
    async def process(self, input: IAgentInput) -> str:
        # Define tools
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"The weather in {city} is sunny and 72°F"
        
        def calculate(expression: str) -> float:
            """Calculate a math expression."""
            return eval(expression)  # Use safe eval in production!
        
        # Create LLM input with tools
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions={
                "get_weather": get_weather,
                "calculate": calculate
            }
        )
        
        # LLM will automatically use tools if needed
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", "I couldn't process that.")

# Create tool-enabled agent
tool_agent = ToolAgent(
    llm_client=llm_client,
    system_prompt="You are an assistant with weather and calculation tools."
)

# Test it
async def test_tools():
    queries = [
        "What's the weather in Paris?",
        "Calculate 25 * 4 + 10"
    ]
    
    for query in queries:
        print(f"\n👤 You: {query}")
        response = await tool_agent.process(IAgentInput(message=query))
        print(f"🤖 Agent: {response}")

asyncio.run(test_tools())
```

## Streaming Responses (1 minute)

For real-time responses:

```python
class StreamingAgent(BaseAgent):
    """Agent with streaming support."""
    
    async def process(self, input: IAgentInput):
        """Stream responses in real-time."""
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        # Stream the response
        print("🤖 Agent: ", end="", flush=True)
        full_response = ""
        
        async for chunk in self.llm_client.stream(llm_input):
            if chunk.get("llm_response"):
                text = chunk["llm_response"]
                print(text, end="", flush=True)
                full_response += text
        
        print()  # New line after streaming
        return full_response

# Test streaming
streaming_agent = StreamingAgent(
    llm_client=llm_client,
    system_prompt="You are a storyteller. Tell brief, engaging stories."
)

async def test_streaming():
    response = await streaming_agent.process(
        IAgentInput(message="Tell me a very short story about a robot")
    )

asyncio.run(test_streaming())
```

## Key Concepts Demonstrated

### 1. Direct Instantiation
```python
# You create components explicitly
llm_client = OpenAIClient(config)
agent = AssistantAgent(llm_client, system_prompt)
```

### 2. Dependency Injection
```python
# Dependencies are passed in, not created internally
class AssistantAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, system_prompt: str):
        super().__init__(llm_client, system_prompt)
```

### 3. Async-First Design
```python
# Everything is async for better performance
async def process(self, input: IAgentInput) -> str:
    result = await self.llm_client.chat(llm_input)
```

### 4. Interface-Based
```python
# Components use interfaces (ILLM, IAgentInput, etc.)
def __init__(self, llm_client: ILLM, system_prompt: str)
```

## Common Patterns

### Pattern 1: Error Handling

```python
class RobustAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> str:
        try:
            llm_input = ILLMInput(
                system_prompt=self.system_prompt,
                user_message=input.message
            )
            result = await self.llm_client.chat(llm_input)
            return result.get("llm_response", "No response")
        except Exception as e:
            return f"Error: {str(e)}"
```

### Pattern 2: Context Management

```python
class ContextAwareAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> str:
        # Extract context from metadata
        context = input.metadata.get("context", {}) if input.metadata else {}
        
        # Include context in prompt
        contextualized_message = f"Context: {context}\nUser: {input.message}"
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=contextualized_message
        )
        
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response")
```

### Pattern 3: Multiple LLM Providers

```python
# Easy to switch providers
from arshai.llms.google_genai import GeminiClient
from arshai.llms.azure import AzureClient

# Just change the client
gemini_client = GeminiClient(config)
azure_client = AzureClient(config)

# Same agent works with any client
agent = AssistantAgent(gemini_client, system_prompt)  # or azure_client
```

## What's Next?

You've just built your first Arshai agent! Here's where to go next:

1. **[Core Concepts](core-concepts.md)** - Understand the architecture deeply
2. **[First Agent](first-agent.md)** - Build a more complex custom agent
3. **[Layer Guides](../02-layer-guides/)** - Explore each architectural layer
4. **[Examples](https://github.com/MobileTechLab/ArsHai/tree/main/examples)** - See more working examples

## Quick Tips

- 💡 **Use async/await** for better performance
- 🔧 **Inject dependencies** for testability
- 📝 **Type hints** help catch errors early
- 🧪 **Test with mocks** by injecting test doubles
- 🚀 **Start simple** and add complexity as needed

## Troubleshooting

### API Key Not Set
```bash
export OPENAI_API_KEY="your-api-key"
```

### Import Errors
```bash
pip install arshai[openai]
```

### Async Warnings
```python
# Always use asyncio.run() for async functions
asyncio.run(your_async_function())
```

---

*Congratulations! You've built your first Arshai agent in 5 minutes! 🎉*

*Ready to learn more? Continue to [Core Concepts](core-concepts.md) →*