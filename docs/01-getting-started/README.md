# Getting Started

Welcome to Arshai! This section will get you up and running quickly with the framework. You'll learn how to install Arshai, understand its core concepts, and build your first AI agent.

## Quick Navigation

1. **[Installation](installation.md)** - Get Arshai installed in your environment
2. **[Quickstart](quickstart.md)** - Build a working example in 5 minutes
3. **[Core Concepts](core-concepts.md)** - Understand Agents, LLMs, and Tools
4. **[First Agent](first-agent.md)** - Build your first custom agent

## Choose Your Path

### 🚀 I want to start coding immediately
→ Jump to [Quickstart](quickstart.md) for a 5-minute example

### 📚 I want to understand the concepts first
→ Read [Core Concepts](core-concepts.md) to understand the building blocks

### 🛠️ I want to build something real
→ Follow [First Agent](first-agent.md) to create a custom agent

## What You'll Learn

By the end of this section, you'll understand:

- ✅ How to install and configure Arshai
- ✅ The core components (LLM Clients, Agents, Tools)
- ✅ Direct instantiation patterns
- ✅ How to build and test agents
- ✅ Best practices for production use

## Prerequisites

- Python 3.9 or higher
- Basic Python async/await knowledge
- An API key for at least one LLM provider (OpenAI, Google, etc.)

## Your First Arshai Code

Here's a taste of what you'll be building:

```python
from arshai.llms.openai import OpenAIClient
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.core.interfaces.iagent import IAgentInput

# Direct instantiation - you create and control everything
config = ILLMConfig(model="gpt-4", temperature=0.7)
llm_client = OpenAIClient(config)

# Create a simple agent
class GreetingAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt="You are a friendly greeting bot",
            user_message=input.message
        )
        result = await self.llm_client.chat(llm_input)
        return result["llm_response"]

# Use your agent
agent = GreetingAgent(llm_client, "Friendly bot")
response = await agent.process(IAgentInput(message="Hello!"))
print(response)  # "Hello! How can I help you today?"
```

## Key Principles to Remember

### 1. Direct Instantiation
You explicitly create every component - no hidden factories or magic configuration.

### 2. Explicit Dependencies
Every component clearly shows what it needs in its constructor.

### 3. You're in Control
You decide when components are created, how they're configured, and how they interact.

## Environment Setup

Before you begin, set up your environment variables for LLM providers:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-api-key"

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

## Ready to Start?

Let's begin with [Installation](installation.md) to get Arshai set up in your environment!

---

*Remember: Arshai gives you the building blocks - you're the architect.*