==========
Quickstart
==========

This guide will help you create your first AI agent with Arshai in just a few minutes.

Prerequisites
=============

Make sure you have:

- Arshai installed (:doc:`installation`)
- An OpenAI API key or Azure OpenAI endpoint
- Python 3.11+

Setup Your Environment
======================

First, set up your API credentials. You can use environment variables:

.. code-block:: bash

   export OPENAI_API_KEY="your-openai-api-key-here"

Or for Azure OpenAI:

.. code-block:: bash

   export AZURE_OPENAI_API_KEY="your-azure-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_DEPLOYMENT="your-deployment-name"

Create Your First Agent - Direct Instantiation
================================================

Here's a complete example using the new direct instantiation approach:

.. code-block:: python

   import os
   import asyncio
   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput

   # Set your API key
   os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

   class PythonAssistantAgent(BaseAgent):
       """A helpful Python programming assistant."""
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

   async def main():
       # Create LLM client (Layer 1)
       llm_config = ILLMConfig(
           model="gpt-4o",
           temperature=0.7
       )
       llm_client = OpenAIClient(llm_config)

       # Create agent (Layer 2)  
       agent = PythonAssistantAgent(
           llm_client=llm_client,
           system_prompt="You are a helpful assistant that specializes in Python programming. "
                        "Provide clear, practical answers with code examples when appropriate."
       )

       # Start a conversation
       response = await agent.process(
           IAgentInput(message="How do I create a simple web server in Python?")
       )

       print("Agent Response:")
       print(response)

   # Run the example
   if __name__ == "__main__":
       asyncio.run(main())

Running the Example
===================

Save the code above as ``quickstart_demo.py`` and run it:

.. code-block:: bash

   python quickstart_demo.py

You should see a detailed response about creating a Python web server, along with token usage information.

Understanding the Code - Three-Layer Architecture
===================================================

Let's break down what happened using Arshai's three-layer architecture:

1. **Layer 1 (LLM Client)**: ``OpenAIClient`` provides core AI capabilities with minimal developer authority
2. **Layer 2 (Agent)**: ``PythonAssistantAgent`` wraps the LLM with specific purpose and behavior
3. **Direct Instantiation**: You create each component explicitly with clear dependencies
4. **Async Processing**: ``process()`` method handles user input asynchronously

**Key Benefits**:
- **Complete Control**: You decide how to compose components
- **No Hidden Dependencies**: All parameters explicit in constructors  
- **Easy Testing**: Mock interfaces directly
- **Clear Architecture**: Layer separation makes code maintainable

Adding Memory with Direct Instantiation
=========================================

For conversation memory, use WorkingMemoryAgent directly:

.. code-block:: python

   from arshai.agents.working_memory import WorkingMemoryAgent
   from arshai.memory.working_memory.in_memory_manager import InMemoryManager

   async def create_agent_with_memory():
       # Create LLM client (Layer 1)
       llm_config = ILLMConfig(model="gpt-4o", temperature=0.7)
       llm_client = OpenAIClient(llm_config)
       
       # Create memory manager (Supporting Component)
       memory_manager = InMemoryManager(ttl=3600)  # 1 hour memory
       
       # Create agent with memory (Layer 2)
       agent = WorkingMemoryAgent(
           llm_client=llm_client,
           memory_manager=memory_manager,
           system_prompt="You are a helpful Python programming assistant."
       )
       
       return agent

   async def test_memory():
       agent = await create_agent_with_memory()
       
       # First message
       response1 = await agent.process(
           IAgentInput(
               message="How do I create a web server in Python?",
               metadata={"conversation_id": "demo_session"}
           )
       )
       print("Response 1:", response1)
       
       # Follow-up message - agent will remember context
       response2 = await agent.process(
           IAgentInput(
               message="Can you show me a more advanced example with authentication?",
               metadata={"conversation_id": "demo_session"}  # Same conversation
           )
       )
       print("Response 2:", response2)

   asyncio.run(test_memory())

The agent will remember the previous context about web servers.

Configuration with Direct Instantiation
========================================

You can customize your agent's behavior explicitly:

.. code-block:: python

   # Detailed configuration example
   async def create_custom_agent():
       # Configure LLM client precisely
       llm_config = ILLMConfig(
           model="gpt-4o",
           temperature=0.1,  # More deterministic
           max_tokens=1000
       )
       llm_client = OpenAIClient(llm_config)
       
       # Create agent with detailed system prompt
       agent = PythonAssistantAgent(
           llm_client=llm_client,
           system_prompt="You are a senior Python developer and mentor. "
                        "Always explain concepts clearly and provide working code examples. "
                        "If you're unsure about something, say so rather than guessing."
       )
       
       return agent

Optional Configuration Loading
==============================

For production, you can optionally use configuration files:

.. code-block:: yaml

   # config.yaml
   llm:
     provider: openai
     model: gpt-4o
     temperature: 0.7

   memory:
     provider: in_memory
     ttl: 3600

Load and use configuration:

.. code-block:: python

   from arshai.config import load_config

   async def create_configured_agent():
       # Optional configuration loading  
       config = load_config("config.yaml")  # Returns {} if no file
       llm_settings = config.get("llm", {})
       
       # Use config data directly in components
       llm_config = ILLMConfig(
           model=llm_settings.get("model", "gpt-4o"),
           temperature=llm_settings.get("temperature", 0.7)
       )
       llm_client = OpenAIClient(llm_config)
       
       # You control how configuration is used
       return PythonAssistantAgent(llm_client, "You are a helpful assistant.")

What's Next?
============

Now that you have a basic agent running, you can:

1. **Add Tools**: Learn how to extend your agent with custom tools in :doc:`first-agent`
2. **Build Workflows**: Create multi-agent workflows in the :doc:`../user-guide/workflows/index`
3. **Explore Examples**: Check out more examples in the :doc:`../examples/basic-usage`
4. **Production Setup**: Learn about deployment in :doc:`../deployment/production`

Common Next Steps
================

**Add Tools with Function Calling**:

.. code-block:: python

   from arshai.tools import WebSearchTool

   async def create_agent_with_tools():
       # Create components
       llm_client = OpenAIClient(llm_config)
       web_search = WebSearchTool()  # Or pass search client directly
       
       class ResearchAgent(BaseAgent):
           async def process(self, input: IAgentInput) -> str:
               # Define tools
               def search_web(query: str) -> str:
                   return web_search.search(query)
               
               # Use tools via LLM function calling
               llm_input = ILLMInput(
                   system_prompt="You are a research assistant. Use web search when needed.",
                   user_message=input.message,
                   regular_functions={"search_web": search_web}
               )
               
               result = await self.llm_client.chat(llm_input)
               return result['llm_response']
       
       return ResearchAgent(llm_client, "You are a research assistant.")

**Enable Persistent Memory**:

.. code-block:: python

   from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
   
   # Using Redis for persistent memory  
   memory_manager = RedisMemoryManager(
       redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
       ttl=86400  # 24 hours
   )

**Create Custom Tools**:

.. code-block:: python

   def my_custom_tool(input_data: str) -> str:
       """Your custom tool logic."""
       return f"Processed: {input_data}"
   
   # Use in agent via function calling
   llm_input = ILLMInput(
       system_prompt="You have access to a custom tool.",
       user_message=user_message,
       regular_functions={"my_custom_tool": my_custom_tool}
   )

Get Help
========

- **Documentation**: Continue reading this documentation
- **Examples**: Check the ``examples/`` directory in the repository
- **Issues**: Report problems on `GitHub <https://github.com/nimunzn/arshai/issues>`_
- **Community**: Join discussions on GitHub Discussions