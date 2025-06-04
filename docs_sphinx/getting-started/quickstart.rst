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

Create Your First Agent
========================

Here's a complete example of creating and using a conversational agent:

.. code-block:: python

   from arshai import Settings, IAgentConfig, IAgentInput

   # Initialize Arshai settings
   settings = Settings()

   # Configure your agent
   agent_config = IAgentConfig(
       task_context="You are a helpful assistant that specializes in Python programming. "
                   "Provide clear, practical answers with code examples when appropriate.",
       tools=[]  # We'll add tools later
   )

   # Create the agent
   agent = settings.create_agent("conversation", agent_config)

   # Start a conversation
   response, usage = agent.process_message(
       IAgentInput(
           message="How do I create a simple web server in Python?",
           conversation_id="quickstart_demo"
       )
   )

   print("Agent Response:")
   print(response)
   print(f"\\nTokens used: {usage}")

Running the Example
===================

Save the code above as ``quickstart_demo.py`` and run it:

.. code-block:: bash

   python quickstart_demo.py

You should see a detailed response about creating a Python web server, along with token usage information.

Understanding the Code
======================

Let's break down what happened:

1. **Settings**: The ``Settings`` class manages configuration and creates components
2. **Agent Configuration**: ``IAgentConfig`` defines the agent's behavior and capabilities
3. **Agent Creation**: ``create_agent("conversation", config)`` creates a conversational agent
4. **Message Processing**: ``process_message()`` handles user input and returns a response

Adding Memory
=============

Agents automatically maintain conversation memory. Try sending multiple messages:

.. code-block:: python

   # Continue the conversation
   response2, usage2 = agent.process_message(
       IAgentInput(
           message="Can you show me a more advanced example with authentication?",
           conversation_id="quickstart_demo"  # Same conversation ID
       )
   )

   print("Agent Response 2:")
   print(response2)

The agent will remember the previous context about web servers.

Configuration Options
=====================

You can customize your agent's behavior:

.. code-block:: python

   # More detailed configuration
   agent_config = IAgentConfig(
       task_context="You are a senior Python developer and mentor. "
                   "Always explain concepts clearly and provide working code examples. "
                   "If you're unsure about something, say so rather than guessing.",
       tools=[],
       # Additional settings will be available in the full API
   )

Using Configuration Files
=========================

For production applications, use configuration files:

.. code-block:: yaml

   # config.yaml
   llm:
     provider: openai
     model: gpt-4
     temperature: 0.7

   memory:
     working_memory:
       provider: in_memory
       ttl: 3600

Then load the configuration:

.. code-block:: python

   settings = Settings(config_path="config.yaml")

What's Next?
============

Now that you have a basic agent running, you can:

1. **Add Tools**: Learn how to extend your agent with custom tools in :doc:`first-agent`
2. **Build Workflows**: Create multi-agent workflows in the :doc:`../user-guide/workflows/index`
3. **Explore Examples**: Check out more examples in the :doc:`../examples/basic-usage`
4. **Production Setup**: Learn about deployment in :doc:`../deployment/production`

Common Next Steps
================

**Add Web Search**:

.. code-block:: python

   from arshai.tools.web_search_tool import WebSearchTool
   
   web_search = WebSearchTool(settings)
   agent_config = IAgentConfig(
       task_context="You are a research assistant with web search capabilities.",
       tools=[web_search]
   )

**Enable Persistent Memory**:

.. code-block:: python

   # Using Redis for persistent memory
   settings = Settings(config_path="config.yaml")  # Configure Redis in YAML

**Create Custom Tools**:

.. code-block:: python

   from arshai.core.interfaces import ITool
   
   class MyCustomTool(ITool):
       # Implement your custom tool
       pass

Get Help
========

- **Documentation**: Continue reading this documentation
- **Examples**: Check the ``examples/`` directory in the repository
- **Issues**: Report problems on `GitHub <https://github.com/nimunzn/arshai/issues>`_
- **Community**: Join discussions on GitHub Discussions