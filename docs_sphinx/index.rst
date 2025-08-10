.. Arshai documentation master file

================================
Arshai: AI Agent Framework
================================

.. image:: https://img.shields.io/pypi/v/arshai.svg
   :target: https://pypi.org/project/arshai/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/arshai.svg
   :target: https://pypi.org/project/arshai/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/nimunzn/arshai.svg
   :target: https://github.com/nimunzn/arshai/blob/main/LICENSE
   :alt: License

Arshai is a powerful, extensible framework for building conversational AI systems with advanced agent capabilities, workflow orchestration, and memory management.

✨ **Key Features**
==================

🤖 **Agent Framework**
   Create intelligent conversational agents with advanced memory management and tool integration

🔄 **Workflow Orchestration** 
   Design complex multi-agent systems with directed graph workflows

🧠 **Memory Management**
   Sophisticated conversation memory with multiple storage options (Redis, in-memory)

🛠️ **Tool Integration**
   Extend agent capabilities with custom tools and external integrations

🔌 **Plugin System**
   Extensible architecture with hooks for customization and plugin development

🔗 **LLM Integration**
   Connect with leading LLM providers (OpenAI, Azure OpenAI) with consistent APIs

📚 **RAG Capabilities**
   Build powerful retrieval-augmented generation systems with document processing

⚡ **Quick Start**
================

Installation
------------

.. code-block:: bash

   pip install arshai[openai]

Basic Usage - Direct Instantiation
-----------------------------------

.. code-block:: python

   import os
   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput

   # Set your API key
   os.environ["OPENAI_API_KEY"] = "your-api-key-here"

   # Create LLM client directly (Layer 1)
   llm_config = ILLMConfig(
       model="gpt-4o",
       temperature=0.7
   )
   llm_client = OpenAIClient(llm_config)

   # Create agent directly (Layer 2)
   class SimpleAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

   # Use your agent
   agent = SimpleAgent(
       llm_client=llm_client, 
       system_prompt="You are a helpful assistant."
   )

   # Process a message
   response = await agent.process(
       IAgentInput(message="Hello! How can you help me?")
   )
   
   print(f"Agent: {response}")

📖 **Documentation**
===================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart
   getting-started/first-agent

.. toctree::
   :maxdepth: 2
   :caption: Three-Layer Architecture

   layer-guides/layer1-llm-clients
   layer-guides/layer2-agents
   layer-guides/layer3-systems
   
.. toctree::
   :maxdepth: 2
   :caption: Components & Tools

   components/memory
   components/tools
   components/extensions

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Migration

   migration/from-v0.1
   migration/petrochemical-rag
   migration/breaking-changes

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic-usage
   examples/advanced-workflows
   examples/plugins
   examples/integrations

.. toctree::
   :maxdepth: 2
   :caption: Deployment

   deployment/index

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing/development
   contributing/testing
   contributing/documentation
   contributing/release-process

🔗 **Links**
============

* **PyPI**: https://pypi.org/project/arshai/
* **GitHub**: https://github.com/nimunzn/arshai
* **Issues**: https://github.com/nimunzn/arshai/issues

📄 **License**
==============

This project is licensed under the MIT License - see the `LICENSE <https://github.com/nimunzn/arshai/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`