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

   pip install arshai

Basic Usage
-----------

.. code-block:: python

   from arshai import Settings, IAgentConfig, IAgentInput

   # Initialize settings
   settings = Settings()

   # Create agent configuration
   agent_config = IAgentConfig(
       task_context="You are a helpful assistant.",
       tools=[]
   )

   # Create conversation agent
   agent = settings.create_agent("conversation", agent_config)

   # Process a message
   response, usage = agent.process_message(
       IAgentInput(
           message="Hello! How can you help me?",
           conversation_id="chat_123"
       )
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
   :caption: User Guide

   user-guide/agents/index
   user-guide/workflows/index
   user-guide/memory/index
   user-guide/tools/index
   user-guide/llms/index
   user-guide/extensions/index

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