=============
API Reference
=============

This section provides detailed documentation for all classes, functions, and modules in the Arshai package.

.. toctree::
   :maxdepth: 2
   :caption: Core Components

   core/index
   agents/index
   workflows/index
   memory/index
   
.. toctree::
   :maxdepth: 2
   :caption: Tools & Extensions

   tools/index
   extensions/index
   factories/index
   
.. toctree::
   :maxdepth: 2
   :caption: Data Processing

   document_loaders/index
   embeddings/index
   vector_db/index
   rerankers/index
   
.. toctree::
   :maxdepth: 2
   :caption: LLMs & Communication

   llms/index
   speech/index
   web_search/index
   
.. toctree::
   :maxdepth: 2
   :caption: Configuration & Utilities

   config/index
   utils/index
   callbacks/index
   clients/index

Quick Reference
===============

Core Classes
------------

.. currentmodule:: arshai

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Settings
   IAgentConfig
   IAgentInput

Main Interfaces
---------------

.. currentmodule:: arshai.core.interfaces

.. autosummary::
   :toctree: generated/
   :template: class.rst

   IAgent
   ILLM
   ITool
   IWorkflow
   IMemoryManager

Factory Classes
---------------

.. currentmodule:: arshai.factories

.. autosummary::
   :toctree: generated/

   agent_factory.AgentFactory
   llm_factory.LLMFactory
   memory_factory.MemoryFactory