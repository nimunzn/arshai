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

.. currentmodule:: arshai.core.interfaces

.. autosummary::
   :toctree: generated/
   :template: class.rst

   IAgentConfig
   IAgentInput
   ILLMConfig

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

LLM Clients
-----------

.. currentmodule:: arshai.llms

.. autosummary::
   :toctree: generated/
   :template: class.rst

   openai.OpenAIClient
   azure.AzureClient