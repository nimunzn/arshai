============
Installation
============

System Requirements
===================

- Python 3.11 or higher
- pip or Poetry for package management

Basic Installation
==================

The easiest way to install Arshai is using pip:

.. code-block:: bash

   pip install arshai

This installs the core Arshai framework with basic dependencies.

Installation with Optional Dependencies
=======================================

Arshai supports optional features that require additional dependencies:

All Features
------------

Install with all optional dependencies:

.. code-block:: bash

   pip install arshai[all]

This includes:
- Redis support for distributed memory
- Milvus support for vector databases
- FlashRank for result reranking

Individual Features
-------------------

You can also install specific optional dependencies:

**Redis Support** (for distributed memory):

.. code-block:: bash

   pip install arshai[redis]

**Milvus Support** (for vector databases):

.. code-block:: bash

   pip install arshai[milvus]

**Reranking Support** (for search result reranking):

.. code-block:: bash

   pip install arshai[rerankers]

Using Poetry
============

If you're using Poetry for dependency management:

.. code-block:: bash

   poetry add arshai

Or with optional dependencies:

.. code-block:: bash

   poetry add arshai[all]

Development Installation
========================

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/nimunzn/arshai.git
   cd arshai
   pip install -e .[all]

Verification
============

Verify your installation by importing Arshai:

.. code-block:: python

   import arshai
   print(f"Arshai version: {arshai.__version__}")

You should see the version number printed without any errors.

Troubleshooting
===============

Common Issues
-------------

**Python Version Error**
   Ensure you're using Python 3.11 or higher:
   
   .. code-block:: bash
   
      python --version

**Permission Errors**
   Use ``--user`` flag if you don't have admin privileges:
   
   .. code-block:: bash
   
      pip install --user arshai

**Virtual Environment Recommended**
   It's recommended to use a virtual environment:
   
   .. code-block:: bash
   
      python -m venv arshai-env
      source arshai-env/bin/activate  # On Windows: arshai-env\Scripts\activate
      pip install arshai

Next Steps
==========

Once you have Arshai installed, continue with the :doc:`quickstart` guide to build your first AI agent!