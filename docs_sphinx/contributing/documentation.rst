============================
Contributing to Documentation
============================

This guide explains how to contribute to the Arshai documentation.

Getting Started
===============

Prerequisites
-------------

- Python 3.11+
- Poetry (recommended) or pip
- Git

Setting Up the Environment
--------------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/nimunzn/arshai.git
   cd arshai

2. Install dependencies:

.. code-block:: bash

   # Using Poetry (recommended)
   poetry install --with docs
   poetry shell

   # Using pip
   pip install -e .[docs]
   pip install -r docs_sphinx/requirements.txt

3. Generate API documentation:

.. code-block:: bash

   python scripts/generate_api_docs.py

4. Build documentation:

.. code-block:: bash

   # Quick build
   cd docs_sphinx && make html

   # Or use the build script
   ./scripts/build_docs.sh

   # For development with auto-reload
   ./scripts/build_docs.sh --watch

Documentation Structure
=======================

Our documentation is organized as follows:

.. code-block::

   docs_sphinx/
   ├── index.rst                 # Main landing page
   ├── getting-started/          # New user tutorials
   │   ├── installation.rst
   │   ├── quickstart.rst
   │   └── first-agent.rst
   ├── user-guide/              # Detailed guides
   ├── api/                     # Auto-generated API docs
   ├── examples/                # Code examples
   ├── deployment/              # Production deployment
   ├── migration/               # Migration guides
   └── contributing/            # Contribution guides

Writing Guidelines
==================

Style Guide
-----------

- Use clear, concise language
- Write in present tense
- Use active voice when possible
- Include practical examples
- Test all code examples

reStructuredText Format
-----------------------

We use reStructuredText (RST) format. Key conventions:

**Headings**:

.. code-block:: rst

   ==================
   Page Title (H1)
   ==================

   Section Title (H2)
   ==================

   Subsection Title (H3)
   ---------------------

   Sub-subsection Title (H4)
   ~~~~~~~~~~~~~~~~~~~~~~~~~

**Code Blocks**:

.. code-block:: rst

   .. code-block:: python

      from arshai import Settings
      settings = Settings()

**Links**:

.. code-block:: rst

   :doc:`internal-document`
   :ref:`section-reference`
   `External Link <https://example.com>`_

**Notes and Warnings**:

.. code-block:: rst

   .. note::
      This is a note.

   .. warning::
      This is a warning.

   .. tip::
      This is a tip.

Code Examples
-------------

All code examples should be:

- Complete and runnable
- Well-commented
- Include error handling where appropriate
- Follow the project's coding standards

Example:

.. code-block:: python

   from arshai import Settings, IAgentConfig, IAgentInput

   def create_example_agent():
       """Create a simple example agent."""
       try:
           # Initialize settings
           settings = Settings()
           
           # Configure the agent
           config = IAgentConfig(
               task_context="You are a helpful assistant.",
               tools=[]
           )
           
           # Create the agent
           agent = settings.create_agent("conversation", config)
           return agent
           
       except Exception as e:
           print(f"Failed to create agent: {e}")
           raise

API Documentation
=================

API documentation is auto-generated from docstrings using Sphinx autodoc.

Writing Docstrings
------------------

Use Google-style docstrings:

.. code-block:: python

   def example_function(param1: str, param2: int = 10) -> bool:
       """Brief description of the function.

       Longer description explaining the function's purpose,
       behavior, and any important details.

       Args:
           param1: Description of the first parameter.
           param2: Description of the second parameter.
               Defaults to 10.

       Returns:
           Description of the return value.

       Raises:
           ValueError: When param2 is negative.
           TypeError: When param1 is not a string.

       Example:
           >>> result = example_function("hello", 5)
           >>> print(result)
           True
       """
       if param2 < 0:
           raise ValueError("param2 must be non-negative")
       
       return len(param1) > param2

Updating API Documentation
--------------------------

1. Write or update docstrings in the source code
2. Regenerate API docs:

.. code-block:: bash

   python scripts/generate_api_docs.py

3. Rebuild documentation:

.. code-block:: bash

   cd docs_sphinx && make html

Building Documentation
======================

Local Development
-----------------

For rapid development with auto-reload:

.. code-block:: bash

   ./scripts/build_docs.sh --watch

This starts a development server at http://localhost:8000 that automatically
rebuilds when you make changes.

Production Build
----------------

For a production-ready build:

.. code-block:: bash

   ./scripts/build_docs.sh

The built documentation will be in ``docs_sphinx/_build/html/``.

Testing Documentation
=====================

Before submitting changes:

1. **Build locally** and verify everything looks correct
2. **Test all links**:

.. code-block:: bash

   cd docs_sphinx && make linkcheck

3. **Check for warnings** during build
4. **Verify code examples** work as expected

Continuous Integration
======================

Documentation is automatically built and deployed via GitHub Actions:

- **Pull Requests**: Documentation is built and artifacts are uploaded
- **Main Branch**: Documentation is deployed to GitHub Pages
- **Link Checking**: External links are validated

The workflow is defined in ``.github/workflows/docs.yml``.

Publishing Changes
==================

1. Create a new branch for your changes:

.. code-block:: bash

   git checkout -b docs/improve-getting-started

2. Make your changes and test locally
3. Commit your changes:

.. code-block:: bash

   git add docs_sphinx/
   git commit -m "docs: improve getting started guide"

4. Push and create a pull request:

.. code-block:: bash

   git push origin docs/improve-getting-started

5. The documentation will be automatically built and reviewed

Tips and Best Practices
========================

Content Organization
--------------------

- Start with the user's goal
- Provide context before details
- Include practical examples
- Link to related topics
- Keep pages focused and digestible

Visual Elements
---------------

- Use code blocks for all code
- Include diagrams for complex concepts
- Use admonitions (notes, warnings) sparingly
- Keep line length reasonable (80-100 characters)

Maintenance
-----------

- Keep examples up to date with the latest API
- Review and update links regularly
- Remove or update deprecated features
- Test documentation with new releases

Getting Help
============

- **Documentation Issues**: Open an issue on GitHub
- **Style Questions**: Check existing documentation for consistency
- **Technical Problems**: Consult the Sphinx documentation
- **Community**: Join discussions on GitHub Discussions

Thank you for contributing to Arshai documentation!