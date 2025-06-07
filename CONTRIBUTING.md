# Contributing to Arshai

Thank you for your interest in contributing to Arshai! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Project Architecture](#project-architecture)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Adding Components](#adding-components)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Bug Reports & Feature Requests](#bug-reports--feature-requests)
- [Documentation](#documentation)
- [Releases](#releases)
- [License](#license)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful, inclusive, and considerate in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Set up the development environment**:
   ```bash
   # Install Poetry if you don't have it
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install

   # Install development extras
   poetry install --with dev
   ```

## Project Architecture

Arshai follows a clean architecture approach with interface-first design principles. This means we define interfaces before implementations, ensuring loose coupling and high cohesion.

### Core Architecture Principles

```
┌──────────────────────────────────────────────────────────┐
│                     Application Layer                     │
│         (Examples, End-User Applications, Services)       │
└───────────────────────────┬──────────────────────────────┘
                            │ uses
                            ▼
┌──────────────────────────────────────────────────────────┐
│                       Domain Layer                        │
│          (Interfaces and Base Implementations)            │
└───────────────────────────┬──────────────────────────────┘
                            │ implements
                            ▼
┌──────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                    │
│      (Concrete Implementations and External Services)     │
└──────────────────────────────────────────────────────────┘
```

### Project Structure

```
arshai/
├── seedwork/            # Core domain interfaces (Domain Layer)
│   └── interfaces/      # Protocol classes defining contracts
├── src/                 # Implementation code (Infrastructure Layer)
│   ├── agents/          # Agent implementations
│   ├── llms/            # LLM client implementations
│   ├── memory/          # Memory management implementations
│   ├── tools/           # Tool implementations
│   ├── document_loaders/ # Document loader implementations
│   ├── embeddings/      # Embedding model implementations
│   ├── rerankers/       # Reranker implementations
│   ├── speech/          # Speech processing implementations
│   ├── websearch/       # Web search implementations
│   ├── factories/       # Component factories
│   ├── config/          # Configuration management
│   ├── workflows/       # Workflow orchestration 
│   └── utils/           # Utility functions and helpers
├── examples/            # Usage examples (Application Layer)
└── tests/               # Test suite
```

### Component Interaction Model

```
┌──────────────────┐                     ┌──────────────────┐
│    Interfaces    │                     │ Implementations  │
│   (seedwork)     │─────implements─────▶│     (src)        │
└──────────────────┘                     └──────────────────┘
         ▲                                       │
         │                                       │
         │                                       │
         └───────────uses───────────────────────┘
```

### Design Principles

1. **Interface-First Design**: Always define interfaces in `seedwork/interfaces` before implementation
2. **Dependency Inversion**: Depend on abstractions, not concrete implementations
3. **Direct Instantiation**: Use direct instantiation for custom components
4. **Factory Pattern**: Use factories only for predefined components
5. **Separation of Configuration and Instantiation**: Separate configuration from instantiation using settings
6. **Composition Over Inheritance**: Prefer composition patterns over inheritance
7. **Clean Architecture**: Keep domain logic separate from infrastructure concerns

## Development Workflow

1. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and write tests if applicable

3. **Run the tests**:
   ```bash
   poetry run pytest
   ```

4. **Format your code**:
   ```bash
   poetry run black .
   poetry run isort .
   ```

5. **Run type checking**:
   ```bash
   poetry run mypy src seedwork
   ```

6. **Verify documentation**:
   ```bash
   poetry run pydocstyle src seedwork
   ```

7. **Commit your changes** with a clear and descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

8. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Open a pull request** from your fork to the main repository

## Coding Standards

### Code Style

We follow PEP 8 and use Black for code formatting with a line length of 88 characters.

```bash
# Format code
poetry run black .

# Check imports
poetry run isort .
```

### Type Annotations

All code should use type annotations. We use `mypy` for static type checking:

```bash
poetry run mypy src seedwork
```

### Protocol Classes

When defining interfaces, use `Protocol` classes from `typing` and document each method's contract:

```python
from typing import Protocol, List

class IExample(Protocol):
    """Interface defining the contract for example components."""
    
    def process(self, data: str) -> List[str]:
        """
        Process input data and return a list of results.
        
        Args:
            data: The input data to process
            
        Returns:
            A list of processed results
            
        Raises:
            ValueError: If the data is invalid
        """
        ...
```

### Documentation

Document all public classes and functions using docstrings following the Google style guide:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Short description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    ...
```

## Testing Guidelines

### Test Organization

- Tests should mirror the structure of the code
- Test files should be named `test_*.py`
- Test functions should be named `test_*`

### Test Coverage

- Aim for at least 80% code coverage
- Cover both normal and error cases
- Consider edge cases

```bash
# Run tests with coverage report
poetry run pytest --cov=src --cov=seedwork
```

### Test Types

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test components working together
3. **End-to-End Tests**: Test complete workflows

## Adding Components

### Component Development Pattern

The recommended pattern for developing new components is:

```
┌───────────────────────────────────────────────────────────────┐
│                      Development Flow                          │
└───────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────┐
│ Define Interface in     │
│ seedwork/interfaces     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Implement in src/       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Register in Factory     │ (For predefined components only)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Add to Settings         │ (For components created via settings)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Write Tests             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Create Example Usage    │
└─────────────────────────┘
```

### Adding a New LLM Provider

1. Create a new file in `src/llms/` that implements the `ILLM` protocol
2. Register the new provider in `src/factories/llm_factory.py`
3. Update the settings handling in `src/config/settings.py` if needed
4. Write tests for the new provider
5. Add example usage

```python
# 1. Implement the ILLM protocol
from seedwork.interfaces.illm import ILLM, ILLMConfig

class NewLLMProvider(ILLM):
    """Implementation of the ILLM protocol for a new provider."""
    
    def __init__(self, config: ILLMConfig):
        """Initialize with configuration."""
        self.config = config
        # Additional initialization...
    
    def chat(self, messages, stream=False):
        """Chat method implementation."""
        # Implementation...
    
    def chat_with_tools(self, messages, tools, stream=False):
        """Chat with tools method implementation."""
        # Implementation...

# 2. Register in the factory
from src.factories.llm_factory import LLMFactory
from src.llms.new_provider import NewLLMProvider

LLMFactory.register("new_provider", NewLLMProvider)
```

### Adding a New Agent Type

1. Create a new file in `src/agents/` that implements the `IAgent` protocol
2. Register the new agent type in `src/factories/agent_factory.py`
3. Update the settings handling in `src/config/settings.py` if needed
4. Write tests for the new agent
5. Add example usage

```python
# 1. Implement the IAgent protocol
from seedwork.interfaces.iagent import IAgent, IAgentConfig, IAgentInput, IAgentOutput

class NewAgent(IAgent):
    """A new agent implementation."""
    
    def __init__(self, config: IAgentConfig, settings):
        """Initialize with configuration and settings."""
        self.config = config
        self.settings = settings
        # Additional initialization...
    
    def process_message(self, input: IAgentInput) -> IAgentOutput:
        """Process a message and return a response."""
        # Implementation...
    
    async def aprocess_message(self, input: IAgentInput) -> IAgentOutput:
        """Async version of process_message."""
        # Implementation...

# 2. Register in the factory
from src.factories.agent_factory import AgentFactory
from src.agents.new_agent import NewAgent

AgentFactory.register("new_agent", NewAgent)
```

## Pull Request Guidelines

When submitting a pull request, please:

1. **Provide a clear description** of the changes
2. **Link to related issues** if applicable
3. **Include tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Follow the code style** guidelines

## Bug Reports & Feature Requests

When reporting bugs or requesting features, please:

1. **Search existing issues** before creating a new one
2. **Provide detailed information** about the bug or feature
3. **Include reproduction steps** for bugs
4. **Explain the value** of requested features

## Documentation

Documentation is a crucial part of the project. Please follow these guidelines:

1. **Document all public interfaces** with clear docstrings
2. **Update relevant documentation files** when adding features
3. **Include code examples** where appropriate
4. **Make diagrams and visuals** for complex concepts

## Releases and PyPI Deployment

Arshai uses an automated CI/CD pipeline for PyPI deployment. **Only maintainers** should create releases.

### Release Process

1. **Update version numbers**:
   ```bash
   # Update version in pyproject.toml
   version = "X.Y.Z"
   
   # Update version in arshai/_version.py
   __version__ = "X.Y.Z"
   __version_info__ = (X, Y, Z)
   ```

2. **Commit and push changes**:
   ```bash
   git add pyproject.toml arshai/_version.py
   git commit -m "chore: bump version to X.Y.Z"
   git push origin main
   ```

3. **Create and push git tag**:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. **Create GitHub Release**:
   ```bash
   gh release create vX.Y.Z --title "Release vX.Y.Z" --notes "Release notes here"
   ```

5. **Automated deployment**: Creating the GitHub release automatically triggers the PyPI deployment via GitHub Actions.

### PyPI Deployment Workflow

The deployment process is fully automated:

- **Trigger**: GitHub release creation
- **Workflow**: `.github/workflows/publish.yml`
- **Requirements**: `PYPI_TOKEN` secret configured in repository settings
- **Process**:
  1. Checkout code
  2. Set up Python and Poetry
  3. Install dependencies
  4. Build package with `poetry build`
  5. Publish to PyPI with `poetry publish`

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **Major (X.0.0)**: Breaking changes
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

### Prerequisites for Maintainers

To deploy releases, maintainers need:
1. **Write access** to the repository
2. **PyPI token** configured as `PYPI_TOKEN` secret
3. **PyPI project permissions** for the arshai package

For detailed deployment instructions, see the [Deployment Guide](docs/deployment/pypi-deployment.md).

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.

Thank you for contributing to Arshai! 