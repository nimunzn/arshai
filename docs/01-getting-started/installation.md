# Installation

This guide will help you install Arshai and set up your development environment.

## Requirements

- Python 3.9 or higher
- pip or poetry package manager
- Virtual environment (recommended)

## Basic Installation

### Using pip

```bash
# Install the core package
pip install arshai

# Install with specific LLM provider support
pip install arshai[openai]     # For OpenAI support
pip install arshai[google]     # For Google Gemini support
pip install arshai[azure]      # For Azure OpenAI support

# Install with all optional dependencies
pip install arshai[all]
```

### Using poetry

```bash
# Add to your project
poetry add arshai

# With specific providers
poetry add arshai[openai]
poetry add arshai[google]
poetry add arshai[azure]

# With all optional dependencies
poetry add arshai[all]
```

## Development Installation

For contributing or modifying Arshai:

```bash
# Clone the repository
git clone https://github.com/MobileTechLab/ArsHai.git
cd ArsHai

# Install with poetry (recommended for development)
poetry install

# Or install with pip in editable mode
pip install -e .

# Install with all optional dependencies for development
poetry install -E all
```

## Optional Dependencies

Arshai has optional dependencies for specific features:

### Memory Backends
```bash
# Redis support for distributed memory
pip install arshai[redis]
```

### Vector Databases
```bash
# Milvus support for vector storage
pip install arshai[milvus]
```

### Document Processing
```bash
# Enhanced document processing capabilities
pip install arshai[documents]
```

### Observability
```bash
# OpenTelemetry support for monitoring
pip install arshai[observability]
```

### All Features
```bash
# Install everything
pip install arshai[all]
```

## Environment Configuration

### Setting Up API Keys

Arshai reads API keys from environment variables. Set them up based on your LLM provider:

#### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```

#### Google Gemini
```bash
export GOOGLE_API_KEY="..."
```

#### Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

#### OpenRouter
```bash
export OPENROUTER_API_KEY="..."
```

### Using a .env File

For local development, you can use a `.env` file:

```bash
# Create a .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
EOF

# Load it in your Python code
from dotenv import load_dotenv
load_dotenv()

# Now environment variables are available
import os
print(os.getenv("OPENAI_API_KEY"))
```

## Verifying Installation

### Check Installation

```python
# verify_installation.py
import arshai

# Check version
print(f"Arshai version: {arshai.__version__}")

# Check core imports
from arshai.llms.openai import OpenAIClient
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.illm import ILLMConfig

print("✅ Core imports successful")

# Check optional dependencies
try:
    import redis
    print("✅ Redis support available")
except ImportError:
    print("⚠️ Redis not installed (optional)")

try:
    import pymilvus
    print("✅ Milvus support available")
except ImportError:
    print("⚠️ Milvus not installed (optional)")
```

### Test Basic Functionality

```python
# test_basic.py
import asyncio
import os
from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput

async def test_llm():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not set")
        return
    
    # Create LLM client
    config = ILLMConfig(model="gpt-3.5-turbo", temperature=0.7)
    client = OpenAIClient(config)
    
    # Test chat
    input_data = ILLMInput(
        system_prompt="You are a helpful assistant",
        user_message="Say 'Installation successful!'"
    )
    
    result = await client.chat(input_data)
    print(f"✅ LLM Response: {result['llm_response']}")

# Run the test
asyncio.run(test_llm())
```

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Error: ModuleNotFoundError: No module named 'arshai'
# Solution: Ensure arshai is installed
pip install arshai
```

#### API Key Errors
```python
# Error: OPENAI_API_KEY environment variable is required
# Solution: Set your API key
export OPENAI_API_KEY="your-key"
```

#### Async Errors
```python
# Error: RuntimeWarning: coroutine was never awaited
# Solution: Use asyncio.run() or await
import asyncio
asyncio.run(your_async_function())
```

#### SSL/Certificate Errors
```bash
# Error: SSL certificate verification failed
# Solution: Update certificates
pip install --upgrade certifi
```

## IDE Setup

### VS Code

Install recommended extensions:
- Python
- Pylance (for type checking)
- Python Docstring Generator

Settings for `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

### PyCharm

1. Set Python interpreter to your virtual environment
2. Enable type checking in Settings → Editor → Inspections
3. Configure async debugging support

## Docker Installation (Optional)

For containerized development:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t arshai-app .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY arshai-app
```

## Next Steps

Now that you have Arshai installed:

1. **[Quickstart](quickstart.md)** - Build your first example
2. **[Core Concepts](core-concepts.md)** - Understand the components
3. **[First Agent](first-agent.md)** - Create a custom agent

## Getting Help

- Check the [Troubleshooting](#troubleshooting) section above
- Review [API Reference](../07-api-reference/)
- Open an issue on [GitHub](https://github.com/MobileTechLab/ArsHai/issues)

---

*Ready to build? Continue to [Quickstart](quickstart.md) →*