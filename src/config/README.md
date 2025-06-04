# Configuration

The Config module provides a centralized way to manage settings and configuration for the Arshai framework, following a layered approach with sensible defaults.

## Overview

The configuration system in Arshai:
- Centralizes all configuration in a single `Settings` class
- Provides component factory methods (`create_llm`, `create_agent`, etc.)
- Loads configuration from multiple sources with priority
- Allows extending with application-specific settings

## The Settings Class

The `Settings` class is the central component for configuration management:

```python
from src.config.settings import Settings

# Create default settings
settings = Settings()

# Create settings with a config file
settings = Settings('config.yaml')

# Access configuration values
llm_provider = settings.get_value('llm.provider', 'default_value')

# Create components
llm = settings.create_llm()
memory_manager = settings.create_memory_manager()
agent = settings.create_agent('operator', agent_config)
```

## Configuration Sources and Priority

Configuration values are loaded from multiple sources with the following priority:

1. **Environment Variables**: Highest priority (e.g., `OPENAI_API_KEY`)
2. **Configuration Files**: YAML or JSON files specified at initialization
3. **Default Values**: Hardcoded defaults in the Settings class

This allows for flexibility while maintaining security (keeping sensitive values like API keys in environment variables).

### Environment Variables

Settings can be overridden using environment variables:

```bash
# Set environment variables
export OPENAI_API_KEY=your_api_key_here
export ARSHAI_LLM_PROVIDER=openai
export ARSHAI_LLM_MODEL=gpt-4
```

### Configuration Files

You can specify configuration in YAML or JSON files:

```yaml
# config.yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  api_key: ${OPENAI_API_KEY}  # Reference environment variable

memory:
  working_memory:
    provider: redis
    url: redis://localhost:6379/0
    ttl: 43200  # 12 hours
```

## Extending Settings

For application-specific needs, you can extend the `Settings` class:

```python
from src.config.settings import Settings

class MyAppSettings(Settings):
    def __init__(self, config_path=None):
        # Initialize base settings
        super().__init__(config_path)
        
        # Initialize application-specific resources
        self.db_client = self._create_db_client()
        self.api_client = self._create_api_client()
    
    def _create_db_client(self):
        """Create a database client from configuration."""
        db_config = self.get_value('database', {})
        # Create and return your database client
        return DatabaseClient(db_config)
    
    def _create_api_client(self):
        """Create an API client from configuration."""
        api_config = self.get_value('api', {})
        # Create and return your API client
        return ApiClient(api_config)
    
    def get_db_client(self):
        """Get the database client."""
        return self.db_client
    
    def get_api_client(self):
        """Get the API client."""
        return self.api_client
```

Then use your extended settings:

```python
# Create application settings
settings = MyAppSettings('config.yaml')

# Get application-specific resources
db_client = settings.get_db_client()
api_client = settings.get_api_client()

# Still have access to base functionality
llm = settings.create_llm()
```

## Component Creation

The Settings class provides methods to create various components:

```python
# Create an LLM client
llm = settings.create_llm()

# Create a memory manager
memory_manager = settings.create_memory_manager()

# Create a predefined agent
agent = settings.create_agent('operator', agent_config)

# Create additional components (if configured)
retriever = settings.retriever
reranker = settings.reranker
```

## Best Practices

1. **Centralized Configuration**
   - Use a single Settings instance throughout your application
   - Pass the Settings object to components rather than individual values

2. **Environment Variables for Sensitive Data**
   - Never hardcode API keys or credentials
   - Use environment variables for sensitive information
   - Support referencing environment variables in config files

3. **Explicit Defaults**
   - Always provide sensible default values
   - Document the default configuration

4. **Configuration Validation**
   - Validate configuration values early
   - Provide helpful error messages for missing or invalid configuration

5. **Extension over Modification**
   - Extend the Settings class rather than modifying it
   - Add application-specific resources through extension
``` 