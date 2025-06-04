# Arshai Package Migration Guide

This guide helps you migrate from the old package structure (v0.1.x) to the new unified structure (v0.2.x).

## Overview

Arshai v0.2.0 introduces a new package structure that provides:
- **Unified imports** under the `arshai` package
- **Plugin system** for easy extensibility
- **Backward compatibility** layer for smooth migration
- **Public PyPI distribution** for easy installation

## Quick Migration Checklist

- [ ] Update package installation method
- [ ] Update import statements
- [ ] Test your application
- [ ] Consider using new plugin system
- [ ] Update documentation and examples

## 1. Installation Changes

### Old Way (v0.1.x)
```bash
# Vendored dependency or local path
# Copy entire arshai code into your project
```

### New Way (v0.2.x)
```bash
# Install from PyPI
pip install arshai

# Or with Poetry
poetry add arshai

# With optional dependencies
pip install arshai[all]  # Includes redis, milvus, flashrank
```

## 2. Import Changes

### Core Interfaces

**Old imports:**
```python
from seedwork.interfaces.iagent import IAgent, IAgentConfig, IAgentInput
from seedwork.interfaces.illm import ILLM
from seedwork.interfaces.itool import ITool
from seedwork.interfaces.iworkflow import IWorkflow
```

**New imports:**
```python
from arshai import IAgent, IAgentConfig, IAgentInput, ILLM, ITool, IWorkflow
# OR
from arshai.core.interfaces import IAgent, IAgentConfig, IAgentInput, ILLM, ITool, IWorkflow
```

### Components

**Old imports:**
```python
from src.config.settings import Settings
from src.agents.conversation import ConversationAgent
from src.workflows.workflow_runner import WorkflowRunner
from src.tools.web_search_tool import WebSearchTool
from src.memory.memory_manager import MemoryManager
```

**New imports:**
```python
from arshai import Settings, ConversationAgent, WorkflowRunner
from arshai.tools.web_search_tool import WebSearchTool
from arshai.memory.memory_manager import MemoryManager
```

### Complete Import Mapping

| Old Import | New Import |
|------------|------------|
| `from seedwork.interfaces.*` | `from arshai.core.interfaces.*` |
| `from src.agents.*` | `from arshai.agents.*` |
| `from src.workflows.*` | `from arshai.workflows.*` |
| `from src.memory.*` | `from arshai.memory.*` |
| `from src.tools.*` | `from arshai.tools.*` |
| `from src.llms.*` | `from arshai.llms.*` |
| `from src.embeddings.*` | `from arshai.embeddings.*` |
| `from src.config.*` | `from arshai.config.*` |
| `from src.factories.*` | `from arshai.factories.*` |
| `from src.utils.*` | `from arshai.utils.*` |

## 3. Code Migration Examples

### Basic Agent Usage

**Old code:**
```python
from seedwork.interfaces.iagent import IAgentConfig, IAgentInput
from src.config.settings import Settings

settings = Settings()
agent_config = IAgentConfig(
    task_context="You are a helpful assistant."
)
agent = settings.create_agent("conversation", agent_config)

response, usage = agent.process_message(
    IAgentInput(message="Hello", conversation_id="123")
)
```

**New code:**
```python
from arshai import Settings, IAgentConfig, IAgentInput

settings = Settings()
agent_config = IAgentConfig(
    task_context="You are a helpful assistant."
)
agent = settings.create_agent("conversation", agent_config)

response, usage = agent.process_message(
    IAgentInput(message="Hello", conversation_id="123")
)
```

### Workflow Creation

**Old code:**
```python
from seedwork.interfaces.iworkflow import IWorkflowState, IUserContext
from src.workflows.workflow_config import BaseWorkflowConfig
from src.workflows.workflow_runner import WorkflowRunner
```

**New code:**
```python
from arshai import IWorkflowState, IUserContext, BaseWorkflowConfig, WorkflowRunner
```

### Using Tools

**Old code:**
```python
from src.tools.web_search_tool import WebSearchTool
from src.tools.knowledge_base_tool import KnowledgeBaseRetrievalTool
```

**New code:**
```python
from arshai.tools.web_search_tool import WebSearchTool
from arshai.tools.knowledge_base_tool import KnowledgeBaseRetrievalTool
```

## 4. Automatic Migration

### Using the Migration Script

We provide an automated migration script to update your imports:

```bash
# Download the migration script
curl -O https://raw.githubusercontent.com/mahdirasoulim/arshai/main/scripts/migrate_imports.py

# Run on your project
python migrate_imports.py --path /path/to/your/project

# Dry run to see what would change
python migrate_imports.py --path /path/to/your/project --dry-run
```

### Manual Migration Steps

1. **Find and replace imports** in your IDE:
   - `from seedwork.interfaces` → `from arshai.core.interfaces`
   - `from src.` → `from arshai.`

2. **Update your requirements:**
   ```txt
   # Remove local dependencies
   # Add:
   arshai>=0.2.0
   ```

3. **Test your application** thoroughly after changes.

## 5. Backward Compatibility

For gradual migration, you can enable backward compatibility:

```python
# Enable compatibility mode for old imports
from arshai.compat import enable_compatibility_mode
enable_compatibility_mode()

# Now old imports will work with deprecation warnings
from seedwork.interfaces.iagent import IAgent  # Shows warning
from src.config.settings import Settings       # Shows warning
```

**Note:** Compatibility mode is temporary and will be removed in v0.3.0.

## 6. Configuration Changes

### Environment Variables

Configuration environment variables remain the same:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `REDIS_URL`
- `MILVUS_HOST`, `MILVUS_PORT`

### Configuration Files

YAML configuration format is unchanged:

```yaml
llm:
  provider: azure
  model: gpt-4o-mini
  
memory:
  working_memory:
    provider: redis
    ttl: 86400
    
embedding:
  provider: mgte
  model: Alibaba-NLP/gte-multilingual-base
```

## 7. New Features in v0.2.x

### Plugin System

You can now extend Arshai with plugins:

```python
from arshai.extensions import load_plugin

# Load a custom plugin
plugin = load_plugin("my_custom_plugin", config={
    "api_key": "secret"
})

# Use plugin's tools
custom_tool = plugin.get_tool("my_tool")
```

### Enhanced Public API

```python
# Cleaner imports
from arshai import (
    Settings,
    ConversationAgent,
    WorkflowRunner,
    IAgent,
    ITool
)
```

### Hook System

Extend behavior without modifying core code:

```python
from arshai.extensions.hooks import hook, HookType

@hook(HookType.BEFORE_AGENT_PROCESS)
def log_agent_input(context):
    print(f"Processing: {context.data['input'].message}")
```

## 8. Troubleshooting

### Common Issues

**Import Error: No module named 'seedwork'**
- Solution: Update imports to use `arshai.core.interfaces`

**Import Error: No module named 'src'**
- Solution: Update imports to use `arshai.*`

**AttributeError: module 'arshai' has no attribute 'X'**
- Solution: Check if `X` is exported in the main `arshai` module or import from specific submodule

**Circular Import Error**
- Solution: Use specific imports instead of importing from main `arshai` module

### Getting Help

1. **Check the examples** in the `examples/` directory
2. **Read the API documentation** (coming soon)
3. **Open an issue** on GitHub if you encounter problems
4. **Enable compatibility mode** as a temporary workaround

## 9. Migration Checklist

- [ ] **Backup your project** before starting migration
- [ ] **Update package installation** to use PyPI package
- [ ] **Run migration script** or manually update imports
- [ ] **Test core functionality** works as expected
- [ ] **Update configuration** if needed
- [ ] **Run your test suite** to catch any issues
- [ ] **Update documentation** and examples in your project
- [ ] **Consider using new features** like plugins and hooks
- [ ] **Remove compatibility mode** once migration is complete
- [ ] **Update CI/CD** to use new package version

## 10. Benefits of Migration

After migration, you'll gain:

- ✅ **Cleaner imports** with unified package structure
- ✅ **Automatic updates** via package manager
- ✅ **Plugin system** for easy extensibility
- ✅ **Better documentation** and examples
- ✅ **Type safety** with improved type hints
- ✅ **Future compatibility** with new features
- ✅ **Community plugins** ecosystem

## Support

If you need help with migration:
- 📖 Check this guide and examples
- 🐛 Report issues on GitHub
- 💬 Discussion forums (coming soon)
- 📧 Contact the maintainers

Happy migrating! 🚀