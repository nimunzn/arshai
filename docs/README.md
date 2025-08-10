# Arshai Documentation

Welcome to the Arshai framework documentation! Arshai is a developer-first AI application framework built on **three-layer architecture** principles with **direct instantiation** patterns that give you complete control over component creation and lifecycle.

## 📚 Documentation Structure

### 🎯 [00 - Philosophy](00-philosophy/)
**Start Here** - Understand the vision and principles behind Arshai
- [Three-Layer Architecture](00-philosophy/three-layer-architecture.md) - Core architectural philosophy
- [Developer Authority](00-philosophy/developer-authority.md) - Why developers should have control
- [Design Decisions](00-philosophy/design-decisions.md) - Why direct instantiation over factories

### 🚀 [01 - Getting Started](01-getting-started/)
Quick start guides to get you building immediately
- [Installation](01-getting-started/installation.md) - Install Arshai
- [Quickstart](01-getting-started/quickstart.md) - 5-minute hello world
- [Core Concepts](01-getting-started/core-concepts.md) - Agents, LLMs, and Tools
- [First Agent](01-getting-started/first-agent.md) - Build your first agent

### 📖 [02 - Layer Guides](02-layer-guides/)
Deep dives into each architectural layer

#### [Layer 1: LLM Clients](02-layer-guides/layer1-llm-clients/)
Core AI components with minimal developer authority
- Direct instantiation patterns
- Streaming responses
- Function calling and background tasks
- Provider-specific guides

#### [Layer 2: Agents](02-layer-guides/layer2-agents/)
Purpose-driven wrappers with moderate authority
- Creating agents with BaseAgent
- Stateless design principles
- Agent composition patterns
- Testing strategies

#### [Layer 3: Agentic Systems](02-layer-guides/layer3-systems/)
Orchestration layer with maximum developer authority
- Workflow patterns
- Memory management
- Tool integration
- MCP server configuration

### 🔧 [03 - Patterns](03-patterns/)
Best practices and design patterns
- [Direct Instantiation](03-patterns/direct-instantiation.md) - Core pattern examples
- [Dependency Injection](03-patterns/dependency-injection.md) - DI patterns for testing
- [Error Handling](03-patterns/error-handling.md) - Building resilient systems
- [Configuration](03-patterns/configuration.md) - Environment variables and YAML

### 📦 [04 - Components](04-components/)
Reference documentation for all components
- [LLM Clients](04-components/llm-clients.md) - Available LLM implementations
- [Agents](04-components/agents.md) - Built-in agent reference
- [Memory](04-components/memory.md) - Memory components
- [Tools](04-components/tools.md) - Available tools and integration

### 📝 [05 - Tutorials](05-tutorials/)
Step-by-step guides for common use cases
- Building a chat application
- Creating a RAG system
- Multi-agent orchestration
- Production deployment

### 🔄 [06 - Migration](06-migration/)
Guides for migrating from older patterns
- [From Settings Pattern](06-migration/from-settings.md) - Migrate from Settings to direct instantiation
- [From Factory Pattern](06-migration/from-factories.md) - Migrate from factories
- [Breaking Changes](06-migration/breaking-changes.md) - Version migration guides

### 📋 [07 - API Reference](07-api-reference/)
Complete API documentation
- Core interfaces
- Class documentation
- Utility functions

### 🤝 [08 - Contributing](08-contributing/)
How to contribute to Arshai
- [Contributing Guide](08-contributing/README.md) - How to contribute
- [Adding LLM Providers](08-contributing/adding-llm-provider.md) - Add new LLM providers
- [Creating Agents](08-contributing/creating-agents.md) - Contribute new agents
- [Code Standards](08-contributing/code-standards.md) - Coding standards

### 🚢 [09 - Deployment](09-deployment/)
Production deployment guides
- [Docker](09-deployment/docker.md) - Docker deployment
- [Kubernetes](09-deployment/kubernetes.md) - K8s deployment
- [PyPI](09-deployment/pypi.md) - Publishing to PyPI
- [Monitoring](09-deployment/monitoring.md) - Production monitoring

## 🎓 Learning Path

### For Beginners
1. Start with [Philosophy](00-philosophy/) to understand the vision
2. Follow [Getting Started](01-getting-started/) for quick wins
3. Explore [Layer Guides](02-layer-guides/) for your use case

### For Developers
1. Review [Patterns](03-patterns/direct-instantiation.md) for best practices
2. Check [Components](04-components/) for available tools
3. Follow [Tutorials](05-tutorials/) for complete examples

### For Contributors
1. Read [Contributing Guide](08-contributing/README.md)
2. Review [Code Standards](08-contributing/code-standards.md)
3. Check existing [Components](04-components/) before creating new ones

## 🔑 Key Principles

### Direct Instantiation
```python
# You create and control components directly
llm_client = OpenAIClient(config)
agent = MyAgent(llm_client, system_prompt="...")
```

### Three-Layer Architecture
- **Layer 1**: LLM Clients - Minimal authority, standardized interfaces
- **Layer 2**: Agents - Moderate authority, purpose-driven logic
- **Layer 3**: Systems - Maximum authority, full orchestration control

### Developer Authority
- You control when components are created
- You control how they're configured
- You control their lifecycle
- No hidden magic, no forced patterns

## 📚 Quick Links

- [Quickstart Guide](01-getting-started/quickstart.md) - Get started in 5 minutes
- [Direct Instantiation Patterns](03-patterns/direct-instantiation.md) - Core patterns
- [Migration from Settings](06-migration/from-settings.md) - Upgrade guide
- [API Reference](07-api-reference/) - Complete API docs

## 💬 Need Help?

- 📖 Check the relevant layer guide for your use case
- 🔍 Search the [API Reference](07-api-reference/) for specific components
- 🐛 Report issues on [GitHub](https://github.com/MobileTechLab/ArsHai/issues)
- 💡 See [Examples](https://github.com/MobileTechLab/ArsHai/tree/main/examples) for working code

---

*Built with developer empowerment at its core - you're in control.*