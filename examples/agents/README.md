# Agent Examples

This directory contains comprehensive examples for working with agents in the Arshai framework.

## Getting Started

### Prerequisites
```bash
export OPENROUTER_API_KEY=your_key_here
```

### Quick Start Options

**Option 1: Interactive Demo (5 minutes)**
```bash
python agent_quickstart.py
```
Interactive console demo - perfect for first-time users.

**Option 2: Comprehensive Tutorial**
```bash
python agents_comprehensive_guide.py
```
Single-file tutorial covering all major patterns.

## Focused Examples

For in-depth learning of specific topics:

### 📚 **01_basic_usage.py** - Getting Started
- Simple agent patterns
- Core concepts and interfaces
- Basic LLM integration

### 🎯 **02_custom_agents.py** - Specialized Agents
- Sentiment analysis agent
- Translation agent with language override
- Code review agent with structured output
- Custom return types and validation

### 🧠 **03_memory_patterns.py** - Memory Management
- WorkingMemoryAgent usage patterns  
- Conversation context management
- Memory storage and retrieval
- Multi-conversation handling

### 🔧 **04_tool_integration.py** - Function Calling
- Regular functions (return results to LLM)
- Background tasks (fire-and-forget operations)
- Dynamic tool selection
- Multi-category tool organization

### 🤝 **05_agent_composition.py** - Multi-Agent Orchestration
- Orchestrator pattern (master coordinates specialists)
- Pipeline pattern (sequential processing)
- Mesh pattern (interconnected agents)
- Dynamic agent creation

### ✅ **06_testing_agents.py** - Testing Strategies
- Unit testing with mocks
- Integration testing with real LLMs
- Performance and load testing
- Error handling and edge cases

## Example Comparison

| Example | Lines | Focus | Best For |
|---------|-------|-------|----------|
| `agent_quickstart.py` | 85 | Interactive demo | First experience |
| `agents_comprehensive_guide.py` | 671 | All-in-one tutorial | Single-file learning |
| `01_basic_usage.py` | 234 | Foundation concepts | Understanding basics |
| `02_custom_agents.py` | 337 | Specialized agents | Custom implementations |
| `03_memory_patterns.py` | 283 | Memory management | Stateful conversations |
| `04_tool_integration.py` | 529 | Tool patterns | External integrations |
| `05_agent_composition.py` | 504 | Multi-agent systems | Complex workflows |
| `06_testing_agents.py` | 577 | Testing patterns | Quality assurance |

## Learning Path

### Beginners
1. `agent_quickstart.py` - Get hands-on quickly
2. `01_basic_usage.py` - Understand core concepts  
3. `02_custom_agents.py` - Create specialized agents

### Intermediate
4. `03_memory_patterns.py` - Add conversation context
5. `04_tool_integration.py` - Integrate external functions

### Advanced
6. `05_agent_composition.py` - Build multi-agent systems
7. `06_testing_agents.py` - Ensure production quality

### Reference
- `agents_comprehensive_guide.py` - Complete reference in one file

## Next Steps

After working through the examples:

1. **Implementation Guide**: See `/arshai/agents/README.md` for critical implementation notes
2. **Technical Architecture**: See `/docs/technical/agent_architecture.md` for architectural decisions
3. **Create Your Own**: Start with BaseAgent and implement the `process()` method
4. **Integration**: Use agents in workflows, applications, and services

## Support

- **Issues**: https://github.com/MobileTechLab/ArsHai/issues
- **Architecture**: `/docs/technical/agent_architecture.md`
- **Implementation Guide**: `/arshai/agents/README.md`