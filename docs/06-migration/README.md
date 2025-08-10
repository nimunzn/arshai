# Migration Guides

This section helps you migrate from older Arshai patterns to the new three-layer architecture with direct instantiation.

## Available Migration Guides

### [From Settings Pattern](from-settings.md)
Complete guide for migrating from Settings-based component creation to direct instantiation patterns. This is the most common migration path.

**Before:**
```python
settings = Settings()
agent = settings.create_agent("operator", config)
```

**After:**
```python
llm_client = OpenAIClient(config)
agent = OperatorAgent(llm_client, system_prompt)
```

## Migration Strategy

### 1. **Assess Your Codebase**
- Identify Settings usage: `grep -r "Settings" your_code/`
- Find factory patterns: `grep -r "Factory" your_code/`
- Locate implicit dependencies

### 2. **Plan Your Migration**
- Start with standalone components
- Work from bottom-up (LLM clients → Agents → Systems)  
- Update tests alongside code

### 3. **Execute Migration**
- Follow the step-by-step guides
- Test each component thoroughly
- Update documentation and examples

## Breaking Changes Summary

### Removed Components
- ❌ `Settings` class and `ISetting` interface
- ❌ All factory classes (`LLMFactory`, `AgentFactory`, etc.)
- ❌ Configuration-driven component creation

### New Patterns
- ✅ Direct instantiation of all components
- ✅ Constructor-based dependency injection
- ✅ Environment variable configuration
- ✅ Optional YAML configuration utilities

## Common Migration Patterns

### Pattern 1: Settings → Direct Instantiation
```python
# Before
settings = Settings()
llm = settings.create_llm()

# After  
config = ILLMConfig(model="gpt-4")
llm = OpenAIClient(config)
```

### Pattern 2: Factory → Direct Creation
```python
# Before
agent = AgentFactory.create("assistant", config)

# After
llm_client = OpenAIClient(llm_config)
agent = AssistantAgent(llm_client, system_prompt)
```

### Pattern 3: Implicit → Explicit Dependencies
```python
# Before
class MyAgent:
    def __init__(self):
        self.llm = get_llm()  # Implicit

# After
class MyAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, prompt: str):
        super().__init__(llm_client, prompt)  # Explicit
```

## Testing Migration

### Before Migration
```python
def test_old_way():
    settings = Mock()
    agent = AgentFactory.create("test", settings)
    # Hard to control what gets created
```

### After Migration
```python
def test_new_way():
    mock_llm = AsyncMock()
    agent = MyAgent(mock_llm, "prompt")
    # Full control over dependencies
```

## Getting Help

- **Detailed Guide**: [From Settings Pattern](from-settings.md)
- **Examples**: Check the [examples](https://github.com/MobileTechLab/ArsHai/tree/main/examples) directory
- **Issues**: Report problems on [GitHub](https://github.com/MobileTechLab/ArsHai/issues)

## Migration Checklist

- [ ] Read the [From Settings Pattern](from-settings.md) guide
- [ ] Identify all Settings/Factory usage in your code
- [ ] Plan migration order (dependencies first)
- [ ] Update LLM client creation
- [ ] Migrate agent instantiation
- [ ] Update system orchestration
- [ ] Verify all tests pass
- [ ] Update documentation

---

*Need help migrating? Start with [From Settings Pattern](from-settings.md) →*