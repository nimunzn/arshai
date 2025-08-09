# Developer Patterns Guide

This guide provides best practices and common patterns for building AI applications with Arshai's three-layer architecture.

## Three-Layer Development Philosophy

### Layer 1: LLM Clients - Use As Provided
- **What they are**: Core AI capabilities (OpenAI, Azure, Gemini, OpenRouter)
- **Your authority**: Minimal - use as framework provides them
- **Best practice**: Create directly, follow interfaces exactly
- **Example**: `OpenAIClient(config)` - no customization needed

### Layer 2: Agents - Extend and Customize  
- **What they are**: Purpose-driven wrappers over LLM clients
- **Your authority**: Moderate - extend base classes or implement interfaces
- **Best practice**: Create agents that do one thing well
- **Example**: Custom agents for specific business domains

### Layer 3: Agentic Systems - Complete Control
- **What they are**: Orchestration and application-level patterns
- **Your authority**: Maximum - design exactly what you need
- **Best practice**: Use framework utilities only if they fit
- **Example**: Multi-agent workflows, custom orchestration

## Core Development Patterns

### Pattern 1: Direct Instantiation (Primary)

**Always prefer explicit component creation:**

```python
# ✅ GOOD: Direct instantiation
from arshai.llms.openai import OpenAIClient
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
from arshai.agents.working_memory import WorkingMemoryAgent

config = ILLMConfig(model="gpt-4o", temperature=0.7)
llm = OpenAIClient(config)

memory = RedisMemoryManager(redis_url="redis://localhost:6379")
agent = WorkingMemoryAgent(llm, memory, "You are helpful")

# ❌ AVOID: Hidden dependencies (old Settings pattern)
# settings = Settings()
# agent = settings.create_agent("working_memory", config)
```

**Benefits:**
- All dependencies explicit
- Easy to test with mocks
- Clear understanding of what's being used
- No hidden configuration

### Pattern 2: Environment Variable Management

**Each component manages its own environment variables:**

```python
# ✅ GOOD: Components read their own env vars
import os

# OpenAI client reads OPENAI_API_KEY automatically
llm = OpenAIClient(config)

# Redis manager reads REDIS_URL if provided
memory = RedisMemoryManager(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    ttl=int(os.getenv("REDIS_TTL", "3600"))
)

# ❌ AVOID: Centralized env var management
# settings.redis_url  # Hidden dependency
```

### Pattern 3: Optional Configuration Loading

**Use configuration files only when they add value:**

```python
from arshai.config import load_config

# ✅ GOOD: Optional configuration
config_data = load_config("app.yaml")  # Returns {} if file doesn't exist

llm_config = ILLMConfig(
    model=config_data.get("llm", {}).get("model", "gpt-4o"),
    temperature=config_data.get("llm", {}).get("temperature", 0.7)
)
llm = OpenAIClient(llm_config)

# ❌ AVOID: Required configuration files
# settings = Settings("required_config.yaml")  # Fails if file missing
```

### Pattern 4: Dependency Injection

**Pass dependencies explicitly through constructors:**

```python
# ✅ GOOD: Explicit dependency injection
class AnalysisService:
    def __init__(self, llm_client: ILLM, memory_manager: IMemoryManager):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
    
    async def analyze(self, data: str) -> str:
        # Use injected dependencies
        pass

# Create dependencies explicitly
llm = OpenAIClient(config)
memory = RedisMemoryManager(redis_url="redis://localhost:6379")
service = AnalysisService(llm, memory)

# ❌ AVOID: Hidden dependencies
# class AnalysisService:
#     def __init__(self):
#         self.settings = Settings()  # Hidden global dependency
```

## Agent Development Patterns

### Pattern 1: Extending BaseAgent (Recommended)

```python
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput

class CustomerServiceAgent(BaseAgent):
    """Specialized agent for customer service interactions."""
    
    def __init__(self, llm_client: ILLM, knowledge_base: KnowledgeBase):
        super().__init__(llm_client, "You are a helpful customer service agent")
        self.knowledge_base = knowledge_base
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # Define tools specific to this agent
        async def search_knowledge(query: str) -> str:
            return await self.knowledge_base.search(query)
        
        async def log_interaction(query: str) -> None:
            """BACKGROUND TASK: Log customer interaction."""
            print(f"Customer query: {query}")
        
        # Prepare LLM input with agent-specific tools
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions={"search_knowledge": search_knowledge},
            background_tasks={"log_interaction": log_interaction}
        )
        
        result = await self.llm_client.chat(llm_input)
        return {
            "response": result["llm_response"],
            "confidence": self._assess_confidence(result),
            "sources_used": self._extract_sources(result)
        }
```

### Pattern 2: Direct Interface Implementation

```python
from arshai.core.interfaces.iagent import IAgent, IAgentInput

class MinimalAgent(IAgent):
    """Minimal agent with complete control over behavior."""
    
    def __init__(self, llm_client: ILLM, custom_config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = custom_config
    
    async def process(self, input: IAgentInput) -> str:
        # Complete control over processing logic
        prompt = self._build_custom_prompt(input.message)
        
        llm_input = ILLMInput(
            system_prompt=prompt,
            user_message=input.message
        )
        
        result = await self.llm_client.chat(llm_input)
        return self._post_process_response(result["llm_response"])
    
    def _build_custom_prompt(self, message: str) -> str:
        # Custom prompt engineering logic
        return f"Custom instructions for: {message}"
    
    def _post_process_response(self, response: str) -> str:
        # Custom response processing
        return response.upper()  # Example transformation
```

### Pattern 3: Agent Composition

```python
class MultiAgentOrchestrator:
    """Layer 3: Orchestrate multiple specialized agents."""
    
    def __init__(self, agents: Dict[str, IAgent]):
        self.agents = agents
    
    async def route_request(self, request_type: str, message: str) -> Dict[str, Any]:
        """Route requests to appropriate specialized agent."""
        
        if request_type in self.agents:
            agent = self.agents[request_type]
            input_data = IAgentInput(
                message=message,
                metadata={"request_type": request_type}
            )
            result = await agent.process(input_data)
            return {"agent_used": request_type, "result": result}
        else:
            # Fallback to general agent
            general_agent = self.agents.get("general")
            if general_agent:
                result = await general_agent.process(IAgentInput(message=message))
                return {"agent_used": "general", "result": result}
            else:
                return {"error": "No suitable agent found"}

# Usage: Developer controls all composition
customer_agent = CustomerServiceAgent(llm, knowledge_base)
technical_agent = TechnicalSupportAgent(llm, issue_tracker)
billing_agent = BillingAgent(llm, billing_system)

orchestrator = MultiAgentOrchestrator({
    "customer_service": customer_agent,
    "technical_support": technical_agent,
    "billing": billing_agent,
    "general": customer_agent  # Fallback
})
```

## Memory Management Patterns

### Pattern 1: Direct Memory Manager Usage

```python
from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
from arshai.memory.working_memory.in_memory_manager import InMemoryManager

# Choose appropriate memory implementation
def create_memory_manager(environment: str) -> IMemoryManager:
    if environment == "production":
        return RedisMemoryManager(
            redis_url=os.getenv("REDIS_URL"),
            ttl=3600,
            key_prefix="prod_app"
        )
    elif environment == "testing":
        return InMemoryManager(ttl=300)  # Short TTL for tests
    else:
        return InMemoryManager(ttl=1800)  # Development

# Usage in agents
memory_manager = create_memory_manager(os.getenv("ENVIRONMENT", "development"))
agent = WorkingMemoryAgent(llm, memory_manager, system_prompt)
```

### Pattern 2: Memory Integration with Custom Agents

```python
class ConversationAgent(BaseAgent):
    """Agent with sophisticated memory management."""
    
    def __init__(self, llm_client: ILLM, memory_manager: IMemoryManager, user_id: str):
        super().__init__(llm_client, "You are a conversational assistant")
        self.memory_manager = memory_manager
        self.user_id = user_id
    
    async def process(self, input: IAgentInput) -> str:
        conversation_id = input.metadata.get("conversation_id")
        
        # Retrieve conversation history
        memory_data = await self._get_conversation_memory(conversation_id)
        
        # Build context-aware prompt
        context_prompt = self._build_contextual_prompt(memory_data)
        
        llm_input = ILLMInput(
            system_prompt=context_prompt,
            user_message=input.message,
            background_tasks={"update_memory": self._update_memory}
        )
        
        result = await self.llm_client.chat(llm_input)
        return result["llm_response"]
    
    async def _get_conversation_memory(self, conversation_id: str) -> Dict[str, Any]:
        if conversation_id:
            memory = await self.memory_manager.retrieve({
                "conversation_id": conversation_id,
                "user_id": self.user_id
            })
            return memory[0].data if memory else {}
        return {}
    
    async def _update_memory(self, conversation_summary: str) -> None:
        """BACKGROUND TASK: Update conversation memory."""
        if hasattr(self, '_current_conversation_id'):
            await self.memory_manager.store({
                "conversation_id": self._current_conversation_id,
                "user_id": self.user_id,
                "working_memory": conversation_summary
            })
```

## Tool Integration Patterns

### Pattern 1: Simple Function Tools

```python
def get_current_time() -> str:
    """Get current time in UTC."""
    from datetime import datetime
    return datetime.utcnow().isoformat()

def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price."""
    return price * (1 - discount_percent / 100)

# Integration with agent
class ShoppingAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> str:
        tools = {
            "get_current_time": get_current_time,
            "calculate_discount": calculate_discount
        }
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions=tools
        )
        
        result = await self.llm_client.chat(llm_input)
        return result["llm_response"]
```

### Pattern 2: Class-Based Tools

```python
class DatabaseTool:
    """Tool for database operations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def search_products(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Search for products in database."""
        # Database search logic
        return [{"name": "Product 1", "price": 29.99}]
    
    def get_user_orders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's order history."""
        # Database query logic
        return [{"order_id": "123", "total": 59.99}]

class EmailTool:
    """Tool for email operations."""
    
    def __init__(self, email_service: EmailService):
        self.email_service = email_service
    
    async def send_confirmation_email(self, user_email: str, order_id: str) -> None:
        """BACKGROUND TASK: Send order confirmation email."""
        await self.email_service.send(
            to=user_email,
            subject=f"Order Confirmation #{order_id}",
            body=f"Your order {order_id} has been confirmed."
        )

# Integration pattern
class EcommerceAgent(BaseAgent):
    def __init__(self, llm_client: ILLM, db_tool: DatabaseTool, email_tool: EmailTool):
        super().__init__(llm_client, "You are an e-commerce assistant")
        self.db_tool = db_tool
        self.email_tool = email_tool
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        # Developer decides which tools to expose to LLM
        regular_functions = {
            "search_products": self.db_tool.search_products,
            "get_user_orders": self.db_tool.get_user_orders,
        }
        
        background_tasks = {
            "send_confirmation_email": self.email_tool.send_confirmation_email
        }
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions=regular_functions,
            background_tasks=background_tasks
        )
        
        result = await self.llm_client.chat(llm_input)
        return {"response": result["llm_response"]}
```

### Pattern 3: Dynamic Tool Selection

```python
class AdaptiveAgent(BaseAgent):
    """Agent that selects tools based on context."""
    
    def __init__(self, llm_client: ILLM, all_tools: Dict[str, Callable]):
        super().__init__(llm_client, "You are an adaptive assistant")
        self.all_tools = all_tools
    
    async def process(self, input: IAgentInput) -> str:
        # Select tools based on input context
        selected_tools = self._select_tools(input.message)
        
        llm_input = ILLMInput(
            system_prompt=self._adapt_prompt(selected_tools),
            user_message=input.message,
            regular_functions=selected_tools
        )
        
        result = await self.llm_client.chat(llm_input)
        return result["llm_response"]
    
    def _select_tools(self, message: str) -> Dict[str, Callable]:
        """Select relevant tools based on message content."""
        selected = {}
        
        if "weather" in message.lower():
            selected["get_weather"] = self.all_tools["get_weather"]
        
        if any(word in message.lower() for word in ["calculate", "math", "compute"]):
            selected["calculate"] = self.all_tools["calculate"]
        
        if "email" in message.lower():
            selected["send_email"] = self.all_tools["send_email"]
        
        return selected
    
    def _adapt_prompt(self, tools: Dict[str, Callable]) -> str:
        """Adapt system prompt based on available tools."""
        base_prompt = "You are a helpful assistant"
        
        if tools:
            tool_names = ", ".join(tools.keys())
            return f"{base_prompt} with access to: {tool_names}"
        
        return base_prompt
```

## Testing Patterns

### Pattern 1: Component Unit Testing

```python
import pytest
from unittest.mock import Mock, AsyncMock
from arshai.core.interfaces.iagent import IAgentInput

@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.chat.return_value = {
        "llm_response": "Test response",
        "usage": {"total_tokens": 100}
    }
    return mock

@pytest.fixture
def mock_memory():
    mock = AsyncMock()
    mock.retrieve.return_value = []
    mock.store.return_value = "memory_id_123"
    return mock

async def test_customer_service_agent(mock_llm, mock_memory):
    """Test agent with mocked dependencies."""
    # Create agent with mocks - easy with direct instantiation
    agent = CustomerServiceAgent(mock_llm, mock_memory)
    
    # Test agent behavior
    result = await agent.process(IAgentInput(
        message="I need help with my order",
        metadata={"conversation_id": "conv_123"}
    ))
    
    # Verify interactions
    assert result["response"] == "Test response"
    mock_llm.chat.assert_called_once()
    mock_memory.store.assert_called_once()

async def test_llm_client_integration():
    """Test with real LLM client for integration testing."""
    config = ILLMConfig(model="gpt-4o-mini", temperature=0.1)
    client = OpenAIClient(config)
    
    input_data = ILLMInput(
        system_prompt="You are a test assistant",
        user_message="Say 'integration test successful'"
    )
    
    response = await client.chat(input_data)
    assert "integration test successful" in response["llm_response"].lower()
```

### Pattern 2: End-to-End Testing

```python
async def test_complete_application_flow():
    """Test entire application with real components."""
    
    # Create real components for integration test
    llm = OpenAIClient(ILLMConfig(model="gpt-4o-mini", temperature=0.1))
    memory = InMemoryManager(ttl=300)  # Use in-memory for tests
    
    # Test tools
    def test_calculator(a: float, b: float) -> float:
        return a + b
    
    # Create agent with real dependencies
    class TestAgent(BaseAgent):
        async def process(self, input: IAgentInput) -> str:
            llm_input = ILLMInput(
                system_prompt="You are a test assistant with a calculator",
                user_message=input.message,
                regular_functions={"add": test_calculator}
            )
            result = await self.llm_client.chat(llm_input)
            return result["llm_response"]
    
    agent = TestAgent(llm, "Test agent")
    
    # Test complete flow
    result = await agent.process(IAgentInput(message="What is 5 + 3?"))
    
    assert "8" in result
```

## Performance Patterns

### Pattern 1: Connection Pooling

```python
class ServiceManager:
    """Manage shared resources across application."""
    
    def __init__(self):
        # Create shared clients for connection reuse
        self.llm_client = OpenAIClient(ILLMConfig(model="gpt-4o"))
        self.memory_manager = RedisMemoryManager(
            redis_url=os.getenv("REDIS_URL"),
            ttl=3600
        )
    
    def create_agent(self, agent_type: str, **kwargs) -> IAgent:
        """Create agents with shared resources."""
        if agent_type == "customer_service":
            return CustomerServiceAgent(
                self.llm_client, 
                self.memory_manager,
                **kwargs
            )
        # ... other agent types
    
# Usage across application
service_manager = ServiceManager()
agent1 = service_manager.create_agent("customer_service", knowledge_base=kb1)
agent2 = service_manager.create_agent("customer_service", knowledge_base=kb2)
# Both agents share the same LLM client connection
```

### Pattern 2: Async Concurrency

```python
import asyncio

async def process_multiple_requests(agents: List[IAgent], requests: List[str]) -> List[str]:
    """Process multiple requests concurrently."""
    
    tasks = []
    for i, request in enumerate(requests):
        agent = agents[i % len(agents)]  # Round-robin agent selection
        task = agent.process(IAgentInput(message=request))
        tasks.append(task)
    
    # Execute all requests concurrently
    results = await asyncio.gather(*tasks)
    return results

# Usage
agents = [create_customer_agent(), create_technical_agent()]
requests = ["Help with billing", "Technical issue", "General question"]
results = await process_multiple_requests(agents, requests)
```

## Configuration Patterns

### Pattern 1: Environment-Based Configuration

```python
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class AppConfig:
    """Application configuration based on environment."""
    
    def __init__(self, environment: Environment = None):
        self.env = environment or Environment(os.getenv("ENVIRONMENT", "development"))
        
    def get_llm_config(self) -> ILLMConfig:
        if self.env == Environment.PRODUCTION:
            return ILLMConfig(
                model="gpt-4o",
                temperature=0.3,
                max_tokens=2000
            )
        elif self.env == Environment.TESTING:
            return ILLMConfig(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=500
            )
        else:  # Development
            return ILLMConfig(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            )
    
    def get_memory_manager(self) -> IMemoryManager:
        if self.env == Environment.PRODUCTION:
            return RedisMemoryManager(
                redis_url=os.getenv("REDIS_URL"),
                ttl=3600
            )
        else:
            return InMemoryManager(ttl=1800)

# Usage
config = AppConfig()
llm = OpenAIClient(config.get_llm_config())
memory = config.get_memory_manager()
```

### Pattern 2: Configuration File Pattern

```python
from arshai.config import load_config
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class LLMSettings:
    model: str
    temperature: float
    max_tokens: Optional[int] = None

@dataclass
class AppSettings:
    llm: LLMSettings
    redis_url: str
    debug: bool = False

def load_app_settings(config_file: str = "app.yaml") -> AppSettings:
    """Load and validate application settings."""
    config_data = load_config(config_file)
    
    llm_data = config_data.get("llm", {})
    llm_settings = LLMSettings(
        model=llm_data.get("model", "gpt-4o"),
        temperature=llm_data.get("temperature", 0.7),
        max_tokens=llm_data.get("max_tokens")
    )
    
    return AppSettings(
        llm=llm_settings,
        redis_url=config_data.get("redis_url", "redis://localhost:6379"),
        debug=config_data.get("debug", False)
    )

# Usage
settings = load_app_settings()
llm_config = ILLMConfig(
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    max_tokens=settings.llm.max_tokens
)
llm_client = OpenAIClient(llm_config)
```

## Error Handling Patterns

### Pattern 1: Graceful Degradation

```python
class ResilientAgent(BaseAgent):
    """Agent with fallback capabilities."""
    
    def __init__(self, primary_llm: ILLM, fallback_llm: ILLM, **kwargs):
        super().__init__(primary_llm, **kwargs)
        self.fallback_llm = fallback_llm
    
    async def process(self, input: IAgentInput) -> str:
        try:
            # Try primary LLM first
            return await self._process_with_llm(self.llm_client, input)
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}, falling back")
            try:
                # Fallback to secondary LLM
                return await self._process_with_llm(self.fallback_llm, input)
            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed: {fallback_error}")
                return f"I'm experiencing technical difficulties. Please try again later."
    
    async def _process_with_llm(self, llm_client: ILLM, input: IAgentInput) -> str:
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        result = await llm_client.chat(llm_input)
        return result["llm_response"]

# Usage
primary_llm = OpenAIClient(ILLMConfig(model="gpt-4o"))
fallback_llm = OpenRouterClient(ILLMConfig(model="anthropic/claude-3.5-sonnet"))
agent = ResilientAgent(primary_llm, fallback_llm, system_prompt="You are helpful")
```

### Pattern 2: Circuit Breaker Pattern

```python
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for LLM calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class CircuitBreakerAgent(BaseAgent):
    """Agent with circuit breaker protection."""
    
    def __init__(self, llm_client: ILLM, **kwargs):
        super().__init__(llm_client, **kwargs)
        self.circuit_breaker = CircuitBreaker()
    
    async def process(self, input: IAgentInput) -> str:
        if not self.circuit_breaker.can_execute():
            return "Service temporarily unavailable. Please try again later."
        
        try:
            llm_input = ILLMInput(
                system_prompt=self.system_prompt,
                user_message=input.message
            )
            result = await self.llm_client.chat(llm_input)
            self.circuit_breaker.record_success()
            return result["llm_response"]
        except Exception as e:
            self.circuit_breaker.record_failure()
            return f"Request failed: {str(e)}"
```

## Summary

These patterns demonstrate how to build robust, scalable AI applications using Arshai's three-layer architecture:

1. **Layer 1 (LLM Clients)**: Use direct instantiation, respect interfaces
2. **Layer 2 (Agents)**: Extend base classes or implement interfaces directly
3. **Layer 3 (Agentic Systems)**: Design exactly what you need

**Key Principles:**
- **Explicit over implicit**: Make all dependencies visible
- **Direct instantiation**: No hidden Settings or factory magic
- **Interface respect**: Follow contracts but customize freely
- **Developer control**: You decide how components work together

The goal is to give you complete authority over your AI application architecture while providing powerful, reliable building blocks.