"""
Basic Usage of Arshai Components

This example demonstrates:
1. Creating and using predefined agents through settings
2. Implementing and using custom agents directly
3. Using factory patterns for predefined components (LLMs, Memory)
4. Working with settings

The simplified architecture emphasizes direct instantiation for custom components
while using factories only for predefined components.
"""

from arshai.core.interfaces import IAgentConfig, IAgentInput, IAgentOutput, IAgent
from arshai.core.interfaces import ILLMConfig
from arshai.core.interfaces import ISetting
from arshai.config.settings import Settings
from arshai.factories import LLMFactory, MemoryFactory


# =========================================================================
# PART 1: Simple Settings Implementation
# =========================================================================

class SimpleSettings(ISetting):
    """A simple implementation of ISetting for demonstration."""
    
    def __init__(self, llm_provider="demo", api_key=None):
        self.llm_provider = llm_provider
        self.api_key = api_key
    
    def create_llm(self):
        """Create a demo LLM client."""
        # In a real implementation, you would return an actual LLM client
        return DemoLLM()
    
    def create_memory_manager(self):
        """Create a demo memory manager."""
        # In a real implementation, you would return an actual memory manager
        return DemoMemoryManager()
    
    def create_agent(self, agent_type, agent_config):
        """Create a predefined agent."""
        # In a real implementation, you would create the actual agent
        # Here we're using a mock implementation for demonstration
        if agent_type == "operator":
            return DemoAgent(config=agent_config, settings=self)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


# =========================================================================
# PART 2: Demo Components for Example
# =========================================================================

class DemoLLM:
    """A demo LLM implementation for the example."""
    
    def chat_with_tools(self, input):
        """Demo implementation that simply echoes the input message."""
        return IAgentOutput(
            agent_message="This is a demo response to: " + input.user_message,
            memory=input.system_prompt
        )


class DemoMemoryManager:
    """A demo memory manager implementation for the example."""
    
    def retrieve(self, input):
        """Demo implementation that returns empty memory."""
        return []
    
    def store(self, input):
        """Demo implementation that does nothing."""
        pass


class DemoAgent(IAgent):
    """A demo agent implementation."""
    
    def __init__(self, config, settings):
        self.config = config
        self.settings = settings
        self.task_context = config.get("task_context", "Demo task")
    
    def process_message(self, input: IAgentInput) -> IAgentOutput:
        """Process a message and return a response."""
        return IAgentOutput(
            agent_message=f"Demo agent responding to: {input.message}\nMy task: {self.task_context}"
        )
    
    async def aprocess_message(self, input: IAgentInput) -> IAgentOutput:
        """Async version of process_message."""
        return self.process_message(input)


# =========================================================================
# PART 3: Custom Agent Implementation
# =========================================================================

class CustomAgent(IAgent):
    """A custom agent implementation showcasing direct instantiation pattern."""
    
    def __init__(self, config, settings):
        """Initialize the custom agent."""
        self.config = config
        self.settings = settings
        self.llm = settings.create_llm()
        self.task_context = config.get("task_context", "Default task context")
    
    def process_message(self, input: IAgentInput) -> IAgentOutput:
        """Process a message and return a response."""
        # In a real implementation, you would use an LLM and more complex logic
        return IAgentOutput(
            agent_message=f"Custom agent responding to: {input.message}\nMy task: {self.task_context}"
        )
    
    async def aprocess_message(self, input: IAgentInput) -> IAgentOutput:
        """Async version of process_message."""
        return self.process_message(input)


# =========================================================================
# PART 4: Using Factories for Predefined Components
# =========================================================================

def demonstrate_llm_factory():
    """Demonstrate using the LLM factory."""
    print("\n=== LLM Factory Examples ===")
    
    # Create OpenAI LLM
    openai_config = ILLMConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )
    openai_llm = LLMFactory.create("openai", openai_config)
    print(f"Created OpenAI LLM: {openai_llm.__class__.__name__}")
    
    # Create Azure LLM
    azure_config = ILLMConfig(
        model="gpt-35-turbo",
        temperature=0.5,
        max_tokens=800
    )
    azure_llm = LLMFactory.create(
        "azure", 
        azure_config, 
        azure_deployment="my-deployment",
        api_version="2023-05-15"
    )
    print(f"Created Azure LLM: {azure_llm.__class__.__name__}")


def demonstrate_memory_factory():
    """Demonstrate using the Memory factory."""
    print("\n=== Memory Factory Examples ===")
    
    # Create in-memory working memory
    in_memory = MemoryFactory.create_working_memory("in_memory", ttl=3600)
    print(f"Created in-memory working memory: {in_memory.__class__.__name__}")
    
    # Create Redis working memory
    redis_memory = MemoryFactory.create_working_memory(
        "redis", 
        storage_url="redis://localhost:6379/0",
        ttl=3600
    )
    print(f"Created Redis working memory: {redis_memory.__class__.__name__}")
    
    # Create memory manager service
    memory_config = {
        "working_memory": {
            "provider": "in_memory",
            "ttl": 3600
        }
    }
    memory_service = MemoryFactory.create_memory_manager_service(memory_config)
    print(f"Created memory manager service: {memory_service.__class__.__name__}")


# =========================================================================
# PART 5: Creating and Using Agents
# =========================================================================

def demonstrate_predefined_agent():
    """Demonstrate using a predefined agent through settings."""
    print("\n=== Predefined Agent Example ===")
    
    # Create settings
    settings = SimpleSettings(llm_provider="demo")
    
    # Create agent config
    config = IAgentConfig(
        task_context="You are a helpful assistant",
        tools=[]
    )
    
    # Create agent using settings
    agent = settings.create_agent("operator", config)
    
    # Process a message
    input_message = IAgentInput(
        conversation_id="demo-123",
        message="Hello, can you help me with something?"
    )
    
    # Get response
    response = agent.process_message(input_message)
    print(f"Agent response: {response.agent_message}")


def demonstrate_custom_agent():
    """Demonstrate creating and using a custom agent directly."""
    print("\n=== Custom Agent Example ===")
    
    # Create settings
    settings = SimpleSettings(llm_provider="demo")
    
    # Create agent config
    config = IAgentConfig(
        task_context="You are a specialized assistant for tax advice",
        tools=[]
    )
    
    # Create custom agent directly
    agent = CustomAgent(config=config, settings=settings)
    
    # Process a message
    input_message = IAgentInput(
        conversation_id="custom-123",
        message="What tax deductions can I claim?"
    )
    
    # Get response
    response = agent.process_message(input_message)
    print(f"Agent response: {response.agent_message}")


# =========================================================================
# PART 6: Using Real Settings
# =========================================================================

def demonstrate_real_settings():
    """Demonstrate using the actual Settings class from the framework."""
    print("\n=== Using Real Settings ===")
    
    try:
        # Create settings
        settings = Settings()
        print("Successfully created Settings instance")
        
        # Create a predefined agent
        agent_config = IAgentConfig(
            task_context="You are a helpful assistant",
            tools=[]
        )
        
        # This will try to create a real agent - may fail in demo environment
        # without proper configuration
        try:
            agent = settings.create_agent("operator", agent_config)
            print(f"Successfully created agent: {agent.__class__.__name__}")
        except Exception as e:
            print(f"Note: Could not create real agent: {str(e)}")
            print("This is expected in a demo environment without full configuration")
    
    except Exception as e:
        print(f"Note: Could not initialize real settings: {str(e)}")
        print("This is expected in a demo environment without full configuration")


# =========================================================================
# Main Example Runner
# =========================================================================

def main():
    """Run all example code."""
    print("Arshai Basic Usage Examples\n")
    
    try:
        # Factory examples
        demonstrate_llm_factory()
        demonstrate_memory_factory()
        
        # Agent examples
        demonstrate_predefined_agent()
        demonstrate_custom_agent()
        
        # Real settings example
        demonstrate_real_settings()
        
        print("\nAll examples completed!")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Note: Some examples may fail in environments without proper configuration")


if __name__ == "__main__":
    main() 