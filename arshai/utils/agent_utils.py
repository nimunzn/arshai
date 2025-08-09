"""
Optional utilities for creating agent components.

These are convenience functions for developers who want simplified agent creation.
Direct instantiation is always preferred, but these utilities can help in some cases.
"""

from typing import Dict, Type
from arshai.core.interfaces.iagent import IAgent
from ..agents.working_memory import WorkingMemoryAgent


# Registry of agent types
_AGENT_TYPES = {
    "working_memory": WorkingMemoryAgent,
}


def register_agent_type(name: str, agent_class: Type[IAgent]) -> None:
    """
    Register a new agent type for use with create_agent().
    
    Args:
        name: Agent type name
        agent_class: Class implementing IAgent
    """
    _AGENT_TYPES[name.lower()] = agent_class


def create_agent(agent_type: str, **kwargs) -> IAgent:
    """
    Optional utility to create an agent instance.
    
    Note: Direct instantiation is preferred (e.g., WorkingMemoryAgent(llm, memory, prompt)).
    Use this utility only when you need dynamic agent type selection.
    
    Args:
        agent_type: Type of agent to create (e.g., "working_memory")
        **kwargs: Agent-specific configuration and dependencies
        
    Returns:
        An instance implementing IAgent
        
    Raises:
        ValueError: If the agent type is not supported
        
    Example:
        # ✅ PREFERRED: Direct instantiation
        from arshai.agents.working_memory import WorkingMemoryAgent
        from arshai.llms.openai import OpenAIClient
        from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
        
        llm = OpenAIClient(config)
        memory = RedisMemoryManager(redis_url="redis://localhost:6379")
        agent = WorkingMemoryAgent(
            llm_client=llm,
            memory_manager=memory,
            system_prompt="You are helpful"
        )
        
        # ⚠️ OPTIONAL: Utility function for dynamic selection
        agent = create_agent("working_memory", 
                           llm_client=llm, 
                           memory_manager=memory,
                           system_prompt="You are helpful")
    """
    agent_type = agent_type.lower()
    
    if agent_type not in _AGENT_TYPES:
        raise ValueError(
            f"Unsupported agent type: {agent_type}. "
            f"Supported types: {', '.join(_AGENT_TYPES.keys())}"
        )
    
    agent_class = _AGENT_TYPES[agent_type]
    
    # Create agent with provided dependencies
    return agent_class(**kwargs)


def get_available_agent_types() -> Dict[str, Type[IAgent]]:
    """
    Get all registered agent types.
    
    Returns:
        Dictionary mapping agent type names to their classes
    """
    return _AGENT_TYPES.copy()