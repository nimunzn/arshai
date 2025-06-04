"""
Factory for creating agent components.
"""

from typing import Dict, Any, Type
from arshai.core.interfaces import IAgent, IAgentConfig
from arshai.core.interfaces import ISetting
from ..agents.conversation import ConversationAgent

class AgentFactory:
    """Factory for creating agent instances."""
    
    # Registry of agent types
    _agent_types = {
        "conversation": ConversationAgent,
    }
    
    @classmethod
    def register(cls, name: str, agent_class: Type[IAgent]) -> None:
        """
        Register a new agent type.
        
        Args:
            name: Agent type name
            agent_class: Class implementing IAgent
        """
        cls._agent_types[name.lower()] = agent_class
    
    @classmethod
    def create(cls, agent_type: str, config: IAgentConfig, settings: ISetting) -> IAgent:
        """
        Create an agent instance based on the specified type.
        
        Args:
            agent_type: Type of agent to create (e.g., "conversation")
            config: Agent configuration
            settings: Settings for the agent
            
        Returns:
            An instance implementing IAgent
            
        Raises:
            ValueError: If the agent type is not supported
        """
        agent_type = agent_type.lower()
        
        if agent_type not in cls._agent_types:
            raise ValueError(
                f"Unsupported agent type: {agent_type}. "
                f"Supported types: {', '.join(cls._agent_types.keys())}"
            )
        
        agent_class = cls._agent_types[agent_type]
        
        return agent_class(config=config, settings=settings) 