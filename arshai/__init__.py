"""
Arshai: A powerful agent framework for building conversational AI systems.
"""

from arshai._version import __version__, __version_info__, __author__, __email__

# Core exports
__all__ = [
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
    # Config
    "Settings",
    # Interfaces  
    "IAgent",
    "IAgentConfig",
    "IAgentInput",
    "IAgentOutput",
    "IWorkflow",
    "IWorkflowState",
    "ITool",
    "IMemoryManager", 
    "ILLM",
    # Implementations
    "ConversationAgent",
    "WorkflowRunner",
    "BaseWorkflowConfig",
]

# Lazy loading to avoid import issues with optional dependencies
def __getattr__(name):
    """Lazy import for better dependency handling."""
    if name == "Settings":
        try:
            from arshai.config.settings import Settings
            return Settings
        except ImportError as e:
            raise ImportError(f"Cannot import Settings: {e}")
    
    elif name in ["IAgent", "IAgentConfig", "IAgentInput", "IAgentOutput", 
                  "IWorkflow", "IWorkflowState", "ITool", "IMemoryManager", "ILLM"]:
        try:
            from arshai.core.interfaces import __dict__ as interfaces
            if name in interfaces:
                return interfaces[name]
            else:
                raise ImportError(f"Interface {name} not found")
        except ImportError as e:
            raise ImportError(f"Cannot import interface {name}: {e}")
    
    elif name == "ConversationAgent":
        try:
            from arshai.agents.conversation import ConversationAgent
            return ConversationAgent
        except ImportError as e:
            raise ImportError(f"Cannot import ConversationAgent: {e}")
    
    elif name == "WorkflowRunner":
        try:
            from arshai.workflows.workflow_runner import WorkflowRunner
            return WorkflowRunner
        except ImportError as e:
            raise ImportError(f"Cannot import WorkflowRunner: {e}")
    
    elif name == "BaseWorkflowConfig":
        try:
            from arshai.workflows.workflow_config import BaseWorkflowConfig
            return BaseWorkflowConfig
        except ImportError as e:
            raise ImportError(f"Cannot import BaseWorkflowConfig: {e}")
    
    raise AttributeError(f"module 'arshai' has no attribute '{name}'")
