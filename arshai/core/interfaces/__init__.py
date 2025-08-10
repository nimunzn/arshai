"""
Core interfaces for the Arshai framework.
"""

# Agent interfaces
from .iagent import IAgent, IAgentInput

# LLM interfaces  
from .illm import ILLM, ILLMConfig, ILLMInput

# Memory interfaces
from .imemorymanager import IMemoryManager, IWorkingMemory, ConversationMemoryType, IMemoryInput

# Tool interfaces
from .itool import ITool

# Workflow interfaces
from .iworkflow import IWorkflowState, IUserContext, IWorkflowOrchestrator, IWorkflowConfig, INode
from .iworkflowrunner import IWorkflowRunner

# Document interfaces
from .idocument import Document

# Other interfaces
from .iembedding import IEmbedding
from .ivector_db_client import IVectorDBClient
from .isearch_client import ISearchClient
from .idto import IDTO, IStreamDTO

# All available interfaces
__all__ = [
    # Agent
    "IAgent", "IAgentInput",
    # LLM
    "ILLM", "ILLMConfig", "ILLMInput",
    # Memory
    "IMemoryManager", "IWorkingMemory", "ConversationMemoryType", "IMemoryInput",
    # Tool
    "ITool",
    # Workflow
    "IWorkflowState", "IUserContext", "IWorkflowOrchestrator", "IWorkflowConfig", "INode", "IWorkflowRunner",
    # Document
    "Document",
    # Other
    "IEmbedding", "IVectorDBClient", "ISearchClient", "IDTO", "IStreamDTO",
]

# Backward compatibility
IWorkflow = IWorkflowConfig
__all__.append("IWorkflow")