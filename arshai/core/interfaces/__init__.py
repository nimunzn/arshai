"""
Core interfaces for the Arshai framework.

This module exports the key interfaces for extending Arshai.
"""

# Import interfaces only if they exist
def _import_if_exists(module_name, class_names):
    """Import classes from module if they exist."""
    try:
        module = __import__(f"arshai.core.interfaces.{module_name}", fromlist=class_names)
        return {name: getattr(module, name, None) for name in class_names if hasattr(module, name)}
    except ImportError:
        return {}

# Agent interfaces
_agent_imports = _import_if_exists("iagent", [
    "IAgent", "IAgentConfig", "IAgentInput", "IAgentOutput", "IAgentStreamOutput"
])

# Workflow interfaces
_workflow_imports = _import_if_exists("iworkflow", [
    "IWorkflowState", "IUserContext", "IWorkflowOrchestrator", "IWorkflowConfig", "INode"
])
_workflow_runner_imports = _import_if_exists("iworkflowrunner", ["IWorkflowRunner"])

# Memory interfaces
_memory_imports = _import_if_exists("imemorymanager", ["IMemoryManager", "IWorkingMemory", "ConversationMemoryType", "IMemoryInput"])

# Tool interfaces
_tool_imports = _import_if_exists("itool", ["ITool", "IToolExecutor"])

# LLM interfaces
_llm_imports = _import_if_exists("illm", ["ILLM", "ILLMResponse", "ILLMConfig", "ILLMInput", "LLMInputType", "ILLMOutput", "ILLMStreamOutput"])

# Document interfaces
_document_imports = _import_if_exists("idocument", ["IDocument", "IDocumentMetadata", "Document"])

# Other interfaces
_embedding_imports = _import_if_exists("iembedding", ["IEmbedding"])
_vector_db_imports = _import_if_exists("ivector_db_client", ["IVectorDBClient"])
_search_imports = _import_if_exists("isearch_client", ["ISearchClient"])
_setting_imports = _import_if_exists("isetting", ["ISetting"])
_dto_imports = _import_if_exists("idto", ["IDTO", "IStreamDTO"])

# Combine all imports
_all_imports = {}
for imports in [
    _agent_imports, _workflow_imports, _workflow_runner_imports, _memory_imports,
    _tool_imports, _llm_imports, _document_imports, _embedding_imports,
    _vector_db_imports, _search_imports, _setting_imports, _dto_imports
]:
    _all_imports.update(imports)

# Export available interfaces
locals().update(_all_imports)

# Build __all__ dynamically
__all__ = [name for name, obj in _all_imports.items() if obj is not None]

# Provide backward compatibility aliases
IWorkflow = _workflow_imports.get("IWorkflowConfig")  # Alias for compatibility
if IWorkflow:
    __all__.append("IWorkflow")