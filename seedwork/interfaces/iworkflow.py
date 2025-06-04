from typing_extensions import Protocol
from ..interfaces.imemorymanager import IWorkingMemory
from ..interfaces.inotification import INotificationState
from .idto import IDTO
from copy import deepcopy
from typing import Dict, List, Optional, Any, Callable, Set, Type
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class IUserContext(IDTO):
    """User context information."""
    user_id: str
    last_active: datetime = Field(default_factory=datetime.utcnow)
    interaction_count: int = 0

    def dict(self, *args, **kwargs) -> dict:
        """Convert to dictionary with serializable datetime."""
        d = super().dict(*args, **kwargs)
        d['last_active'] = d['last_active'].isoformat()
        return d


class IWorkflowState(IDTO):
    """Complete workflow state.
    
    This state model follows the pattern from the previous project's WorkflowState,
    providing a central object that flows through all nodes in the workflow and 
    captures the complete state of the workflow execution.
    """
    # User and context information
    user_context: IUserContext
    working_memories: Dict[str, IWorkingMemory] = Field(default_factory=dict)
    notification_state: INotificationState = Field(default_factory=INotificationState)
    
    # Agent-specific data storage
    agent_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Graph execution metadata
    current_step: str = "start"
    step_count: int = 0
    errors: List[Dict] = Field(default_factory=list)
    processing_path: str = "none"
    
    # Workflow execution data that can be customized by specific workflows
    # This will hold any workflow-specific data that doesn't fit the standard fields
    workflow_data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

    def update_from(self, other: 'IWorkflowState') -> None:
        """Update this state from another state instance.
        
        This method allows for easy copying of state from one instance to another,
        which is useful when nodes create new state instances.
        
        Args:
            other: The other state instance to copy from
        """
        for key, value in vars(other).items():
            if not key.startswith('_'):  # Skip private attributes
                setattr(self, key, value)


class INode(Protocol):
    """Interface for workflow nodes that wrap agents.
    
    Nodes are callable objects that process workflow state and return 
    updated state along with any other results. They form the building 
    blocks of workflows.
    """
    
    def __init__(self, node_id: str, name: str, **kwargs):
        """Initialize the node with its identifier and name."""
        ...
    
    def get_id(self) -> str:
        """Get the node's unique identifier."""
        ...
    
    def get_name(self) -> str:
        """Get the node's name."""
        ...
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and return updated state and results.
        
        Args:
            input_data: Dictionary containing the input data including state
            
        Returns:
            Dictionary with updated state and processing results
        """
        ...
    
    def get_agent_settings(self) -> Dict[str, Any]:
        """Get the settings associated with the agent for this node."""
        ...
        
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the node callable, directly delegating to the process method.
        
        This allows nodes to be called as functions, matching the pattern 
        from the previous project.
        
        Args:
            input_data: Dictionary containing the input data including state
            
        Returns:
            Dictionary with processing results
        """
        ...


class IWorkflowOrchestrator(Protocol):
    """Orchestrates the workflow execution.
    
    The orchestrator manages the flow of control and data between nodes in a workflow.
    It is responsible for routing input to the appropriate entry node, executing 
    nodes in sequence based on edge connections, and collecting the final results.
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize workflow orchestrator."""
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.entry_nodes: Dict[str, str] = {}
        self.router: Callable = None
        self._debug_mode = debug_mode
    
    def add_node(self, name: str, node: Callable) -> None:
        """Add a node to the workflow."""
        self.nodes[name] = node
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between nodes."""
        self.edges[from_node] = to_node
    
    def set_entry_points(self, router_func: Callable, entry_mapping: Dict[str, str]) -> None:
        """Set the workflow entry points with routing logic."""
        self.router = router_func
        self.entry_nodes = entry_mapping
    
    def _update_state(self, state: IWorkflowState, updates: Dict[str, Any]) -> None:
        """Update state with node result while preserving model types."""
        ...
    
    async def execute(self, input_data: Dict[str, Any], callbacks: Dict[str, Any], is_streaming: bool = False) -> Dict[str, Any]:
        """
        Execute the workflow with given input.
        
        Args:
            input_data: Dictionary containing the input data including state
            
        Returns:
            Dictionary with final state and workflow results
        """
        ...


class IWorkflowConfig(Protocol):
    """Configuration for workflow orchestration.
    
    This interface defines how workflows are configured, including their structure
    (nodes and edges) and routing logic.
    """
    
    def __init__(
        self, 
        settings: Any,
        debug_mode: bool = False,
        **kwargs: Any
    ):
        """Initialize workflow configuration."""
        self.settings = settings
        self.debug_mode = debug_mode
        self._kwargs = kwargs
    
    def create_workflow(self) -> IWorkflowOrchestrator:
        """Create and configure the workflow."""
        ...
    
    def _configure_workflow(self, workflow: IWorkflowOrchestrator) -> None:
        """Configure the workflow with nodes, edges, and entry points."""
        ...
    
    def _route_input(self, input_data: Dict[str, Any]) -> str:
        """Route to appropriate node based on input."""
        ...
        
    def _create_nodes(self) -> Dict[str, INode]:
        """Create all nodes for the workflow."""
        ...
        
    def _define_edges(self) -> Dict[str, str]:
        """Define the edges between nodes."""
        ...    