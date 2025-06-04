from typing import Dict, Any, List, Type, Optional
from arshai.core.interfaces import IWorkflowConfig, IWorkflowOrchestrator, INode
from arshai.core.interfaces import ISetting
from arshai.workflows.workflow_orchestrator import BaseWorkflowOrchestrator
from arshai.utils import get_logger

class WorkflowConfig(IWorkflowConfig):
    """Base implementation of workflow configuration.
    
    This implementation follows the pattern from the previous project where:
    - The config creates and configures the workflow orchestrator
    - The config defines the workflow structure (nodes and edges)
    - The config provides routing logic for input
    """
    
    def __init__(
        self,
        settings: ISetting,
        debug_mode: bool = False,
        **kwargs: Any
    ):
        """Initialize workflow configuration.
        
        Args:
            settings: Application settings to be used for nodes
            debug_mode: Whether to enable debug mode for verbose logging
            **kwargs: Additional configuration options
        """
        self.settings = settings
        self.debug_mode = debug_mode
        self._kwargs = kwargs
        self._logger = get_logger(__name__)
        
        # Nodes and edges will be created in _configure_workflow
        self.nodes: Dict[str, INode] = {}
        self.edges: Dict[str, str] = {}
    
    def create_workflow(self) -> IWorkflowOrchestrator:
        """Create and configure the workflow orchestrator.
        
        This method:
        1. Creates a new workflow orchestrator
        2. Configures it with nodes, edges, and entry points
        3. Returns the configured orchestrator
        
        Returns:
            Configured workflow orchestrator ready for execution
        """
        self._logger.debug("Creating workflow orchestrator")
        
        # Create the workflow orchestrator
        workflow = BaseWorkflowOrchestrator(debug_mode=self.debug_mode)
        
        # Configure it with nodes, edges, and entry points
        self._configure_workflow(workflow)
        
        return workflow
    
    def _configure_workflow(self, workflow: IWorkflowOrchestrator) -> None:
        """Configure the workflow with nodes, edges, and entry points.
        
        This method must be implemented by subclasses to define:
        1. What nodes the workflow contains
        2. How nodes are connected with edges
        3. Entry points and routing logic
        
        Args:
            workflow: The workflow orchestrator to configure
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _configure_workflow")
    
    def _route_input(self, input_data: Dict[str, Any]) -> str:
        """Route to appropriate entry node based on input.
        
        This method must be implemented by subclasses to define the routing logic
        that determines which entry node to start with based on the input data.
        
        Args:
            input_data: The input data to route
            
        Returns:
            The name of the entry node to start with
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _route_input")
    
    def _create_nodes(self) -> Dict[str, INode]:
        """Create all nodes for the workflow.
        
        This method must be implemented by subclasses to create all the nodes
        that will be used in the workflow.
        
        Returns:
            Dictionary mapping node names to node instances
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _create_nodes")
    
    def _define_edges(self) -> Dict[str, str]:
        """Define the edges between nodes.
        
        This method must be implemented by subclasses to define the edges
        that connect nodes in the workflow.
        
        Returns:
            Dictionary mapping source node names to destination node names
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _define_edges") 