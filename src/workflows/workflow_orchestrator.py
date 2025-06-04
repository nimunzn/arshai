from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from copy import deepcopy

from arshai.core.interfaces import IWorkflowOrchestrator, IWorkflowState
from arshai.utils import get_logger

class BaseWorkflowOrchestrator(IWorkflowOrchestrator):
    """Base implementation of workflow orchestrator.
    
    This implementation follows the pattern from the previous project where:
    - Nodes are callable classes that operate on workflow state
    - State is passed between nodes and updated in each step
    - The orchestrator manages the flow between nodes using edges
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize workflow orchestrator."""
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.entry_nodes: Dict[str, str] = {}
        self.router: Optional[Callable] = None
        self._debug_mode = debug_mode
        self._logger = get_logger(__name__)

    def add_node(self, name: str, node: Callable) -> None:
        """Add a node to the workflow."""
        if self._debug_mode:
            self._logger.debug(f"Adding node: {name}")
        self.nodes[name] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between nodes."""
        if self._debug_mode:
            self._logger.debug(f"Adding edge: {from_node} -> {to_node}")
        self.edges[from_node] = to_node

    def set_entry_points(self, router_func: Callable, entry_mapping: Dict[str, str]) -> None:
        """Set the workflow entry points with routing logic."""
        if self._debug_mode:
            self._logger.debug(f"Setting entry points with mapping: {entry_mapping}")
        self.router = router_func
        self.entry_nodes = entry_mapping

    def _update_state(self, state: IWorkflowState, updates: Dict[str, Any]) -> None:
        """Update state with node result while preserving model types."""
        if not updates:
            return
        
        # If there's no state key in updates, nothing to do
        if "state" not in updates:
            return
            
        # Get the updated state from the node result
        updated_state = updates["state"]
        
        # If it's a WorkflowState object, copy its attributes to our state
        if isinstance(updated_state, IWorkflowState):
            # Copy all attributes from updated_state to state
            for key, value in vars(updated_state).items():
                if not key.startswith("_"):  # Skip private attributes
                    setattr(state, key, value)

    async def execute(self, input_data: Dict[str, Any], callbacks: Optional[Dict[str, Any]] = None, is_streaming: bool = False) -> Dict[str, Any]:
        """Execute the workflow with given input data.
        
        This method:
        1. Routes to the appropriate entry node based on input data
        2. Executes each node in sequence based on edge connections
        3. Passes the workflow state between nodes
        4. Returns the final state and results
        
        Args:
            input_data: Dictionary containing the input data including state
            callbacks: Optional callback functions for nodes to use
            
        Returns:
            Dictionary with final state and workflow results
        """
        if "state" not in input_data:
            raise ValueError("State must be provided in input data")
        
        # Create a copy of the input data to avoid modifying the original
        input_data = deepcopy(input_data)
        state: IWorkflowState = input_data["state"]
        current_node = None
        
        try:
            # Debug logging before routing
            if self._debug_mode:
                self._logger.debug("Starting workflow execution")
                self._logger.debug(f"Input data keys: {list(input_data.keys())}")

            # Route to initial node
            if not self.router:
                raise ValueError("No router function set")
            
            route_key = self.router(input_data)
            if route_key not in self.entry_nodes:
                raise ValueError(f"Invalid entry node: {route_key}")
            
            current_node = self.entry_nodes[route_key]
            
            # Set processing path based on entry node
            state.processing_path = route_key
            state.current_step = "start"
            
            # Execute workflow by moving from node to node
            while current_node:
                try:
                    # Update current step in state
                    state.current_step = current_node
                    
                    # Debug logging before node execution
                    if self._debug_mode:
                        self._logger.debug(f"Executing node: {current_node}")
                        self._logger.debug(f"Current state step: {state.current_step}")

                    # Ensure state is in the input data
                    input_data["state"] = state
                    
                    # Execute current node
                    node_func = self.nodes[current_node]
                    node_result = await node_func(input_data, callbacks, is_streaming)
                    
                    # Debug logging after node execution
                    if self._debug_mode:
                        self._logger.debug(f"Node {current_node} execution completed")
                        self._logger.debug(f"Node result keys: {list(node_result.keys())}")

                    # Update state with node result
                    if "state" in node_result:
                        state = node_result["state"]
                    else:
                        self._logger.warning(f"Node {current_node} did not return state")
                    
                    # Update step information
                    state.step_count += 1
                    
                    # Add any routing information from the node to the input_data
                    # This allows nodes to influence the workflow path
                    for key, value in node_result.items():
                        if key != "state":
                            input_data[key] = value
                    
                    # Update input data with the latest state
                    input_data["state"] = state
                    
                    # Get next node based on routing information if provided
                    next_node = None
                    if "route" in node_result:
                        # If the node provided explicit routing
                        route_value = node_result["route"]
                        # Check if this is a valid node name
                        if route_value in self.nodes:
                            next_node = route_value
                    
                    # If no explicit route was provided, use the edge mapping
                    if next_node is None:
                        next_node = self.edges.get(current_node)
                    
                    current_node = next_node
                    
                except Exception as e:
                    # Debug logging on node error
                    if self._debug_mode:
                        self._logger.error(f"Error in node {current_node}: {str(e)}")

                    # Update state with error and break workflow
                    state.errors.append({
                        "step": current_node,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    break
            
        except Exception as e:
            # Debug logging on routing error
            if self._debug_mode:
                self._logger.error(f"Error in workflow routing: {str(e)}")

            # Handle routing errors
            state.errors.append({
                "step": "routing",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Update final state in input data
        input_data["state"] = state
        
        # Prepare final result with state and any results
        final_result = {"state": state}
        
        # Add any other keys from input_data that might be useful
        for key, value in input_data.items():
            if key != "state" and not key.startswith("_"):
                final_result[key] = value
        
        return final_result 