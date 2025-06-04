from typing import Dict, Any, Optional, Callable
from arshai.core.interfaces.iworkflow import INode, IWorkflowState
from arshai.core.interfaces.iagent import IAgent, IAgentInput, IAgentConfig
from arshai.core.interfaces.isetting import ISetting
from arshai.utils import get_logger

class BaseNode(INode):
    """Base implementation of the INode interface.
    
    Wraps an agent and provides the node interface for workflow integration.
    This implementation follows the pattern where nodes directly operate on workflow state.
    """
    
    def __init__(
        self,
        node_id: str,
        name: str,
        agent: IAgent,
        settings: Optional[ISetting] = None,
        **kwargs: Any
    ):
        """Initialize a node with an agent.
        
        Args:
            node_id: Unique identifier for this node
            name: Descriptive name for this node
            agent: The agent implementation this node will use
            settings: Optional settings to override agent defaults
            **kwargs: Additional node-specific settings
        """
        self._node_id = node_id
        self._name = name
        self._agent = agent
        self._settings = settings
        self._kwargs = kwargs
        self._logger = get_logger(__name__)
        
    def get_id(self) -> str:
        """Get the node's unique identifier."""
        return self._node_id
    
    def get_name(self) -> str:
        """Get the node's name."""
        return self._name
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data using the underlying agent.
        
        Following the workflow state pattern, this method:
        1. Extracts the workflow state from input data
        2. Processes the input using the agent
        3. Updates the state with the results
        4. Returns the updated state and results
        
        Args:
            input_data: Dictionary containing the input data including state
            
        Returns:
            Dictionary with updated workflow state and processing results
        """
        self._logger.debug(f"Node {self._name} processing input data")
        
        # Extract the workflow state from input data
        if "state" not in input_data:
            self._logger.error("No workflow state found in input data")
            raise ValueError("Input data must contain 'state'")
            
        state = input_data["state"]
        if not isinstance(state, IWorkflowState):
            self._logger.error(f"Invalid state type: {type(state)}")
            raise ValueError("State must be an instance of IWorkflowState")
        
        try:
            # Extract inputs required for the agent
            # - If message is directly in input_data, use it
            # - Otherwise look for query
            message = input_data.get("message", input_data.get("query", ""))
            
            # Create agent input using the user_id from state as conversation_id
            agent_input = IAgentInput(
                message=message,
                conversation_id=state.user_context.user_id,
                stream=False
            )
            
            # Update state with current step info
            state.current_step = self._name
            
            # Process with agent
            self._logger.debug(f"Calling agent {self._agent.__class__.__name__} with message: {message[:50]}...")
            agent_response = self._agent.process_message(agent_input)
            
            # Initialize agent_data dictionary if needed
            if not state.agent_data:
                state.agent_data = {}
                
            # Update state with agent response
            state.agent_data[self._node_id] = agent_response
            
            # Create result dictionary with the updated state and any additional outputs
            result = {
                "state": state,
                "result": agent_response,
                "node_id": self._node_id,
                "agent_data": state.agent_data  # This provides direct access to all agent data
            }
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error in node processing: {str(e)}")
            
            # Record error in state
            state.errors.append({
                "node": self._node_id,
                "error": str(e),
                "step": state.current_step
            })
            
            # Return the state with error information
            return {
                "state": state,
                "error": str(e),
                "node_id": self._node_id
            }
    
    def get_agent_settings(self) -> Dict[str, Any]:
        """Get the settings associated with the agent for this node."""
        if self._settings:
            return self._settings.model_dump()
        return {}
        
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make the node callable, directly delegating to the process method.
        
        This allows nodes to be called as functions, matching the pattern from the previous project.
        
        Args:
            input_data: Dictionary containing the input data including state
            
        Returns:
            Dictionary with processing results
        """
        return self.process(input_data) 