from typing import Protocol, Dict, Any, Optional
from datetime import datetime
from .iworkflow import IWorkflowState, IWorkflowConfig

class IWorkflowRunner(Protocol):
    """Interface for workflow runners.
    
    Workflow runners are responsible for initializing workflow state,
    executing workflows, and handling the results. They act as the 
    bridge between the application and the workflow orchestrator.
    """
    
    def __init__(
        self,
        workflow_config: IWorkflowConfig,
        debug_mode: bool = False,
        **kwargs: Any
    ):
        """Initialize the workflow runner.
        
        Args:
            workflow_config: Configuration for the workflow
            debug_mode: Whether to enable debug mode
            **kwargs: Additional configuration options
        """
        self.workflow_config = workflow_config
        self.workflow = workflow_config.create_workflow()
        self.debug_mode = debug_mode
        self.last_state = None
    
    async def _initialize_state(self, user_id: str) -> IWorkflowState:
        """Initialize workflow state for a user.
        
        Args:
            user_id: The ID of the user for whom to initialize state
            
        Returns:
            A new workflow state instance
        """
        ...
    
    async def execute_workflow(
        self, 
        user_id: str, 
        input_data: Dict[str, Any],
        callbacks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow with the given input.
        
        This method:
        1. Initializes workflow state if not provided
        2. Prepares input data with state and callbacks
        3. Executes the workflow
        4. Processes and returns the results
        
        Args:
            user_id: The ID of the user initiating the workflow
            input_data: Input data for the workflow
            callbacks: Optional callback functions to be called during workflow execution
            
        Returns:
            Dict with workflow execution results including state
        """
        ...
    
    async def _handle_workflow_error(
        self,
        error: Exception,
        user_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle errors that occur during workflow execution.
        
        Args:
            error: The exception that was raised
            user_id: The ID of the user who initiated the workflow
            input_data: The original input data for the workflow
            
        Returns:
            Dict with error information in a standardized format
        """
        ...
    
    async def _send_notification(
        self,
        user_id: str,
        notification_type: str,
        content: Dict[str, Any],
        state: IWorkflowState
    ) -> None:
        """
        Send notification to the user during workflow execution.
        
        Args:
            user_id: The ID of the user to send the notification to
            notification_type: The type of notification to send
            content: The content of the notification
            state: The current workflow state
        """
        ...
    
    async def cleanup(self) -> None:
        """
        Clean up any resources used by the workflow runner.
        """
        ...
    