from typing import Dict, Any, Optional
import logging
from datetime import datetime
import traceback

from arshai.core.interfaces import IWorkflowState, IUserContext, IWorkflowConfig
from arshai.core.interfaces import IWorkflowRunner
from arshai.core.interfaces import INotificationState
from arshai.utils import get_logger

class BaseWorkflowRunner(IWorkflowRunner):
    """Base implementation of workflow runner with common functionality.
    
    This implementation follows the pattern from the previous project where:
    - The runner initializes and manages workflow state
    - The runner delegates to the workflow for execution
    - The runner handles errors and returns standardized responses
    """
    
    def __init__(
        self,
        workflow_config: IWorkflowConfig,
        debug_mode: bool = False,
        **kwargs: Any
    ):
        """Initialize the workflow runner with config and settings."""
        self.workflow_config = workflow_config
        self.workflow = workflow_config.create_workflow()
        self.debug_mode = debug_mode
        self._logger = get_logger(__name__)
        self._kwargs = kwargs
        self.last_state = None  # Store the last workflow state for reference
    
    async def _initialize_state(self, user_id: str) -> IWorkflowState:
        """Initialize workflow state for a user.
        
        Creates a new workflow state with default values for a user.
        
        Args:
            user_id: The ID of the user for whom to initialize state
            
        Returns:
            A new workflow state instance
        """
        self._logger.info(f"Initializing workflow state for user {user_id}")
        
        # Create user context
        user_context = IUserContext(
            user_id=user_id,
            last_active=datetime.utcnow(),
            interaction_count=0
        )
        
        # Create notification state
        notification_state = INotificationState()
        
        # Create workflow state
        state = IWorkflowState(
            user_context=user_context,
            notification_state=notification_state,
            current_step="start",
            step_count=0,
            errors=[],
            processing_path="none",
            agent_data={},
            working_memories={}
        )
        
        return state
    
    async def execute_workflow(
        self, 
        user_id: str, 
        input_data: Dict[str, Any],
        callbacks: Optional[Dict[str, Any]] = None,
        is_streaming: bool = False
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
        start_time = datetime.now()
        self._logger.info(f"Executing workflow for user {user_id}")
        self._logger.info(f"Input data: {input_data}")
        
        try:
            # Initialize state if not provided in input_data
            if "state" not in input_data:
                self._logger.debug(f"State not provided in input_data, initializing new state for user {user_id}")
                try:
                    state = await self._initialize_state(user_id)
                    self._logger.debug(f"New state initialized: {state}")
                    input_data["state"] = state
                except Exception as e:
                    self._logger.error(f"Error initializing state: {str(e)}")
                    self._logger.error(f"Detailed traceback: {traceback.format_exc()}")
                    raise
            else:
                # Use the provided state
                state = input_data["state"]
                self._logger.debug(f"Using provided state: {state}")
                
                # Ensure user_id matches
                if state.user_context.user_id != user_id:
                    self._logger.warning(f"User ID mismatch in state. Expected {user_id}, got {state.user_context.user_id}")
                    state.user_context.user_id = user_id
            
            # Prepare workflow input without callbacks
            workflow_input = {**input_data}
            
            # Execute workflow, passing callbacks separately
            self._logger.debug(f"Executing workflow with input data keys: {workflow_input.keys()}")
            try:
                result = await self.workflow.execute(workflow_input, callbacks, is_streaming)
            except Exception as e:
                self._logger.error(f"Exception during workflow execution: {str(e)}")
                self._logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
            
            # Extract final state from result
            final_state = result.get("state")
            
            # Store the final state for reference
            if final_state and isinstance(final_state, IWorkflowState):
                self.last_state = final_state
            
            # Check for errors
            if final_state and final_state.errors:
                error_details = []
                for error in final_state.errors:
                    error_details.append(f"Step: {error.get('step', 'unknown')}, Error: {error.get('error', 'No error message')}")
                
                error_str = "\n".join(error_details)
                self._logger.warning(f"Workflow completed with errors:\n{error_str}\nTraceback (if available):\n{traceback.format_exc()}")
                return {
                    "success": False,
                    "errors": final_state.errors,
                    "state": final_state,
                    "result": result
                }
            
            # Log completion
            execution_time = (datetime.now() - start_time).total_seconds()
            self._logger.info(f"Workflow executed successfully in {execution_time:.2f}s")
            
            # Return success response with state and results
            return {
                "success": True,
                "state": final_state,
                "result": result
            }
            
        except Exception as e:
            # Log error and return error response
            self._logger.error(f"Error executing workflow: {str(e)}")
            self._logger.info(traceback.format_exc())
            
            return await self._handle_workflow_error(e, user_id, input_data)
    
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
        self._logger.error(f"Workflow error for user {user_id}: {str(error)}")
        
        # Get state if available
        state = input_data.get("state")
        
        # If we have a state, add the error to it
        if state and isinstance(state, IWorkflowState):
            state.errors.append({
                "step": state.current_step,
                "error": str(error),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Create error response
        error_response = {
            "success": False,
            "error": str(error),
            "error_type": error.__class__.__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add state if available
        if state:
            error_response["state"] = state
        
        # Add original input for debugging (excluding sensitive data)
        debug_input = {k: v for k, v in input_data.items() if k not in ["state", "credentials", "password", "token"]}
        error_response["debug_info"] = {
            "input": debug_input
        }
        
        return error_response
    
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
        try:
            # Create notification payload
            timestamp = datetime.utcnow().isoformat()
            notification = {
                "type": notification_type,
                "content": content,
                "timestamp": timestamp
            }
            
            # Add to notifications list
            state.notification_state.notifications.append(notification)
            
            # Record the notification attempt using the proper method
            state.notification_state.record_attempt(
                notification=notification,
                had_active_connection=True,
                error=None
            )
            
            self._logger.info(f"Notification sent to user {user_id}: {notification_type}")
            
        except Exception as e:
            self._logger.error(f"Error sending notification: {str(e)}")
            
            # Record failed attempt if we have state and notification data
            if state and 'notification' in locals():
                state.notification_state.record_attempt(
                    notification=notification,
                    had_active_connection=False,
                    error=str(e)
                )
    
    async def cleanup(self) -> None:
        """
        Clean up any resources used by the workflow runner.
        """
        self._logger.info("Cleaning up workflow runner resources")
        # Reset the last_state when cleaning up
        self.last_state = None 