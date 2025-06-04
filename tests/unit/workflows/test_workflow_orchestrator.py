"""Unit tests for workflow orchestrator."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
import sys

from arshai.core.interfaces import IWorkflowState
from arshai.workflows.workflow_orchestrator import BaseWorkflowOrchestrator


# Don't try to inherit from IWorkflowState directly, use MagicMock for testing
@pytest.fixture
def mock_state():
    """Create a mock workflow state for testing."""
    state = MagicMock(spec=IWorkflowState)
    
    # Create a special mock for errors that we can track
    errors_mock = MagicMock()
    errors_mock.__len__.return_value = 0
    errors_mock.__getitem__.side_effect = IndexError
    errors_mock.append = MagicMock()
    
    state.errors = errors_mock
    state.agent_data = {}
    state.step_count = 0
    state.processing_path = None
    state.current_step = None
    state.user_context = MagicMock()
    state.user_context.user_id = "test-user-123"
    
    # Mock __deepcopy__ to return a new mock with the same attributes
    def mock_deepcopy(memo=None):
        new_state = MagicMock(spec=IWorkflowState)
        new_state.errors = errors_mock  # Share the same errors mock for tracking
        new_state.agent_data = state.agent_data.copy()
        new_state.step_count = state.step_count
        new_state.processing_path = state.processing_path
        new_state.current_step = state.current_step
        new_state.user_context = state.user_context
        return new_state
    
    state.__deepcopy__ = mock_deepcopy
    return state


@pytest.fixture
def workflow_orchestrator():
    """Create a basic workflow orchestrator for testing."""
    return BaseWorkflowOrchestrator(debug_mode=True)


@pytest.fixture
def mock_nodes(mock_state):
    """Create mock nodes for testing."""
    # Create three nodes: start, middle, end
    start_node = AsyncMock()
    start_node.return_value = {
        "state": mock_state,
        "result": "Start node result"
    }
    
    middle_node = AsyncMock()
    middle_node.return_value = {
        "state": mock_state,
        "result": "Middle node result"
    }
    
    end_node = AsyncMock()
    end_node.return_value = {
        "state": mock_state,
        "result": "End node result"
    }
    
    return {
        "start": start_node,
        "middle": middle_node,
        "end": end_node
    }


@pytest.fixture
def configured_workflow(workflow_orchestrator, mock_nodes):
    """Create a configured workflow with nodes and edges."""
    # Add nodes
    for name, node in mock_nodes.items():
        workflow_orchestrator.add_node(name, node)
    
    # Add edges
    workflow_orchestrator.add_edge("start", "middle")
    workflow_orchestrator.add_edge("middle", "end")
    
    # Set router and entry points
    def router(input_data):
        return input_data.get("route_key", "default")
    
    workflow_orchestrator.set_entry_points(
        router_func=router,
        entry_mapping={
            "default": "start",
            "alt": "middle"
        }
    )
    
    return workflow_orchestrator


def test_add_node(workflow_orchestrator):
    """Test adding a node to the workflow."""
    node = AsyncMock()
    workflow_orchestrator.add_node("test_node", node)
    
    assert "test_node" in workflow_orchestrator.nodes
    assert workflow_orchestrator.nodes["test_node"] == node


def test_add_edge(workflow_orchestrator):
    """Test adding an edge between nodes."""
    workflow_orchestrator.add_edge("node_a", "node_b")
    
    assert "node_a" in workflow_orchestrator.edges
    assert workflow_orchestrator.edges["node_a"] == "node_b"


def test_set_entry_points(workflow_orchestrator):
    """Test setting entry points and router function."""
    router_func = lambda x: x.get("type", "default")
    entry_mapping = {"default": "start_node", "special": "special_node"}
    
    workflow_orchestrator.set_entry_points(router_func, entry_mapping)
    
    assert workflow_orchestrator.router == router_func
    assert workflow_orchestrator.entry_nodes == entry_mapping


@pytest.mark.asyncio
async def test_execute_basic_flow(configured_workflow, mock_state):
    """Test executing a basic workflow with default entry point."""
    input_data = {
        "state": mock_state,
        "message": "Test input"
    }
    
    result = await configured_workflow.execute(input_data)
    
    # Verify each node was called in sequence
    configured_workflow.nodes["start"].assert_called_once()
    configured_workflow.nodes["middle"].assert_called_once()
    configured_workflow.nodes["end"].assert_called_once()
    
    # Verify state is in result
    assert "state" in result


@pytest.mark.asyncio
async def test_execute_alternative_entry(configured_workflow, mock_state):
    """Test executing workflow with alternative entry point."""
    input_data = {
        "state": mock_state,
        "message": "Test input",
        "route_key": "alt"  # Use alternative entry point
    }
    
    result = await configured_workflow.execute(input_data)
    
    # Verify only middle and end nodes were called (skipping start)
    configured_workflow.nodes["start"].assert_not_called()
    configured_workflow.nodes["middle"].assert_called_once()
    configured_workflow.nodes["end"].assert_called_once()


@pytest.mark.asyncio
async def test_execute_explicit_routing(configured_workflow, mock_state):
    """Test executing workflow with explicit node routing."""
    # Modify middle node to provide explicit routing to end node
    middle_node = configured_workflow.nodes["middle"]
    middle_node.return_value = {
        "state": mock_state,
        "result": "Middle node result",
        "route": "end"  # Explicit routing
    }
    
    input_data = {
        "state": mock_state,
        "message": "Test input"
    }
    
    result = await configured_workflow.execute(input_data)
    
    # Verify each node was called in expected sequence
    configured_workflow.nodes["start"].assert_called_once()
    configured_workflow.nodes["middle"].assert_called_once()
    configured_workflow.nodes["end"].assert_called_once()


@pytest.mark.asyncio
async def test_execute_node_error(configured_workflow, mock_state):
    """Test workflow handling of node errors."""
    # Reset any previous calls
    mock_state.errors.append.reset_mock()
    
    # Make middle node raise an exception
    middle_node = configured_workflow.nodes["middle"]
    middle_node.side_effect = Exception("Test error")
    
    input_data = {
        "state": mock_state,
        "message": "Test input"
    }
    
    result = await configured_workflow.execute(input_data)
    
    # Verify first node was called but execution stopped at error
    configured_workflow.nodes["start"].assert_called_once()
    configured_workflow.nodes["middle"].assert_called_once()
    configured_workflow.nodes["end"].assert_not_called()
    
    # Verify error was recorded
    mock_state.errors.append.assert_called()
    # Get the first call arguments
    error_info = mock_state.errors.append.call_args[0][0]
    assert error_info["error"] == "Test error"
    assert error_info["step"] == "middle"


@pytest.mark.asyncio
async def test_execute_missing_state(configured_workflow):
    """Test that execute raises error when state is missing."""
    input_data = {
        "message": "Test message"  # No state provided
    }
    
    # Should raise ValueError about missing state
    with pytest.raises(ValueError, match="State must be provided in input data"):
        await configured_workflow.execute(input_data)


@pytest.mark.asyncio
async def test_execute_invalid_router(workflow_orchestrator, mock_state):
    """Test that execute handles missing router function."""
    # Reset any previous calls
    mock_state.errors.append.reset_mock()
    
    # Configure a workflow without setting router
    node = AsyncMock()
    workflow_orchestrator.add_node("test_node", node)
    
    input_data = {
        "state": mock_state,
        "message": "Test message"
    }
    
    # Execute should handle the missing router error
    result = await workflow_orchestrator.execute(input_data)
    
    # Verify error was recorded
    mock_state.errors.append.assert_called()
    # Get the first call arguments
    error_info = mock_state.errors.append.call_args[0][0]
    assert "No router function set" in error_info["error"]


@pytest.mark.asyncio
async def test_execute_invalid_entry_node(configured_workflow, mock_state):
    """Test workflow handling of invalid entry node."""
    # Reset any previous calls
    mock_state.errors.append.reset_mock()
    
    input_data = {
        "state": mock_state,
        "message": "Test message",
        "route_key": "invalid"  # This route key doesn't exist
    }
    
    # Execute should handle the invalid entry error
    result = await configured_workflow.execute(input_data)
    
    # Verify error was recorded
    mock_state.errors.append.assert_called()
    # Get the first call arguments
    error_info = mock_state.errors.append.call_args[0][0]
    assert "Invalid entry node" in error_info["error"]
    
    # No nodes should have been called
    configured_workflow.nodes["start"].assert_not_called()
    configured_workflow.nodes["middle"].assert_not_called()
    configured_workflow.nodes["end"].assert_not_called() 