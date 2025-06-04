"""Unit tests for workflow nodes."""

import pytest
from unittest.mock import MagicMock, patch

from arshai.core.interfaces import IWorkflowState
from arshai.core.interfaces import IAgent, IAgentInput
from arshai.core.interfaces import ISetting
from arshai.workflows.node import BaseNode


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=IAgent)
    agent.process_message.return_value = {
        "content": "Agent response",
        "metadata": {"tokens": 50}
    }
    return agent


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=ISetting)
    settings.model_dump.return_value = {"temperature": 0.7, "model": "gpt-4"}
    return settings


@pytest.fixture
def mock_workflow_state():
    """Create a mock workflow state for testing."""
    state = MagicMock(spec=IWorkflowState)
    state.user_context = MagicMock()
    state.user_context.user_id = "test-user-123"
    state.current_step = None
    state.agent_data = {}
    state.errors = []
    return state


@pytest.fixture
def test_node(mock_agent, mock_settings):
    """Create a test node for testing."""
    return BaseNode(
        node_id="test-node-123",
        name="Test Node",
        agent=mock_agent,
        settings=mock_settings
    )


def test_node_initialization(mock_agent, mock_settings):
    """Test node initialization with correct parameters."""
    node = BaseNode(
        node_id="test-node-123",
        name="Test Node",
        agent=mock_agent,
        settings=mock_settings,
        extra_param="test"
    )
    
    assert node.get_id() == "test-node-123"
    assert node.get_name() == "Test Node"
    assert node._agent == mock_agent
    assert node._settings == mock_settings
    assert node._kwargs == {"extra_param": "test"}


@pytest.mark.asyncio
async def test_node_process(test_node, mock_workflow_state, mock_agent):
    """Test the node's process method with valid input."""
    # Create input data with state
    input_data = {
        "state": mock_workflow_state,
        "message": "Test message"
    }
    
    # Process the input
    result = await test_node.process(input_data)
    
    # Verify agent was called with correct input
    agent_input = mock_agent.process_message.call_args[0][0]
    assert isinstance(agent_input, IAgentInput)
    assert agent_input.message == "Test message"
    assert agent_input.conversation_id == "test-user-123"
    assert agent_input.stream is False
    
    # Verify state was updated correctly
    assert mock_workflow_state.current_step == "Test Node"
    assert "test-node-123" in mock_workflow_state.agent_data
    assert mock_workflow_state.agent_data["test-node-123"] == {
        "content": "Agent response",
        "metadata": {"tokens": 50}
    }
    
    # Verify the result structure
    assert result["state"] == mock_workflow_state
    assert result["result"] == {
        "content": "Agent response",
        "metadata": {"tokens": 50}
    }
    assert result["node_id"] == "test-node-123"
    assert result["agent_data"] == mock_workflow_state.agent_data


@pytest.mark.asyncio
async def test_node_process_with_query(test_node, mock_workflow_state, mock_agent):
    """Test the node's process method with query instead of message."""
    # Create input data with query instead of message
    input_data = {
        "state": mock_workflow_state,
        "query": "Test query"
    }
    
    # Process the input
    result = await test_node.process(input_data)
    
    # Verify agent was called with query as message
    agent_input = mock_agent.process_message.call_args[0][0]
    assert agent_input.message == "Test query"


@pytest.mark.asyncio
async def test_node_process_missing_state(test_node):
    """Test that process raises error when state is missing."""
    # Create input data without state
    input_data = {
        "message": "Test message"
    }
    
    # Verify it raises ValueError
    with pytest.raises(ValueError, match="Input data must contain 'state'"):
        await test_node.process(input_data)


@pytest.mark.asyncio
async def test_node_process_invalid_state(test_node):
    """Test that process raises error when state is invalid."""
    # Create input data with invalid state
    input_data = {
        "state": {"not": "a workflow state"},
        "message": "Test message"
    }
    
    # Verify it raises ValueError
    with pytest.raises(ValueError, match="State must be an instance of IWorkflowState"):
        await test_node.process(input_data)


@pytest.mark.asyncio
async def test_node_process_agent_exception(test_node, mock_workflow_state, mock_agent):
    """Test that process handles agent exceptions properly."""
    # Make the agent raise an exception
    mock_agent.process_message.side_effect = Exception("Agent error")
    
    # Create input data
    input_data = {
        "state": mock_workflow_state,
        "message": "Test message"
    }
    
    # Process should not raise but should return error in result
    result = await test_node.process(input_data)
    
    # Verify error is recorded in state
    assert len(mock_workflow_state.errors) == 1
    assert mock_workflow_state.errors[0]["node"] == "test-node-123"
    assert mock_workflow_state.errors[0]["error"] == "Agent error"
    
    # Verify result contains error
    assert result["state"] == mock_workflow_state
    assert result["error"] == "Agent error"
    assert result["node_id"] == "test-node-123"


def test_node_get_agent_settings(test_node, mock_settings):
    """Test retrieving agent settings."""
    settings = test_node.get_agent_settings()
    
    # Verify settings are returned
    assert settings == {"temperature": 0.7, "model": "gpt-4"}
    mock_settings.model_dump.assert_called_once()


def test_node_get_agent_settings_no_settings():
    """Test retrieving agent settings when no settings were provided."""
    agent = MagicMock(spec=IAgent)
    node = BaseNode(
        node_id="test-node-123",
        name="Test Node",
        agent=agent
    )
    
    settings = node.get_agent_settings()
    assert settings == {} 