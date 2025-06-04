"""Unit tests for workflow configuration."""

import pytest
from unittest.mock import MagicMock, patch

from arshai.core.interfaces import ISetting
from arshai.core.interfaces import IWorkflowOrchestrator, INode
from arshai.workflows.workflow_config import WorkflowConfig
from arshai.workflows.workflow_orchestrator import BaseWorkflowOrchestrator


class TestWorkflowConfig(WorkflowConfig):
    """Test implementation of WorkflowConfig for testing."""
    
    def __init__(self, settings, debug_mode=False, **kwargs):
        super().__init__(settings, debug_mode, **kwargs)
        self.test_nodes = {}
        self.test_edges = {}
        
    def _configure_workflow(self, workflow):
        """Test implementation to configure workflow."""
        # Add nodes to the workflow
        for node_name, node in self.test_nodes.items():
            workflow.add_node(node_name, node)
        
        # Add edges
        for source, target in self.test_edges.items():
            workflow.add_edge(source, target)
    
    def _route_input(self, input_data):
        """Test implementation to route input."""
        return list(self.test_nodes.keys())[0] if self.test_nodes else "default_entry"
    
    def _create_nodes(self):
        """Test implementation to create nodes."""
        return self.test_nodes
    
    def _define_edges(self):
        """Test implementation to define edges."""
        return self.test_edges


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return MagicMock(spec=ISetting)


@pytest.fixture
def test_node():
    """Create a test node for the workflow."""
    node = MagicMock(spec=INode)
    node.process.return_value = {"output": "test_result"}
    return node


@pytest.fixture
def workflow_config(mock_settings, test_node):
    """Create a test workflow configuration."""
    config = TestWorkflowConfig(mock_settings, debug_mode=True)
    config.test_nodes = {"test_node": test_node}
    config.test_edges = {"test_node": "output_node"}
    return config


def test_workflow_config_initialization(mock_settings):
    """Test workflow config initialization."""
    config = TestWorkflowConfig(mock_settings, debug_mode=True, extra_param="value")
    
    assert config.settings == mock_settings
    assert config.debug_mode is True
    assert config._kwargs == {"extra_param": "value"}
    assert isinstance(config.nodes, dict)
    assert isinstance(config.edges, dict)


def test_create_workflow(workflow_config):
    """Test creating a workflow from config."""
    # Test creating a workflow
    workflow = workflow_config.create_workflow()
    
    # Verify it's the right type
    assert isinstance(workflow, BaseWorkflowOrchestrator)
    
    # Verify configuration was applied
    assert "test_node" in workflow.nodes
    assert workflow.edges == {"test_node": "output_node"}


def test_base_workflow_abstract_methods():
    """Test that the base class requires abstract methods to be implemented."""
    mock_settings = MagicMock(spec=ISetting)
    base_config = WorkflowConfig(mock_settings)
    
    # Abstract methods should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        base_config._configure_workflow(MagicMock())
    
    with pytest.raises(NotImplementedError):
        base_config._route_input({})
    
    with pytest.raises(NotImplementedError):
        base_config._create_nodes()
    
    with pytest.raises(NotImplementedError):
        base_config._define_edges() 