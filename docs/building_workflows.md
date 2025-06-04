# Building Workflows

Workflows are a key feature of the Arshai framework, allowing you to create complex, multi-agent systems that work together to solve problems. This document explains how to build and use workflows in your applications.

## Workflow Architecture

Arshai workflows are based on a directed graph model where:

- **Nodes** are the processing units, typically wrapping around agents
- **Edges** define the flow of information between nodes
- **State** is passed between nodes during execution
- **Entry Points** determine how messages enter the workflow

```
┌───────────────────────────────────────────────────────────────┐
│                    Workflow Orchestrator                       │
└────────────┬───────────────────┬───────────────┬──────────────┘
             │                   │               │
             │                   │               │
             ▼                   ▼               ▼
      ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
      │    Nodes    │◄────┤    Edges    │    │Entry Points │
      └─────────────┘     └─────────────┘    └─────────────┘
             │
             │ (typically wraps)
             ▼
      ┌─────────────┐
      │   Agents    │
      └─────────────┘
```

## Core Components

### Workflow Interfaces

All workflows implement interfaces from the `seedwork.interfaces.iworkflow` module:

```python
class IWorkflowOrchestrator(Protocol):
    """Interface for workflow orchestrators that manage workflow execution."""
    
    def add_node(self, node_id: str, node: Any) -> None:
        """Add a node to the workflow"""
        ...
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between nodes"""
        ...
    
    def set_entry_points(self, router_function: Callable, default_routes: Dict[str, str]) -> None:
        """Set entry points for the workflow"""
        ...
    
    def process(self, input_data: Dict[str, Any], entry_point: str = "default") -> Dict[str, Any]:
        """Process input through the workflow"""
        ...
    
    async def aprocess(self, input_data: Dict[str, Any], entry_point: str = "default") -> Dict[str, Any]:
        """Process input through the workflow asynchronously"""
        ...
```

### WorkflowOrchestrator

The `WorkflowOrchestrator` is the central component that manages the workflow execution:

```python
class WorkflowOrchestrator(IWorkflowOrchestrator):
    """
    Orchestrator that manages workflow execution.
    
    The orchestrator maintains a directed graph of nodes and handles the
    flow of data between them during execution.
    """
    
    def __init__(self):
        """Initialize the workflow orchestrator."""
        self.nodes = {}
        self.edges = {}
        self.entry_router = None
        self.default_routes = {}
        
    # Implementation of IWorkflowOrchestrator methods...
```

### Nodes

Nodes are processing units in the workflow. The base node class provides common functionality:

```python
class BaseNode(INode):
    """
    Base implementation of a node in a workflow.
    
    A node is responsible for processing input data and producing output.
    This base implementation provides common functionality for all nodes.
    """
    
    def __init__(self, node_id: str, name: str, settings):
        """
        Initialize the node.
        
        Args:
            node_id: Unique identifier for the node
            name: Human-readable name for the node
            settings: Settings object for creating components
        """
        self.node_id = node_id
        self.name = name
        self.settings = settings
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and produce output.
        
        This method should be overridden by subclasses to define
        the node's specific processing logic.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Output data from processing
        """
        raise NotImplementedError("Subclasses must implement process()")
        
    async def aprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously process input data.
        
        By default, this calls the synchronous process method.
        Subclasses should override this for true async processing.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Output data from processing
        """
        return self.process(input_data)
```

#### Common Node Types

The framework provides several common node types:

**AgentNode**
```python
class AgentNode(BaseNode):
    """A node that wraps an agent in a workflow."""
    
    def __init__(self, node_id: str, name: str, agent: IAgent, settings):
        super().__init__(node_id, name, settings)
        self.agent = agent
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through the agent."""
        
        # Extract message and conversation ID from input data
        message = input_data.get("message", "")
        conversation_id = input_data.get("conversation_id", "default")
        
        # Process through agent
        agent_input = IAgentInput(
            message=message,
            conversation_id=conversation_id
        )
        
        agent_output = self.agent.process_message(agent_input)
        
        # Return agent output as node result
        return {
            "message": agent_output.agent_message,
            "tool_calls": agent_output.tool_calls,
            "memory": agent_output.memory
        }
```

**FunctionNode**
```python
class FunctionNode(BaseNode):
    """A node that executes a function."""
    
    def __init__(self, node_id: str, name: str, function: Callable, settings):
        super().__init__(node_id, name, settings)
        self.function = function
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through the function."""
        return self.function(input_data)
```

**ConditionalNode**
```python
class ConditionalNode(BaseNode):
    """A node that routes based on a condition."""
    
    def __init__(self, node_id: str, name: str, condition: Callable, settings):
        super().__init__(node_id, name, settings)
        self.condition = condition
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and add a routing key based on the condition."""
        result = self.condition(input_data)
        return {**input_data, "_route": result}
```

### Edges

Edges define the flow between nodes:

```python
def add_edge(self, from_node: str, to_node: str) -> None:
    """
    Add an edge between nodes.
    
    Args:
        from_node: ID of the source node
        to_node: ID of the destination node
        
    Raises:
        ValueError: If either node doesn't exist
    """
    if from_node not in self.nodes:
        raise ValueError(f"Source node '{from_node}' doesn't exist")
    
    if to_node not in self.nodes:
        raise ValueError(f"Destination node '{to_node}' doesn't exist")
    
    if from_node not in self.edges:
        self.edges[from_node] = []
    
    self.edges[from_node].append(to_node)
```

### Entry Points

Entry points determine how messages enter the workflow:

```python
def set_entry_points(self, router_function: Callable, default_routes: Dict[str, str]) -> None:
    """
    Set entry points for the workflow.
    
    Args:
        router_function: Function that determines the entry point
        default_routes: Mapping of entry point names to node IDs
    """
    self.entry_router = router_function
    self.default_routes = default_routes
```

## Creating Workflows

### Using WorkflowConfig

The recommended way to create workflows is by extending the `WorkflowConfig` class:

```python
class CustomWorkflowConfig(WorkflowConfig):
    """Configuration for a custom workflow."""
    
    def __init__(self, settings, debug_mode=False, **kwargs):
        super().__init__(settings, debug_mode, **kwargs)
    
    def _configure_workflow(self, orchestrator: IWorkflowOrchestrator) -> None:
        """Configure the workflow structure."""
        
        # Create nodes
        nodes = self._create_nodes()
        
        # Add nodes to orchestrator
        for node_id, node in nodes.items():
            orchestrator.add_node(node_id, node)
        
        # Add edges
        self._add_edges(orchestrator)
        
        # Set entry points
        orchestrator.set_entry_points(
            self._route_input,
            {"default": "first_node"}
        )
    
    def _create_nodes(self) -> Dict[str, INode]:
        """Create nodes for the workflow."""
        
        # Create an operator agent
        operator_config = IAgentConfig(
            task_context="You are an operator agent in a workflow",
            tools=[]
        )
        operator_agent = self.settings.create_agent("operator", operator_config)
        
        # Create a specialized agent
        specialized_config = IAgentConfig(
            task_context="You are a specialized agent focusing on a specific task",
            tools=[]
        )
        specialized_agent = self.settings.create_agent("specialized", specialized_config)
        
        # Create nodes
        return {
            "operator_node": AgentNode(
                node_id="operator",
                name="Operator Agent",
                agent=operator_agent,
                settings=self.settings
            ),
            "specialized_node": AgentNode(
                node_id="specialized",
                name="Specialized Agent",
                agent=specialized_agent,
                settings=self.settings
            ),
            "formatter_node": FunctionNode(
                node_id="formatter",
                name="Response Formatter",
                function=self._format_response,
                settings=self.settings
            )
        }
    
    def _add_edges(self, orchestrator: IWorkflowOrchestrator) -> None:
        """Add edges between nodes."""
        orchestrator.add_edge("operator_node", "specialized_node")
        orchestrator.add_edge("specialized_node", "formatter_node")
    
    def _route_input(self, input_data: Dict[str, Any]) -> str:
        """Route input to the appropriate entry point."""
        # Simple routing logic
        return "default"
    
    def _format_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response."""
        return {
            "formatted_message": f"Final response: {input_data.get('message', '')}"
        }
```

### Using WorkflowRunner

To execute a workflow, use the `WorkflowRunner`:

```python
class WorkflowRunner(IWorkflowRunner):
    """
    Runner for executing workflows.
    
    The runner is responsible for initializing and executing workflows
    based on a configuration.
    """
    
    def __init__(self, workflow_config: WorkflowConfig):
        """
        Initialize the workflow runner.
        
        Args:
            workflow_config: Configuration for the workflow
        """
        self.config = workflow_config
        self.orchestrator = self.config.create_workflow()
    
    def process(self, input_data: Dict[str, Any], entry_point: str = "default") -> Dict[str, Any]:
        """
        Process input through the workflow.
        
        Args:
            input_data: Input data to process
            entry_point: Entry point to use (optional)
            
        Returns:
            Output data from the workflow
        """
        return self.orchestrator.process(input_data, entry_point)
    
    async def aprocess(self, input_data: Dict[str, Any], entry_point: str = "default") -> Dict[str, Any]:
        """
        Process input through the workflow asynchronously.
        
        Args:
            input_data: Input data to process
            entry_point: Entry point to use (optional)
            
        Returns:
            Output data from the workflow
        """
        return await self.orchestrator.aprocess(input_data, entry_point)
```

## Usage Examples

### Simple Linear Workflow

A simple linear workflow with three nodes:

```python
from arshai import Settings
from src.workflows import WorkflowConfig, WorkflowRunner
from src.workflows.nodes import AgentNode, FunctionNode
from seedwork.interfaces.iagent import IAgentConfig
from seedwork.interfaces.iworkflow import IWorkflowOrchestrator
from typing import Dict, Any

class SimpleWorkflowConfig(WorkflowConfig):
    """Configuration for a simple linear workflow."""
    
    def _configure_workflow(self, orchestrator: IWorkflowOrchestrator) -> None:
        """Configure the workflow structure."""
        
        # Create agents
        operator_config = IAgentConfig(
            task_context="You are the initial operator who understands user requests",
            tools=[]
        )
        operator_agent = self.settings.create_agent("operator", operator_config)
        
        researcher_config = IAgentConfig(
            task_context="You research and provide detailed information on topics",
            tools=[]
        )
        researcher_agent = self.settings.create_agent("operator", researcher_config)
        
        # Create nodes
        orchestrator.add_node("intake", AgentNode(
            node_id="intake",
            name="Intake Agent",
            agent=operator_agent,
            settings=self.settings
        ))
        
        orchestrator.add_node("research", AgentNode(
            node_id="research",
            name="Research Agent",
            agent=researcher_agent,
            settings=self.settings
        ))
        
        orchestrator.add_node("format", FunctionNode(
            node_id="format",
            name="Format Response",
            function=self._format_response,
            settings=self.settings
        ))
        
        # Add edges
        orchestrator.add_edge("intake", "research")
        orchestrator.add_edge("research", "format")
        
        # Set entry points
        orchestrator.set_entry_points(
            lambda _: "default",
            {"default": "intake"}
        )
    
    def _format_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response."""
        return {
            "final_response": input_data.get("message", "")
        }

# Create and use the workflow
settings = Settings()
workflow_config = SimpleWorkflowConfig(settings)
runner = WorkflowRunner(workflow_config)

# Process a message
result = runner.process({
    "message": "Tell me about artificial intelligence",
    "conversation_id": "workflow_demo"
})

print(result["final_response"])
```

### Branching Workflow

A workflow with conditional branching:

```python
from arshai import Settings
from src.workflows import WorkflowConfig, WorkflowRunner
from src.workflows.nodes import AgentNode, ConditionalNode
from seedwork.interfaces.iagent import IAgentConfig
from seedwork.interfaces.iworkflow import IWorkflowOrchestrator
from typing import Dict, Any

class BranchingWorkflowConfig(WorkflowConfig):
    """Configuration for a workflow with branching logic."""
    
    def _configure_workflow(self, orchestrator: IWorkflowOrchestrator) -> None:
        """Configure the workflow structure."""
        
        # Create agents
        operator_config = IAgentConfig(
            task_context="You are the initial operator who categorizes requests",
            tools=[]
        )
        operator_agent = self.settings.create_agent("operator", operator_config)
        
        tech_config = IAgentConfig(
            task_context="You are a technology specialist",
            tools=[]
        )
        tech_agent = self.settings.create_agent("operator", tech_config)
        
        finance_config = IAgentConfig(
            task_context="You are a finance specialist",
            tools=[]
        )
        finance_agent = self.settings.create_agent("operator", finance_config)
        
        # Create nodes
        orchestrator.add_node("intake", AgentNode(
            node_id="intake",
            name="Intake Agent",
            agent=operator_agent,
            settings=self.settings
        ))
        
        orchestrator.add_node("categorize", ConditionalNode(
            node_id="categorize",
            name="Categorize Request",
            condition=self._categorize_request,
            settings=self.settings
        ))
        
        orchestrator.add_node("tech", AgentNode(
            node_id="tech",
            name="Technology Specialist",
            agent=tech_agent,
            settings=self.settings
        ))
        
        orchestrator.add_node("finance", AgentNode(
            node_id="finance",
            name="Finance Specialist",
            agent=finance_agent,
            settings=self.settings
        ))
        
        # Add edges
        orchestrator.add_edge("intake", "categorize")
        orchestrator.add_edge("categorize", "tech")
        orchestrator.add_edge("categorize", "finance")
        
        # Set entry points
        orchestrator.set_entry_points(
            lambda _: "default",
            {"default": "intake"}
        )
    
    def _categorize_request(self, input_data: Dict[str, Any]) -> str:
        """Categorize the request as tech or finance."""
        message = input_data.get("message", "").lower()
        
        if any(word in message for word in ["computer", "software", "technology", "code"]):
            return "tech"
        elif any(word in message for word in ["money", "finance", "investment", "stock"]):
            return "finance"
        else:
            return "tech"  # Default to tech

# Create and use the workflow
settings = Settings()
workflow_config = BranchingWorkflowConfig(settings)
runner = WorkflowRunner(workflow_config)

# Process messages
tech_result = runner.process({
    "message": "How do computers work?",
    "conversation_id": "workflow_tech"
})

finance_result = runner.process({
    "message": "What stocks should I invest in?",
    "conversation_id": "workflow_finance"
})

print(tech_result["message"])
print(finance_result["message"])
```

### Advanced Workflow with State Management

A more advanced workflow with state management between nodes:

```python
class AdvancedWorkflowConfig(WorkflowConfig):
    """Configuration for an advanced workflow with state management."""
    
    def _configure_workflow(self, orchestrator: IWorkflowOrchestrator) -> None:
        """Configure the workflow structure."""
        
        # Create all required nodes
        nodes = self._create_nodes()
        
        # Add nodes to orchestrator
        for node_id, node in nodes.items():
            orchestrator.add_node(node_id, node)
        
        # Add edges
        orchestrator.add_edge("intake", "planner")
        orchestrator.add_edge("planner", "router")
        orchestrator.add_edge("router", "research")
        orchestrator.add_edge("router", "coding")
        orchestrator.add_edge("router", "writing")
        orchestrator.add_edge("research", "final")
        orchestrator.add_edge("coding", "final")
        orchestrator.add_edge("writing", "final")
        
        # Set entry points
        orchestrator.set_entry_points(
            self._route_input,
            {"default": "intake", "direct_research": "research"}
        )
    
    def _create_nodes(self) -> Dict[str, INode]:
        """Create nodes for the workflow."""
        # Create agent configs
        intake_config = IAgentConfig(
            task_context="You are the initial intake agent who gathers requirements",
            tools=[]
        )
        
        planner_config = IAgentConfig(
            task_context="You create a plan based on requirements",
            tools=[]
        )
        
        research_config = IAgentConfig(
            task_context="You research topics and provide information",
            tools=[self._create_web_search_tool()]
        )
        
        coding_config = IAgentConfig(
            task_context="You write code to solve problems",
            tools=[]
        )
        
        writing_config = IAgentConfig(
            task_context="You write content based on requirements",
            tools=[]
        )
        
        final_config = IAgentConfig(
            task_context="You create a final response based on all previous work",
            tools=[]
        )
        
        # Create agents
        intake_agent = self.settings.create_agent("operator", intake_config)
        planner_agent = self.settings.create_agent("operator", planner_config)
        research_agent = self.settings.create_agent("operator", research_config)
        coding_agent = self.settings.create_agent("operator", coding_config)
        writing_agent = self.settings.create_agent("operator", writing_config)
        final_agent = self.settings.create_agent("operator", final_config)
        
        # Create and return nodes
        return {
            "intake": AgentNode("intake", "Intake Agent", intake_agent, self.settings),
            "planner": AgentNode("planner", "Planner Agent", planner_agent, self.settings),
            "router": ConditionalNode("router", "Task Router", self._route_task, self.settings),
            "research": AgentNode("research", "Research Agent", research_agent, self.settings),
            "coding": AgentNode("coding", "Coding Agent", coding_agent, self.settings),
            "writing": AgentNode("writing", "Writing Agent", writing_agent, self.settings),
            "final": AgentNode("final", "Final Agent", final_agent, self.settings)
        }
    
    def _create_web_search_tool(self):
        """Create a web search tool."""
        # Implementation details...
        return WebSearchTool(self.settings)
    
    def _route_input(self, input_data: Dict[str, Any]) -> str:
        """Route initial input to the appropriate entry point."""
        if input_data.get("direct_research"):
            return "direct_research"
        return "default"
    
    def _route_task(self, input_data: Dict[str, Any]) -> str:
        """Route to the appropriate specialist based on the plan."""
        plan = input_data.get("plan", "").lower()
        
        if "research" in plan or "information" in plan:
            return "research"
        elif "code" in plan or "programming" in plan:
            return "coding"
        elif "write" in plan or "content" in plan:
            return "writing"
        else:
            return "research"  # Default
```

## Best Practices

### Design Principles

1. **Single Responsibility**: Each node should have a single, well-defined responsibility
2. **Clear Data Flow**: Design clear paths for data to flow through the workflow
3. **Stateless Nodes**: Keep nodes stateless, storing state in the data passed between nodes
4. **Appropriate Granularity**: Neither too fine-grained nor too coarse-grained nodes
5. **Error Handling**: Include error handling at appropriate points in the workflow

### Implementation Patterns

1. **Configuration Over Code**: Use configuration to define workflow structure
2. **Composition**: Build complex workflows by composing simple components
3. **Testing**: Test individual nodes before integrating them into workflows
4. **Monitoring**: Add monitoring to track workflow execution and performance
5. **Documentation**: Document the purpose and behavior of each node and the overall workflow

### Common Patterns

1. **Linear Workflow**: A simple sequence of nodes
2. **Branching Workflow**: Nodes that conditionally route to different paths
3. **Hub-and-Spoke**: A central node that coordinates with specialist nodes
4. **Pipeline Workflow**: A processing pipeline with transformations at each step
5. **Parallel Processing**: Multiple nodes processing in parallel

## Advanced Topics

### Asynchronous Workflow Processing

For asynchronous workflow processing, use the `aprocess` methods:

```python
async def run_workflow():
    """Run a workflow asynchronously."""
    settings = Settings()
    workflow_config = CustomWorkflowConfig(settings)
    runner = WorkflowRunner(workflow_config)
    
    # Process asynchronously
    result = await runner.aprocess({
        "message": "Process this asynchronously",
        "conversation_id": "async_workflow"
    })
    
    return result
```

### Debugging Workflows

To debug workflows, enable debug mode in the workflow configuration:

```python
# Enable debug mode
workflow_config = CustomWorkflowConfig(settings, debug_mode=True)
runner = WorkflowRunner(workflow_config)

# Process with debugging
result = runner.process({
    "message": "Debug this workflow",
    "conversation_id": "debug_workflow"
})
```

With debug mode enabled, the workflow will log detailed information about each step, including:
- Input to each node
- Output from each node
- Execution time for each node
- Any errors that occur

### Integration with Memory Systems

To integrate with memory systems, use the shared conversation IDs:

```python
def process_with_memory(message, conversation_id):
    """Process a message with memory integration."""
    settings = Settings()
    workflow_config = CustomWorkflowConfig(settings)
    runner = WorkflowRunner(workflow_config)
    
    # Process with memory
    result = runner.process({
        "message": message,
        "conversation_id": conversation_id
    })
    
    return result
```

By using the same conversation ID across multiple workflow invocations, the agents within the workflow can maintain context through their memory systems. 