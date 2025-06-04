"""
Advanced Workflow Implementation using Arshai

This example demonstrates:
1. Combined usage of predefined agents and a custom agent
2. Simple routing between nodes based on message content
3. Direct message passing to ensure reliable node communication
4. Clear and explicit workflow structure
"""

from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime
import json

from arshai.core.interfaces import IWorkflowState, IWorkflowOrchestrator, INode
from arshai.core.interfaces import IAgent, IAgentConfig, IAgentInput, IAgentOutput
from arshai.core.interfaces import ISetting
from arshai.core.interfaces import ILLMInput, LLMInputType

from arshai.workflows.node import BaseNode
from arshai.workflows.workflow_config import WorkflowConfig
from arshai.workflows.workflow_runner import BaseWorkflowRunner
from arshai.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================================================================
# PART 1: Custom Agent Implementation
# =========================================================================

class ProductSpecialistAgent(IAgent):
    """
    Custom agent implementation specializing in product information.
    """
    
    def __init__(self, config: IAgentConfig, settings: ISetting):
        """Initialize with product knowledge."""
        self.settings = settings
        self.llm = self.settings.create_llm()
        self.task_context = config.get("task_context", "You are a product specialist")
        
        # Product database (in real case, would be retrieved from external source)
        self.products = {
            "basic_plan": {
                "name": "Basic Plan",
                "price": "$9.99/month",
                "features": ["Core functionality", "Email support", "5GB storage"]
            },
            "pro_plan": {
                "name": "Professional Plan",
                "price": "$29.99/month",
                "features": ["All Basic features", "Priority support", "25GB storage", "Analytics"]
            },
            "enterprise_plan": {
                "name": "Enterprise Plan",
                "price": "Custom pricing",
                "features": ["All Pro features", "Dedicated support", "Unlimited storage", "Custom integrations"]
            }
        }
    
    def _prepare_system_prompt(self) -> str:
        """Prepare a system prompt with product knowledge."""
        return f"""
        You are a Product Specialist AI assistant focused on helping customers understand our offerings.
        
        {self.task_context}
        
        Our products:
        {json.dumps(self.products, indent=2)}
        
        Provide accurate product information based on the catalog above.
        """
    
    def process_message(self, input: IAgentInput) -> IAgentOutput:
        """Process message with product expertise."""
        logger.info(f"ProductSpecialistAgent processing message")
        
        try:
            # Prepare system prompt with product knowledge
            system_prompt = self._prepare_system_prompt()
            
            # Call LLM with our specialized prompt
            llm_response = self.llm.generate_text(
                input=ILLMInput(
                    llm_type=LLMInputType.CHAT,
                    system_prompt=system_prompt,
                    user_message=input.message
                )
            )
            
            # Return the response
            return IAgentOutput(agent_message=llm_response)
            
        except Exception as e:
            logger.error(f"Error in product specialist agent: {str(e)}")
            return IAgentOutput(
                agent_message="I'm sorry, I'm having trouble accessing our product information."
            )
    
    async def aprocess_message(self, input: IAgentInput) -> IAgentOutput:
        """Async version of process_message."""
        return self.process_message(input)


# =========================================================================
# PART 2: Simple Routing Node
# =========================================================================

class RoutingNode(BaseNode):
    """
    Node that analyzes messages and routes them to appropriate specialists.
    
    This node extends BaseNode and uses a regular agent but sets the 'route' 
    key in the result to tell the workflow orchestrator where to go next.
    """
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and determine routing."""
        logger.info("RoutingNode processing")
        
        # Extract message
        message = input_data.get("message", "")
        message_lower = message.lower()
        
        # Simple keyword-based routing
        next_node = "general_node"  # Default route
        
        # Check for product-related keywords
        if any(word in message_lower for word in ["product", "plan", "pricing", "cost", "buy"]):
            next_node = "product_node"
            logger.info("Routing to product_node")
        
        # Check for support-related keywords
        elif any(word in message_lower for word in ["help", "support", "issue", "problem", "error"]):
            next_node = "support_node"
            logger.info("Routing to support_node")
            
        # Let agent process the message normally
        result = await super().process(input_data)
        
        # Add routing information to result
        result["route"] = next_node
        
        # Make sure we preserve the original message for the next node
        if "state" in result:
            state = result["state"]
            if not state.workflow_data:
                state.workflow_data = {}
            state.workflow_data["original_message"] = message
        
        return result


# =========================================================================
# PART 3: Specialized Agent Nodes
# =========================================================================

class ProductNode(BaseNode):
    """Product specialist node."""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with product expertise."""
        logger.info("ProductNode processing")
        
        # Get original message if possible
        state = input_data.get("state")
        if state and state.workflow_data and "original_message" in state.workflow_data:
            original_message = state.workflow_data["original_message"]
            # Override current message with original
            input_data = input_data.copy()
            input_data["message"] = original_message
        
        # Let the agent handle the response
        return await super().process(input_data)


class SupportNode(BaseNode):
    """Support specialist node."""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with support expertise."""
        logger.info("SupportNode processing")
        
        # Get original message if possible
        state = input_data.get("state")
        if state and state.workflow_data and "original_message" in state.workflow_data:
            original_message = state.workflow_data["original_message"]
            # Override current message with original
            input_data = input_data.copy()
            input_data["message"] = original_message
        
        # Let the agent handle the response
        return await super().process(input_data)


class GeneralNode(BaseNode):
    """General assistant node."""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process general inquiries."""
        logger.info("GeneralNode processing")
        
        # Get original message if possible
        state = input_data.get("state")
        if state and state.workflow_data and "original_message" in state.workflow_data:
            original_message = state.workflow_data["original_message"]
            # Override current message with original
            input_data = input_data.copy()
            input_data["message"] = original_message
        
        # Let the agent handle the response
        return await super().process(input_data)


# =========================================================================
# PART 4: Workflow Configuration
# =========================================================================

class AdvancedWorkflowConfig(WorkflowConfig):
    """
    Configuration for workflow with routing and specialized nodes.
    """
    
    def __init__(self, settings: ISetting, debug_mode: bool = False, **kwargs: Any):
        """Initialize workflow configuration."""
        super().__init__(settings, debug_mode, **kwargs)
    
    def _configure_workflow(self, workflow: IWorkflowOrchestrator) -> None:
        """Configure the workflow with nodes and edges."""
        # Create nodes
        nodes = self._create_nodes()
        
        # Add nodes to workflow
        for name, node in nodes.items():
            workflow.add_node(name, node)
        
        # Define edges - no direct edges needed except from router
        # The router will use the 'route' key to determine the next node
        edges = self._define_edges()
        for from_node, to_node in edges.items():
            workflow.add_edge(from_node, to_node)
        
        # Set entry points - always start with the router
        workflow.set_entry_points(
            self._route_input,
            {"default": "router_node"}
        )
    
    def _create_nodes(self) -> Dict[str, Any]:
        """Create all nodes for the workflow."""
        # Create router node with predefined agent
        router_agent_config = IAgentConfig(
            task_context="You analyze user messages to identify the type of query.",
            tools=[]
        )
        router_agent = self.settings.create_agent("operator", router_agent_config)
        
        router_node = RoutingNode(
            node_id="router_node",
            name="Message Router",
            agent=router_agent,
            settings=self.settings
        )
        
        # Create product node with custom agent
        product_agent_config = IAgentConfig(
            task_context="You are a product specialist who provides detailed information about our products.",
            tools=[]
        )
        product_agent = ProductSpecialistAgent(product_agent_config, self.settings)
        
        product_node = ProductNode(
            node_id="product_node",
            name="Product Specialist",
            agent=product_agent,
            settings=self.settings
        )
        
        # Create support node with predefined agent
        support_agent_config = IAgentConfig(
            task_context="You are a support specialist who helps users resolve technical issues.",
            tools=[]
        )
        support_agent = self.settings.create_agent("operator", support_agent_config)
        
        support_node = SupportNode(
            node_id="support_node",
            name="Support Specialist",
            agent=support_agent,
            settings=self.settings
        )
        
        # Create general node with predefined agent
        general_agent_config = IAgentConfig(
            task_context="You are a general assistant who helps with a wide range of questions.",
            tools=[]
        )
        general_agent = self.settings.create_agent("operator", general_agent_config)
        
        general_node = GeneralNode(
            node_id="general_node",
            name="General Assistant",
            agent=general_agent,
            settings=self.settings
        )
        
        return {
            "router_node": router_node,
            "product_node": product_node,
            "support_node": support_node,
            "general_node": general_node
        }
    
    def _define_edges(self) -> Dict[str, str]:
        """
        Define the edges between nodes.
        
        No direct edges needed - routing is handled via route key.
        """
        # Return empty dict - no direct edges needed
        return {}
    
    def _route_input(self, input_data: Dict[str, Any]) -> str:
        """Route input to the appropriate entry point."""
        # Always start with the router node
        return "default"


# =========================================================================
# PART 5: Running the Workflow
# =========================================================================

async def run_advanced_workflow():
    """Run the advanced workflow with example messages."""
    print("\n=== Advanced Workflow with Routing Example ===")
    
    # Create settings
    settings = Settings()
    
    # Create workflow configuration
    workflow_config = AdvancedWorkflowConfig(settings, debug_mode=True)
    
    # Create workflow runner
    runner = BaseWorkflowRunner(workflow_config, debug_mode=True)
    
    # Sample messages to demonstrate different routing scenarios
    examples = [
        "What products do you offer and how much do they cost?",
        "I'm having trouble logging into my account, can you help?",
        "Tell me more about your company."
    ]
    
    for i, message in enumerate(examples):
        print(f"\n[Example {i+1}] Processing: \"{message}\"")
        
        # Execute the workflow
        result = await runner.execute_workflow(
            user_id=f"user-{i+1}",
            input_data={"message": message}
        )
        
        # Process the results
        if result.get("success", False):
            workflow_result = result.get("result", {})
            state = result.get("state")
            
            # Extract execution path
            execution_path = []
            if state and state.agent_data:
                execution_path = list(state.agent_data.keys())
            
            # Get final response from the last node that processed
            final_response = None
            final_node = None
            
            if state and state.agent_data:
                # Find the response from the specialized node (not router)
                for node_id, response in state.agent_data.items():
                    if node_id != "router_node" and hasattr(response, "agent_message"):
                        final_response = response.agent_message
                        final_node = node_id
            
            # Display results
            print("\nWorkflow Execution:")
            print(f"Path: {' -> '.join(execution_path)}")
            print(f"Final node: {final_node}")
            
            print("\nResponse:")
            print("-" * 60)
            print(final_response or "No response generated")
            print("-" * 60)
        else:
            # Handle errors
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\nAdvanced workflow example completed!")


if __name__ == "__main__":
    # Run the workflow example
    asyncio.run(run_advanced_workflow()) 