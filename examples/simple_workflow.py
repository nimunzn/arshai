"""
Simple Workflow Example

This demonstrates a basic workflow with direct edges only:
1. Greeting Node -> Analysis Node -> Response Node

This is a linear workflow without any conditional routing.
"""

from typing import Dict, Any, Optional
import logging
import asyncio
from datetime import datetime

from arshai.core.interfaces import IAgent, IAgentConfig, IAgentInput, IAgentOutput
from arshai.core.interfaces import ISetting
from arshai.core.interfaces import IWorkflowOrchestrator, IWorkflowState
from arshai.core.interfaces import ILLMInput, LLMInputType

from arshai.config.settings import Settings
from arshai.workflows.node import BaseNode
from arshai.workflows.workflow_config import WorkflowConfig
from arshai.workflows.workflow_runner import BaseWorkflowRunner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================================================================
# PART 1: Node Implementation using BaseNode
# =========================================================================

class GreetingNode(BaseNode):
    """
    Node that provides a personalized greeting based on the input message.
    
    This node simply wraps a basic agent with a greeting-specific task context.
    """
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input with a greeting-focused approach."""
        logger.info(f"GreetingNode processing input")
        
        # Extract the message
        message = input_data.get("message", "")
        
        # Add a greeting-specific prefix to the message
        enhanced_message = f"Greet the user and acknowledge their message: {message}"
        
        # Update the input data with the enhanced message
        enhanced_input = input_data.copy()
        enhanced_input["message"] = enhanced_message
        
        # Process with the standard BaseNode implementation
        return await super().process(enhanced_input)


class AnalysisNode(BaseNode):
    """
    Node that analyzes the input message for topics and sentiments.
    
    This node adds simple analysis results to the workflow state.
    """
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input by analyzing content and sentiment."""
        logger.info(f"AnalysisNode processing input")
        
        # Extract state and message
        state = input_data.get("state")
        if not state:
            raise ValueError("Input data must contain 'state'")
        
        message = input_data.get("message", "")
        
        # Perform simple content analysis
        topics = self._detect_topics(message)
        sentiment = self._analyze_sentiment(message)
        
        # Store analysis in workflow state
        if not state.workflow_data:
            state.workflow_data = {}
        
        state.workflow_data["analysis"] = {
            "topics": topics,
            "sentiment": sentiment,
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Add analysis context to the message
        enhanced_message = (
            f"Original message: {message}\n\n"
            f"Analysis results:\n"
            f"- Topics: {', '.join(topics)}\n"
            f"- Sentiment: {sentiment}"
        )
        
        # Update the input data
        enhanced_input = input_data.copy()
        enhanced_input["message"] = enhanced_message
        
        # Process with the standard BaseNode implementation
        return await super().process(enhanced_input)
    
    def _detect_topics(self, message: str) -> list:
        """Detect topics in the message."""
        topics = []
        message_lower = message.lower()
        
        # Simple keyword-based topic detection
        if any(word in message_lower for word in ["product", "service", "offer", "pricing", "price"]):
            topics.append("products")
        
        if any(word in message_lower for word in ["help", "support", "issue", "problem", "error"]):
            topics.append("support")
        
        if any(word in message_lower for word in ["account", "login", "password", "sign in", "sign up"]):
            topics.append("account")
        
        if not topics:
            topics.append("general")
        
        return topics
    
    def _analyze_sentiment(self, message: str) -> str:
        """Analyze sentiment in the message."""
        message_lower = message.lower()
        
        # Simple rule-based sentiment analysis
        positive_words = ["good", "great", "excellent", "happy", "like", "love", "thank", "thanks"]
        negative_words = ["bad", "poor", "terrible", "unhappy", "dislike", "hate", "problem", "issue"]
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"


class ResponseNode(BaseNode):
    """
    Node that generates a comprehensive response based on analysis.
    
    This node takes the analysis from the previous node and generates
    a tailored response.
    """
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input by generating a tailored response."""
        logger.info(f"ResponseNode processing input")
        
        # Extract state and analysis
        state = input_data.get("state")
        if not state:
            raise ValueError("Input data must contain 'state'")
        
        # Get analysis from state
        analysis = {}
        if state.workflow_data and "analysis" in state.workflow_data:
            analysis = state.workflow_data["analysis"]
        
        # Original message
        original_message = input_data.get("original_message", input_data.get("message", ""))
        
        # Create a response prompt based on analysis
        response_prompt = (
            f"Generate a helpful response to: '{original_message}'\n\n"
            f"Topics: {analysis.get('topics', ['general'])}\n"
            f"Sentiment: {analysis.get('sentiment', 'neutral')}\n\n"
            f"Make your response friendly, informative, and tailored to both "
            f"the topics and sentiment detected."
        )
        
        # Update the input data
        enhanced_input = input_data.copy()
        enhanced_input["message"] = response_prompt
        
        # Process with the standard BaseNode implementation
        return await super().process(enhanced_input)


# =========================================================================
# PART 2: Workflow Configuration
# =========================================================================

class SimpleWorkflowConfig(WorkflowConfig):
    """
    Configuration for a simple linear workflow.
    
    This workflow uses direct edges only without conditional routing.
    """
    
    def __init__(self, settings: ISetting, debug_mode: bool = False, **kwargs: Any):
        """Initialize workflow configuration."""
        super().__init__(settings, debug_mode, **kwargs)
    
    def _configure_workflow(self, workflow: IWorkflowOrchestrator) -> None:
        """
        Configure the workflow with nodes and direct edges.
        """
        # Create nodes
        nodes = self._create_nodes()
        
        # Add nodes to workflow
        for name, node in nodes.items():
            workflow.add_node(name, node)
        
        # Define direct edges
        edges = self._define_edges()
        for from_node, to_node in edges.items():
            workflow.add_edge(from_node, to_node)
        
        # Set entry points
        workflow.set_entry_points(
            self._route_input,
            {"default": "greeting_node"}
        )
    
    def _create_nodes(self) -> Dict[str, Any]:
        """Create all nodes for the workflow."""
        # Create greeting node with predefined agent
        greeting_agent_config = IAgentConfig(
            task_context="You are a friendly assistant who greets users and sets a positive tone.",
            tools=[]
        )
        greeting_agent = self.settings.create_agent("operator", greeting_agent_config)
        
        greeting_node = GreetingNode(
            node_id="greeting_node",
            name="Greeting Node",
            agent=greeting_agent,
            settings=self.settings
        )
        
        # Create analysis node with predefined agent
        analysis_agent_config = IAgentConfig(
            task_context="You analyze messages to identify topics and sentiment.",
            tools=[]
        )
        analysis_agent = self.settings.create_agent("operator", analysis_agent_config)
        
        analysis_node = AnalysisNode(
            node_id="analysis_node",
            name="Analysis Node",
            agent=analysis_agent,
            settings=self.settings
        )
        
        # Create response node with predefined agent
        response_agent_config = IAgentConfig(
            task_context="You provide helpful, tailored responses based on message analysis.",
            tools=[]
        )
        response_agent = self.settings.create_agent("operator", response_agent_config)
        
        response_node = ResponseNode(
            node_id="response_node",
            name="Response Node",
            agent=response_agent,
            settings=self.settings
        )
        
        return {
            "greeting_node": greeting_node,
            "analysis_node": analysis_node,
            "response_node": response_node
        }
    
    def _define_edges(self) -> Dict[str, str]:
        """Define direct edges between nodes."""
        # Linear flow: greeting -> analysis -> response
        return {
            "greeting_node": "analysis_node",
            "analysis_node": "response_node"
        }
    
    def _route_input(self, input_data: Dict[str, Any]) -> str:
        """Route input to the appropriate entry point."""
        # Always start with the greeting node
        return "default"


# =========================================================================
# PART 3: Running the Workflow
# =========================================================================

async def run_simple_workflow():
    """Run the simple workflow with example messages."""
    print("\n=== Simple Linear Workflow Example ===")
    
    # Create settings
    settings = Settings()
    
    # Create workflow configuration
    workflow_config = SimpleWorkflowConfig(settings, debug_mode=True)
    
    # Create workflow runner
    runner = BaseWorkflowRunner(workflow_config, debug_mode=True)
    
    # Sample messages to test
    examples = [
        "Hello, can you help me with your products?",
        "I'm having a problem with my account.",
        "Thank you for your excellent service!"
    ]
    
    for i, message in enumerate(examples):
        print(f"\n[Example {i+1}] Processing: \"{message}\"")
        
        # Save original message for later reference
        input_data = {
            "message": message,
            "original_message": message
        }
        
        # Execute the workflow
        result = await runner.execute_workflow(
            user_id=f"user-{i+1}",
            input_data=input_data
        )
        
        # Process the results
        if result.get("success", False):
            workflow_result = result.get("result", {})
            state = result.get("state")
            
            # Get the final response
            final_response = None
            
            if state and state.agent_data and "response_node" in state.agent_data:
                response_data = state.agent_data["response_node"]
                if hasattr(response_data, "agent_message"):
                    final_response = response_data.agent_message
            
            # Get the analysis results
            analysis = None
            if state and state.workflow_data and "analysis" in state.workflow_data:
                analysis = state.workflow_data["analysis"]
            
            # Display results
            print("\nAnalysis Results:")
            print(f"Topics: {', '.join(analysis.get('topics', ['unknown']))}")
            print(f"Sentiment: {analysis.get('sentiment', 'unknown')}")
            
            print("\nFinal Response:")
            print("-" * 40)
            print(final_response or "No response generated")
            print("-" * 40)
        else:
            # Handle errors
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\nSimple workflow example completed!")


if __name__ == "__main__":
    # Run the workflow example
    asyncio.run(run_simple_workflow()) 