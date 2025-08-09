"""
Example 5: Agent Composition Patterns
======================================

This example demonstrates how to compose multiple agents together to create
complex, orchestrated AI workflows.

Prerequisites:
- Set OPENROUTER_API_KEY environment variable
- Install arshai package
"""

import os
import json
import asyncio
from typing import Dict, Any, List
from arshai.agents.base import BaseAgent
from arshai.agents.working_memory import WorkingMemoryAgent
from arshai.core.interfaces.iagent import IAgent, IAgentInput
from arshai.core.interfaces.illm import ILLMInput, ILLMConfig, ILLM
from arshai.llms.openrouter import OpenRouterClient


# Mock memory manager for demonstrations
class SimpleMemoryManager:
    def __init__(self):
        self.memories = {}
    
    async def store(self, data: Dict[str, Any]):
        conv_id = data.get("conversation_id")
        if conv_id:
            self.memories[conv_id] = data.get("working_memory", "")
    
    async def retrieve(self, query: Dict[str, Any]):
        conv_id = query.get("conversation_id")
        if conv_id and conv_id in self.memories:
            return [type('obj', (), {'working_memory': self.memories[conv_id]})()]
        return None


class DataAnalysisAgent(BaseAgent):
    """Specialized agent for data analysis tasks."""
    
    def __init__(self, llm_client: ILLM, **kwargs):
        system_prompt = """You are a data analysis expert.
        Analyze data patterns, provide insights, and make recommendations.
        Return structured analysis results."""
        super().__init__(llm_client, system_prompt, **kwargs)
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Analyze data and return structured insights."""
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=f"Analyze this data: {input.message}"
        )
        
        result = await self.llm_client.chat(llm_input)
        
        # Mock structured analysis
        return {
            "analysis_type": "data_pattern_analysis",
            "insights": result.get('llm_response', ''),
            "confidence": 0.85,
            "recommendations": ["Consider trend analysis", "Look for outliers"],
            "data_quality": "good"
        }


class ReportGenerationAgent(BaseAgent):
    """Specialized agent for generating reports."""
    
    def __init__(self, llm_client: ILLM, **kwargs):
        system_prompt = """You are a report writing expert.
        Create clear, structured reports based on provided data and analysis.
        Format reports professionally with sections and summaries."""
        super().__init__(llm_client, system_prompt, **kwargs)
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Generate a structured report."""
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=f"Create a report based on: {input.message}"
        )
        
        result = await self.llm_client.chat(llm_input)
        
        return {
            "report_type": "analytical_report",
            "content": result.get('llm_response', ''),
            "sections": ["Executive Summary", "Analysis", "Recommendations"],
            "word_count": len(result.get('llm_response', '').split()),
            "status": "completed"
        }


class KnowledgeBaseAgent(BaseAgent):
    """Agent that simulates knowledge base search."""
    
    def __init__(self, llm_client: ILLM, **kwargs):
        system_prompt = """You are a knowledge base search expert.
        Find relevant information from the knowledge base and provide accurate answers."""
        super().__init__(llm_client, system_prompt, **kwargs)
    
    async def process(self, input: IAgentInput) -> str:
        """Search knowledge base and return relevant information."""
        # Mock knowledge base search
        query = input.message.lower()
        
        if "customer" in query:
            return "Customer data shows 85% satisfaction rate with average response time of 2.3 hours."
        elif "sales" in query:
            return "Q3 sales figures: $2.4M revenue, 15% growth over previous quarter."
        elif "product" in query:
            return "Product catalog includes 150+ items across 5 categories with 95% availability."
        else:
            # Use LLM for general queries
            llm_input = ILLMInput(
                system_prompt=self.system_prompt,
                user_message=f"Search for: {input.message}"
            )
            result = await self.llm_client.chat(llm_input)
            return result.get('llm_response', 'No information found.')


class OrchestratorAgent(BaseAgent):
    """
    Master agent that orchestrates multiple specialized agents.
    
    This demonstrates the composition pattern where agents work together
    to complete complex tasks.
    """
    
    def __init__(self, llm_client: ILLM, specialized_agents: Dict[str, IAgent], **kwargs):
        system_prompt = """You are an intelligent orchestrator that coordinates multiple specialized agents.
        
        Available agents:
        - data_analyst: Analyzes data patterns and provides insights
        - report_generator: Creates structured reports
        - knowledge_search: Searches knowledge base for information
        - memory_manager: Manages conversation context
        
        Decide which agents to use based on the user's request and coordinate their work."""
        
        super().__init__(llm_client, system_prompt, **kwargs)
        self.agents = specialized_agents
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Orchestrate multiple agents to complete complex tasks."""
        
        # Define agent functions for LLM to call
        async def analyze_data(data_description: str) -> str:
            """Use the data analysis agent to analyze data."""
            result = await self.agents['data_analyst'].process(
                IAgentInput(message=data_description)
            )
            return json.dumps(result, indent=2)
        
        async def generate_report(content: str) -> str:
            """Use the report generator agent to create reports."""
            result = await self.agents['report_generator'].process(
                IAgentInput(message=content)
            )
            return result['content']
        
        async def search_knowledge(query: str) -> str:
            """Use the knowledge base agent to find information."""
            result = await self.agents['knowledge_search'].process(
                IAgentInput(message=query)
            )
            return str(result)
        
        # Background task for memory management
        async def update_memory(interaction: str) -> None:
            """Update conversation memory in background."""
            if 'memory_manager' in self.agents:
                await self.agents['memory_manager'].process(IAgentInput(
                    message=interaction,
                    metadata=input.metadata
                ))
        
        # Create LLM input with agent functions
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions={
                "analyze_data": analyze_data,
                "generate_report": generate_report,
                "search_knowledge": search_knowledge
            },
            background_tasks={
                "update_memory": update_memory
            }
        )
        
        # Orchestrate the task
        result = await self.llm_client.chat(llm_input)
        
        return {
            "orchestrator_response": result.get('llm_response', ''),
            "agents_available": list(self.agents.keys()),
            "coordination_metadata": {
                "input_length": len(input.message),
                "agents_used": "determined by LLM function calls",
                "complexity": "multi_agent_orchestration"
            }
        }


class PipelineAgent(BaseAgent):
    """
    Agent that implements a processing pipeline pattern.
    
    Demonstrates sequential agent composition where the output of one
    agent becomes the input to the next.
    """
    
    def __init__(self, llm_client: ILLM, pipeline_agents: List[IAgent], **kwargs):
        system_prompt = "You coordinate a processing pipeline of specialized agents."
        super().__init__(llm_client, system_prompt, **kwargs)
        self.pipeline = pipeline_agents
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Process input through a pipeline of agents."""
        
        pipeline_results = []
        current_input = input.message
        
        print(f"  🏭 Pipeline starting with input: {current_input[:100]}...")
        
        # Process through each agent in the pipeline
        for i, agent in enumerate(self.pipeline):
            print(f"  ⚙️  Stage {i+1}: Processing with {agent.__class__.__name__}")
            
            # Create input for this stage
            stage_input = IAgentInput(
                message=current_input,
                metadata=input.metadata
            )
            
            # Process with current agent
            stage_result = await agent.process(stage_input)
            
            # Store result
            stage_info = {
                "stage": i + 1,
                "agent": agent.__class__.__name__,
                "input": current_input[:100] + "..." if len(current_input) > 100 else current_input,
                "output": str(stage_result)[:200] + "..." if len(str(stage_result)) > 200 else str(stage_result)
            }
            pipeline_results.append(stage_info)
            
            # Prepare input for next stage
            if isinstance(stage_result, dict):
                # Extract the main content from dict results
                if 'content' in stage_result:
                    current_input = stage_result['content']
                elif 'insights' in stage_result:
                    current_input = stage_result['insights']
                else:
                    current_input = json.dumps(stage_result)
            else:
                current_input = str(stage_result)
            
            print(f"  ✅ Stage {i+1} completed")
        
        return {
            "final_result": current_input,
            "pipeline_stages": len(self.pipeline),
            "stage_results": pipeline_results,
            "processing_complete": True
        }


async def main():
    """Demonstrate agent composition patterns."""
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Initialize LLM client
    config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.7)
    llm_client = OpenRouterClient(config)
    
    # Create memory manager
    memory_manager = SimpleMemoryManager()
    
    # Create specialized agents
    print("=" * 60)
    print("CREATING SPECIALIZED AGENTS")
    print("=" * 60)
    
    agents = {
        'data_analyst': DataAnalysisAgent(llm_client),
        'report_generator': ReportGenerationAgent(llm_client),
        'knowledge_search': KnowledgeBaseAgent(llm_client),
        'memory_manager': WorkingMemoryAgent(llm_client, memory_manager=memory_manager)
    }
    
    for name, agent in agents.items():
        print(f"✅ Created {name}: {agent.__class__.__name__}")
    
    # Example 1: Orchestrator Agent Pattern
    print("\n" + "=" * 60)
    print("EXAMPLE 1: ORCHESTRATOR AGENT PATTERN")
    print("=" * 60)
    
    orchestrator = OrchestratorAgent(llm_client, agents)
    
    complex_requests = [
        "I need a comprehensive analysis of our customer satisfaction data and a report on the findings",
        "Search for sales information and create a summary report",
        "Find product information and analyze customer feedback patterns"
    ]
    
    for request in complex_requests:
        print(f"\n🎯 Complex Request: {request}")
        
        result = await orchestrator.process(IAgentInput(
            message=request,
            metadata={"conversation_id": "orchestrator_demo", "user_id": "demo_user"}
        ))
        
        print(f"🎭 Orchestrator Response: {result['orchestrator_response']}")
        print(f"⚙️  Available Agents: {result['agents_available']}")
        print(f"📊 Coordination Metadata: {result['coordination_metadata']}")
    
    # Example 2: Pipeline Agent Pattern
    print("\n" + "=" * 60)
    print("EXAMPLE 2: PIPELINE AGENT PATTERN")
    print("=" * 60)
    
    # Create a pipeline: Data Analysis → Report Generation
    pipeline_agents = [
        agents['data_analyst'],
        agents['report_generator']
    ]
    
    pipeline = PipelineAgent(llm_client, pipeline_agents)
    
    pipeline_inputs = [
        "Monthly sales data: January $120K, February $135K, March $142K, showing consistent growth",
        "Customer feedback scores: UI 4.2/5, Performance 3.8/5, Support 4.5/5, overall satisfaction trending upward"
    ]
    
    for pipeline_input in pipeline_inputs:
        print(f"\n🏭 Pipeline Input: {pipeline_input}")
        
        result = await pipeline.process(IAgentInput(
            message=pipeline_input,
            metadata={"pipeline_id": "demo_pipeline"}
        ))
        
        print(f"🏆 Final Result: {result['final_result'][:300]}...")
        print(f"📊 Pipeline Stats:")
        print(f"   Stages: {result['pipeline_stages']}")
        print(f"   Processing: {'✅' if result['processing_complete'] else '❌'}")
        
        # Show pipeline flow
        for stage in result['stage_results']:
            print(f"   Stage {stage['stage']} ({stage['agent']}): {stage['output'][:100]}...")
    
    # Example 3: Agent Mesh Pattern
    print("\n" + "=" * 60)
    print("EXAMPLE 3: AGENT MESH PATTERN")
    print("=" * 60)
    
    class MeshCoordinatorAgent(BaseAgent):
        """Agent that coordinates a mesh of interconnected agents."""
        
        def __init__(self, llm_client: ILLM, agent_mesh: Dict[str, IAgent], **kwargs):
            system_prompt = """You coordinate a mesh of interconnected agents.
            Agents can communicate with each other to solve complex problems collaboratively."""
            super().__init__(llm_client, system_prompt, **kwargs)
            self.mesh = agent_mesh
        
        async def process(self, input: IAgentInput) -> Dict[str, Any]:
            """Coordinate agents in a mesh pattern."""
            
            # Agents can call each other
            async def get_analysis(data: str) -> str:
                """Get analysis from data analyst."""
                result = await self.mesh['analyst'].process(IAgentInput(message=data))
                return json.dumps(result)
            
            async def search_info(query: str) -> str:
                """Search knowledge base."""
                result = await self.mesh['searcher'].process(IAgentInput(message=query))
                return str(result)
            
            async def cross_reference(analysis: str, knowledge: str) -> str:
                """Cross-reference analysis with knowledge base."""
                combined = f"Analysis: {analysis}\nKnowledge: {knowledge}\nProvide cross-referenced insights:"
                result = await self.mesh['reporter'].process(IAgentInput(message=combined))
                return result['content']
            
            # All agents available to each other
            mesh_functions = {
                "get_analysis": get_analysis,
                "search_info": search_info,
                "cross_reference": cross_reference
            }
            
            llm_input = ILLMInput(
                system_prompt=self.system_prompt + f"\n\nMesh functions: {list(mesh_functions.keys())}",
                user_message=input.message,
                regular_functions=mesh_functions
            )
            
            result = await self.llm_client.chat(llm_input)
            
            return {
                "mesh_response": result.get('llm_response', ''),
                "mesh_agents": list(self.mesh.keys()),
                "interconnected": True
            }
    
    # Create agent mesh
    mesh_agents = {
        'analyst': agents['data_analyst'],
        'searcher': agents['knowledge_search'],
        'reporter': agents['report_generator']
    }
    
    mesh_coordinator = MeshCoordinatorAgent(llm_client, mesh_agents)
    
    mesh_requests = [
        "I need to understand customer satisfaction trends and cross-reference with our product knowledge base",
        "Analyze recent sales performance and correlate with available market research"
    ]
    
    for request in mesh_requests:
        print(f"\n🕸️  Mesh Request: {request}")
        
        result = await mesh_coordinator.process(IAgentInput(message=request))
        
        print(f"🤝 Mesh Response: {result['mesh_response']}")
        print(f"🔗 Interconnected Agents: {result['mesh_agents']}")
        print(f"🌐 Mesh Status: {'✅' if result['interconnected'] else '❌'}")
    
    # Example 4: Agent Factory Pattern
    print("\n" + "=" * 60)
    print("EXAMPLE 4: DYNAMIC AGENT CREATION")
    print("=" * 60)
    
    class AgentFactory:
        """Factory for creating agents dynamically."""
        
        def __init__(self, llm_client: ILLM):
            self.llm_client = llm_client
            self.agent_cache = {}
        
        def create_specialist_agent(self, specialty: str) -> IAgent:
            """Create a specialist agent for a specific domain."""
            
            if specialty in self.agent_cache:
                return self.agent_cache[specialty]
            
            class SpecialistAgent(BaseAgent):
                def __init__(self, llm_client, specialty):
                    prompt = f"You are a {specialty} specialist. Provide expert advice and analysis in this domain."
                    super().__init__(llm_client, prompt)
                    self.specialty = specialty
                
                async def process(self, input: IAgentInput) -> str:
                    llm_input = ILLMInput(
                        system_prompt=self.system_prompt,
                        user_message=f"As a {self.specialty} expert, help with: {input.message}"
                    )
                    result = await self.llm_client.chat(llm_input)
                    return result.get('llm_response', '')
            
            agent = SpecialistAgent(self.llm_client, specialty)
            self.agent_cache[specialty] = agent
            return agent
    
    factory = AgentFactory(llm_client)
    
    # Dynamically create specialists
    specialties = ["cybersecurity", "marketing", "finance"]
    
    for specialty in specialties:
        print(f"\n🏭 Creating {specialty} specialist...")
        specialist = factory.create_specialist_agent(specialty)
        
        test_query = f"What are the key trends in {specialty} for 2024?"
        result = await specialist.process(IAgentInput(message=test_query))
        
        print(f"🎓 {specialty.title()} Expert: {result[:200]}...")
    
    print(f"\n📊 Factory Stats: {len(factory.agent_cache)} agents cached")
    
    print("\n✅ Agent composition examples completed!")
    print("\nKey Composition Patterns Demonstrated:")
    print("• 🎭 Orchestrator: Master agent coordinates specialists")
    print("• 🏭 Pipeline: Sequential processing through agents")
    print("• 🕸️  Mesh: Interconnected agents that communicate")
    print("• 🏭 Factory: Dynamic agent creation for specialization")
    print("• 🤝 All patterns leverage LLM function calling for coordination")


if __name__ == "__main__":
    asyncio.run(main())