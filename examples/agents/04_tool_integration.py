"""
Example 4: Tool Integration with Agents
========================================

This example demonstrates how agents can integrate with tools and external functions.
Shows both regular functions and background tasks.

Prerequisites:
- Set OPENROUTER_API_KEY environment variable
- Install arshai package
"""

import os
import asyncio
import math
import random
from typing import Dict, Any, List
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput, ILLMConfig, ILLM
from arshai.llms.openrouter import OpenRouterClient


class CalculatorAgent(BaseAgent):
    """
    Agent with mathematical calculation capabilities.
    
    Capabilities:
    - Basic arithmetic operations
    - Advanced mathematical functions
    - Statistical calculations
    
    Returns:
        Dict[str, Any]: Response with calculation results and metadata
    """
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Process mathematical queries with tool support."""
        
        # Define calculation tools
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b
        
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b
        
        def divide(a: float, b: float) -> float:
            """Divide two numbers."""
            if b == 0:
                return float('inf')  # Handle division by zero
            return a / b
        
        def sqrt(x: float) -> float:
            """Calculate square root."""
            if x < 0:
                return float('nan')  # Handle negative input
            return math.sqrt(x)
        
        def power(base: float, exponent: float) -> float:
            """Calculate base raised to exponent."""
            return math.pow(base, exponent)
        
        def factorial(n: int) -> int:
            """Calculate factorial of n."""
            if n < 0:
                return -1  # Invalid input
            return math.factorial(min(n, 170))  # Prevent overflow
        
        # Prepare tools for LLM
        tools = {
            "add": add,
            "multiply": multiply,
            "divide": divide,
            "sqrt": sqrt,
            "power": power,
            "factorial": factorial
        }
        
        # Create LLM input with tools
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions=tools
        )
        
        # Process with LLM and tools
        result = await self.llm_client.chat(llm_input)
        
        return {
            "response": result.get('llm_response', ''),
            "tools_available": list(tools.keys()),
            "usage": result.get('usage', {})
        }


class ResearchAgent(BaseAgent):
    """
    Agent with research and data analysis capabilities.
    
    Capabilities:
    - Mock web search
    - Data analysis
    - Report generation
    - Background logging
    
    Returns:
        Dict[str, Any]: Research results with citations and metadata
    """
    
    def __init__(self, llm_client: ILLM, **kwargs):
        """Initialize research agent with specialized prompt."""
        system_prompt = """You are a research assistant with access to search and analysis tools.
        Help users find information, analyze data, and generate reports.
        Always cite your sources and provide structured responses."""
        
        super().__init__(llm_client, system_prompt, **kwargs)
        self.research_log = []
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Process research requests with tools and background tasks."""
        
        # Regular tools (return results to conversation)
        def search_web(query: str) -> str:
            """Search the web for information."""
            # Mock search results
            mock_results = [
                f"Article: '{query}' - Comprehensive overview from Wikipedia",
                f"Research paper: 'Analysis of {query}' - Academic Journal 2024",
                f"News: Recent developments in {query} - Tech News Today"
            ]
            return " | ".join(mock_results)
        
        def analyze_data(data_description: str) -> Dict[str, Any]:
            """Analyze provided data description."""
            # Mock analysis
            return {
                "data_type": data_description,
                "sample_size": random.randint(100, 1000),
                "key_findings": ["Significant correlation found", "Outliers detected", "Trend confirmed"],
                "confidence": random.randint(80, 95)
            }
        
        def get_citations(topic: str) -> List[str]:
            """Get academic citations for a topic."""
            return [
                f"Smith, J. (2024). {topic}: A Comprehensive Study. Journal of Science.",
                f"Johnson, A. (2023). Understanding {topic}. Academic Press.",
                f"Williams, B. (2024). {topic} in Modern Context. Research Quarterly."
            ]
        
        # Background tasks (fire-and-forget)
        async def log_research_activity(query: str, user_id: str = "unknown") -> None:
            """Log research activity for analytics."""
            log_entry = {
                "query": query,
                "user_id": user_id,
                "timestamp": "2024-current-time"  # Mock timestamp
            }
            self.research_log.append(log_entry)
            print(f"  📊 [BACKGROUND] Logged research activity: {query}")
        
        async def generate_usage_report(activity: str) -> None:
            """Generate usage statistics."""
            print(f"  📈 [BACKGROUND] Generating usage report for: {activity}")
            # In real implementation, this might update a database or send metrics
        
        # Extract user info from metadata
        user_id = input.metadata.get("user_id", "anonymous") if input.metadata else "anonymous"
        
        # Prepare regular functions and background tasks
        regular_functions = {
            "search_web": search_web,
            "analyze_data": analyze_data,
            "get_citations": get_citations
        }
        
        background_tasks = {
            "log_research_activity": lambda q: log_research_activity(q, user_id),
            "generate_usage_report": generate_usage_report
        }
        
        # Create LLM input with both types of functions
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message,
            regular_functions=regular_functions,
            background_tasks=background_tasks
        )
        
        # Process request
        result = await self.llm_client.chat(llm_input)
        
        return {
            "response": result.get('llm_response', ''),
            "research_tools": list(regular_functions.keys()),
            "background_services": list(background_tasks.keys()),
            "user_id": user_id,
            "usage": result.get('usage', {})
        }
    
    def get_research_log(self) -> List[Dict[str, Any]]:
        """Get the research activity log."""
        return self.research_log.copy()


class MultiToolAgent(BaseAgent):
    """
    Agent with multiple tool categories for complex tasks.
    
    Capabilities:
    - File operations (mock)
    - Network requests (mock)
    - Data processing
    - System monitoring
    
    Returns:
        Dict[str, Any]: Complete task results with tool usage summary
    """
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Process complex requests with multiple tool categories."""
        
        # File operations
        def read_file(filename: str) -> str:
            """Read file content."""
            return f"Mock content of {filename}: Lorem ipsum dolor sit amet..."
        
        def write_file(filename: str, content: str) -> bool:
            """Write content to file."""
            print(f"  📄 [TOOL] Writing to {filename}: {content[:50]}...")
            return True
        
        def list_files(directory: str = ".") -> List[str]:
            """List files in directory."""
            return ["file1.txt", "file2.py", "data.json", "config.yaml"]
        
        # Network operations
        def fetch_url(url: str) -> Dict[str, Any]:
            """Fetch data from URL."""
            return {
                "url": url,
                "status": 200,
                "content_length": 1024,
                "content_preview": "Mock response data..."
            }
        
        def send_notification(recipient: str, message: str) -> bool:
            """Send notification."""
            print(f"  📧 [TOOL] Notification sent to {recipient}: {message}")
            return True
        
        # Data processing
        def process_json(json_data: str) -> Dict[str, Any]:
            """Process JSON data."""
            return {
                "processed": True,
                "records_found": random.randint(10, 100),
                "processing_time": f"{random.uniform(0.1, 2.0):.2f}s"
            }
        
        def validate_data(data_type: str, data: str) -> Dict[str, Any]:
            """Validate data format."""
            return {
                "data_type": data_type,
                "valid": True,
                "errors": [],
                "warnings": ["Minor formatting inconsistency"]
            }
        
        # System monitoring
        def check_system_status() -> Dict[str, Any]:
            """Check system status."""
            return {
                "cpu_usage": f"{random.randint(10, 80)}%",
                "memory_usage": f"{random.randint(30, 70)}%",
                "disk_space": f"{random.randint(40, 90)}%",
                "status": "healthy"
            }
        
        # Background monitoring
        async def monitor_performance(task_type: str) -> None:
            """Monitor task performance."""
            print(f"  🔍 [BACKGROUND] Monitoring performance for: {task_type}")
        
        async def update_metrics(operation: str, success: bool = True) -> None:
            """Update operational metrics."""
            status = "success" if success else "failure"
            print(f"  📊 [BACKGROUND] Updated metrics: {operation} - {status}")
        
        # Organize tools by category
        file_tools = {
            "read_file": read_file,
            "write_file": write_file,
            "list_files": list_files
        }
        
        network_tools = {
            "fetch_url": fetch_url,
            "send_notification": send_notification
        }
        
        data_tools = {
            "process_json": process_json,
            "validate_data": validate_data
        }
        
        system_tools = {
            "check_system_status": check_system_status
        }
        
        # Combine all regular tools
        all_tools = {**file_tools, **network_tools, **data_tools, **system_tools}
        
        # Background tasks
        background_tasks = {
            "monitor_performance": monitor_performance,
            "update_metrics": update_metrics
        }
        
        # Create enhanced system prompt
        enhanced_prompt = f"""{self.system_prompt}
        
        You have access to the following tool categories:
        - File Operations: {list(file_tools.keys())}
        - Network Operations: {list(network_tools.keys())}
        - Data Processing: {list(data_tools.keys())}
        - System Monitoring: {list(system_tools.keys())}
        
        Use these tools appropriately to complete user requests."""
        
        # Create LLM input
        llm_input = ILLMInput(
            system_prompt=enhanced_prompt,
            user_message=input.message,
            regular_functions=all_tools,
            background_tasks=background_tasks
        )
        
        # Process request
        result = await self.llm_client.chat(llm_input)
        
        return {
            "response": result.get('llm_response', ''),
            "tool_categories": {
                "file_operations": list(file_tools.keys()),
                "network_operations": list(network_tools.keys()),
                "data_processing": list(data_tools.keys()),
                "system_monitoring": list(system_tools.keys())
            },
            "background_services": list(background_tasks.keys()),
            "total_tools": len(all_tools),
            "usage": result.get('usage', {})
        }


async def main():
    """Demonstrate tool integration patterns with agents."""
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Initialize LLM client
    config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.3)
    llm_client = OpenRouterClient(config)
    
    # Example 1: Calculator Agent
    print("=" * 60)
    print("CALCULATOR AGENT WITH MATHEMATICAL TOOLS")
    print("=" * 60)
    
    calc_agent = CalculatorAgent(
        llm_client=llm_client,
        system_prompt="You are a mathematical assistant with access to calculation tools. Help users solve math problems."
    )
    
    math_problems = [
        "What is 15 + 27?",
        "Calculate the square root of 144",
        "What is 2 to the power of 10?",
        "Find the factorial of 6"
    ]
    
    for problem in math_problems:
        print(f"\n🧮 Problem: {problem}")
        result = await calc_agent.process(IAgentInput(message=problem))
        print(f"🤖 Response: {result['response']}")
        print(f"🔧 Tools used: {result.get('tools_available', [])}")
    
    # Example 2: Research Agent with Background Tasks
    print("\n" + "=" * 60)
    print("RESEARCH AGENT WITH TOOLS AND BACKGROUND TASKS")
    print("=" * 60)
    
    research_agent = ResearchAgent(llm_client)
    
    research_queries = [
        "Find information about artificial intelligence trends in 2024",
        "Research the impact of renewable energy on the economy"
    ]
    
    for query in research_queries:
        print(f"\n🔍 Research Query: {query}")
        result = await research_agent.process(IAgentInput(
            message=query,
            metadata={"user_id": "researcher_001"}
        ))
        print(f"📝 Response: {result['response']}")
        print(f"🔧 Research Tools: {result.get('research_tools', [])}")
        print(f"⚙️ Background Services: {result.get('background_services', [])}")
    
    # Check research log
    print(f"\n📊 Research Activity Log: {research_agent.get_research_log()}")
    
    # Example 3: Multi-Tool Agent
    print("\n" + "=" * 60)
    print("MULTI-TOOL AGENT WITH COMPLEX CAPABILITIES")
    print("=" * 60)
    
    multi_agent = MultiToolAgent(
        llm_client=llm_client,
        system_prompt="You are a system administrator with access to various operational tools."
    )
    
    complex_requests = [
        "Check the system status and create a summary report",
        "Process the data in config.json and validate its format"
    ]
    
    for request in complex_requests:
        print(f"\n⚙️ Complex Request: {request}")
        result = await multi_agent.process(IAgentInput(message=request))
        print(f"🤖 Response: {result['response']}")
        print(f"📦 Tool Categories Available:")
        for category, tools in result.get('tool_categories', {}).items():
            print(f"   {category}: {tools}")
        print(f"📊 Total Tools: {result.get('total_tools', 0)}")
    
    # Example 4: Dynamic Tool Selection
    print("\n" + "=" * 60)
    print("DYNAMIC TOOL SELECTION PATTERN")
    print("=" * 60)
    
    class AdaptiveAgent(BaseAgent):
        """Agent that adapts its tools based on the request."""
        
        async def process(self, input: IAgentInput) -> Dict[str, Any]:
            """Dynamically select tools based on input analysis."""
            
            message = input.message.lower()
            tools = {}
            
            # Math tools for mathematical queries
            if any(word in message for word in ['calculate', 'math', 'number', 'sum']):
                def calculate(expression: str) -> float:
                    """Safe calculation."""
                    try:
                        return eval(expression.replace('^', '**'))  # Basic safety
                    except:
                        return 0
                tools['calculate'] = calculate
            
            # Text tools for text processing
            if any(word in message for word in ['text', 'word', 'sentence', 'analyze']):
                def count_words(text: str) -> int:
                    """Count words in text."""
                    return len(text.split())
                
                def analyze_text(text: str) -> Dict[str, Any]:
                    """Analyze text properties."""
                    return {
                        "characters": len(text),
                        "words": len(text.split()),
                        "sentences": text.count('.') + text.count('!') + text.count('?')
                    }
                
                tools.update({
                    'count_words': count_words,
                    'analyze_text': analyze_text
                })
            
            # Create LLM input with selected tools
            llm_input = ILLMInput(
                system_prompt=f"{self.system_prompt}\n\nAvailable tools for this request: {list(tools.keys())}",
                user_message=input.message,
                regular_functions=tools if tools else {}
            )
            
            result = await self.llm_client.chat(llm_input)
            
            return {
                "response": result.get('llm_response', ''),
                "selected_tools": list(tools.keys()),
                "tool_selection_reason": f"Based on keywords in: {message}"
            }
    
    adaptive_agent = AdaptiveAgent(
        llm_client=llm_client,
        system_prompt="You are an adaptive assistant that selects appropriate tools for each task."
    )
    
    adaptive_requests = [
        "Calculate 25 * 4 + 10",
        "Analyze the text: 'Hello world! How are you today?'",
        "Tell me about the weather"  # Should get no tools
    ]
    
    for request in adaptive_requests:
        print(f"\n🎯 Adaptive Request: {request}")
        result = await adaptive_agent.process(IAgentInput(message=request))
        print(f"🤖 Response: {result['response']}")
        print(f"🔧 Selected Tools: {result.get('selected_tools', [])}")
        print(f"💡 Selection Reason: {result.get('tool_selection_reason', 'N/A')}")
    
    print("\n✅ Tool integration examples completed!")
    print("\nKey Takeaways:")
    print("• Agents can use regular_functions for tools that return results")
    print("• background_tasks are for fire-and-forget operations")
    print("• Tools are defined as Python functions within the agent")
    print("• Agents can dynamically select tools based on input")
    print("• Complex agents can have multiple tool categories")
    print("• Background tasks enable monitoring and logging")


if __name__ == "__main__":
    asyncio.run(main())