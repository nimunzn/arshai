"""
Quick Start Guide for Arshai Agents
====================================

This is a minimal example to get you started with agents in 5 minutes.

For more examples, see:
- agents_comprehensive_guide.py - Single-file comprehensive tutorial  
- 01_basic_usage.py through 06_testing_agents.py - Detailed topic-specific examples
"""

import os
import asyncio
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMConfig
from arshai.agents.base import BaseAgent
from arshai.llms.openrouter import OpenRouterClient


# Step 1: Create your custom agent
class MyFirstAgent(BaseAgent):
    """Your first custom agent - it's this simple!"""
    
    async def process(self, input: IAgentInput) -> str:
        """Process user input and return a response."""
        from arshai.core.interfaces.illm import ILLMInput
        
        # Prepare the LLM request
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        # Get response from LLM
        result = await self.llm_client.chat(llm_input)
        
        # Return the response
        return result.get('llm_response', 'No response')


async def main():
    """Quick start example - get an agent running in seconds."""
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  Please set OPENROUTER_API_KEY environment variable")
        print("   export OPENROUTER_API_KEY=your_key_here")
        return
    
    # Configure LLM (using OpenRouter with GPT-4o Mini)
    config = ILLMConfig(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        max_tokens=150
    )
    
    # Initialize LLM client
    llm_client = OpenRouterClient(config)
    
    # Create your agent
    agent = MyFirstAgent(
        llm_client=llm_client,
        system_prompt="You are a helpful AI assistant. Be concise and friendly."
    )
    
    # Use your agent
    print("🤖 Agent is ready! Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        user_message = input("You: ")
        
        if user_message.lower() == 'quit':
            break
        
        # Process with agent
        agent_input = IAgentInput(message=user_message)
        response = await agent.process(agent_input)
        
        # Display response
        print(f"Agent: {response}\n")
    
    print("Goodbye! 👋")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())