"""
Example 1: Basic Agent Usage
============================

This example demonstrates the simplest way to create and use an agent in Arshai.
Perfect for getting started quickly.

Prerequisites:
- Set OPENROUTER_API_KEY environment variable
- Install arshai package
"""

import os
import asyncio
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput, ILLMConfig
from arshai.llms.openrouter import OpenRouterClient


class SimpleAgent(BaseAgent):
    """A basic agent that processes messages and returns responses."""
    
    async def process(self, input: IAgentInput) -> str:
        """
        Process user input and return a string response.
        
        This is the simplest possible agent implementation.
        """
        # Create LLM input
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        # Get response from LLM
        result = await self.llm_client.chat(llm_input)
        
        # Return the response
        return result.get('llm_response', 'No response generated')


async def main():
    """Demonstrate basic agent usage."""
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  Please set OPENROUTER_API_KEY environment variable")
        print("   export OPENROUTER_API_KEY=your_key_here")
        return
    
    print("=" * 60)
    print("EXAMPLE 1: Basic Agent Usage")
    print("=" * 60)
    
    # Step 1: Configure LLM client
    llm_config = ILLMConfig(
        model="openai/gpt-4o-mini",  # Using OpenRouter with GPT-4o Mini
        temperature=0.7,              # Balanced creativity
        max_tokens=150                # Reasonable response length
    )
    
    # Step 2: Initialize LLM client
    print("\n🔄 Initializing OpenRouter client...")
    llm_client = OpenRouterClient(llm_config)
    
    # Step 3: Create agent
    print("🤖 Creating simple agent...")
    agent = SimpleAgent(
        llm_client=llm_client,
        system_prompt="You are a helpful AI assistant. Be concise and friendly."
    )
    
    # Step 4: Test the agent with various inputs
    test_messages = [
        "Hello! How are you today?",
        "What is the capital of France?",
        "Can you explain what an AI agent is in simple terms?"
    ]
    
    print("\n" + "-" * 40)
    for message in test_messages:
        print(f"\n👤 User: {message}")
        
        # Create input
        agent_input = IAgentInput(message=message)
        
        # Process with agent
        response = await agent.process(agent_input)
        
        print(f"🤖 Agent: {response}")
    
    print("\n" + "=" * 60)
    print("✅ Basic agent example completed!")
    
    # Demonstrate metadata usage
    print("\n" + "=" * 60)
    print("BONUS: Using Metadata")
    print("=" * 60)
    
    # Input with metadata
    input_with_metadata = IAgentInput(
        message="Tell me about Python",
        metadata={
            "user_id": "user_123",
            "session_id": "session_456",
            "max_length": 100
        }
    )
    
    print(f"\n📋 Input with metadata: {input_with_metadata.metadata}")
    response = await agent.process(input_with_metadata)
    print(f"🤖 Agent: {response}")
    
    print("\n✨ Metadata can be used to pass context without changing the API!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())