"""
Simple agent for testing purposes.
A basic wrapper over LLM client that provides straightforward message processing.
"""

from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput


class SimpleAgent(BaseAgent):
    """
    Simple agent that wraps LLM client for basic message processing.
    
    This agent provides a straightforward implementation for testing:
    - Takes a message and returns a simple string response
    - Uses the provided system prompt
    - No tools, memory, or complex functionality
    """
    
    async def process(self, input: IAgentInput) -> str:
        """
        Process the input message and return a simple text response.
        
        Args:
            input: Input containing the user message
            
        Returns:
            str: The LLM's text response
        """
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=input.message
        )
        
        result = await self.llm_client.chat(llm_input)
        
        # Extract response from the result dictionary
        response = result.get('llm_response', '')
        
        # Handle different response types
        if isinstance(response, str):
            return response
        else:
            return str(response)