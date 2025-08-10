"""
Working Memory Agent for the Arshai framework.

This module provides an agent specialized in managing conversation working memory.
It handles fetching conversation history, updating memory in storage systems,
and generating memory summaries.
"""

from typing import Dict, Any, Optional
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLM, ILLMInput
from arshai.core.interfaces.imemorymanager import IMemoryManager
from arshai.utils.logging import get_logger

logger = get_logger(__name__)


class WorkingMemoryAgent(BaseAgent):
    """
    Agent specialized in managing conversation working memory.
    
    This agent is responsible for maintaining and updating the working memory
    of conversations. It fetches conversation history, generates summaries,
    and stores updated memory in persistent storage.
    
    Capabilities:
    - Fetches conversation history from chat history storage
    - Retrieves current working memory from storage (e.g., Redis)
    - Generates revised working memory based on new interactions
    - Stores updated memory for future reference
    
    Requirements:
    - Memory manager for storage operations
    - Chat history client for conversation retrieval (optional)
    
    Example:
        memory_manager = RedisMemoryManager(redis_client)
        agent = WorkingMemoryAgent(
            llm_client,
            "You are a memory management assistant",
            memory_manager=memory_manager
        )
        
        # Update memory for a conversation
        result = await agent.process(IAgentInput(
            message="User asked about product pricing", 
            metadata={"conversation_id": "123"}
        ))
        # Returns "success" if successful, or "error: <description>" if failed
    """
    
    def __init__(
        self, 
        llm_client: ILLM, 
        system_prompt: str = None,
        memory_manager: IMemoryManager = None,
        chat_history_client: Any = None,
        **kwargs
    ):
        """
        Initialize the working memory agent.
        
        Args:
            llm_client: The LLM client for generating memory updates
            system_prompt: Optional custom prompt (uses default if not provided)
            memory_manager: Memory manager for storage operations
            chat_history_client: Optional client for fetching conversation history
            **kwargs: Additional configuration
        """
        # Use default memory management prompt if not provided
        if system_prompt is None:
            system_prompt = """You are a memory management assistant responsible for maintaining conversation context.

Your tasks:
1. Analyze conversation history and current interaction
2. Extract key information, facts, and context
3. Generate a concise working memory summary
4. Focus on information relevant for future interactions

Keep the memory:
- Concise but comprehensive
- Focused on actionable information
- Updated with latest context
- Free from redundancy"""
        
        super().__init__(llm_client, system_prompt, **kwargs)
        self.memory_manager = memory_manager
        self.chat_history = chat_history_client
    
    async def process(self, input: IAgentInput) -> str:
        """
        Process memory update request.
        
        This method:
        1. Extracts conversation_id from metadata
        2. Fetches current memory and history
        3. Generates updated memory using LLM
        4. Stores the updated memory
        
        Args:
            input: Input containing the new interaction and metadata with conversation_id
            
        Returns:
            str: Status of the operation - "success" if successful, 
                 "error: <description>" if failed, or None if no conversation_id
        """
        # Extract conversation ID from metadata
        conversation_id = input.metadata.get("conversation_id") if input.metadata else None
        
        if not conversation_id:
            # No conversation ID, nothing to do for memory management
            logger.warning("WorkingMemoryAgent: No conversation_id provided in metadata, skipping memory update")
            return "error: no conversation_id provided"
        
        # Fetch current working memory if available
        current_memory = ""
        if self.memory_manager:
            try:
                memory_data = await self.memory_manager.retrieve({"conversation_id": conversation_id})
                if memory_data:
                    current_memory = memory_data[0].working_memory if hasattr(memory_data[0], 'working_memory') else str(memory_data[0])
                    logger.debug(f"WorkingMemoryAgent: Retrieved existing memory for conversation {conversation_id}")
                else:
                    logger.debug(f"WorkingMemoryAgent: No existing memory found for conversation {conversation_id}")
            except Exception as e:
                # If retrieval fails, continue without current memory
                logger.warning(f"WorkingMemoryAgent: Failed to retrieve memory for conversation {conversation_id}: {e}")
                pass
        
        # Fetch conversation history if available
        conversation_history = ""
        if self.chat_history:
            try:
                history = await self.chat_history.get(conversation_id)
                if history:
                    conversation_history = str(history)
                    logger.debug(f"WorkingMemoryAgent: Retrieved conversation history for {conversation_id}")
                else:
                    logger.debug(f"WorkingMemoryAgent: No conversation history found for {conversation_id}")
            except Exception as e:
                # If history retrieval fails, continue without it
                logger.warning(f"WorkingMemoryAgent: Failed to retrieve conversation history for {conversation_id}: {e}")
                pass
        
        # Prepare context for memory update
        context = f"""
Current Working Memory:
{current_memory if current_memory else "No existing memory"}

Conversation History:
{conversation_history if conversation_history else "No previous history"}

New Interaction:
{input.message}

Please generate an updated working memory that incorporates the new information while maintaining relevant context from the existing memory.
"""
        
        # Generate updated memory using LLM
        logger.debug(f"WorkingMemoryAgent: Generating updated memory for conversation {conversation_id}")
        
        try:
            llm_input = ILLMInput(
                system_prompt=self.system_prompt,
                user_message=context
            )
            
            result = await self.llm_client.chat(llm_input)
            updated_memory = result.get('llm_response', '')
            logger.debug(f"Updated memory: {updated_memory}")

            if updated_memory and updated_memory.strip():
                logger.info(f"WorkingMemoryAgent: Successfully generated updated memory for conversation {conversation_id}")
            else:
                logger.warning(f"WorkingMemoryAgent: LLM returned empty response for conversation {conversation_id}")
                return "error: empty memory response"
                
        except Exception as e:
            logger.error(f"WorkingMemoryAgent: Failed to generate updated memory for conversation {conversation_id}: {e}")
            return f"error: {str(e)}"
        
        # Store updated memory if memory manager is available
        if self.memory_manager and conversation_id:
            try:
                await self.memory_manager.store({
                    "conversation_id": conversation_id,
                    "working_memory": updated_memory,
                    "metadata": input.metadata
                })
                logger.info(f"WorkingMemoryAgent: Successfully stored updated memory for conversation {conversation_id}")
            except Exception as e:
                # Log error and return error status
                logger.error(f"WorkingMemoryAgent: Failed to store memory for conversation {conversation_id}: {e}")
                return f"error: storage failed - {str(e)}"
        elif not self.memory_manager:
            logger.warning("WorkingMemoryAgent: No memory manager configured, cannot store updated memory")
            # Still return success if memory was generated, just warn about storage
        
        logger.debug(f"WorkingMemoryAgent: Completed memory update task for conversation {conversation_id}")
        
        # Return success status
        return "success"