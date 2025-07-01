from typing import Dict, List, Optional, Any, Tuple, TypeVar, AsyncGenerator, Union
import json
from arshai.core.interfaces.illm import ILLMInput, LLMInputType
from arshai.core.interfaces.imemorymanager import IWorkingMemory, ConversationMemoryType, IMemoryInput
from arshai.core.interfaces.iagent import IAgent, IAgentOutput, IAgentInput, IAgentConfig
from arshai.core.interfaces.isetting import ISetting
from arshai.prompts.human_like_conversation import HUMAN_LIKE_CONVERSATION_PROMPT
from arshai.prompts.working_memory import MEMORY_PROMPT
from arshai.prompts.human_intervention import HUMAN_INTERVENTION_PROMPT
from arshai.prompts.general import TOOL_USAGE_PROMPT, STRUCTURED_OUTPUT_PROMPT
from arshai.prompts.guardrails import GENERAL_GUARDRAILS_PROMPT, MEMORY_GUARDRAILS_PROMPT, CONTEXT_GUARDRAILS_PROMPT
from datetime import datetime
import traceback
from arshai.utils.logging import get_logger

T = TypeVar('T')

logger = get_logger(__name__)


class ConversationAgent(IAgent):
    """Agent responsible for handling user interactions and managing conversation flow"""

    def __init__(self, config: IAgentConfig, settings: ISetting):
        """Initialize the agent with the given configuration

        Args:
            config: Configuration containing task context and available tools
        """
        self.settings = settings
        self.llm = self.settings.create_llm()
        self.context_manager = self.settings.create_memory_manager()
        self.task_context = config.get("task_context")
        self.available_tools = config.get("tools")
        self.output_structure = config.get("output_structure")

        logger.info(f"Available tools: {self.available_tools}")

    def _get_function_description(self) -> List[Dict]:
        """
        Get list of all available function definitions from configured tools.
        """
        return [tool.function_definition for tool in self.available_tools]

    def _get_callable_functions(self) -> Dict:
        """
        Get implementations of all available functions from configured tools.
        """
        return {
            tool.function_definition["name"]: tool.aexecute
            for tool in self.available_tools
        }

    def _prepare_system_prompt(self, working_memory: IWorkingMemory) -> str:
        """Prepare system prompt with context and response structure"""
        prompt_start = datetime.now()
        model_schema = self.output_structure.model_json_schema()
        # logger.info(f"Model schema: {model_schema}")

        # Build the base system prompt
        base_prompt = f"""
        You are a conversational AI Conversation Agent who speaks with users in the same language they use. You are friendly, approachable, and human-like, with a deep understanding of conversation context. You maintain dynamic memory to guide natural, adaptive, and emotionally aware interactions.
        You are responsible for managing the conversation flow and Assist the user with their questions and requests.

        ### YOUR TASK, IDENTITY AND ROLE:
        {self.task_context}

        ### CURRENT TIME:
        {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        ### CRITICAL REQUIREMENTS:
        - Maintain strict focus on the domain and task specified in your identity
        """
        
        # Add working memory prompt
        base_prompt += "\n\n" + MEMORY_PROMPT(working_memory.working_memory)

        # Add human-like conversation guidelines
        base_prompt += "\n\n" + HUMAN_LIKE_CONVERSATION_PROMPT()

        # Add tool usage requirements if tools are available
        if self.available_tools or self.output_structure:
            base_prompt += "\n\n" + TOOL_USAGE_PROMPT()

        # Add human intervention prompt if enabled in config
        if self.settings.get("human_intervention", {}).get("enabled", False):
            base_prompt += "\n\n" + HUMAN_INTERVENTION_PROMPT()

        # Add structured output requirements
        if model_schema:
            base_prompt += "\n\n" + STRUCTURED_OUTPUT_PROMPT(json.dumps(model_schema, indent=2))

        # Add guardrails
        base_prompt += "\n\n" + GENERAL_GUARDRAILS_PROMPT() + "\n\n" + MEMORY_GUARDRAILS_PROMPT() + "\n\n" + CONTEXT_GUARDRAILS_PROMPT()

        logger.info(f"Prompt prepared in {(datetime.now() - prompt_start).total_seconds()}s")
        return base_prompt

    async def _get_llm_response(self, system_prompt: str, user_message: str) -> Any:
        """Get response from LLM with tools"""
        llm_start = datetime.now()

        logger.debug(f"callable functions: {self._get_callable_functions()}")
        logger.debug(f"function description: {self._get_function_description()}")

        llm_output = await self.llm.chat_with_tools(
            input=ILLMInput(
                llm_type=LLMInputType.CHAT_WITH_TOOLS,
                system_prompt=system_prompt,
                user_message=user_message,
                tools_list=self._get_function_description(),
                callable_functions=self._get_callable_functions(),
                structure_type=self.output_structure
            )
        )
        llm_response = llm_output.get("llm_response")
        llm_usage = llm_output.get("usage")
        logger.debug(f"LLM response: {llm_response}")
        logger.info(f"LLM response received in {(datetime.now() - llm_start).total_seconds()}s")

        if isinstance(llm_response, str):
            raise Exception(llm_response)

        return llm_response, llm_usage
    
    async def _stream_llm_response(self, system_prompt: str, user_message: str) -> AsyncGenerator[Tuple[str, Dict], None]:
        """Get response from LLM with tools"""
        llm_start = datetime.now()

        logger.debug(f"callable functions: {self._get_callable_functions()}")
        logger.debug(f"function description: {self._get_function_description()}")

        llm_input=ILLMInput(
                llm_type=LLMInputType.CHAT_WITH_TOOLS,
                system_prompt=system_prompt,
                user_message=user_message,
                tools_list=self._get_function_description(),
                callable_functions=self._get_callable_functions(),
            structure_type=self.output_structure
        )
        async for llm_output in self.llm.stream_with_tools(input=llm_input):
            llm_response = llm_output.get("llm_response")
            llm_usage = llm_output.get("usage")
            yield llm_response, llm_usage

        logger.info(f"LLM response received in {(datetime.now() - llm_start).total_seconds()}s")


    async def stream_process_message(
        self,
        input: IAgentInput,
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Async process incoming message and generate streaming response.
        """
        start_time = datetime.now()
        logger.info(f"Starting message processing for conversation: {input.conversation_id}")

        try:
            # Load conversation context
            context_start = datetime.now()

            working_memory = self.context_manager.retrieve_working_memory(input.conversation_id)
            logger.debug(f"Working memory data: {working_memory}")
           
            logger.info(f"Working memory loaded in {(datetime.now() - context_start).total_seconds()}s")

            # Prepare system prompt
            system_prompt = self._prepare_system_prompt(working_memory=working_memory)

            updated_working_memory = None
            async for llm_response, llm_usage in self._stream_llm_response(system_prompt, input.message):
                if isinstance(llm_response, str):
                    yield llm_response, llm_usage
                if isinstance(llm_response, dict):                    # Handle structured response
                    if 'agent_message' in llm_response:
                        yield llm_response['agent_message'], llm_usage
                    if 'memory' in llm_response:
                        updated_working_memory = llm_response['memory']  

            yield llm_response, llm_usage
            
            if updated_working_memory and input.conversation_id:
                save_start = datetime.now()
                self.context_manager.store_working_memory(input.conversation_id, updated_working_memory)
                logger.info(f"Working memory saved in {(datetime.now() - save_start).total_seconds()}s")

            total_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Total processing time: {total_duration}s")

        except Exception as e:
            logger.error(f"Error streaming message: {str(e)}")
            logger.error(traceback.format_exc())
            if hasattr(self, '_current_context'):
                delattr(self, '_current_context')
            raise

    async def process_message(
            self,
            input: IAgentInput,
    ) -> Tuple[IAgentOutput, str]:
        """Process incoming user message and generate response

        Args:
            input: IAgentInput containing message and conversation details

        Returns:
            Tuple[IAgentOutput, str]: Response and message type
        """
        start_time = datetime.now()
        logger.info(f"Starting message processing for conversation: {input.conversation_id}")

        try:
            # Load conversation context
            context_start = datetime.now()

            working_memory = self.context_manager.retrieve_working_memory(input.conversation_id)
            logger.debug(f"Working memory data: {working_memory}")
           
            logger.debug(f"Working memory loaded in {(datetime.now() - context_start).total_seconds()}s")

            # Prepare system prompt
            system_prompt = self._prepare_system_prompt(working_memory=working_memory)

            # Get LLM response
            llm_response, llm_usage = await self._get_llm_response(system_prompt, input.message)

            # Extract context and response
            updated_working_memory = llm_response.memory
            logger.debug(
                f"Updated working memory: {json.dumps(updated_working_memory.model_dump(), ensure_ascii=False)}")
            agent_response = llm_response.agent_message
            logger.debug(f"Agent response: {agent_response}")
            # Save updated context
            if updated_working_memory and input.conversation_id:
                save_start = datetime.now()
                self.context_manager.store_working_memory(input.conversation_id, updated_working_memory)
                logger.info(f"Working memory saved in {(datetime.now() - save_start).total_seconds()}s")

            total_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Total processing time: {total_duration}s")
            return agent_response, llm_usage

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            if hasattr(self, '_current_context'):
                delattr(self, '_current_context')
            raise
