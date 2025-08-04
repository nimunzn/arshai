"""
OpenRouter implementation of the LLM interface using openai SDK.
Supports both standard function calling and background tasks with manual tool orchestration.
Follows the same interface pattern as the Azure and OpenAI clients for consistency.
"""

import os
import logging
import json
import traceback
import time
import inspect
from typing import Dict, Any, TypeVar, Type, Union, AsyncGenerator, List, Tuple
from openai import OpenAI
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import (
    is_json_complete,
    standardize_usage_metadata,
    accumulate_usage_safely,
    parse_to_structure,
    build_enhanced_instructions,
)

T = TypeVar("T")

class OpenRouterClient(BaseLLMClient):
    """OpenRouter implementation of the LLM interface"""
    
    def __init__(self, config: ILLMConfig):
        """
        Initialize the OpenRouter client.
        
        Args:
            config: Configuration for the LLM
        """
        # Initialize base client (handles common setup)
        super().__init__(config)
    
    def _initialize_client(self) -> Any:
        """
        Initialize the OpenRouter client.
        
        OpenRouter uses an OpenAI-compatible API with custom base URL and headers.
        The client reads OPENROUTER_API_KEY from environment variables.
        
        Returns:
            OpenAI client instance configured for OpenRouter
            
        Raises:
            ValueError: If OPENROUTER_API_KEY is not set in environment variables
        """
        # Check if API key is available in environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.logger.error("OpenRouter API key not found in environment variables")
            raise ValueError(
                "OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable."
            )
        
        # Get optional site URL and app name for OpenRouter headers
        site_url = os.environ.get("OPENROUTER_SITE_URL", "")
        app_name = os.environ.get("OPENROUTER_APP_NAME", "arshai")
        
        # Create client with OpenRouter-specific configuration
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            }
        )
    
    def _python_function_to_openai_tool(self, func, name: str, function_type: str = "tool") -> Dict[str, Any]:
        """
        Convert a Python function to OpenAI tool format using function inspection.
        
        Args:
            func: Python callable function
            name: Function name
            function_type: "tool" for regular tools, "background_task" for background tasks
            
        Returns:
            Dictionary in OpenAI tool format
        """
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Extract description from docstring
            description = func.__doc__ or f"Execute {name} function"
            description = description.strip()
            
            # Enhance description for background tasks
            if function_type == "background_task":
                description = f"BACKGROUND TASK: {description}. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."
            
            # Build parameters schema
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                # Skip 'self' parameter
                if param_name == 'self':
                    continue
                    
                # Get parameter type
                param_type = "string"  # default
                if param.annotation != inspect.Parameter.empty:
                    param_type = self._python_type_to_json_schema_type(param.annotation)
                
                # Build parameter definition
                param_def = {
                    "type": param_type,
                    "description": f"{param_name} parameter"
                }
                
                # Add to required if no default value
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                else:
                    param_def["description"] += f" (default: {param.default})"
                
                properties[param_name] = param_def
            
            return {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to inspect function {name}: {str(e)}")
            # Return basic fallback schema
            description = f"Execute {name} function"
            if function_type == "background_task":
                description = f"BACKGROUND TASK: {description}. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."
            
            return {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
    
    def _python_type_to_json_schema_type(self, python_type) -> str:
        """
        Convert Python type annotations to JSON schema types.
        
        Args:
            python_type: Python type annotation
            
        Returns:
            JSON schema type string
        """
        # Handle common types
        if python_type == str:
            return "string"
        elif python_type == int:
            return "integer"  
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == list or (hasattr(python_type, '__origin__') and python_type.__origin__ == list):
            return "array"
        elif python_type == dict or (hasattr(python_type, '__origin__') and python_type.__origin__ == dict):
            return "object"
        else:
            # Default to string for unknown types
            return "string"
    
    def _convert_functions_to_llm_format(
        self, 
        functions: Union[List[Dict], Dict[str, Any]], 
        function_type: str = "tool"
    ) -> List[Dict]:
        """
        Convert functions to OpenAI tool format.
        
        Args:
            functions: Either List of JSON schema tool dictionaries or Dict of callable functions
            function_type: "tool" for regular tools, "background_task" for background tasks
            
        Returns:
            List of function definition dictionaries in OpenAI format
        """
        openrouter_functions = []
        
        # Handle JSON schema tool dictionaries (from tools_list)
        if isinstance(functions, list):
            for tool in functions:
                # Get tool properties
                description = tool.get("description", "")
                
                # Enhance description for background tasks
                if function_type == "background_task":
                    description = f"BACKGROUND TASK: {description}. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."
                
                # Ensure additionalProperties is False for strict mode
                parameters = tool.get("parameters", {})
                if isinstance(parameters, dict) and "additionalProperties" not in parameters:
                    parameters["additionalProperties"] = False

                # Create standardized tool format for OpenRouter
                openrouter_functions.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": description,
                        "parameters": parameters
                    }
                })
                
        # Handle callable Python functions (from background_tasks or callable_functions)
        elif isinstance(functions, dict):
            for name, callable_func in functions.items():
                try:
                    # Use function inspection helper
                    function_tool = self._python_function_to_openai_tool(callable_func, name, function_type)
                    # Wrap in OpenRouter format
                    openrouter_functions.append({
                        "type": "function",
                        "function": function_tool
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert function {name}: {str(e)}")
                    continue
        
        return openrouter_functions
    
    def _create_structure_function(self, structure_type: Type[T]) -> Dict[str, Any]:
        """
        Create a function definition from the structure type for OpenRouter.
        This follows the same pattern as openrouter_old.py but in the new architecture.
        
        Args:
            structure_type: The Pydantic model or TypedDict class to convert
            
        Returns:
            Function definition dictionary in OpenRouter format
        """
        function_name = structure_type.__name__.lower()
        description = structure_type.__doc__ or f"Create a {structure_type.__name__} response"
        
        # Get the JSON schema from the structure type
        if hasattr(structure_type, 'model_json_schema'):
            # Pydantic model
            schema = structure_type.model_json_schema()
        elif hasattr(structure_type, '__annotations__'):
            # TypedDict - build basic schema
            properties = {}
            required = []
            for field_name, field_type in structure_type.__annotations__.items():
                properties[field_name] = {
                    "type": self._python_type_to_json_schema_type(field_type),
                    "description": f"{field_name} field"
                }
                required.append(field_name)
            schema = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            }
        else:
            # Fallback schema
            schema = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        
        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": schema
            }
        }
    
    def _prepare_tools_context(self, input: ILLMInput) -> Tuple[List, str]:
        """
        Prepare OpenRouter-specific tools and enhanced context instructions.
        
        Args:
            input: The LLM input
            
        Returns:
            Tuple of (openrouter_tools_list, enhanced_context_instructions)
        """
        # Convert tools to OpenRouter format using unified method
        openrouter_tools = []
        
        # Add structure function if structure_type is provided (but don't count as regular function)
        if input.structure_type:
            structure_function = self._create_structure_function(input.structure_type)
            openrouter_tools.append(structure_function)
        
        if input.tools_list and len(input.tools_list) > 0:
            openrouter_tools.extend(
                self._convert_functions_to_llm_format(input.tools_list, function_type="tool")
            )
        
        if input.background_tasks and len(input.background_tasks) > 0:
            openrouter_tools.extend(
                self._convert_functions_to_llm_format(input.background_tasks, function_type="background_task")
            )
        
        # Build enhanced instructions using generic utility
        enhanced_instructions = build_enhanced_instructions(
            structure_type=input.structure_type,
            background_tasks=input.background_tasks
        )
        
        # Add structure function instructions if needed
        if input.structure_type:
            function_name = input.structure_type.__name__.lower()
            enhanced_instructions += f"""\n\nYou MUST ALWAYS use the {function_name} tool/function to format your response.
Your response ALWAYS MUST be returned using the tool, independently of what the message or response are.
You MUST ALWAYS CALL TOOLS FOR RETURNING RESPONSE
The response Must be in JSON format"""
        
        return openrouter_tools, enhanced_instructions

    async def _process_function_calls(self, tool_calls, input: ILLMInput, messages: List[Dict]) -> Tuple[bool, Any]:
        """
        Process function calls from LLM response and add results to messages.
        Handles both regular tools, background tasks, and structure functions consistently.
        
        Args:
            tool_calls: Tool calls from LLM response (OpenAI format)
            input: The original LLM input
            messages: List of OpenAI-style messages to append function results to
            
        Returns:
            Tuple of (has_regular_functions, structured_response)
            - has_regular_functions: True if regular tools were processed (continue conversation)
            - structured_response: Parsed structure object if structure function was called, None otherwise
        """
        if not tool_calls:
            return False, None
            
        # Create function call objects for orchestrator
        class FunctionCall:
            def __init__(self, name: str, args: dict):
                self.name = name
                self.args = args
        
        generic_function_calls = []
        structured_response = None
        has_regular_functions = False
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            
            # Check if it's the structure function
            if input.structure_type and function_name == input.structure_type.__name__.lower():
                try:
                    structured_response = input.structure_type(**function_args)
                    self.logger.info(f"Created structured response: {function_name}")
                    continue  # Don't process structure function as regular function
                except Exception as e:
                    self.logger.error(f"Error creating structured response from {function_name}: {str(e)}")
                    continue
            
            # Track if we have regular functions (affects continuation logic)
            if function_name in (input.callable_functions or {}):
                has_regular_functions = True
            elif function_name not in (input.background_tasks or {}):
                self.logger.warning(f"Function {function_name} not found in available functions or background tasks")
                continue
                
            generic_function_calls.append(FunctionCall(name=function_name, args=function_args))
        
        # Execute regular functions and background tasks (but not structure functions)
        if generic_function_calls:
            # Execute all functions (regular + background) via orchestrator
            execution_results = await self._function_orchestrator.process_function_calls_from_response(
                generic_function_calls,
                input.callable_functions or {},
                input.background_tasks or {}
            )
            
            # Add results to messages
            self._add_function_results_to_messages(execution_results, messages, tool_calls)
            
            self.logger.info(f"Processed {len(generic_function_calls)} function calls, regular tools: {has_regular_functions}")
        
        return has_regular_functions, structured_response

    def _add_function_results_to_messages(self, execution_results: Dict, messages: List[Dict], tool_calls) -> None:
        """
        Add function execution results to messages in OpenAI chat format.
        
        Args:
            execution_results: Results from function orchestrator
            messages: List of OpenAI-style messages to append to
            tool_calls: Original tool calls with call_id info
        """
        # Add function results as tool messages
        function_results = execution_results.get('function_results', [])
        function_names = execution_results.get('function_names', [])
        function_args = execution_results.get('function_args', [])
        for (name, result, args) in zip(function_names, function_results, function_args):
            messages.append({
                "role": "function",
                "name": name,
                "content": f"Function '{name}' called with arguments {args} returned: {result}"
            })
    
        # Add background task notifications as tool messages (if any)
        for bg_message in execution_results.get('background_initiated', []):
            messages.append({
                "type": "message",
                "role": "user",
                "content": f"Background task initiated: {bg_message}"
            })
        
        if function_results:
            completion_msg = f"All {len(function_results)} function(s) completed. Please provide your response based on these results."
            messages.append({
                "type": "message",
                "role": "user",
                "content": completion_msg
            })

    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Handle simple chat without tools or background tasks using chat.completions.
        For structured output, we use function calling approach consistently.
        
        Args:
            input: The LLM input
            
        Returns:
            Dict containing the LLM response and usage information
        """
        # Build base context
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Prepare arguments for chat.completions.create
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
        }
        
        # For structured output, use function calling approach (not JSON mode)
        if input.structure_type:
            structure_function = self._create_structure_function(input.structure_type)
            kwargs["tools"] = [structure_function]
            
            # Add structure function instructions
            function_name = input.structure_type.__name__.lower()
            messages[0]["content"] += f"""\n\nYou MUST ALWAYS use the {function_name} tool/function to format your response.
Your response ALWAYS MUST be returned using the tool, independently of what the message or response are.
You MUST ALWAYS CALL TOOLS FOR RETURNING RESPONSE
The response Must be in JSON format"""
                
        response = self._client.chat.completions.create(**kwargs)
        
        # Process usage metadata
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = standardize_usage_metadata(response.usage, provider="openrouter")

        message = response.choices[0].message
        
        # Handle response based on whether it's structured or not
        if input.structure_type:
            # Check for structure function call
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    if function_name == input.structure_type.__name__.lower():
                        try:
                            function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                            structured_response = input.structure_type(**function_args)
                            self.logger.info(f"Created structured response via function call: {function_name}")
                            return {"llm_response": structured_response, "usage": usage}
                        except Exception as e:
                            self.logger.error(f"Error creating structured response from {function_name}: {str(e)}")
                            return {"llm_response": f"Error creating structured response: {str(e)}", "usage": usage}
            
            # Fallback: no function call received for structured output
            self.logger.warning("Expected structure function call but none received")
            return {"llm_response": "Failed to generate structured response", "usage": usage}
        else:
            # For unstructured output, use the message content
            return {"llm_response": message.content, "usage": usage}

    async def _chat_with_tools(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Handle complex chat with tools and/or background tasks using chat.completions.
        
        Args:
            input: The LLM input
            
        Returns:
            Dict containing the LLM response and usage information
        """
        # Build base context and prepare tools
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        openrouter_tools, enhanced_instructions = self._prepare_tools_context(input)
        messages[0]["content"] += enhanced_instructions
        
        # Handle complex cases with function calling (multi-turn)
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Current turn: {current_turn}")
            has_regular_functions = False
            try:
                start_time = time.time()
                
                # Prepare arguments for chat.completions.create
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": openrouter_tools if openrouter_tools else None,
                }
                
                # Note: For structured output, we use function calling instead of response_format
                # The structure function is added to tools in _prepare_tools_context

                response = self._client.chat.completions.create(**kwargs)
                self.logger.info(f"🔍response time: {time.time() - start_time}")
                
                # Process usage metadata safely
                if hasattr(response, "usage") and response.usage:
                    current_usage = standardize_usage_metadata(response.usage, provider="openrouter")
                    accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)

                message = response.choices[0].message
                
                # Check for function calls
                if message.tool_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(message.tool_calls)} function calls")
                    
                    # Process function calls using unified method
                    has_regular_functions, structured_response = await self._process_function_calls(message.tool_calls, input, messages)
                    
                    # If we got a structured response, return it immediately
                    if structured_response is not None:
                        self.logger.info(f"Turn {current_turn}: Received structured response via function call")
                        return {"llm_response": structured_response, "usage": accumulated_usage}
                    
                    if has_regular_functions:
                        current_turn += 1
                        continue
                
                # For structured output, we should have received a function call
                # If no function call was received but structure_type is expected, that's an error
                if input.structure_type:
                    self.logger.warning(f"Turn {current_turn}: Expected structure function call but none received")
                    return {"llm_response": "Failed to generate structured response via function call", "usage": accumulated_usage}
                
                # Return text response (only for non-structured output)
                if message.content:
                    self.logger.info(f"Turn {current_turn}: Received text response")
                    return {"llm_response": message.content, "usage": accumulated_usage}

            except Exception as e:
                self.logger.error(
                    f"Error in OpenRouter chat_with_tools turn {current_turn}: {str(e)}"
                )
                self.logger.error(traceback.format_exc())
                return {
                    "llm_response": f"An error occurred: {str(e)}",
                    "usage": accumulated_usage,
                }

        # Handle max turns reached
        return {
            "llm_response": "Maximum number of function calling turns reached",
            "usage": accumulated_usage,
        }
    
    async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle simple streaming without tools or background tasks using chat.completions.
        For structured output, we use function calling approach consistently.
        
        Args:
            input: The LLM input
            
        Yields:
            Dict containing streaming response chunks and usage information
        """
        # Build base context
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Prepare arguments for chat.completions.create
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
            "stream": True,
        }
        
        # For structured output, use function calling approach (not JSON mode)
        if input.structure_type:
            structure_function = self._create_structure_function(input.structure_type)
            kwargs["tools"] = [structure_function]
            
            # Add structure function instructions
            function_name = input.structure_type.__name__.lower()
            messages[0]["content"] += f"""\n\nYou MUST ALWAYS use the {function_name} tool/function to format your response.
Your response ALWAYS MUST be returned using the tool, independently of what the message or response are.
You MUST ALWAYS CALL TOOLS FOR RETURNING RESPONSE
The response Must be in JSON format"""
        
        # Track usage and collected data
        accumulated_usage = None
        collected_text = ""
        collected_tool_calls = []
        
        # Process streaming response
        for chunk in self._client.chat.completions.create(**kwargs):
            # Handle usage data if available
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                current_usage = standardize_usage_metadata(chunk.usage, provider="openrouter")
                accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)
            
            # Skip chunks without choices
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            # Handle content streaming
            if hasattr(delta, 'content') and delta.content is not None:
                collected_text += delta.content
                if not input.structure_type:
                    yield {"llm_response": collected_text}
            
            # Handle tool calls streaming for structured output
            if hasattr(delta, 'tool_calls') and delta.tool_calls and input.structure_type:
                for i, tool_delta in enumerate(delta.tool_calls):
                    # Initialize or get current tool call
                    if i >= len(collected_tool_calls):
                        collected_tool_calls.append({
                            "id": tool_delta.id or "",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    current_tool_call = collected_tool_calls[i]
                    
                    # Update tool call with new delta information
                    if tool_delta.id:
                        current_tool_call["id"] = tool_delta.id
                        
                    if hasattr(tool_delta, 'function'):
                        if tool_delta.function.name:
                            current_tool_call["function"]["name"] = tool_delta.function.name
                            
                        if tool_delta.function.arguments:
                            current_tool_call["function"]["arguments"] += tool_delta.function.arguments
                            
                            # Try to parse and yield structured response when complete
                            if current_tool_call["function"]["name"] == input.structure_type.__name__.lower():
                                is_complete, fixed_json = is_json_complete(current_tool_call["function"]["arguments"])
                                if is_complete:
                                    try:
                                        function_args = json.loads(fixed_json)
                                        structured_response = input.structure_type(**function_args)
                                        yield {"llm_response": structured_response}
                                    except (json.JSONDecodeError, TypeError, ValueError):
                                        continue
        
        # Final yield with usage information
        yield {"llm_response": None, "usage": accumulated_usage}

    async def _stream_with_tools(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle complex streaming with tools and/or background tasks using chat.completions.
        
        Args:
            input: The LLM input
            
        Yields:
            Dict containing streaming response chunks and usage information
        """
        # Build base context and prepare tools
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        openrouter_tools, enhanced_instructions = self._prepare_tools_context(input)
        messages[0]["content"] += enhanced_instructions
        
        # Handle complex cases with function calling (multi-turn)
        current_turn = 0
        accumulated_usage = None

        while current_turn < input.max_turns:
            self.logger.info(f"Current turn: {current_turn}")
            has_regular_functions = False
            try:
                # Prepare arguments for chat.completions.create
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": openrouter_tools if openrouter_tools else None,
                    "stream": True,
                }
                
                # Store collected message and tool calls
                collected_message = {"content": "", "tool_calls": []}
                collected_text = ""
                chunk_count = 0

                self.logger.debug(f"Starting stream processing for turn {current_turn}")

                # Process streaming response
                for chunk in self._client.chat.completions.create(**kwargs):
                    chunk_count += 1
                    # Handle usage metadata
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        current_usage = standardize_usage_metadata(chunk.usage, provider="openrouter")
                        accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)
                    
                    # Skip chunks without choices
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta

                    # Handle content streaming
                    if hasattr(delta, 'content') and delta.content is not None:
                        collected_message["content"] += delta.content
                        collected_text += delta.content
                        # For structured output, we only yield content via function calls, not direct content
                        if not input.structure_type:
                            yield {"llm_response": collected_text}

                    # Handle tool calls streaming
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_delta in delta.tool_calls:
                            # Use the index from the tool_delta, not enumerate
                            tool_index = tool_delta.index
                            
                            # Ensure we have enough slots in the array
                            while len(collected_message["tool_calls"]) <= tool_index:
                                collected_message["tool_calls"].append({
                                    "id": "",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            current_tool_call = collected_message["tool_calls"][tool_index]
                            
                            # Update tool call with new delta information
                            if tool_delta.id:
                                current_tool_call["id"] = tool_delta.id
                                
                            if hasattr(tool_delta, 'function'):
                                if tool_delta.function.name:
                                    current_tool_call["function"]["name"] = tool_delta.function.name
                                    
                                if tool_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += tool_delta.function.arguments

                self.logger.debug(f"Turn {current_turn}: Stream ended. Processed {chunk_count} chunks. Tool calls: {len(collected_message['tool_calls'])}, Text collected: {len(collected_text)} chars")

                # Process function calls if any were collected (similar to _chat_with_tools)
                if collected_message["tool_calls"]:
                    # Create mock tool calls for processing
                    class MockToolCall:
                        def __init__(self, tc_dict):
                            self.id = tc_dict["id"]
                            self.function = type('obj', (object,), {
                                'name': tc_dict["function"]["name"],
                                'arguments': tc_dict["function"]["arguments"]
                            })()
                    
                    mock_tool_calls = [MockToolCall(tc) for tc in collected_message["tool_calls"] if tc["function"]["name"]]
                    self.logger.info(f"tool to call {mock_tool_calls}")
                    has_regular_functions, structured_response = await self._process_function_calls(mock_tool_calls, input, messages)
                    
                    # If we got a structured response, yield it and break
                    if structured_response is not None:
                        yield {"llm_response": structured_response, "usage": accumulated_usage}
                        break
                    
                    if has_regular_functions:
                        current_turn += 1
                        continue
                
                # Stream completed
                self.logger.debug(f"Turn {current_turn}: Stream completed")
                break

            except Exception as e:
                self.logger.error(f"Error in OpenRouter stream_with_tools turn {current_turn}: {str(e)}")
                self.logger.error(traceback.format_exc())
                yield {
                    "llm_response": f"An error occurred: {str(e)}",
                    "usage": accumulated_usage,
                }
                return

        # Handle max turns reached
        if current_turn >= input.max_turns:
            self.logger.warning(f"Maximum turns reached: {current_turn} >= {input.max_turns}")
            yield {
                "llm_response": "Maximum number of function calling turns reached",
                "usage": accumulated_usage,
            }
        else:
            # Final usage yield
            yield {"llm_response": None, "usage": accumulated_usage}