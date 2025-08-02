"""
OpenAI implementation of the LLM interface using openai SDK.
Supports both standard function calling and background tasks with manual tool orchestration.
Follows the same interface pattern as the Gemini client for consistency.
"""

import os
import logging
import json
import traceback
import time
import inspect
from typing import Dict, Any, TypeVar, Type, Union, AsyncGenerator, List, Tuple, get_type_hints
from openai import OpenAI
from openai.types.responses import ParsedResponse
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import (
    is_json_complete,
    standardize_usage_metadata,
    accumulate_usage_safely,
    parse_to_structure,
    build_enhanced_instructions,
    convert_typeddict_to_basemodel,
)

T = TypeVar("T")

class OpenAIClient(BaseLLMClient):
    """OpenAI implementation of the LLM interface"""
    
    def __init__(self, config: ILLMConfig):
        """
        Initialize the OpenAI client.
        
        Args:
            config: Configuration for the LLM
        """
        # -specific configuration
        # Initialize base client (handles common setup)
        super().__init__(config)
    
    def _initialize_client(self) -> Any:
        """
        Initialize the OpenAI client.
        
        The client automatically uses OPENAI_API_KEY from environment variables.
        If the API key is not found, a clear error is raised.
        
        Returns:
            OpenAI client instance
        
        Raises:
            ValueError: If OPENAI_API_KEY is not set in environment variables
        """
        # Check if API key is available in environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OpenAI API key not found in environment variables")
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
            
        return OpenAI(api_key=api_key)
    
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
                
                # For strict mode, ALL parameters must be in required array
                # regardless of default values
                required.append(param_name)
                
                # Add default value info to description
                if param.default != inspect.Parameter.empty:
                    param_def["description"] += f" (default: {param.default})"
                
                properties[param_name] = param_def
            
            # Build the function definition (without outer wrapper)
            return {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                },
                "strict": True
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
                },
                "strict": True
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
        azure_functions = []
        
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

                # Create standardized flat tool format
                azure_functions.append({
                    "type": "function",
                    "name": tool.get("name"),
                    "description": description,
                    "parameters": parameters,
                    "strict": True  # Required for structured output
                })
                
        # Handle callable Python functions (from background_tasks or callable_functions)
        elif isinstance(functions, dict):
            for name, callable_func in functions.items():
                try:
                    # Use function inspection helper for background tasks
                    if function_type == "background_task":
                        function_tool = self._python_function_to_openai_tool(callable_func, name, function_type)
                        # Add type to create flat structure
                        function_tool["type"] = "function"
                        azure_functions.append(function_tool)
                    else:
                        # For regular callable functions, use basic parameter extraction
                        description = callable_func.__doc__ or name
                        
                        parameters = {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False  # Required for strict mode
                        }
                        
                        azure_functions.append({
                            "type": "function",
                            "name": name,
                            "description": description,
                            "parameters": parameters,
                            "strict": True  # Required for structured output
                        })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert function {name}: {str(e)}")
                    continue
        
        return azure_functions
    
    def _prepare_tools_context(self, input: ILLMInput) -> Tuple[List, str]:
        """
        Prepare OpenAI-specific tools and enhanced context instructions.
        
        Args:
            input: The LLM input
            
        Returns:
            Tuple of (openai_tools_list, enhanced_context_instructions)
        """
        # Convert tools to OpenAI format using unified method
        openai_tools = []
        if input.tools_list and len(input.tools_list) > 0:
            openai_tools.extend(
                self._convert_functions_to_llm_format(input.tools_list, function_type="tool")
            )
        
        if input.background_tasks and len(input.background_tasks) > 0:
            openai_tools.extend(
                self._convert_functions_to_llm_format(input.background_tasks, function_type="background_task")
            )
        
        # Build enhanced instructions using generic utility
        enhanced_instructions = build_enhanced_instructions(
            structure_type=input.structure_type,
            background_tasks=input.background_tasks
        )
        
        return openai_tools, enhanced_instructions

    def _convert_messages_to_response_input(self, messages: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-style messages to ResponseInputParam format.
        
        Args:
            messages: List of OpenAI-style message dictionaries
            
        Returns:
            List of ResponseInputParam items
        """
        response_input = []
        for message in messages:
            # Convert to EasyInputMessageParam format
            response_message = {
                "type": "message",
                "role": message["role"],
                "content": message["content"]
            }
            response_input.append(response_message)
        return response_input

    async def _process_function_calls(self, function_calls, input: ILLMInput, response_input: List[Dict]) -> bool:
        """
        Process function calls from LLM response and add results to response_input.
        Handles both regular tools and background tasks consistently.
        
        Args:
            function_calls: Function calls from LLM response (any format)
            input: The original LLM input
            response_input: ResponseInputParam list to append function results to
            
        Returns:
            bool: True if regular tools were processed (continue conversation), False otherwise
        """
        if not function_calls:
            return False
            
        # Create function call objects for orchestrator
        class FunctionCall:
            def __init__(self, name: str, args: dict):
                self.name = name
                self.args = args
        
        generic_function_calls = []
        has_regular_functions = False
        
        for func_call in function_calls:
            # Extract function details (handles both responses.parse and responses.stream formats)
            if hasattr(func_call, 'arguments'):
                function_args = json.loads(func_call.arguments) if func_call.arguments else {}
            else:
                function_args = {}
            function_name = func_call.name
            
            # Track if we have regular functions (affects continuation logic)
            if function_name in (input.callable_functions or {}):
                has_regular_functions = True
            elif function_name not in (input.background_tasks or {}):
                self.logger.warning(f"Function {function_name} not found in available functions or background tasks")
                continue
                
            generic_function_calls.append(FunctionCall(name=function_name, args=function_args))
        
        if not generic_function_calls:
            return False
            
        # Execute all functions (regular + background) via orchestrator
        execution_results = await self._function_orchestrator.process_function_calls_from_response(
            generic_function_calls,
            input.callable_functions or {},
            input.background_tasks or {}
        )
        
        # Add results to response_input
        self._add_function_results_to_response_input(execution_results, response_input, generic_function_calls)
        
        self.logger.info(f"Processed {len(generic_function_calls)} function calls, regular tools: {has_regular_functions}")
        
        return has_regular_functions

    def _add_function_results_to_response_input(self, execution_results: Dict, response_input: List[Dict], function_calls: List) -> None:
        """
        Add function execution results to response_input in responses API format.
        
        Args:
            execution_results: Results from function orchestrator
            response_input: ResponseInputParam list to append to
            function_calls: Original function calls with call_id info
        """
        # Add function results as FunctionCallOutput items
        function_results = execution_results.get('function_results', [])
        function_names = execution_results.get('function_names', [])
        for (index, (name, result)) in enumerate(zip(function_names, function_results)):
            function_output = {
                "type": "message",
                "role": "user",
                "content": f"Function '{name}' called with arguments {function_calls[index].args} returned: {result}"
            }
            response_input.append(function_output)
    
        # Add background task notifications as user messages (if any)
        for bg_message in execution_results.get('background_initiated', []):
            response_input.append({
                "type": "message",
                "role": "user",
                "content": f"Background task initiated: {bg_message}"
            })

        if function_results:
            completion_msg = f"All {len(function_results)} function(s) completed. Please provide your response based on these results."
            response_input.append({
                "type": "message",
                "role": "user",
                "content": completion_msg
            })

    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Handle simple chat without tools or background tasks using responses.parse().
        
        Args:
            input: The LLM input
            
        Returns:
            Dict containing the LLM response and usage information
        """
        # Build base context in OpenAI format first
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Convert to ResponseInputParam format
        response_input = self._convert_messages_to_response_input(messages)
        
        # Prepare arguments for responses.parse()
        kwargs = {
            "model": self.config.model,
            "input": response_input,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens if self.config.max_tokens else None,
        }
        
        # Add text_format for structured output if needed
        if input.structure_type:
            kwargs["text_format"] = input.structure_type
        
        response: ParsedResponse = self._client.responses.parse(**kwargs)
        
        # Process usage metadata
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = standardize_usage_metadata(response.usage, provider="azure")

        # Handle response based on whether it's structured or not
        if input.structure_type:
            # For structured output, extract the parsed content
            llm_response = None
            for output_item in response.output:
                if output_item.type == "message":
                    for content in output_item.content:
                        if content.type == "output_text" and hasattr(content, 'parsed'):
                            llm_response = content.parsed
                            break
                    if llm_response:
                        break
            
            return {"llm_response": llm_response, "usage": usage}
        else:
            # For unstructured output, use the output_text property
            return {"llm_response": response.output_text, "usage": usage}

    async def _chat_with_tools(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Handle complex chat with tools and/or background tasks using responses.parse().
        
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
        azure_tools, enhanced_instructions = self._prepare_tools_context(input)
        messages[0]["content"] += enhanced_instructions
        
        # Convert to ResponseInputParam format
        response_input = self._convert_messages_to_response_input(messages)
        
        # Handle complex cases with function calling (multi-turn using previous_response_id)
        current_turn = 0
        accumulated_usage = None
        has_regular_functions = False
        while current_turn < input.max_turns:
            self.logger.info(f"Current turn: {current_turn}")

            try:
                start_time = time.time()
                
                # Prepare arguments for responses.parse()
                kwargs = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": azure_tools if azure_tools else None,
                    "input": response_input
                }
                # Add text_format for structured output if needed
                if input.structure_type:
                    kwargs["text_format"] = input.structure_type
                
                response =  self._client.responses.parse(**kwargs)
                self.logger.info(f"🔍response time: {time.time() - start_time}")
                
                # Process usage metadata safely
                if hasattr(response, "usage") and response.usage:
                    current_usage = standardize_usage_metadata(response.usage, provider="azure")
                    accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)

                # Check for function calls in output
                function_calls = [output_item for output_item in response.output if output_item.type == "function_call"]

                if function_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(function_calls)} function calls")
                    
                    # Process function calls using unified method
                    has_regular_functions = await self._process_function_calls(function_calls, input, response_input)
                    if has_regular_functions:
                        current_turn += 1
                        continue
                    
                # Check for structured response
                if input.structure_type:
                    llm_response = None
                    for output_item in response.output:
                        if output_item.type == "message":
                            for content in output_item.content:
                                if content.type == "output_text" and hasattr(content, 'parsed'):
                                    try:
                                        llm_response = content.parsed
                                    except Exception:
                                        is_complete, fixed_json = is_json_complete(content.text)
                                        if is_complete:
                                            try:
                                                llm_response = parse_to_structure(fixed_json, input.structure_type)
                                            except Exception as e:
                                                self.logger.error(f"Error parsing structured response: {str(e)}")
                                                llm_response = content.text
                    
                    if llm_response is not None:
                        self.logger.info(f"Turn {current_turn}: Received structured response")
                        return {"llm_response": llm_response, "usage": accumulated_usage}
                
                # Check for text response
                else:
                    self.logger.info(f"Turn {current_turn}: Received text response")
                    self.logger.info(f"🔍response: {response.output_text}")
                    return {"llm_response": response.output_text, "usage": accumulated_usage}

            except Exception as e:
                self.logger.error(
                    f"Error in OpenAI chat_with_tools turn {current_turn}: {str(e)}"
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
        Handle simple streaming without tools or background tasks using responses.stream().
        
        Args:
            input: The LLM input
            
        Yields:
            Dict containing streaming response chunks and usage information
        """
        # Build base context in OpenAI format first
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Convert to ResponseInputParam format  
        response_input = self._convert_messages_to_response_input(messages)
        
        # Prepare arguments for responses.stream()
        kwargs = {
            "model": self.config.model,
            "input": response_input,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens if self.config.max_tokens else None,
        }
        
        # Add text_format for structured output if needed
        if input.structure_type:
            sdk_structure_type = convert_typeddict_to_basemodel(input.structure_type)
            kwargs["text_format"] = sdk_structure_type
        
        # Use proper ResponseStreamManager pattern
        with self._client.responses.stream(**kwargs) as stream:
            accumulated_usage = None
            collected_text = ""
            # Process streaming events
            for event in stream:
                # self.logger.info(f"🔍event: {event}")
                if hasattr(event, "response") and hasattr(event.response, "usage") and event.response.usage:
                    # Usage from ResponseIncompleteEvent or ResponseCompletedEvent
                    current_usage = standardize_usage_metadata(event.response.usage, provider="openai")
                    accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)
                
                # Handle text delta events - progressive streaming
                if "response.output_text.delta" in event.type and hasattr(event, "delta"):
                    collected_text += event.delta
                    if not input.structure_type:
                        yield {"llm_response": collected_text}
                    else:
                        is_complete, fixed_json = is_json_complete(collected_text)
                        if is_complete:
                            try:
                                final_response = parse_to_structure(fixed_json, input.structure_type)
                                yield {"llm_response": final_response}
                            except ValueError:
                                pass
            
            # Final yield with usage information
            yield {"llm_response": None, "usage": accumulated_usage}

    async def _stream_with_tools(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle complex streaming with tools and/or background tasks using responses.stream().
        
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
        azure_tools, enhanced_instructions = self._prepare_tools_context(input)
        messages[0]["content"] += enhanced_instructions
        
        # Convert to ResponseInputParam format
        response_input = self._convert_messages_to_response_input(messages)
        
        # Handle complex cases with function calling (multi-turn using previous_response_id)
        current_turn = 0
        accumulated_usage = None
        has_regular_functions = False

        while current_turn < input.max_turns:
            self.logger.info(f"Current turn: {current_turn}")
            has_regular_functions = False
            
            try:
                # Prepare arguments for responses.stream()
                kwargs = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": azure_tools if azure_tools else None,
                    "parallel_tool_calls": True,
                    "input": response_input
                }
                
                # Add text_format for structured output if needed (convert TypedDict to BaseModel)
                if input.structure_type:
                    # Convert TypedDict to BaseModel for SDK compatibility
                    sdk_structure_type = convert_typeddict_to_basemodel(input.structure_type)
                    kwargs["text_format"] = sdk_structure_type
                
                # Generate streaming content with tools using responses API
                with self._client.responses.stream(**kwargs) as stream:
                    
                    # Store function calls collected during streaming
                    function_calls = []
                    collected_text = ""
                    chunk_count = 0

                    self.logger.debug(f"Starting stream processing for turn {current_turn}")

                    # Process streaming response events - collect function calls and text
                    for event in stream:
                        chunk_count += 1                        
                        # Handle usage metadata from response completion events
                        if hasattr(event, "response") and hasattr(event.response, "usage") and event.response.usage:
                            current_usage = standardize_usage_metadata(event.response.usage, provider="azure")
                            accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)
                        
                        # Handle function call arguments completion  
                        if event.type == "response.output_item.done" and event.item.type == 'function_call':
                            # Collect function calls for unified processing
                            function_calls.append(event.item)
                                        
                        # Handle text content from ResponseContentPartDoneEvent (for final responses)
                        if "response.output_text.delta" in event.type and hasattr(event, "delta"):
                            collected_text += event.delta
                            if not input.structure_type:
                                yield {"llm_response": collected_text}
                            else:
                                is_complete, fixed_json = is_json_complete(collected_text)
                                if is_complete:
                                    try:
                                        final_response = parse_to_structure(fixed_json, input.structure_type)
                                        yield {"llm_response": final_response}
                                    except ValueError:
                                        pass
                        
                    self.logger.debug(f"Turn {current_turn}: Stream ended. Processed {chunk_count} chunks. Function calls: {len(function_calls)}, Text collected: {len(collected_text)} chars")

                    # Process function calls if any were collected using unified method
                    if function_calls:
                        has_regular_functions = await self._process_function_calls(function_calls, input, response_input)
                        
                    self.logger.info(f"🔍has_regular_functions: {has_regular_functions}")
                    if not has_regular_functions:
                        # Stream completed
                        self.logger.debug(f"Turn {current_turn}: Stream completed")
                        break
                
                    current_turn += 1
                    self.logger.debug(f"Turn {current_turn}: Continuing - regular tools need processing")

            except Exception as e:
                self.logger.error(f"Error in OpenAI stream_with_tools turn {current_turn}: {str(e)}")
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
