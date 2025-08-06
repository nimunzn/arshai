"""
OpenRouter implementation using the new BaseLLMClient framework.

Migrated to use structured function orchestration, dual interface support,
and standardized patterns from the Arshai framework.
"""

import os
import json
import time
import inspect
from typing import Dict, Any, TypeVar, Type, Union, AsyncGenerator, List
from openai import OpenAI

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import (
    is_json_complete,
    parse_to_structure,
)
from arshai.llms.utils.function_execution import FunctionCall, FunctionExecutionInput

T = TypeVar("T")

# Structure instructions template used across methods
STRUCTURE_INSTRUCTIONS_TEMPLATE = """

You MUST ALWAYS use the {function_name} tool/function to format your response.
Your response ALWAYS MUST be returned using the tool, independently of what the message or response are.
You MUST ALWAYS CALL TOOLS FOR RETURNING RESPONSE
The response Must be in JSON format"""


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter implementation using the new framework architecture.
    
    This client demonstrates how to implement a provider using the new
    BaseLLMClient framework with minimal code duplication.
    """
    
    def _initialize_client(self) -> Any:
        """
        Initialize the OpenRouter client with safe HTTP configuration.
        
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
        
        try:
            # Import the safe factory for better HTTP handling
            from arshai.clients.utils.safe_http_client import SafeHttpClientFactory
            
            self.logger.info("Creating OpenRouter client with safe HTTP configuration")
            
            # Create safe httpx client first
            import httpx
            httpx_version = getattr(httpx, '__version__', '0.0.0')
            
            # Get safe HTTP configuration
            limits_config = SafeHttpClientFactory._get_safe_limits_config(httpx_version)
            timeout_config = SafeHttpClientFactory._get_safe_timeout_config(httpx_version)
            additional_config = SafeHttpClientFactory._get_additional_httpx_config(httpx_version)
            
            safe_http_client = httpx.Client(
                limits=limits_config,
                timeout=timeout_config,
                **additional_config
            )
            
            # Create OpenRouter client with safe HTTP client
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": app_name,
                },
                http_client=safe_http_client,
                max_retries=3
            )
            
            self.logger.info("OpenRouter client created successfully with safe configuration")
            return client
            
        except ImportError as e:
            self.logger.warning(f"Safe HTTP client factory not available: {e}, using default OpenRouter client")
            # Fallback to original implementation
            return OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": app_name,
                },
                timeout=30.0,
                max_retries=2
            )
        
        except Exception as e:
            self.logger.error(f"Failed to create safe OpenRouter client: {e}")
            # Final fallback to ensure system keeps working
            self.logger.info("Using fallback OpenRouter client configuration")
            try:
                # At least try to set a timeout for basic safety
                return OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": site_url,
                        "X-Title": app_name,
                    },
                    timeout=30.0,
                    max_retries=2
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback OpenRouter client also failed: {fallback_error}")
                # Last resort - basic client
                return OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": site_url,
                        "X-Title": app_name,
                    }
                )

    # ========================================================================
    # PROVIDER-SPECIFIC HELPER METHODS
    # ========================================================================

    def _accumulate_usage_safely(self, current_usage: Dict[str, Any], accumulated_usage: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Safely accumulate usage metadata without in-place mutations.
        
        Args:
            current_usage: Current usage metadata
            accumulated_usage: Previously accumulated usage (optional)
        
        Returns:
            New accumulated usage dictionary
        """
        if accumulated_usage is None:
            return current_usage
        
        return {
            "input_tokens": accumulated_usage["input_tokens"] + current_usage["input_tokens"],
            "output_tokens": accumulated_usage["output_tokens"] + current_usage["output_tokens"],
            "total_tokens": accumulated_usage["total_tokens"] + current_usage["total_tokens"],
            "thinking_tokens": accumulated_usage["thinking_tokens"] + current_usage["thinking_tokens"],
            "tool_calling_tokens": accumulated_usage["tool_calling_tokens"] + current_usage["tool_calling_tokens"],
            "provider": current_usage["provider"],
            "model": current_usage["model"],
            "request_id": current_usage["request_id"]
        }

    def _create_openai_messages(self, input: ILLMInput) -> List[Dict[str, Any]]:
        """Create OpenAI-compatible messages from input."""
        return [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]

    def _convert_functions_to_openai_format(self, functions: Union[List[Dict], Dict[str, Any]], is_background: bool = False) -> List[Dict]:
        """
        Unified method to convert functions to OpenAI format.
        
        Args:
            functions: Either list of tool schemas or dict of callable functions
            is_background: Whether these are background tasks
        
        Returns:
            List of OpenAI-formatted function definitions
        """
        openai_functions = []
        
        if isinstance(functions, list):
            # Handle pre-defined tool schemas
            for tool in functions:
                # Ensure additionalProperties is False for strict mode
                parameters = tool.get("parameters", {})
                if isinstance(parameters, dict) and "additionalProperties" not in parameters:
                    parameters["additionalProperties"] = False

                openai_functions.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": parameters
                    }
                })
        
        elif isinstance(functions, dict):
            # Handle callable Python functions
            for name, func in functions.items():
                try:
                    function_def = self._python_function_to_openai_function(func, name, is_background=is_background)
                    openai_functions.append({
                        "type": "function",
                        "function": function_def
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to convert {'background task' if is_background else 'function'} {name}: {str(e)}")
                    continue
        
        return openai_functions

    def _python_function_to_openai_function(self, func, name: str, is_background: bool = False) -> Dict[str, Any]:
        """Convert a Python function to OpenAI function format using introspection - matches old implementation."""
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Extract description from docstring
            description = func.__doc__ or f"Execute {name} function"
            description = description.strip()
            
            # Enhance description for background tasks
            if is_background:
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
            if is_background:
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
        """Convert Python type annotations to JSON schema types."""
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
            return "string"  # Default to string for unknown types

    def _create_structure_function_openai(self, structure_type: Type[T]) -> Dict[str, Any]:
        """Create OpenAI function definition for structured output."""
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

    def _extract_function_calls_from_response(self, response) -> List:
        """Extract function calls from OpenAI response."""
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return message.tool_calls
        return []

    def _process_function_calls_for_orchestrator(self, tool_calls, input: ILLMInput) -> tuple:
        """
        Process function calls and prepare them for the orchestrator using object-based approach.
        
        CRITICAL FIX: Uses List[FunctionCall] to handle multiple calls to the same function.
        This solves the infinite loop issue where dictionary-based approach was dropping duplicate function names.
        
        Returns:
            Tuple of (function_calls, structured_response)
        """
        function_calls = []
        structured_response = None
        
        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse function arguments for {function_name}")
                function_args = {}
            
            # Check if it's the structure function
            if input.structure_type and function_name == input.structure_type.__name__.lower():
                try:
                    structured_response = input.structure_type(**function_args)
                    self.logger.info(f"Created structured response: {function_name}")
                    continue  # Don't process structure function as regular function
                except Exception as e:
                    self.logger.error(f"Error creating structured response from {function_name}: {str(e)}")
                    continue
            
            # Create unique call_id to track individual function calls
            call_id = f"{function_name}_{i}"
            
            # Check if it's a background task
            if function_name in (input.background_tasks or {}):
                function_calls.append(FunctionCall(
                    name=function_name,
                    args=function_args,
                    call_id=call_id,
                    is_background=True
                ))
            # Check if it's a regular function
            elif function_name in (input.callable_functions or {}):
                function_calls.append(FunctionCall(
                    name=function_name,
                    args=function_args,
                    call_id=call_id,
                    is_background=False
                ))
            else:
                self.logger.warning(f"Function {function_name} not found in available functions or background tasks")
        
        return function_calls, structured_response

    # ========================================================================
    # FRAMEWORK-REQUIRED ABSTRACT METHODS
    # ========================================================================

    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle simple chat without tools or background tasks."""
        messages = self._create_openai_messages(input)
        
        # Prepare OpenAI request arguments
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
        }
        
        # Add structure function if needed
        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            kwargs["tools"] = [structure_function]
            
            # Add structure instructions to system prompt
            function_name = input.structure_type.__name__.lower()
            kwargs["messages"][0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)
        
        # Make the API call
        response = self._client.chat.completions.create(**kwargs)
        
        # Process usage metadata
        usage = self._standardize_usage_metadata(
            response.usage if hasattr(response, 'usage') else None,
            self._get_provider_name(),
            self.config.model,
            getattr(response, 'id', None)
        )
        
        # Handle structured output
        if input.structure_type:
            tool_calls = self._extract_function_calls_from_response(response)
            if tool_calls:
                _, structured_response = self._process_function_calls_for_orchestrator(tool_calls, input)
                if structured_response is not None:
                    return {"llm_response": structured_response, "usage": usage}
            
            # Fallback for failed structured output
            return {"llm_response": f"Failed to generate structured response of type {input.structure_type.__name__}", "usage": usage}
        
        # Handle regular text response
        message = response.choices[0].message
        return {"llm_response": message.content, "usage": usage}

    async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle complex chat with tools and/or background tasks."""
        messages = self._create_openai_messages(input)
        
        # Prepare tools for OpenAI
        openai_tools = []
        
        # Add structure function if needed
        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            openai_tools.append(structure_function)
            
            # Add structure instructions
            function_name = input.structure_type.__name__.lower()
            messages[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)
        
        # Add regular tools
        if input.tools_list:
            openai_tools.extend(self._convert_functions_to_openai_format(input.tools_list, is_background=False))
        
        # Add background tasks
        if input.background_tasks:
            openai_tools.extend(self._convert_functions_to_openai_format(input.background_tasks, is_background=True))
        
        # Multi-turn conversation for function calling
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Function calling turn: {current_turn}")
            
            try:
                start_time = time.time()
                
                # Prepare arguments for OpenAI
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": openai_tools if openai_tools else None,
                }
                
                response = self._client.chat.completions.create(**kwargs)
                self.logger.info(f"Response time: {time.time() - start_time:.2f}s")
                
                # Process usage metadata using framework standardization
                if hasattr(response, "usage") and response.usage:
                    current_usage = self._standardize_usage_metadata(
                        response.usage, self._get_provider_name(), self.config.model, getattr(response, 'id', None)
                    )
                    # Use safe accumulation helper
                    accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                
                message = response.choices[0].message
                
                # Check for function calls
                tool_calls = self._extract_function_calls_from_response(response)
                if tool_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(tool_calls)} function calls")
                    
                    # Process function calls for orchestrator
                    function_calls, structured_response = self._process_function_calls_for_orchestrator(tool_calls, input)
                    
                    # If we got a structured response, return it immediately
                    if structured_response is not None:
                        self.logger.info(f"Turn {current_turn}: Received structured response via function call")
                        return {"llm_response": structured_response, "usage": accumulated_usage}
                    
                    # Execute functions via orchestrator using new object-based approach
                    if function_calls:
                        # Create execution input
                        execution_input = FunctionExecutionInput(
                            function_calls=function_calls,
                            available_functions=input.callable_functions or {},
                            available_background_tasks=input.background_tasks or {}
                        )
                        
                        execution_result = await self._execute_functions_with_orchestrator(execution_input)
                        
                        # Add function results to conversation
                        self._add_function_results_to_messages(execution_result, messages)
                        
                        # Continue if we have regular functions (need to continue conversation)
                        regular_function_calls = [call for call in function_calls if not call.is_background]
                        if regular_function_calls:
                            current_turn += 1
                            continue
                
                # Handle structured output expectation
                if input.structure_type:
                    self.logger.warning(f"Turn {current_turn}: Expected structure function call but none received")
                    return {"llm_response": f"Failed to generate structured response of type {input.structure_type.__name__}", "usage": accumulated_usage}
                
                # Return text response
                if message.content:
                    self.logger.info(f"Turn {current_turn}: Received text response")
                    return {"llm_response": message.content, "usage": accumulated_usage}
                
            except Exception as e:
                self.logger.error(f"Error in OpenRouter chat_with_functions turn {current_turn}: {str(e)}")
                return {
                    "llm_response": f"An error occurred: {str(e)}",
                    "usage": accumulated_usage,
                }
        
        # Handle max turns reached
        return {
            "llm_response": "Maximum number of function calling turns reached",
            "usage": accumulated_usage,
        }

    def _add_function_results_to_messages(self, execution_result: Dict, messages: List[Dict]) -> None:
        """Add function execution results to messages in OpenAI chat format."""
        # Add function results as function messages
        for result in execution_result.get('regular_results', []):
            messages.append({
                "role": "function",
                "name": result['name'],
                "content": f"Function '{result['name']}' called with arguments {result['args']} returned: {result['result']}"
            })
        
        # Add background task notifications
        for bg_message in execution_result.get('background_initiated', []):
            messages.append({
                "role": "user",
                "content": f"Background task initiated: {bg_message}"
            })
        
        # Add completion message if we have results
        if execution_result.get('regular_results'):
            completion_msg = f"All {len(execution_result['regular_results'])} function(s) completed. Please provide your response based on these results."
            messages.append({
                "role": "user",
                "content": completion_msg
            })

    async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle simple streaming without tools or background tasks."""
        messages = self._create_openai_messages(input)
        
        # Prepare OpenAI request arguments
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
            "stream": True,
        }
        
        # Add structure function if needed
        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            kwargs["tools"] = [structure_function]
            
            # Add structure instructions
            function_name = input.structure_type.__name__.lower()
            kwargs["messages"][0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)
        
        # Track usage and collected data
        accumulated_usage = None
        collected_text = ""
        collected_tool_calls = []
        
        # Process streaming response
        for chunk in self._client.chat.completions.create(**kwargs):
            # Handle usage data if available
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                current_usage = self._standardize_usage_metadata(
                    chunk.usage, self._get_provider_name(), self.config.model
                )
                accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
            
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

    async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex streaming with tools and/or background tasks - full streaming implementation."""
        messages = self._create_openai_messages(input)
        
        # Prepare tools for OpenAI
        openai_tools = []
        
        # Add structure function if needed
        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            openai_tools.append(structure_function)
            
            # Add structure instructions
            function_name = input.structure_type.__name__.lower()
            messages[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)
        
        # Add regular tools
        if input.tools_list:
            openai_tools.extend(self._convert_functions_to_openai_format(input.tools_list, is_background=False))
        
        # Add background tasks
        if input.background_tasks:
            openai_tools.extend(self._convert_functions_to_openai_format(input.background_tasks, is_background=True))
        
        # Multi-turn streaming conversation for function calling  
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Stream function calling turn: {current_turn}")
            has_regular_functions = False
            try:
                # Prepare arguments for streaming
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": openai_tools if openai_tools else None,
                    "stream": True,
                }
                
                # Store collected message and tool calls (matching old implementation exactly)
                collected_message = {"content": "", "tool_calls": []}
                collected_text = ""
                chunk_count = 0
                
                self.logger.debug(f"Starting stream processing for turn {current_turn}")
                
                # Process streaming response (matching old implementation)
                for chunk in self._client.chat.completions.create(**kwargs):
                    chunk_count += 1
                    # Handle usage metadata
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        current_usage = self._standardize_usage_metadata(
                            chunk.usage, self._get_provider_name(), self.config.model
                        )
                        accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                    
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
                    
                    # Handle tool calls streaming (EXACT match to old implementation)
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_delta in delta.tool_calls:
                            # Use the index from the tool_delta, not enumerate (critical for OpenRouter)
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
                
                # Process function calls if any were collected (matching old implementation)
                if collected_message["tool_calls"]:
                    # Create mock tool calls for processing (exact match to old implementation)
                    class MockToolCall:
                        def __init__(self, tc_dict):
                            self.id = tc_dict["id"]
                            self.function = type('obj', (object,), {
                                'name': tc_dict["function"]["name"],
                                'arguments': tc_dict["function"]["arguments"]
                            })()
                    
                    mock_tool_calls = [MockToolCall(tc) for tc in collected_message["tool_calls"] if tc["function"]["name"]]
                    self.logger.info(f"Functions to call: {len(mock_tool_calls)}")
                    
                    # Process function calls for orchestrator
                    function_calls, structured_response = self._process_function_calls_for_orchestrator(mock_tool_calls, input)
                    
                    # If we got a structured response, yield it and break
                    if structured_response is not None:
                        yield {"llm_response": structured_response, "usage": accumulated_usage}
                        break
                    
                    # Execute functions via orchestrator using new object-based approach
                    if function_calls:
                        # Create execution input
                        execution_input = FunctionExecutionInput(
                            function_calls=function_calls,
                            available_functions=input.callable_functions or {},
                            available_background_tasks=input.background_tasks or {}
                        )
                        
                        execution_result = await self._execute_functions_with_orchestrator(execution_input)
                        
                        # Add function results to conversation
                        self._add_function_results_to_messages(execution_result, messages)
                        
                        # Continue if we have regular functions
                        regular_function_calls = [call for call in function_calls if not call.is_background]
                        if regular_function_calls:
                            has_regular_functions = True
                            current_turn += 1
                            continue
                
                # Stream completed for this turn
                self.logger.debug(f"Turn {current_turn}: Stream completed")
                break
                
            except Exception as e:
                self.logger.error(f"Error in OpenRouter stream_with_functions turn {current_turn}: {str(e)}")
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
            # Final usage yield if no structured response was returned
            yield {"llm_response": None, "usage": accumulated_usage}