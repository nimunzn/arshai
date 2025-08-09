"""
OpenAI implementation using the new BaseLLMClient framework.

Migrated to use structured function orchestration, dual interface support,
and standardized patterns from the Arshai framework.
"""

import os
import json
import time
import inspect
from typing import Dict, Any, TypeVar, Type, Union, AsyncGenerator, List, get_type_hints, Callable
from openai import OpenAI
from openai.types.responses import ParsedResponse

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import (
    is_json_complete,
    parse_to_structure,
    convert_typeddict_to_basemodel,
)
from arshai.llms.utils.function_execution import FunctionCall, FunctionExecutionInput, StreamingExecutionState

T = TypeVar("T")

# Structure instructions template used across methods
STRUCTURE_INSTRUCTIONS_TEMPLATE = """

You MUST ALWAYS use the {function_name} tool/function to format your response.
Your response ALWAYS MUST be returned using the tool, independently of what the message or response are.
You MUST ALWAYS CALL TOOLS FOR RETURNING RESPONSE
The response Must be in JSON format"""


class OpenAIClient(BaseLLMClient):
    """
    OpenAI implementation using the new framework architecture.
    
    This client demonstrates how to implement the base OpenAI provider using 
    the new BaseLLMClient framework with minimal code and maximum clarity.
    """
    
    def __del__(self):
        """Cleanup connections when the client is destroyed."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup connections."""
        self.close()
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup connections."""
        self.close()
        return False
    
    def close(self):
        """Close the OpenAI client and cleanup connections."""
        try:
            # Close the underlying httpx client if it exists
            if hasattr(self._client, '_client') and hasattr(self._client._client, 'close'):
                self._client._client.close()
                self.logger.info("Closed OpenAI httpx client")
            elif hasattr(self._client, 'close'):
                self._client.close()
                self.logger.info("Closed OpenAI client")
        except Exception as e:
            self.logger.warning(f"Error closing OpenAI client: {e}")
    
    def _initialize_client(self) -> Any:
        """
        Initialize the OpenAI client with safe HTTP configuration.
        
        Returns:
            OpenAI client instance configured with API key
            
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
        
        try:
            # Import the safe factory for better HTTP handling
            from arshai.clients.utils.safe_http_client import SafeHttpClientFactory
            
            self.logger.info("Creating OpenAI client with safe HTTP configuration")
            client = SafeHttpClientFactory.create_openai_client(api_key=api_key)
            
            self.logger.info("OpenAI client created successfully with safe configuration")
            return client
            
        except ImportError as e:
            self.logger.warning(f"Safe HTTP client factory not available: {e}, using default client")
            # Fallback to original implementation
            return OpenAI(api_key=api_key)
        
        except Exception as e:
            self.logger.error(f"Failed to create safe OpenAI client: {e}")
            # Final fallback to ensure system keeps working
            self.logger.info("Using fallback OpenAI client configuration")
            try:
                # At least try to set a timeout for basic safety
                return OpenAI(api_key=api_key, timeout=30.0)
            except Exception as fallback_error:
                self.logger.error(f"Fallback client also failed: {fallback_error}")
                # Last resort - basic client
                return OpenAI(api_key=api_key)

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

    def _convert_callables_to_provider_format(self, functions: Dict[str, Callable]) -> List[Dict]:
        """
        Convert python callables to OpenAI function format.
        Pure conversion without execution metadata.
        
        Args:
            functions: Dictionary of callable functions to convert
        
        Returns:
            List of OpenAI-formatted function definitions
        """
        openai_functions = []
        
        # Handle callable Python functions
        for name, func in functions.items():
            try:
                # Check if it's a background task by docstring
                original_description = func.__doc__ or f"Execute {name} function"
                is_background_task = original_description.startswith("BACKGROUND TASK:")
                
                function_def = self._python_function_to_openai_function(func, name, is_background=is_background_task)
                openai_functions.append({
                    "type": "function",
                    "function": function_def
                })
            except Exception as e:
                self.logger.warning(f"Failed to convert function {name}: {str(e)}")
                continue
        
        return openai_functions

    def _python_function_to_openai_function(self, func, name: str, is_background: bool = False) -> Dict[str, Any]:
        """Convert a Python function to OpenAI function format using introspection."""
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Extract description from docstring (keeping original for background tasks)
            description = func.__doc__ or f"Execute {name} function"
            description = description.strip()
            
            # Build parameters schema
            properties = {}
            required = []
            
            # Get type hints if available
            type_hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
            
            for param_name, param in sig.parameters.items():
                # Skip 'self' parameter
                if param_name == 'self':
                    continue
                    
                # Get parameter type from type hints or annotation
                param_type = "string"  # default
                if param_name in type_hints:
                    param_type = self._python_type_to_json_schema_type(type_hints[param_name])
                elif param.annotation != inspect.Parameter.empty:
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
                },
                "strict": True
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to inspect function {name}: {str(e)}")
            # Return basic fallback schema
            description = func.__doc__ or f"Execute {name} function"
            
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
                "parameters": schema,
                "strict": True
            }
        }

    def _process_function_calls_for_orchestrator(self, tool_calls, input: ILLMInput) -> tuple:
        """
        Process function calls and prepare them for the orchestrator using object-based approach.
        
        Args:
            tool_calls: Tool calls from OpenAI response
            input: The LLM input
            
        Returns:
            Tuple of (function_calls_list, structured_response)
        """
        function_calls_list = []
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
                function_calls_list.append(FunctionCall(
                    name=function_name,
                    args=function_args,
                    call_id=call_id,
                    is_background=True
                ))
            # Check if it's a regular function
            elif function_name in (input.regular_functions or {}):
                function_calls_list.append(FunctionCall(
                    name=function_name,
                    args=function_args,
                    call_id=call_id,
                    is_background=False
                ))
            else:
                self.logger.warning(f"Function {function_name} not found in available functions or background tasks")
        
        return function_calls_list, structured_response

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
        
        # Use response_format for structured output if needed
        if input.structure_type:
            # Convert TypedDict to BaseModel for SDK compatibility if needed
            sdk_structure_type = convert_typeddict_to_basemodel(input.structure_type)
            
            # Use beta.chat.completions.parse for structured output
            response: ParsedResponse = self._client.beta.chat.completions.parse(
                **kwargs,
                response_format=sdk_structure_type
            )
        else:
            # Regular chat completion
            response = self._client.chat.completions.create(**kwargs)
        
        # Process usage metadata
        usage = self._standardize_usage_metadata(
            response.usage if hasattr(response, 'usage') else None,
            self._get_provider_name(),
            self.config.model,
            getattr(response, 'id', None)
        )
        
        # Handle response based on whether it's structured or not
        if input.structure_type:
            # For structured output using parse(), extract the parsed content
            if hasattr(response, 'parsed') and response.parsed:
                return {"llm_response": response.parsed, "usage": usage}
            else:
                # Fallback for failed structured output
                return {"llm_response": f"Failed to generate structured response of type {input.structure_type.__name__}", "usage": usage}
        else:
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
        
        # Convert all functions using the new unified approach
        all_functions = {}
        if input.regular_functions:
            all_functions.update(input.regular_functions)
        if input.background_tasks:
            all_functions.update(input.background_tasks)
        
        if all_functions:
            openai_tools.extend(self._convert_callables_to_provider_format(all_functions))
        
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
                    accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                
                message = response.choices[0].message
                
                # Check for function calls
                if message.tool_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(message.tool_calls)} function calls")
                    
                    # Process function calls for orchestrator
                    function_calls_list, structured_response = self._process_function_calls_for_orchestrator(message.tool_calls, input)
                    
                    # If we got a structured response, return it immediately
                    if structured_response is not None:
                        self.logger.info(f"Turn {current_turn}: Received structured response via function call")
                        return {"llm_response": structured_response, "usage": accumulated_usage}
                    
                    # Execute functions via orchestrator using new object-based approach
                    if function_calls_list:
                        # Create execution input
                        execution_input = FunctionExecutionInput(
                            function_calls=function_calls_list,
                            available_functions=input.regular_functions or {},
                            available_background_tasks=input.background_tasks or {}
                        )
                        
                        execution_result = await self._execute_functions_with_orchestrator(execution_input)
                        
                        # Add function results to conversation
                        self._add_function_results_to_messages(execution_result, messages)
                        
                        # Continue if we have regular functions (need to continue conversation)
                        regular_function_calls = [call for call in function_calls_list if not call.is_background]
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
                self.logger.error(f"Error in OpenAI chat_with_functions turn {current_turn}: {str(e)}")
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
        
        # Track usage and collected data
        accumulated_usage = None
        collected_text = ""
        
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
                yield {"llm_response": collected_text}
        
        # Final yield with usage information
        yield {"llm_response": None, "usage": accumulated_usage}

    async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex streaming with tools and/or background tasks."""
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
        
        # Convert all functions using the new unified approach
        all_functions = {}
        if input.regular_functions:
            all_functions.update(input.regular_functions)
        if input.background_tasks:
            all_functions.update(input.background_tasks)
        
        if all_functions:
            openai_tools.extend(self._convert_callables_to_provider_format(all_functions))
        
        # Multi-turn streaming conversation for function calling  
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Stream function calling turn: {current_turn}")
            
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
                
                # Progressive streaming state management
                streaming_state = StreamingExecutionState()
                collected_text = ""
                chunk_count = 0
                # Track tool calls for progressive processing
                tool_calls_in_progress = {}
                
                self.logger.debug(f"Starting progressive stream processing for turn {current_turn}")
                
                # Process streaming response with progressive function execution
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
                        collected_text += delta.content
                        # For structured output, we only yield content via function calls, not direct content
                        if not input.structure_type:
                            yield {"llm_response": collected_text}
                    
                    # Handle tool calls streaming with progressive execution
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_delta in delta.tool_calls:
                            # Use the index from the tool_delta
                            tool_index = tool_delta.index
                            
                            # Initialize tool call tracking if needed
                            if tool_index not in tool_calls_in_progress:
                                tool_calls_in_progress[tool_index] = {
                                    "id": "",
                                    "function": {"name": "", "arguments": ""}
                                }
                            
                            current_tool_call = tool_calls_in_progress[tool_index]
                            
                            # Update tool call with new delta information
                            if tool_delta.id:
                                current_tool_call["id"] = tool_delta.id
                                
                            if hasattr(tool_delta, 'function'):
                                if tool_delta.function.name:
                                    current_tool_call["function"]["name"] = tool_delta.function.name
                                    
                                if tool_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += tool_delta.function.arguments
                            
                            # Check if this is a structure output function
                            if (input.structure_type and 
                                current_tool_call["function"]["name"] and 
                                current_tool_call["function"]["name"].lower() == input.structure_type.__name__.lower()):
                                
                                try:
                                    # For structure functions, double-check JSON completeness
                                    is_complete, fixed_json = is_json_complete(current_tool_call["function"]["arguments"])
                                    if not is_complete:
                                        continue  # Wait for more data
                                    else:
                                        # Use parse_to_structure for consistent parsing
                                        structured_response = parse_to_structure(fixed_json, input.structure_type)
                                        # Yield the structured response immediately
                                        yield {"llm_response": structured_response, "usage": accumulated_usage}
                                        # Mark that we've yielded a structure response
                                except Exception as e:
                                    self.logger.error(f"Failed to parse structure output progressively: {str(e)}")
                                continue
                            
                            # Progressive execution: check if function is complete
                            if self._is_function_complete(current_tool_call["function"]):
                                function_name = current_tool_call["function"]["name"]
                                
                                # Regular function - execute progressively
                                function_call = FunctionCall(
                                    name=function_name,
                                    args=json.loads(current_tool_call["function"]["arguments"]) if current_tool_call["function"]["arguments"] else {},
                                    call_id=current_tool_call["id"] or f"{function_name}_{tool_index}",
                                    is_background=function_name in (input.background_tasks or {})
                                )
                                
                                # Execute progressively if not already executed
                                if not streaming_state.is_already_executed(function_call):
                                    self.logger.info(f"Executing function progressively: {function_call.name} at {time.time()}")
                                    try:
                                        task = await self._execute_function_progressively(function_call, input)
                                        streaming_state.add_function_task(task, function_call)
                                    except Exception as e:
                                        self.logger.error(f"Progressive execution failed for {function_call.name}: {str(e)}")
                
                self.logger.debug(f"Turn {current_turn}: Stream ended. Processed {chunk_count} chunks. "
                               f"Tool calls: {len(tool_calls_in_progress)}, Text collected: {len(collected_text)} chars, "
                               f"Progressive tasks: {len(streaming_state.active_function_tasks)}")
                
                # Progressive streaming: gather results from executed functions
                if streaming_state.active_function_tasks:
                    self.logger.info(f"Gathering results from {len(streaming_state.active_function_tasks)} progressive function executions")
                    
                    # Gather progressive execution results
                    execution_result = await self._gather_progressive_results(streaming_state.active_function_tasks)
                    
                    # Add failed functions to context for model awareness
                    if execution_result.get('failed_functions'):
                        context_messages = []
                        for msg in messages:
                            if isinstance(msg, dict) and 'content' in msg:
                                context_messages.append(msg['content'])
                        self._add_failed_functions_to_context(execution_result['failed_functions'], context_messages)
                        # Update messages with context
                        if context_messages:
                            messages.append({"role": "user", "content": context_messages[-1]})
                    
                    # Add function results to conversation
                    self._add_function_results_to_messages(execution_result, messages)
                    
                    # Check if we have regular function calls that require conversation continuation
                    regular_results = execution_result.get('regular_results', [])
                    if regular_results:
                        current_turn += 1
                        continue
                
                # Stream completed for this turn
                self.logger.debug(f"Turn {current_turn}: Stream completed")
                break
                
            except Exception as e:
                self.logger.error(f"Error in OpenAI stream_with_functions turn {current_turn}: {str(e)}")
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