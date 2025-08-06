"""
Azure OpenAI implementation using the new BaseLLMClient framework.

Migrated to use structured function orchestration, dual interface support,
and standardized patterns from the Arshai framework.
"""

import os
import json
import time
import inspect
from typing import Dict, Any, TypeVar, Type, Union, AsyncGenerator, List
from openai import AzureOpenAI
from openai.types.responses import ParsedResponse

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import (
    is_json_complete,
    parse_to_structure,
    convert_typeddict_to_basemodel,
)
from arshai.llms.utils.function_execution import FunctionCall, FunctionExecutionInput

T = TypeVar("T")

# Structure instructions template used across methods
STRUCTURE_INSTRUCTIONS_TEMPLATE = """

You MUST use structured output formatting as specified.
Follow the required structure format exactly.
The response must be properly formatted according to the schema."""


class AzureClient(BaseLLMClient):
    """
    Azure OpenAI implementation using the new framework architecture.
    
    This client demonstrates how to implement a provider using the new
    BaseLLMClient framework with Azure-specific optimizations.
    """
    
    def __init__(self, config: ILLMConfig, azure_deployment: str = None, api_version: str = None):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            config: Configuration for the LLM
            azure_deployment: Optional deployment name (if not provided, will be read from env)
            api_version: Optional API version (if not provided, will be read from env)
        """
        # Azure-specific configuration
        self.azure_deployment = azure_deployment or os.environ.get("AZURE_DEPLOYMENT")
        self.api_version = api_version or os.environ.get("AZURE_API_VERSION")
        
        if not self.azure_deployment:
            raise ValueError("Azure deployment is required. Set AZURE_DEPLOYMENT environment variable.")
        
        if not self.api_version:
            raise ValueError("Azure API version is required. Set AZURE_API_VERSION environment variable.")
        
        # Initialize base client (handles common setup)
        super().__init__(config)
    
    def _initialize_client(self) -> Any:
        """
        Initialize the Azure OpenAI client with safe HTTP configuration.
        
        Returns:
            AzureOpenAI client instance configured with deployment and API version
            
        Raises:
            ValueError: If required Azure configuration is missing
        """
        try:
            # Import the safe factory for better HTTP handling
            from arshai.clients.utils.safe_http_client import SafeHttpClientFactory
            
            self.logger.info("Creating Azure OpenAI client with safe HTTP configuration")
            
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
            
            # Try to create Azure client with safe HTTP client
            try:
                client = AzureOpenAI(
                    azure_deployment=self.azure_deployment,
                    api_version=self.api_version,
                    http_client=safe_http_client,
                    max_retries=3
                )
                self.logger.info("Azure OpenAI client created successfully with safe configuration")
                return client
            except TypeError as e:
                if 'http_client' in str(e) or 'max_retries' in str(e):
                    self.logger.warning("AzureOpenAI does not support http_client or max_retries parameter in this version")
                    # Close the unused httpx client
                    safe_http_client.close()
                    # Fallback to basic Azure client
                    return AzureOpenAI(
                        azure_deployment=self.azure_deployment,
                        api_version=self.api_version
                    )
                else:
                    raise
                    
        except ImportError as e:
            self.logger.warning(f"Safe HTTP client factory not available: {e}, using default Azure client")
            # Fallback to original implementation
            return AzureOpenAI(
                azure_deployment=self.azure_deployment,
                api_version=self.api_version
            )
        
        except Exception as e:
            self.logger.error(f"Failed to create safe Azure OpenAI client: {e}")
            # Final fallback to ensure system keeps working
            self.logger.info("Using fallback Azure OpenAI client configuration")
            return AzureOpenAI(
                azure_deployment=self.azure_deployment,
                api_version=self.api_version
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

    def _convert_messages_to_response_input(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style messages to Azure ResponseInputParam format."""
        response_input = []
        for message in messages:
            response_message = {
                "type": "message",
                "role": message["role"],
                "content": message["content"]
            }
            response_input.append(response_message)
        return response_input

    def _convert_functions_to_azure_format(self, functions: Union[List[Dict], Dict[str, Any]], is_background: bool = False) -> List[Dict]:
        """
        Unified method to convert functions to Azure format.
        
        Args:
            functions: Either list of tool schemas or dict of callable functions
            is_background: Whether these are background tasks
        
        Returns:
            List of Azure-formatted function definitions
        """
        azure_functions = []
        
        if isinstance(functions, list):
            # Handle pre-defined tool schemas
            for tool in functions:
                # Get tool properties
                description = tool.get("description", "")
                
                # Enhance description for background tasks
                if is_background:
                    description = f"BACKGROUND TASK: {description}. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."
                
                # Ensure additionalProperties is False for strict mode
                parameters = tool.get("parameters", {})
                if isinstance(parameters, dict) and "additionalProperties" not in parameters:
                    parameters["additionalProperties"] = False

                azure_functions.append({
                    "type": "function",
                    "name": tool.get("name"),
                    "description": description,
                    "parameters": parameters,
                    "strict": True  # Required for structured output
                })
        
        elif isinstance(functions, dict):
            # Handle callable Python functions
            for name, func in functions.items():
                try:
                    function_def = self._python_function_to_azure_function(func, name, is_background=is_background)
                    function_def["type"] = "function"  # Add type to create flat structure
                    azure_functions.append(function_def)
                except Exception as e:
                    self.logger.warning(f"Failed to convert {'background task' if is_background else 'function'} {name}: {str(e)}")
                    continue
        
        return azure_functions

    def _python_function_to_azure_function(self, func, name: str, is_background: bool = False) -> Dict[str, Any]:
        """Convert a Python function to Azure function format using introspection."""
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
                
                # For Azure strict mode, ALL parameters must be in required array
                required.append(param_name)
                
                # Add default value info to description
                if param.default != inspect.Parameter.empty:
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

    def _process_function_calls_for_orchestrator(self, function_calls, input: ILLMInput) -> tuple:
        """
        Process function calls and prepare them for the orchestrator using object-based approach.
        
        Args:
            function_calls: Function calls from Azure response
            input: The LLM input
            
        Returns:
            Tuple of (function_calls_list, structured_response)
        """
        function_calls_list = []
        structured_response = None
        
        for i, func_call in enumerate(function_calls):
            function_name = func_call.name
            try:
                # Extract function args (handles Azure response format)
                if hasattr(func_call, 'arguments'):
                    function_args = json.loads(func_call.arguments) if func_call.arguments else {}
                else:
                    function_args = {}
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse function arguments for {function_name}")
                function_args = {}
            
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
            elif function_name in (input.callable_functions or {}):
                function_calls_list.append(FunctionCall(
                    name=function_name,
                    args=function_args,
                    call_id=call_id,
                    is_background=False
                ))
            else:
                self.logger.warning(f"Function {function_name} not found in available functions or background tasks")
        
        return function_calls_list, structured_response

    def _add_function_results_to_response_input(self, execution_result: Dict, response_input: List[Dict]) -> None:
        """Add function execution results to response_input in Azure format."""
        # Add function results as messages
        for result in execution_result.get('regular_results', []):
            response_input.append({
                "type": "message",
                "role": "user",
                "content": f"Function '{result['name']}' called with arguments {result['args']} returned: {result['result']}"
            })
        
        # Add background task notifications
        for bg_message in execution_result.get('background_initiated', []):
            response_input.append({
                "type": "message",
                "role": "user",
                "content": f"Background task initiated: {bg_message}"
            })
        
        # Add completion message if we have results
        if execution_result.get('regular_results'):
            completion_msg = f"All {len(execution_result['regular_results'])} function(s) completed. Please provide your response based on these results."
            response_input.append({
                "type": "message",
                "role": "user",
                "content": completion_msg
            })

    # ========================================================================
    # FRAMEWORK-REQUIRED ABSTRACT METHODS
    # ========================================================================

    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle simple chat without tools or background tasks."""
        # Build base context in OpenAI format first
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Convert to Azure ResponseInputParam format
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
            # Add structure instructions to system prompt  
            response_input[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE
        
        response: ParsedResponse = self._client.responses.parse(**kwargs)
        
        # Process usage metadata
        usage = self._standardize_usage_metadata(
            response.usage if hasattr(response, 'usage') else None,
            self._get_provider_name(),
            self.config.model,
            getattr(response, 'id', None)
        )
        
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

    async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle complex chat with tools and/or background tasks."""
        # Build base context and prepare tools
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Prepare tools for Azure
        azure_tools = []
        
        # Add regular tools
        if input.tools_list:
            azure_tools.extend(self._convert_functions_to_azure_format(input.tools_list, is_background=False))
        
        # Add background tasks
        if input.background_tasks:
            azure_tools.extend(self._convert_functions_to_azure_format(input.background_tasks, is_background=True))
        
        # Convert to Azure ResponseInputParam format
        response_input = self._convert_messages_to_response_input(messages)
        
        # Multi-turn conversation for function calling
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Function calling turn: {current_turn}")
            
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
                    # Add structure instructions to system prompt
                    response_input[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE
                
                response = self._client.responses.parse(**kwargs)
                self.logger.info(f"Response time: {time.time() - start_time:.2f}s")
                
                # Process usage metadata using framework standardization
                if hasattr(response, "usage") and response.usage:
                    current_usage = self._standardize_usage_metadata(
                        response.usage, self._get_provider_name(), self.config.model, getattr(response, 'id', None)
                    )
                    accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                
                # Check for function calls in output
                function_calls = [output_item for output_item in response.output if output_item.type == "function_call"]
                
                if function_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(function_calls)} function calls")
                    
                    # Process function calls for orchestrator
                    function_calls_list, structured_response = self._process_function_calls_for_orchestrator(function_calls, input)
                    
                    # If we got a structured response, return it immediately
                    if structured_response is not None:
                        self.logger.info(f"Turn {current_turn}: Received structured response via function call")
                        return {"llm_response": structured_response, "usage": accumulated_usage}
                    
                    # Execute functions via orchestrator using new object-based approach
                    if function_calls_list:
                        # Create execution input
                        execution_input = FunctionExecutionInput(
                            function_calls=function_calls_list,
                            available_functions=input.callable_functions or {},
                            available_background_tasks=input.background_tasks or {}
                        )
                        
                        execution_result = await self._execute_functions_with_orchestrator(execution_input)
                        
                        # Add function results to conversation
                        self._add_function_results_to_response_input(execution_result, response_input)
                        
                        # Continue if we have regular functions (need to continue conversation)
                        regular_function_calls = [call for call in function_calls_list if not call.is_background]
                        if regular_function_calls:
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
                
                # Return text response
                if hasattr(response, 'output_text') and response.output_text:
                    self.logger.info(f"Turn {current_turn}: Received text response")
                    return {"llm_response": response.output_text, "usage": accumulated_usage}
                
            except Exception as e:
                self.logger.error(f"Error in Azure chat_with_functions turn {current_turn}: {str(e)}")
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
        # Build base context in OpenAI format first
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Convert to Azure ResponseInputParam format  
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
            # Add structure instructions to system prompt
            response_input[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE
        
        # Use proper ResponseStreamManager pattern
        with self._client.responses.stream(**kwargs) as stream:
            accumulated_usage = None
            collected_text = ""
            
            # Process streaming events
            for event in stream:
                if hasattr(event, "response") and hasattr(event.response, "usage") and event.response.usage:
                    # Usage from ResponseIncompleteEvent or ResponseCompletedEvent
                    current_usage = self._standardize_usage_metadata(
                        event.response.usage, self._get_provider_name(), self.config.model
                    )
                    accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                
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

    async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex streaming with tools and/or background tasks."""
        # Build base context and prepare tools
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Prepare tools for Azure
        azure_tools = []
        
        # Add regular tools
        if input.tools_list:
            azure_tools.extend(self._convert_functions_to_azure_format(input.tools_list, is_background=False))
        
        # Add background tasks
        if input.background_tasks:
            azure_tools.extend(self._convert_functions_to_azure_format(input.background_tasks, is_background=True))
        
        # Convert to Azure ResponseInputParam format
        response_input = self._convert_messages_to_response_input(messages)
        
        # Multi-turn streaming conversation for function calling  
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Stream function calling turn: {current_turn}")
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
                
                # Add text_format for structured output if needed
                if input.structure_type:
                    sdk_structure_type = convert_typeddict_to_basemodel(input.structure_type)
                    kwargs["text_format"] = sdk_structure_type
                    # Add structure instructions to system prompt
                    response_input[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE
                
                # Generate streaming content with tools using responses API
                with self._client.responses.stream(**kwargs) as stream:
                    
                    # Store function calls collected during streaming
                    function_calls = []
                    collected_text = ""
                    chunk_count = 0

                    self.logger.debug(f"Starting stream processing for turn {current_turn}")

                    # Process streaming response events
                    for event in stream:
                        chunk_count += 1                        
                        # Handle usage metadata from response completion events
                        if hasattr(event, "response") and hasattr(event.response, "usage") and event.response.usage:
                            current_usage = self._standardize_usage_metadata(
                                event.response.usage, self._get_provider_name(), self.config.model
                            )
                            accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                        
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

                    # Process function calls if any were collected
                    if function_calls:
                        # Process function calls for orchestrator
                        function_calls_list, structured_response = self._process_function_calls_for_orchestrator(function_calls, input)
                        
                        # If we got a structured response, yield it and break
                        if structured_response is not None:
                            yield {"llm_response": structured_response, "usage": accumulated_usage}
                            break
                        
                        # Execute functions via orchestrator using new object-based approach
                        if function_calls_list:
                            # Create execution input
                            execution_input = FunctionExecutionInput(
                                function_calls=function_calls_list,
                                available_functions=input.callable_functions or {},
                                available_background_tasks=input.background_tasks or {}
                            )
                            
                            execution_result = await self._execute_functions_with_orchestrator(execution_input)
                            
                            # Add function results to conversation
                            self._add_function_results_to_response_input(execution_result, response_input)
                            
                            # Continue if we have regular functions
                            regular_function_calls = [call for call in function_calls_list if not call.is_background]
                            if regular_function_calls:
                                has_regular_functions = True
                                current_turn += 1
                                continue
                
                # Stream completed for this turn
                self.logger.debug(f"Turn {current_turn}: Stream completed")
                break
                
            except Exception as e:
                self.logger.error(f"Error in Azure stream_with_functions turn {current_turn}: {str(e)}")
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