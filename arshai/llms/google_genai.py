"""
Google Gemini implementation using the new BaseLLMClient framework.

Migrated to use structured function orchestration, dual interface support,
and standardized patterns from the Arshai framework. Serves as the reference
implementation as mentioned in CLAUDE.md.
"""

import os
import time
from typing import Dict, Any, TypeVar, Type, Union, AsyncGenerator, List, Callable
from google.oauth2 import service_account
import google.genai as genai
from google.genai.types import (
    GenerateContentConfig,
    ThinkingConfig,
    FunctionDeclaration,
    Tool,
    SpeechConfig,
    Schema,
    AutomaticFunctionCallingConfig,
)

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import (
    is_json_complete,
    parse_to_structure,
)
from arshai.llms.utils.function_execution import FunctionCall, FunctionExecutionInput, StreamingExecutionState

T = TypeVar("T")

# Structure instructions template used across methods
STRUCTURE_INSTRUCTIONS_TEMPLATE = """

You MUST use structured output formatting as specified.
Follow the required structure format exactly.
The response must be properly formatted according to the schema."""


class GeminiClient(BaseLLMClient):
    """
    Google Gemini implementation using the new framework architecture.
    
    This client serves as the reference implementation mentioned in CLAUDE.md
    and demonstrates best practices for the new BaseLLMClient framework.
    """
    
    def __init__(self, config: ILLMConfig, observability_config=None):
        """
        Initialize the Gemini client with configuration.

        Args:
            config: Configuration for the LLM
            observability_config: Optional observability configuration for metrics collection

        Supports dual authentication methods:
        1. API Key (simpler): Set GOOGLE_API_KEY environment variable
        2. Service Account (enterprise): Set VERTEX_AI_SERVICE_ACCOUNT_PATH,
           VERTEX_AI_PROJECT_ID, VERTEX_AI_LOCATION environment variables
        """
        # Gemini-specific configuration
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.service_account_path = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = os.getenv("VERTEX_AI_LOCATION")
        self.model_config = getattr(config, "config", {})

        # Initialize base client (handles common setup including observability)
        super().__init__(config, observability_config=observability_config)
        
        # Get model-specific configuration from config dict
        self.model_config = getattr(config, 'config', {})
        
        self.logger.info(f"Initializing Gemini client with model: {self.config.model}")
        
        # Initialize the client
        self._client = self._initialize_client()
        self._http_client = None  # Track httpx client if using custom one
    
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
        """Close the GenAI client and cleanup connections."""
        try:
            # Close the httpx client if we have one
            if self._http_client is not None:
                self._http_client.close()
                self._http_client = None
                self.logger.info("Closed custom httpx client for Gemini")
            
            # Try to close the GenAI client if it has a close method
            if hasattr(self._client, 'close'):
                self._client.close()
                self.logger.info("Closed Gemini client")
            elif hasattr(self._client, '_transport') and hasattr(self._client._transport, 'close'):
                # Try to close underlying transport
                self._client._transport.close()
                self.logger.info("Closed Gemini client transport")
        except Exception as e:
            self.logger.warning(f"Error closing Gemini client: {e}")    
    def _initialize_client(self) -> Any:
        """
        Initialize the Google GenAI client with safe HTTP configuration.
        
        Returns:
            Google GenAI client instance with safe HTTP configuration
            
        Raises:
            ValueError: If neither authentication method is properly configured
        """
        try:
            # Import the safe factory for better HTTP handling
            from arshai.clients.utils.safe_http_client import SafeHttpClientFactory
            
            # Try API key authentication first (simpler)
            if self.api_key:
                self.logger.info("Creating GenAI client with API key and safe HTTP configuration")
                try:
                    client = SafeHttpClientFactory.create_genai_client(api_key=self.api_key)
                    self._test_client_connection(client)
                    self.logger.info("GenAI client created successfully with safe configuration")
                    return client
                except Exception as e:
                    self.logger.error(f"API key authentication with safe config failed: {str(e)}")
                    # Try fallback with basic client
                    self.logger.info("Trying fallback GenAI client with API key")
                    client = genai.Client(api_key=self.api_key)
                    self._test_client_connection(client)
                    return client
            
            # Try service account authentication
            elif self.service_account_path and self.project_id and self.location:
                self.logger.info("Creating GenAI client with service account and safe HTTP configuration")
                try:
                    # Load service account credentials
                    credentials = service_account.Credentials.from_service_account_file(
                        self.service_account_path,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    
                    client = SafeHttpClientFactory.create_genai_client(
                        vertexai=True,
                        project=self.project_id,
                        location=self.location,
                        credentials=credentials
                    )
                    
                    self._test_client_connection(client)
                    self.logger.info("GenAI service account client created successfully with safe configuration")
                    return client
                    
                except FileNotFoundError:
                    self.logger.error(f"Service account file not found: {self.service_account_path}")
                    raise ValueError(f"Service account file not found: {self.service_account_path}")
                except Exception as e:
                    self.logger.error(f"Service account authentication with safe config failed: {str(e)}")
                    # Try fallback with basic client
                    self.logger.info("Trying fallback GenAI client with service account")
                    credentials = service_account.Credentials.from_service_account_file(
                        self.service_account_path,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    client = genai.Client(
                        vertexai=True,
                        project=self.project_id,
                        location=self.location,
                        credentials=credentials
                    )
                    self._test_client_connection(client)
                    return client
            
            else:
                # No valid authentication method found
                error_msg = (
                    "No valid authentication method found for Gemini. Please set either:\n"
                    "1. GOOGLE_API_KEY for API key authentication, or\n"
                    "2. VERTEX_AI_SERVICE_ACCOUNT_PATH, VERTEX_AI_PROJECT_ID, and VERTEX_AI_LOCATION "
                    "for service account authentication"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        except ImportError as e:
            self.logger.warning(f"Safe HTTP client factory not available: {e}, using default GenAI client")
            
            # Fallback to original implementation without safe HTTP configuration
            if self.api_key:
                self.logger.info("Using API key authentication for Gemini (fallback)")
                try:
                    client = genai.Client(api_key=self.api_key)
                    self._test_client_connection(client)
                    return client
                except Exception as e:
                    self.logger.error(f"API key authentication failed: {str(e)}")
                    raise ValueError(f"Invalid Google API key: {str(e)}")
            
            elif self.service_account_path and self.project_id and self.location:
                self.logger.info("Using service account authentication for Gemini (fallback)")
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        self.service_account_path,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    client = genai.Client(
                        vertexai=True,
                        project=self.project_id,
                        location=self.location,
                        credentials=credentials
                    )
                    self._test_client_connection(client)
                    return client
                except FileNotFoundError:
                    self.logger.error(f"Service account file not found: {self.service_account_path}")
                    raise ValueError(f"Service account file not found: {self.service_account_path}")
                except Exception as e:
                    self.logger.error(f"Service account authentication failed: {str(e)}")
                    raise ValueError(f"Service account authentication failed: {str(e)}")
            
            else:
                error_msg = (
                    "No valid authentication method found for Gemini. Please set either:\n"
                    "1. GOOGLE_API_KEY for API key authentication, or\n"
                    "2. VERTEX_AI_SERVICE_ACCOUNT_PATH, VERTEX_AI_PROJECT_ID, and VERTEX_AI_LOCATION "
                    "for service account authentication"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

    # ========================================================================
    # PROVIDER-SPECIFIC HELPER METHODS
    # ========================================================================

    def _extract_and_standardize_usage(self, response: Any) -> Dict[str, Any]:
        """
        Extract and standardize usage metadata from Gemini response.
        
        Updated for accurate extraction based on 2025 Gemini API format:
        - prompt_token_count -> input_tokens
        - candidates_token_count -> output_tokens (excludes thinking tokens)
        - thoughts_token_count -> thinking_tokens (Gemini 2.5+ models)
        - total_token_count -> total_tokens
        
        Args:
            response: Gemini response object
            
        Returns:
            Standardized usage metadata dictionary
        """
        if not hasattr(response, 'usage_metadata') or not response.usage_metadata:
            return {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "thinking_tokens": 0, "tool_calling_tokens": 0,
                "provider": self._provider_name, "model": self.config.model,
                "request_id": getattr(response, 'id', None)
            }
        
        usage = response.usage_metadata
        
        # Extract base token counts
        input_tokens = getattr(usage, 'prompt_token_count', 0)
        output_tokens = getattr(usage, 'candidates_token_count', 0)
        total_tokens = getattr(usage, 'total_token_count', 0)
        
        # Extract thinking tokens (Gemini 2.5+ models with reasoning)
        thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
        
        # Gemini doesn't separate tool calling tokens from candidates_token_count
        tool_calling_tokens = 0
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "thinking_tokens": thinking_tokens,
            "tool_calling_tokens": tool_calling_tokens,
            "provider": self._provider_name,
            "model": self.config.model,
            "request_id": getattr(response, 'id', None)
        }

    def _test_client_connection(self, client) -> None:
        """Test the client connection with a minimal request."""
        try:
            response = client.models.generate_content(
                model=self.config.model,
                contents=["Test connection"],
                config=GenerateContentConfig(max_output_tokens=1, temperature=0.0),
            )
            self.logger.info("Gemini client connection test successful")
        except Exception as e:
            raise Exception(f"Client connection test failed: {str(e)}")

    def _prepare_base_context(self, input: ILLMInput) -> str:
        """Build base conversation context from system prompt and user message."""
        return f"{input.system_prompt}\n\nUser: {input.user_message}"

    def _convert_callables_to_provider_format(self, functions: Dict[str, Callable]) -> List[FunctionDeclaration]:
        """
        Convert python callables to Gemini FunctionDeclaration format.
        Pure conversion without execution metadata.
        
        Args:
            functions: Dictionary of callable functions to convert
        
        Returns:
            List of Gemini FunctionDeclaration objects
        """
        gemini_declarations = []
        
        # Handle callable Python functions
        for name, callable_func in functions.items():
            try:
                # Use Gemini SDK's auto-generation from callable
                declaration = FunctionDeclaration.from_callable(
                    callable=callable_func, 
                    client=self._client
                )
                
                # Get original description and check if it's a background task
                original_description = declaration.description or callable_func.__doc__ or name
                is_background_task = original_description.startswith("BACKGROUND TASK:")
                
                # Create enhanced declaration with the dictionary key as name (not the function name)
                enhanced_declaration = FunctionDeclaration(
                    name=name,  # Use the dictionary key as function name
                    description=original_description,
                    parameters=declaration.parameters
                )
                
                gemini_declarations.append(enhanced_declaration)
                self.logger.debug(f"Auto-generated declaration for {'background task' if is_background_task else 'function'}: {name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to auto-generate declaration for function {name}: {str(e)}")
                # Fallback: create basic declaration
                original_description = callable_func.__doc__ or name
                        
                gemini_declarations.append(
                    FunctionDeclaration(
                        name=name,
                        description=original_description,
                        parameters={"type": "object", "properties": {}, "required": []},
                    )
                )
        
        return gemini_declarations

    def _create_generation_config(self, structure_type: Type[T] = None, tools=None) -> GenerateContentConfig:
        """Create generation config from model config dict."""
        # Start with base temperature from main config
        config_dict = {"temperature": self.config.temperature}

        # Process all model config parameters and convert nested dicts to proper classes
        for key, value in self.model_config.items():
            if key == "thinking_config" and isinstance(value, dict):
                config_dict["thinking_config"] = ThinkingConfig(**value)
            elif key == "speech_config" and isinstance(value, dict):
                config_dict["speech_config"] = SpeechConfig(**value)
            elif key == "response_schema" and isinstance(value, dict):
                config_dict["response_schema"] = Schema(**value)
            elif key == "response_json_schema" and isinstance(value, dict):
                config_dict["response_json_schema"] = value
            else:
                config_dict[key] = value

        # Add structured output configuration if requested
        if structure_type:
            config_dict["response_mime_type"] = "application/json"
            
            # Create response schema from Pydantic model
            schema_dict = structure_type.model_json_schema()
            
            try:
                config_dict["response_schema"] = Schema(**schema_dict)
            except Exception:
                config_dict["response_schema"] = schema_dict

        # Add tools to config if provided and disable automatic function calling for manual orchestration
        if tools:
            config_dict["tools"] = tools
            config_dict["automatic_function_calling"] = AutomaticFunctionCallingConfig(disable=True)
            self.logger.debug("Disabled automatic function calling for manual orchestration")

        return GenerateContentConfig(**config_dict)

    def _process_function_calls_for_orchestrator(self, function_calls, input: ILLMInput) -> tuple:
        """
        Process function calls and prepare them for the orchestrator using object-based approach.
        
        Args:
            function_calls: Function calls from Gemini response
            input: The LLM input
            
        Returns:
            Tuple of (function_calls_list, structured_response)
        """
        function_calls_list = []
        structured_response = None
        
        for i, func_call in enumerate(function_calls):
            function_name = func_call.name
            function_args = dict(func_call.args) if hasattr(func_call, 'args') and func_call.args else {}
            
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

    def _add_function_results_to_contents(self, execution_result: Dict, contents: List[str]) -> None:
        """Add function execution results to contents list in Gemini format."""
        # Add function results as context
        for result in execution_result.get('regular_results', []):
            contents.append(f"Function '{result['name']}' called with arguments {result['args']} returned: {result['result']}")
        
        # Add background task notifications
        for bg_message in execution_result.get('background_initiated', []):
            contents.append(f"Background task initiated: {bg_message}")
        
        # Add completion message if we have results
        if execution_result.get('regular_results'):
            completion_msg = f"All {len(execution_result['regular_results'])} function(s) completed. Please provide your response based on these results."
            contents.append(completion_msg)

    def _extract_text_from_response(self, response) -> str:
        """
        Extract text content from Gemini response, handling different response formats.
        
        Args:
            response: Gemini API response object
            
        Returns:
            Extracted text content or empty string if not found
        """
        # Try direct text access first
        if hasattr(response, "text") and response.text:
            return response.text
        
        # Try candidates structure
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                if hasattr(candidate.content, "parts") and candidate.content.parts:
                    part = candidate.content.parts[0]
                    if hasattr(part, 'text') and part.text:
                        return part.text
        
        return ""

    # ========================================================================
    # FRAMEWORK-REQUIRED ABSTRACT METHODS
    # ========================================================================

    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle simple chat without tools or background tasks."""
        # Build base context
        contents = [self._prepare_base_context(input)]
        
        # Add structured output instructions if needed
        if input.structure_type:
            contents[0] += STRUCTURE_INSTRUCTIONS_TEMPLATE
        
        # Generate content without tools
        response = self._client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=self._create_generation_config(input.structure_type),
        )
        self.logger.debug(f"Response: {response}")

        # Process usage metadata
        usage = self._extract_and_standardize_usage(response)

        # Extract text from response using robust method
        response_text = self._extract_text_from_response(response)
        
        # Handle structured output
        if input.structure_type:
            if response_text:
                try:
                    final_response = parse_to_structure(response_text, input.structure_type)
                    return {"llm_response": final_response, "usage": usage}
                except ValueError as e:
                    return {"llm_response": f"Failed to parse structured response: {str(e)}", "usage": usage}
        
        # Handle regular text response
        if response_text:
            return {"llm_response": response_text, "usage": usage}
        else:
            return {"llm_response": "No response generated", "usage": usage}

    async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle complex chat with tools and/or background tasks."""
        # Build base context and prepare tools
        contents = [self._prepare_base_context(input)]
        
        # Prepare tools for Gemini
        gemini_tools = []
        
        # Convert all functions using the new unified approach
        all_functions = {}
        if input.regular_functions:
            all_functions.update(input.regular_functions)
        if input.background_tasks:
            all_functions.update(input.background_tasks)
        
        if all_functions:
            gemini_tools.extend(self._convert_callables_to_provider_format(all_functions))
        
        # Note: Background tasks are now included in the unified conversion above
        
        # Multi-turn conversation for function calling
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Function calling turn: {current_turn}")
            
            try:
                start_time = time.time()
                
                # Create tool objects for Gemini
                tools = [Tool(function_declarations=gemini_tools)] if gemini_tools else None
                
                # Generate content with tools (manual mode - no automatic execution)
                response = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=self._create_generation_config(input.structure_type, tools),
                )
                self.logger.info(f"Response time: {time.time() - start_time:.2f}s")
                
                # Process usage metadata using framework standardization
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    current_usage = self._extract_and_standardize_usage(response)
                    accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                
                # Check for function calls
                if hasattr(response, "function_calls") and response.function_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(response.function_calls)} function calls")
                    
                    # Process function calls for orchestrator
                    function_calls_list, structured_response = self._process_function_calls_for_orchestrator(response.function_calls, input)
                    
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
                        self._add_function_results_to_contents(execution_result, contents)
                        
                        # Continue if we have regular functions (need to continue conversation)
                        regular_function_calls = [call for call in function_calls_list if not call.is_background]
                        if regular_function_calls:
                            current_turn += 1
                            continue
                
                # Extract text from response
                response_text = self._extract_text_from_response(response)
                
                # Check for structured response
                if input.structure_type:
                    if response_text:
                        try:
                            final_response = parse_to_structure(response_text, input.structure_type)
                            self.logger.info(f"Turn {current_turn}: Received structured response")
                            return {"llm_response": final_response, "usage": accumulated_usage}
                        except ValueError as e:
                            self.logger.warning(f"Structured parsing failed: {str(e)}")
                            contents.append(response_text)
                
                # Return text response
                if response_text:
                    self.logger.info(f"Turn {current_turn}: Received text response")
                    return {"llm_response": response_text, "usage": accumulated_usage}
                
                current_turn += 1
                
            except Exception as e:
                self.logger.error(f"Error in Gemini chat_with_functions turn {current_turn}: {str(e)}")
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
        # Build base context
        contents = [self._prepare_base_context(input)]
        
        # Add structured output instructions if needed
        if input.structure_type:
            contents[0] += STRUCTURE_INSTRUCTIONS_TEMPLATE
        
        # Generate streaming content
        stream = self._client.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=self._create_generation_config(input.structure_type),
        )

        accumulated_usage = None
        collected_text = ""

        for chunk in stream:
            # Process usage metadata safely
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                current_usage = self._extract_and_standardize_usage(chunk)
                accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)

            # Extract text from chunk
            chunk_text = self._extract_text_from_response(chunk)
            if chunk_text:
                collected_text += chunk_text
                
                # Handle structured vs regular text streaming
                if input.structure_type:                            
                    # Try to parse the accumulated content as JSON for structured output
                    is_complete, fixed_json = is_json_complete(collected_text)
                    if is_complete:
                        try:
                            final_response = parse_to_structure(fixed_json, input.structure_type)
                            yield {"llm_response": final_response}
                        except ValueError:
                            pass
                else:
                    # Regular text streaming
                    yield {"llm_response": collected_text}

        # Final yield with usage information
        yield {"llm_response": None, "usage": accumulated_usage}

    async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex streaming with tools and/or background tasks."""
        # Build base context and prepare tools
        contents = [self._prepare_base_context(input)]
        
        # Prepare tools for Gemini
        gemini_tools = []
        
        # Convert all functions using the new unified approach
        all_functions = {}
        if input.regular_functions:
            all_functions.update(input.regular_functions)
        if input.background_tasks:
            all_functions.update(input.background_tasks)
        
        if all_functions:
            gemini_tools.extend(self._convert_callables_to_provider_format(all_functions))
        
        # Note: Background tasks are now included in the unified conversion above
        
        # Multi-turn streaming conversation for function calling  
        current_turn = 0
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Stream function calling turn: {current_turn}")
            
            try:
                # Create tool objects for Gemini
                tools = [Tool(function_declarations=gemini_tools)] if gemini_tools else None
                
                # Generate streaming content with tools (manual mode)
                stream = self._client.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=self._create_generation_config(input.structure_type, tools),
                )
                
                # Progressive streaming state management
                streaming_state = StreamingExecutionState()
                collected_text = ""
                chunk_count = 0
                # Track tool calls for progressive processing
                tool_calls_in_progress = {}
                
                self.logger.debug(f"Starting progressive stream processing for turn {current_turn}")

                # Process streaming response with progressive function execution
                for chunk in stream:
                    chunk_count += 1
                    
                    # Handle usage metadata
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        current_usage = self._extract_and_standardize_usage(chunk)
                        accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
                    
                    # Extract text from chunk
                    chunk_text = self._extract_text_from_response(chunk)
                    if chunk_text:
                        collected_text += chunk_text
                        # For structured output, we only yield content via function calls, not direct content
                        if not input.structure_type:
                            yield {"llm_response": collected_text}
                        else:
                            # Check if JSON is complete for structured response
                            is_complete, fixed_json = is_json_complete(collected_text)
                            if is_complete:
                                try:
                                    final_response = parse_to_structure(fixed_json, input.structure_type)
                                    yield {"llm_response": final_response, "usage": accumulated_usage}
                                except ValueError:
                                    pass
                    
                    # Handle function calls with progressive execution
                    if hasattr(chunk, "function_calls") and chunk.function_calls:
                        for func_call in chunk.function_calls:
                            # For Gemini, function calls arrive complete in chunks
                            # Track them in progress similar to OpenRouter pattern
                            call_index = len(tool_calls_in_progress)
                            
                            if call_index not in tool_calls_in_progress:
                                tool_calls_in_progress[call_index] = {
                                    "name": func_call.name,
                                    "args": dict(func_call.args) if hasattr(func_call, 'args') and func_call.args else {},
                                    "id": f"{func_call.name}_{call_index}"
                                }
                            
                            current_tool_call = tool_calls_in_progress[call_index]
                            function_name = current_tool_call["name"]
                            function_args = current_tool_call["args"]
                            
                            # Progressive execution: function is complete for Gemini
                            # Create function call object for progressive execution
                            function_call = FunctionCall(
                                name=function_name,
                                args=function_args,
                                call_id=current_tool_call["id"],
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
                        for msg in contents:
                            if isinstance(msg, str):
                                context_messages.append(msg)
                        self._add_failed_functions_to_context(execution_result['failed_functions'], context_messages)
                        # Update contents with context
                        if context_messages:
                            contents.append(context_messages[-1])
                    
                    # Add function results to conversation
                    self._add_function_results_to_contents(execution_result, contents)
                    
                    self.logger.debug(f"execution_result: {execution_result}")

                    # Check if we have regular function calls that require conversation continuation
                    regular_results = execution_result.get('regular_results', [])

                    if regular_results:
                        current_turn += 1
                        continue
                    
                    # For background-tasks-only: if no text response collected, continue to get actual answer
                    background_results = execution_result.get('background_initiated', [])
                    if background_results and not collected_text:
                        self.logger.info(f"Turn {current_turn}: Background tasks completed but no text response - continuing for actual answer")
                        current_turn += 1
                        continue
                
                # Stream completed for this turn
                self.logger.debug(f"Turn {current_turn}: Stream completed")
                break
                
            except Exception as e:
                self.logger.error(f"Error in Gemini stream_with_functions turn {current_turn}: {str(e)}")
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