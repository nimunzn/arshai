"""
Google Gemini implementation of the LLM interface using google-genai SDK.
Supports both API key and service account authentication with manual tool orchestration.
Follows the same interface pattern as the Azure client for consistency.
"""

import os
import logging
import time
import traceback
import asyncio
from typing import Dict, Any, TypeVar, Type, Union, AsyncGenerator, List, Tuple
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

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput, ILLMOutput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import (
    is_json_complete,
    standardize_usage_metadata,
    accumulate_usage_safely,
    parse_to_structure,
    build_enhanced_instructions,
)

T = TypeVar("T")


class GeminiClient(BaseLLMClient):
    """Google Gemini implementation of the LLM interface"""

    def __init__(self, config: ILLMConfig):
        """
        Initialize the Gemini client with configuration.

        Supports dual authentication methods:
        1. API Key (simpler): Set GOOGLE_API_KEY environment variable
        2. Service Account (enterprise): Set GOOGLE_SERVICE_ACCOUNT_PATH,
           VERTEXAI_PROJECT_ID, VERTEXAI_LOCATION environment variables

        Args:
            config: LLM configuration
        """
        # Gemini-specific configuration
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.service_account_path = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = os.getenv("VERTEX_AI_LOCATION")

        # Get model-specific configuration from config dict
        self.model_config = getattr(config, "config", {})

        # Initialize base client (handles common setup)
        super().__init__(config)

    def _initialize_client(self) -> Any:
        """
        Initialize the Google GenAI client with automatic authentication detection.

        Authentication priority:
        1. API Key (GOOGLE_API_KEY) - Simple authentication
        2. Service Account - Enterprise authentication using credentials file

        Returns:
            Google GenAI client instance

        Raises:
            ValueError: If neither authentication method is properly configured
        """
        # Try API key authentication first (simpler)
        if self.api_key:
            self.logger.info("Using API key authentication for Gemini")
            try:
                client = genai.Client(api_key=self.api_key)
                # Test the client with a simple call
                self._test_client_connection(client)
                return client
            except Exception as e:
                self.logger.error(f"API key authentication failed: {str(e)}")
                raise ValueError(f"Invalid Google API key: {str(e)}")

        # Try service account authentication
        elif self.service_account_path and self.project_id and self.location:
            self.logger.info("Using service account authentication for Gemini")
            try:
                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

                client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                    credentials=credentials,
                )

                # Test the client with a simple call
                self._test_client_connection(client)
                return client

            except FileNotFoundError:
                self.logger.error(
                    f"Service account file not found: {self.service_account_path}"
                )
                raise ValueError(
                    f"Service account file not found: {self.service_account_path}"
                )
            except Exception as e:
                self.logger.error(f"Service account authentication failed: {str(e)}")
                raise ValueError(f"Service account authentication failed: {str(e)}")

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

    def _test_client_connection(self, client) -> None:
        """
        Test the client connection with a minimal request.

        Args:
            client: The GenAI client to test

        Raises:
            Exception: If the client connection test fails
        """
        try:
            # Test with a simple content generation request
            response = client.models.generate_content(
                model=self.config.model,
                contents=["Test connection"],
                config=GenerateContentConfig(max_output_tokens=1, temperature=0.0),
            )
            self.logger.info("Gemini client connection test successful")
        except Exception as e:
            raise Exception(f"Client connection test failed: {str(e)}")

    def _prepare_base_context(self, input: ILLMInput) -> str:
        """
        Build base conversation context from system prompt and user message.
        
        Args:
            input: The LLM input
            
        Returns:
            Formatted conversation context string
        """
        return f"{input.system_prompt}\n\nUser: {input.user_message}"

    def _prepare_tools_context(self, input: ILLMInput) -> Tuple[List, str]:
        """
        Prepare Gemini-specific tools and enhanced context instructions.
        
        Args:
            input: The LLM input
            
        Returns:
            Tuple of (gemini_tools_list, enhanced_context_instructions)
        """
        # Convert tools to LLM format using unified method
        gemini_tools = []
        if input.tools_list and len(input.tools_list) > 0:
            gemini_tools.extend(
                self._convert_functions_to_llm_format(input.tools_list, function_type="tool")
            )
        
        if input.background_tasks and len(input.background_tasks) > 0:
            gemini_tools.extend(
                self._convert_functions_to_llm_format(input.background_tasks, function_type="background_task")
            )
        
        # Build enhanced instructions using generic utility
        enhanced_instructions = build_enhanced_instructions(
            structure_type=input.structure_type,
            background_tasks=input.background_tasks
        )
        
        return gemini_tools, enhanced_instructions


    async def _process_function_calls(self, function_calls, input: ILLMInput, contents: List[str]) -> None:
        """
        Process function calls using the generic orchestrator pattern.
        
        Args:
            function_calls: Function calls from the LLM response
            input: The original LLM input
            contents: Conversation contents list to update
        """
        # Get function execution results from orchestrator
        execution_results = await self._function_orchestrator.process_function_calls_from_response(
            function_calls,
            input.callable_functions or {},
            input.background_tasks or {}
        )
        
        # Convert results to Gemini content format and add to conversation
        self._add_function_results_to_contents(execution_results, contents)
    
    def _add_function_results_to_contents(self, execution_results: Dict, contents: List[str]) -> None:
        """
        Convert function execution results to Gemini content format and add to conversation.
        
        Args:
            execution_results: Results from function orchestrator
            contents: Gemini contents list to update
        """
        # Add background task notifications
        for bg_message in execution_results.get('background_initiated', []):
            contents.append(f"Background task initiated: {bg_message}")
        
        # Add function results with enhanced context
        function_results = execution_results.get('function_results', [])
        function_names = execution_results.get('function_names', [])
        function_args = execution_results.get('function_args', [])
        
        for name, result, args in zip(function_names, function_results, function_args):
            # Format function result with context for Gemini
            result_message = f"Function '{name}' called with arguments {args} returned: {result}"
            contents.append(result_message)
        
        # Add completion indicator if functions were executed
        if function_results:
            completion_msg = f"All {len(function_results)} function(s) completed. Please provide your response based on these results."
            contents.append(completion_msg)

    def _create_response_schema(self, structure_type: Type[T]) -> Dict[str, Any]:
        """
        Create a response schema from the structure type for Gemini structured output.
        This is the preferred method over function calling for structured output.

        Args:
            structure_type: Pydantic model class for structured output

        Returns:
            Schema dict compatible with GenerationConfig.responseSchema
        """
        return structure_type.model_json_schema()

    def _convert_functions_to_llm_format(
        self, 
        functions: Union[List[Dict], Dict[str, Any]], 
        function_type: str = "tool"
    ) -> List[FunctionDeclaration]:
        """
        Convert functions to LLM provider-specific format.
        
        Unified method that handles both tool dictionaries and callable functions,
        with enhanced descriptions based on function type. This method signature
        can be standardized across LLM clients for consistent interfaces.

        Args:
            functions: Either List of tool dictionaries or Dict of callable functions
            function_type: "tool" for regular tools, "background_task" for background tasks

        Returns:
            List of provider-specific function declaration objects (FunctionDeclaration for Gemini)
        """
        gemini_declarations = []
        
        # Handle tool dictionaries (from tools_list)
        if isinstance(functions, list):
            for tool in functions:
                description = tool.get("description", "")
                
                # Enhance description based on function type
                if function_type == "background_task":
                    description = f"BACKGROUND TASK: {description}. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."
                
                gemini_declarations.append(
                    FunctionDeclaration(
                        name=tool.get("name"),
                        description=description,
                        parameters=tool.get("parameters", {}),
                    )
                )
                
        # Handle callable functions (from background_tasks or callable_functions)
        elif isinstance(functions, dict):
            for name, callable_func in functions.items():
                try:
                    # Use Gemini SDK's auto-generation from callable
                    declaration = FunctionDeclaration.from_callable(
                        callable=callable_func, 
                        client=self._client
                    )
                    
                    # Get original description
                    original_description = declaration.description or callable_func.__doc__ or name
                    
                    # Enhance description based on function type
                    if function_type == "background_task":
                        enhanced_description = f"BACKGROUND TASK: {original_description}. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."
                    else:
                        enhanced_description = original_description
                    
                    # Create enhanced declaration
                    enhanced_declaration = FunctionDeclaration(
                        name=declaration.name,
                        description=enhanced_description,
                        parameters=declaration.parameters
                    )
                    
                    gemini_declarations.append(enhanced_declaration)
                    self.logger.debug(f"Auto-generated declaration for {function_type}: {name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to auto-generate declaration for {function_type} {name}: {str(e)}")
                    # Fallback: create basic declaration
                    original_description = callable_func.__doc__ or name
                    if function_type == "background_task":
                        enhanced_description = f"BACKGROUND TASK: {original_description}. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."
                    else:
                        enhanced_description = original_description
                        
                    gemini_declarations.append(
                        FunctionDeclaration(
                            name=name,
                            description=enhanced_description,
                            parameters={"type": "object", "properties": {}, "required": []},
                        )
                    )
        
        return gemini_declarations

    def _create_generation_config(
        self,
        structure_type: Type[T] = None,
        tools=None,
    ) -> GenerateContentConfig:
        """
        Create generation config from model config dict.
        Converts nested dict configs to proper class objects based on Google GenAI schema.

        Args:
            structure_type: Optional Pydantic model for structured output
            use_response_schema: Whether to use responseSchema instead of function calling
            tools: Optional list of tools to include in config

        Returns:
            GenerateContentConfig with all specified settings and proper class conversions
        """
        # Start with base temperature from main config
        config_dict = {"temperature": self.config.temperature}

        # Process all model config parameters and convert nested dicts to proper classes
        for key, value in self.model_config.items():
            if key == "thinking_config" and isinstance(value, dict):
                # Convert thinking_config dict to ThinkingConfig object
                config_dict["thinking_config"] = ThinkingConfig(**value)
            elif key == "speech_config" and isinstance(value, dict):
                # Convert speech_config dict to SpeechConfig object
                config_dict["speech_config"] = SpeechConfig(**value)
            elif key == "response_schema" and isinstance(value, dict):
                # Convert response_schema dict to Schema object
                config_dict["response_schema"] = Schema(**value)
            elif key == "response_json_schema" and isinstance(value, dict):
                # Keep as dict for response_json_schema (it expects a dict/object)
                config_dict["response_json_schema"] = value
            else:
                # For all other parameters (primitive types, arrays, etc.)
                # stopSequences, responseMimeType, responseModalities, candidateCount,
                # maxOutputTokens, topP, topK, seed, presencePenalty, frequencyPenalty,
                # responseLogprobs, logprobs, enableEnhancedCivicAnswers, mediaResolution
                config_dict[key] = value

        # Add structured output configuration if requested
        if structure_type:
            # Use responseSchema approach (preferred for structured output)
            # Set MIME type first for JSON structured output
            config_dict["response_mime_type"] = "application/json"
            
            # Create response schema from Pydantic model
            schema_dict = self._create_response_schema(structure_type)
            
            # Convert to the format expected by the SDK
            # According to the SDK docs, it expects either a Schema object or direct dict
            try:
                config_dict["response_schema"] = Schema(**schema_dict)
            except Exception:
                # Fallback to direct dict if Schema construction fails
                config_dict["response_schema"] = schema_dict

        # Add tools to config if provided and disable automatic function calling for manual orchestration
        if tools:
            config_dict["tools"] = tools
            # Disable automatic function calling to prevent conflicts with manual orchestration
            config_dict["automatic_function_calling"] = AutomaticFunctionCallingConfig(
                disable=True
            )
            self.logger.debug(
                "Disabled automatic function calling for manual orchestration"
            )

        # Create the generation config with properly converted objects
        return GenerateContentConfig(**config_dict)


    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Handle simple chat without tools or background tasks.
        
        Args:
            input: The LLM input
            
        Returns:
            Dict containing the LLM response and usage information
        """
        # Build base context
        contents = [self._prepare_base_context(input)]
        
        # Add structured output instructions if needed
        if input.structure_type is not None:
            contents[0] += "\n\nProvide your response as structured JSON matching the expected format."
        
        # Generate content without tools
        response = self._client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=self._create_generation_config(input.structure_type),
        )

        # Process usage metadata
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = standardize_usage_metadata(response.usage_metadata, provider="gemini")

        # Handle structured output
        if input.structure_type is not None:
            if hasattr(response, "text") and response.text:
                try:
                    final_response = parse_to_structure(response.text, input.structure_type)
                    return {"llm_response": final_response, "usage": usage}
                except ValueError as e:
                    return {"llm_response": f"Failed to parse structured response: {str(e)}", "usage": usage}
        
        # Handle regular text response
        if hasattr(response, "text") and response.text:
            return {"llm_response": response.text, "usage": usage}
        else:
            return {"llm_response": "No response generated", "usage": usage}

    async def _chat_with_tools(self, input: ILLMInput) -> Dict[str, Any]:
        """
        Handle complex chat with tools and/or background tasks.
        
        Args:
            input: The LLM input
            
        Returns:
            Dict containing the LLM response and usage information
        """
        # Build base context and prepare tools
        contents = [self._prepare_base_context(input)]
        gemini_tools, enhanced_instructions = self._prepare_tools_context(input)
        contents[0] += enhanced_instructions
        
        # Handle complex cases with function calling (multi-turn)
        current_turn = 0
        final_response = None
        accumulated_usage = None

        while current_turn < input.max_turns:
            self.logger.info(f"Current turn: {current_turn}")

            try:
                # Create tool objects for Gemini
                tools = (
                    [Tool(function_declarations=gemini_tools)]
                    if gemini_tools
                    else None
                )

                # Prepare generation config with tools
                generation_config = self._create_generation_config(
                    input.structure_type, tools
                )

                start_time = time.time()
                # Generate content with tools (manual mode - no automatic execution)
                response = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=generation_config,
                )

                self.logger.info(f"🔍response time: {time.time() - start_time}")
                
                # Process usage metadata safely
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    current_usage = standardize_usage_metadata(response.usage_metadata, provider="gemini")
                    accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)

                # Check for function calls with parallel execution support
                if hasattr(response, "function_calls") and response.function_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(response.function_calls)} function calls")
                    
                    # Process function calls using orchestrator
                    await self._process_function_calls(response.function_calls, input, contents)

                # Check for direct text response
                if hasattr(response, "text") and response.text:
                    self.logger.info(f"Turn {current_turn}: Received text response")
                    
                    if input.structure_type:
                        # Try to parse as structured response
                        try:
                            final_response = parse_to_structure(
                                response.text, input.structure_type
                            )
                            return {
                                "llm_response": final_response,
                                "usage": accumulated_usage,
                            }
                        except ValueError as e:
                            self.logger.warning(f"Structured parsing failed: {str(e)}")
                            contents.append(response.text)
                    else:
                        # Return plain text response
                        return {
                            "llm_response": response.text,
                            "usage": accumulated_usage,
                        }

                current_turn += 1

            except Exception as e:
                self.logger.error(
                    f"Error in Gemini chat_with_tools turn {current_turn}: {str(e)}"
                )
                self.logger.error(traceback.format_exc())
                return {
                    "llm_response": f"An error occurred: {str(e)}",
                    "usage": accumulated_usage,
                }

        # Handle final response or max turns
        if final_response is None:
            return {
                "llm_response": "Maximum number of function calling turns reached",
                "usage": accumulated_usage,
            }

        # Return structured response with usage
        return {"llm_response": final_response, "usage": accumulated_usage}

    async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle simple streaming without tools or background tasks.
        
        Args:
            input: The LLM input
            
        Yields:
            Dict containing streaming response chunks and usage information
        """
        # Build base context
        contents = [self._prepare_base_context(input)]
        
        # Add structured output instructions if needed
        if input.structure_type:
            contents[0] += "\n\nProvide your response as structured JSON matching the expected format."
        
        # Generate streaming content
        stream = self._client.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=self._create_generation_config(input.structure_type),
        )

        usage = None
        collected_text = ""

        for chunk in stream:
            # Process usage metadata safely
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = standardize_usage_metadata(chunk.usage_metadata, provider="gemini")

            if hasattr(chunk, "text") and chunk.text:
                chunk_text = chunk.text
                collected_text += chunk_text
                
                # Handle structured vs regular text streaming
                if input.structure_type:                            
                    # Try to parse the accumulated content as JSON for structured output
                    is_complete, fixed_json = is_json_complete(collected_text)

                    if is_complete:
                        try:
                            final_response = parse_to_structure(fixed_json, input.structure_type)
                            # Send final structured response when JSON is complete
                            yield {"llm_response": final_response}
                        except ValueError:
                            # Continue streaming if parsing fails
                            pass
                else:
                    # Regular text streaming
                    yield {"llm_response": collected_text}

        yield {"llm_response": None, "usage": usage}

    async def _stream_with_tools(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle complex streaming with tools and/or background tasks.
        
        Args:
            input: The LLM input
            
        Yields:
            Dict containing streaming response chunks and usage information
        """
        # Build base context and prepare tools
        contents = [self._prepare_base_context(input)]
        gemini_tools, enhanced_instructions = self._prepare_tools_context(input)
        contents[0] += enhanced_instructions
        
        # Handle complex cases with function calling (multi-turn needed)
        current_turn = 0
        accumulated_usage = None
        is_finished = False

        while current_turn < input.max_turns:
            self.logger.info(f"Current turn: {current_turn}")
            
            # Check if we should exit due to completion
            if is_finished:
                self.logger.debug(f"Breaking out of loop due to is_finished=True")
                break

            try:
                # Create tool objects for Gemini
                tools = (
                    [Tool(function_declarations=gemini_tools)]
                    if gemini_tools
                    else None
                )

                # Prepare generation config from model config dict with tools
                generation_config = self._create_generation_config(
                    input.structure_type, tools
                )

                # Generate streaming content with tools (manual mode)
                stream = self._client.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=generation_config,
                )
                
                # Store function tasks for parallel execution
                function_tasks = []
                function_names = []
                function_args_list = []
                background_tasks_to_execute = {}
                background_args_dict = {}
                collected_text = ""
                chunk_count = 0

                self.logger.debug(f"Starting stream processing for turn {current_turn}")

                # Process streaming response - execute functions as soon as they're complete
                for chunk in stream:
                    chunk_count += 1
                    self.logger.debug(f"Processing chunk {chunk_count} in turn {current_turn}")

                    # Handle usage metadata safely
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        current_usage = standardize_usage_metadata(chunk.usage_metadata, provider="gemini")
                        accumulated_usage = accumulate_usage_safely(current_usage, accumulated_usage)

                    # Check if chunk has direct text access
                    if hasattr(chunk, "text") and chunk.text:
                        chunk_text = chunk.text
                        collected_text += chunk_text
                        self.logger.debug(f"Direct chunk.text: '{chunk_text}'")
                        
                        # Stream content if no structure type required
                        if not input.structure_type:
                            yield {"llm_response": collected_text}
                        else:
                            # Check if JSON is complete for structured response
                            is_complete, fixed_json = is_json_complete(collected_text)
                            if is_complete:
                                try:
                                    final_response = parse_to_structure(fixed_json, input.structure_type)
                                    yield {"llm_response": final_response}
                                except ValueError:
                                    pass

                    # Check for direct function calls - execute immediately when found
                    if hasattr(chunk, "function_calls") and chunk.function_calls:
                        self.logger.info(f"Turn {current_turn}: Found {len(chunk.function_calls)} function calls")
                        
                        # Process function calls using orchestrator
                        for function_call in chunk.function_calls:
                            function_name = function_call.name
                            function_args = dict(function_call.args) if function_call.args else {}
                            
                            self.logger.debug(f"Preparing function: {function_name}")
                            
                            # Check if it's a background task (fire-and-forget)
                            if function_name in input.background_tasks:
                                background_tasks_to_execute[function_name] = input.background_tasks[function_name]
                                background_args_dict[function_name] = function_args
                            # Check if it's a regular tool
                            elif function_name in input.callable_functions:
                                function_tasks.append(input.callable_functions[function_name])
                                function_names.append(function_name)
                                function_args_list.append(function_args)
                            else:
                                raise ValueError(f"Function {function_name} not found in available functions or background tasks")

                    # Check for finish_reason to determine completion
                    if hasattr(chunk, "candidates") and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                            finish_reason = candidate.finish_reason
                            self.logger.debug(f"Turn {current_turn}: Received finish_reason: {finish_reason}")
                            
                            # Check for completion reasons (any finish reason indicates completion)
                            # Handle both enum and integer values
                            if (hasattr(finish_reason, 'name') and finish_reason.name in ['STOP', 'MAX_TOKENS', 'SAFETY', 'RECITATION', 'OTHER']) or \
                               finish_reason in [1, 2, 3, 4, 5]:
                                is_finished = True
                                self.logger.debug(f"Turn {current_turn}: Stream finished with reason: {finish_reason}")

                self.logger.debug(f"Turn {current_turn}: Stream ended. Processed {chunk_count} chunks. Function tasks created: {len(function_tasks)}, Text collected: {len(collected_text)} chars")

                # Execute background tasks
                if background_tasks_to_execute:
                    background_messages = await self._function_orchestrator.execute_background_tasks(
                        background_tasks_to_execute, background_args_dict
                    )
                    contents.extend(background_messages)
                
                # Execute regular functions in parallel
                if function_tasks:
                    self.logger.info(f"Executing {len(function_tasks)} functions in parallel")
                    results = await self._function_orchestrator.execute_parallel_functions(
                        function_tasks, function_args_list
                    )
                    
                    # Build enhanced context with function arguments and results
                    context_messages = self._function_orchestrator.build_function_context_messages(
                        function_names, results, function_args_list
                    )
                    contents.extend(context_messages)
                    
                    # Add completion indicator to guide model's next response
                    completion_message = self._function_orchestrator.get_completion_message(len(function_tasks))
                    if completion_message:
                        contents.append(completion_message)
                        self.logger.info(f"Added completion indicator: {completion_message}")
                    
                    self.logger.info(f"Completed {len(results)} function calls with enhanced context")
                    
                    # Always continue to next turn after function calls to allow response generation
                    is_finished = False

                # Check for text response after function processing
                if collected_text and not function_tasks and not background_tasks_to_execute:
                    # We have a text response and no function calls - this is the final response
                    self.logger.info(f"Turn {current_turn}: Received final text response")
                    if input.structure_type:
                        try:
                            final_response = parse_to_structure(collected_text, input.structure_type)
                            yield {"llm_response": final_response, "usage": accumulated_usage}
                            return
                        except ValueError as e:
                            self.logger.warning(f"Structured parsing failed: {str(e)}")
                            yield {"llm_response": collected_text, "usage": accumulated_usage}
                            return
                    else:
                        yield {"llm_response": collected_text, "usage": accumulated_usage}
                        return

                # Check if we should exit or continue
                if is_finished and not function_tasks and not background_tasks_to_execute:
                    # Model finished and no function calls to process - exit loop
                    self.logger.debug(f"Turn {current_turn}: Stream completed - model finished with no function tasks")
                    break
                else:
                    # Continue to next turn
                    current_turn += 1
                    self.logger.debug(f"Turn {current_turn}: Continuing to next turn")

            except Exception as e:
                self.logger.error(f"Error in Gemini stream_with_tools turn {current_turn}: {str(e)}")
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
        
