"""
Google Gemini implementation of the LLM interface using google-genai SDK.
Supports both API key and service account authentication with manual tool orchestration.
Follows the same interface pattern as the Azure client for consistency.
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, Optional, TypeVar, Type, Union, AsyncGenerator, List
from google.oauth2 import service_account
import google.genai as genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, FunctionDeclaration, Tool, SpeechConfig, Schema, AutomaticFunctionCallingConfig

from arshai.core.interfaces.illm import ILLM, ILLMConfig, ILLMInput, ILLMOutput

T = TypeVar('T')


class GeminiClient(ILLM):
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
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Authentication configuration
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.service_account_path = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = os.getenv("VERTEX_AI_LOCATION")
        
        # Get model-specific configuration from config dict
        self.model_config = getattr(config, 'config', {})
        
        self.logger.info(f"Initializing Gemini client with model: {self.config.model}")
        
        # Initialize the client
        self._client = self._initialize_client()
    
    def _initialize_client(self) -> Any:
        """
        Initialize the Google GenAI client with safe HTTP configuration and authentication detection.
        
        Uses SafeHttpClientFactory to create a client with proper timeouts and connection limits
        to prevent deadlock issues. Falls back gracefully if advanced configuration is not available.
        
        Authentication priority:
        1. API Key (GOOGLE_API_KEY) - Simple authentication
        2. Service Account - Enterprise authentication using credentials file
        
        Returns:
            Google GenAI client instance with safe HTTP configuration
        
        Raises:
            ValueError: If neither authentication method is properly configured
        """
        
        try:
            # Import the safe factory
            from arshai.clients.utils.safe_http_client import SafeHttpClientFactory
            
            # Try API key authentication first (simpler)
            if self.api_key:
                self.logger.info("Creating GenAI client with API key and safe HTTP configuration")
                try:
                    client = SafeHttpClientFactory.create_genai_client(api_key=self.api_key)
                    # Test the client with a simple call
                    self._test_client_connection(client)
                    self.logger.info("GenAI client created successfully with safe configuration")
                    return client
                except Exception as e:
                    self.logger.error(f"API key authentication with safe config failed: {str(e)}")
                    # Try fallback with basic client
                    self.logger.info("Trying fallback GenAI client with API key")
                    try:
                        import google.genai as genai
                        client = genai.Client(api_key=self.api_key)
                        self._test_client_connection(client)
                        return client
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback API key authentication failed: {fallback_error}")
                        raise ValueError(f"Invalid Google API key: {str(e)}")
            
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
                    
                    # Test the client with a simple call
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
                    try:
                        import google.genai as genai
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
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback service account authentication failed: {fallback_error}")
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
        
        except ImportError as e:
            self.logger.warning(f"Safe HTTP client factory not available: {e}, using default GenAI client")
            
            # Fallback to original implementation without safe HTTP configuration
            if self.api_key:
                self.logger.info("Using API key authentication for Gemini (fallback)")
                try:
                    import google.genai as genai
                    client = genai.Client(api_key=self.api_key)
                    self._test_client_connection(client)
                    return client
                except Exception as e:
                    self.logger.error(f"API key authentication failed: {str(e)}")
                    raise ValueError(f"Invalid Google API key: {str(e)}")
            
            elif self.service_account_path and self.project_id and self.location:
                self.logger.info("Using service account authentication for Gemini (fallback)")
                try:
                    import google.genai as genai
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
                config=GenerateContentConfig(
                    max_output_tokens=1,
                    temperature=0.0
                )
            )
            self.logger.info("Gemini client connection test successful")
        except Exception as e:
            raise Exception(f"Client connection test failed: {str(e)}")
    
    def _create_structure_function(self, structure_type: Type[T]) -> FunctionDeclaration:
        """
        Create a function declaration from the structure type for Gemini function calling.
        (Legacy method - use _create_response_schema for better structured output)
        
        Args:
            structure_type: Pydantic model class for structured output
            
        Returns:
            FunctionDeclaration compatible with Gemini
        """
        schema = structure_type.model_json_schema()
        
        return FunctionDeclaration(
            name=structure_type.__name__.lower(),
            description=structure_type.__doc__ or f"Create a {structure_type.__name__} response",
            parameters=schema
        )
    
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
    
    def _should_use_response_schema(self, input: ILLMInput) -> bool:
        """
        Determine whether to use responseSchema (preferred) or function calling for structured output.
        
        Args:
            input: The LLM input
            
        Returns:
            True if should use responseSchema, False if should use function calling
        """
        # Use responseSchema if:
        # 1. We have a structure_type (need structured output)
        # 2. We don't have tools (no function calling needed)
        return input.structure_type is not None and not input.tools_list
    
    def _convert_tools_to_gemini_format(self, tools_list: List[Dict]) -> List[FunctionDeclaration]:
        """
        Convert tools list to Gemini FunctionDeclaration format.
        
        Args:
            tools_list: List of tool definitions
            
        Returns:
            List of FunctionDeclaration objects
        """
        gemini_tools = []
        for tool in tools_list:
            gemini_tools.append(FunctionDeclaration(
                name=tool.get('name'),
                description=tool.get('description', ''),
                parameters=tool.get('parameters', {})
            ))
        return gemini_tools
    
    def _create_generation_config(self, structure_type: Type[T] = None, use_response_schema: bool = False, tools=None) -> GenerateContentConfig:
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
        config_dict = {
            'temperature': self.config.temperature
        }
        
        # Process all model config parameters and convert nested dicts to proper classes
        for key, value in self.model_config.items():
            if key == 'thinking_config' and isinstance(value, dict):
                # Convert thinking_config dict to ThinkingConfig object
                config_dict['thinking_config'] = ThinkingConfig(**value)
            elif key == 'speech_config' and isinstance(value, dict):
                # Convert speech_config dict to SpeechConfig object
                config_dict['speech_config'] = SpeechConfig(**value)
            elif key == 'response_schema' and isinstance(value, dict):
                # Convert response_schema dict to Schema object
                config_dict['response_schema'] = Schema(**value)
            elif key == 'response_json_schema' and isinstance(value, dict):
                # Keep as dict for response_json_schema (it expects a dict/object)
                config_dict['response_json_schema'] = value
            else:
                # For all other parameters (primitive types, arrays, etc.)
                # stopSequences, responseMimeType, responseModalities, candidateCount,
                # maxOutputTokens, topP, topK, seed, presencePenalty, frequencyPenalty,
                # responseLogprobs, logprobs, enableEnhancedCivicAnswers, mediaResolution
                config_dict[key] = value
        
        # Add structured output configuration if requested
        if structure_type and use_response_schema:
            # Use responseSchema approach (preferred for structured output)
            config_dict['response_schema'] = Schema(**self._create_response_schema(structure_type))
            # Ensure JSON MIME type for structured output
            if 'response_mime_type' not in config_dict:
                config_dict['response_mime_type'] = 'application/json'
        
        # Add tools to config if provided and disable automatic function calling for manual orchestration
        if tools:
            config_dict['tools'] = tools
            # Disable automatic function calling to prevent conflicts with manual orchestration
            config_dict['automatic_function_calling'] = AutomaticFunctionCallingConfig(disable=True)
            self.logger.info("Disabled automatic function calling for manual orchestration")
        
        # Create the generation config with properly converted objects
        return GenerateContentConfig(**config_dict)
    
    def _parse_to_structure(self, content: Union[str, dict], structure_type: Type[T]) -> T:
        """
        Parse response content into the specified structure type.
        
        Args:
            content: Response content to parse
            structure_type: Target Pydantic model class
            
        Returns:
            Instance of the structure type
        """
        if isinstance(content, str):
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON response: {str(e)}")
        else:
            parsed_content = content
        
        try:
            return structure_type(**parsed_content)
        except Exception as e:
            raise ValueError(f"Failed to create {structure_type.__name__} from response: {str(e)}")
    
    async def chat_with_tools(self, input: ILLMInput) -> Union[ILLMOutput, str]:
        """
        Process a chat with tools message using manual tool orchestration.
        
        Args:
            input: The LLM input containing system prompt, user message, tools, and options
            
        Returns:
            Dict containing the LLM response and usage information
        """
        try:
            # Build conversation messages (Gemini uses content format)
            contents = [f"{input.system_prompt}\n\nUser: {input.user_message}"]
            
            # Determine approach for structured output
            use_response_schema = self._should_use_response_schema(input)
            
            # Prepare tools for Gemini (only if not using responseSchema)
            gemini_tools = []
            if input.tools_list:
                gemini_tools.extend(self._convert_tools_to_gemini_format(input.tools_list))
            
            # Add structure function if using function calling approach
            if input.structure_type and not use_response_schema:
                structure_function = self._create_structure_function(input.structure_type)
                gemini_tools.append(structure_function)
                contents[0] += f"\n\nYou MUST use the {input.structure_type.__name__.lower()} function to format your response."
            elif input.structure_type and use_response_schema:
                # For responseSchema approach, just add instruction for JSON
                contents[0] += f"\n\nProvide your response as structured JSON matching the expected format."
            
            current_turn = 0
            final_response = None
            accumulated_usage = None
            
            while current_turn < input.max_turns:
                self.logger.info(f"Current turn: {current_turn}")
                
                try:
                    # Determine whether to use responseSchema or function calling
                    use_response_schema = self._should_use_response_schema(input)
                    
                    # Create tool objects for Gemini
                    tools = [Tool(function_declarations=gemini_tools)] if gemini_tools else None
                    
                    # Prepare generation config from model config dict with tools
                    generation_config = self._create_generation_config(input.structure_type, use_response_schema, tools)
                    
                    # Generate content with tools (manual mode - no automatic execution)
                    response = self._client.models.generate_content(
                        model=self.config.model,
                        contents=contents,
                        config=generation_config
                    )
                    
                    # Process usage metadata
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        current_usage = self._process_usage_metadata(response.usage_metadata)
                        if accumulated_usage is None:
                            accumulated_usage = current_usage
                        else:
                            # Accumulate usage metrics (standardized format)
                            accumulated_usage['total_tokens'] += current_usage.get('total_tokens', 0)
                            accumulated_usage['prompt_tokens'] += current_usage.get('prompt_tokens', 0)
                            accumulated_usage['completion_tokens'] += current_usage.get('completion_tokens', 0)
                    
                    # Process response - handle both responseSchema and function calling approaches
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        
                        # Handle responseSchema approach first (direct text response)
                        if use_response_schema and input.structure_type:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            try:
                                                # Parse JSON response directly
                                                final_response = self._parse_to_structure(part.text, input.structure_type)
                                                break
                                            except ValueError as e:
                                                self.logger.warning(f"Failed to parse responseSchema output: {str(e)}")
                                                continue
                        
                        # Check for function calls - only if not using responseSchema
                        elif hasattr(candidate, 'function_calls') and candidate.function_calls:
                            function_call = candidate.function_calls[0]
                            function_name = function_call.name
                            function_args = dict(function_call.args) if function_call.args else {}
                            
                            self.logger.info(f"Function name: {function_name}")
                            
                            # If it's the structure function, validate and create the structured response
                            if input.structure_type and function_name == input.structure_type.__name__.lower():
                                final_response = input.structure_type(**function_args)
                                break
                            
                            # Handle other tool functions manually
                            if function_name in input.callable_functions:
                                # Execute the function manually
                                role, function_response = await input.callable_functions[function_name](**function_args)
                                self.logger.debug(f"Function response: {function_response}")
                                
                                # Add function response to conversation
                                contents.append(f"Function {function_name} result: {function_response}")
                            else:
                                raise ValueError(f"Function {function_name} not found in available functions")
                        
                        # Check for text response (no function calls)
                        elif hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        if not input.structure_type:
                                            return {"llm_response": part.text, "usage": accumulated_usage}
                                        else:
                                            # Try to parse as structured response
                                            try:
                                                final_response = self._parse_to_structure(part.text, input.structure_type)
                                                break
                                            except ValueError as e:
                                                # Continue if parsing fails
                                                self.logger.warning(f"Structured parsing failed: {str(e)}")
                                                contents.append(part.text)
                    
                    current_turn += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in Gemini chat_with_tools turn {current_turn}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    return {"llm_response": f"An error occurred: {str(e)}", "usage": accumulated_usage}
            
            # Handle final response or max turns
            if final_response is None:
                if input.structure_type:
                    try:
                        # Make one final attempt to get structured response
                        contents.append(f"You must provide a final response using the {input.structure_type.__name__.lower()} function.")
                        
                        # Create tools for final attempt
                        final_tools = [Tool(function_declarations=[self._create_structure_function(input.structure_type)])]
                        final_config = self._create_generation_config(input.structure_type, use_response_schema, final_tools)
                        
                        response = self._client.models.generate_content(
                            model=self.config.model,
                            contents=contents,
                            config=final_config
                        )
                        
                        # Accumulate usage info from final call
                        if hasattr(response, 'usage_metadata') and response.usage_metadata:
                            current_usage = self._process_usage_metadata(response.usage_metadata)
                            if accumulated_usage is None:
                                accumulated_usage = current_usage
                            else:
                                accumulated_usage['total_tokens'] += current_usage.get('total_tokens', 0)
                                accumulated_usage['prompt_tokens'] += current_usage.get('prompt_tokens', 0)
                                accumulated_usage['completion_tokens'] += current_usage.get('completion_tokens', 0)
                        
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'function_calls') and candidate.function_calls:
                                function_call = candidate.function_calls[0]
                                function_args = dict(function_call.args) if function_call.args else {}
                                final_response = input.structure_type(**function_args)
                            else:
                                return {"llm_response": "Failed to generate structured response", "usage": accumulated_usage}
                    except Exception as e:
                        return {"llm_response": f"Failed to generate structured response: {str(e)}", "usage": accumulated_usage}
                else:
                    return {"llm_response": "Maximum number of function calling turns reached", "usage": accumulated_usage}
            
            # Return structured response with usage
            return {"llm_response": final_response, "usage": accumulated_usage}
                
        except Exception as e:
            self.logger.error(f"Error in Gemini chat_with_tools: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"llm_response": f"An error occurred: {str(e)}", "usage": None}
    
    async def stream_with_tools(
        self,
        input: ILLMInput
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a streaming chat with tools message using manual tool orchestration.
        Uses Google GenAI SDK direct accessors (chunk.text, chunk.function_calls).
        
        Args:
            input: The LLM input containing system prompt, user message, tools, and options
            
        Yields:
            Dict containing streaming response chunks and usage information
        """
        try:
            # Build conversation contents (Gemini format)
            contents = [f"{input.system_prompt}\n\nUser: {input.user_message}"]
            
            # Prepare tools for Gemini
            gemini_tools = []
            if input.tools_list:
                gemini_tools.extend(self._convert_tools_to_gemini_format(input.tools_list))
            
            # Add structure function if provided
            if input.structure_type:
                structure_function = self._create_structure_function(input.structure_type)
                gemini_tools.append(structure_function)
                contents[0] += f"\n\nYou MUST use the {input.structure_type.__name__.lower()} function to format your response."
            
            current_turn = 0
            final_response = None
            accumulated_usage = None
            is_finished = False
            
            while current_turn < 5: #input.max_turns:
                if is_finished:
                    break
                    
                self.logger.info(f"Current turn: {current_turn}")
                
                try:
                    # Determine whether to use responseSchema or function calling
                    use_response_schema = self._should_use_response_schema(input)
                    
                    # Create tool objects for Gemini
                    tools = [Tool(function_declarations=gemini_tools)] if gemini_tools else None
                    
                    # Prepare generation config from model config dict with tools
                    generation_config = self._create_generation_config(input.structure_type, use_response_schema, tools)
                    
                    # Generate streaming content with tools (manual mode)
                    stream = self._client.models.generate_content_stream(
                        model=self.config.model,
                        contents=contents,
                        config=generation_config
                    )
                    self.logger.info(f"Stream: {stream}")
                    collected_content = {"text": "", "function_calls": []}
                    chunk_count = 0
                    
                    self.logger.info(f"Starting stream processing for turn {current_turn}")
                    
                    # Process streaming response
                    for chunk in stream:
                        chunk_count += 1
                        self.logger.debug(f"Processing chunk {chunk_count} in turn {current_turn}")
                        
                        # Log chunk structure for debugging
                        if hasattr(chunk, 'candidates'):
                            self.logger.debug(f"Chunk has {len(chunk.candidates) if chunk.candidates else 0} candidates")
                        
                        if hasattr(chunk, 'usage_metadata'):
                            self.logger.debug(f"Chunk has usage metadata: {chunk.usage_metadata is not None}")
                        
                        # Debug: Log chunk attributes
                        chunk_attrs = [attr for attr in dir(chunk) if not attr.startswith('_')]
                        self.logger.debug(f"Chunk attributes: {chunk_attrs}")
                        
                        # Check if chunk has direct text access (Google GenAI SDK standard approach)
                        if hasattr(chunk, 'text') and chunk.text:
                            chunk_text = chunk.text
                            self.logger.debug(f"Direct chunk.text: '{chunk_text}'")
                            collected_content["text"] += chunk_text
                            self.logger.info(f"Turn {current_turn}: Collected text via direct access: '{chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}'")
                            
                            # Stream content if no structure type required
                            if not input.structure_type:
                                yield {"llm_response": chunk_text}
                            else:
                                # Check if JSON is complete for structured response
                                is_complete, fixed_json = self._is_json_complete(collected_content["text"])
                                if is_complete:
                                    try:
                                        final_response = self._parse_to_structure(fixed_json, input.structure_type)
                                        yield {"llm_response": final_response}
                                        is_finished = True
                                        break
                                    except ValueError:
                                        continue
                        
                        # Check for direct function calls access (Google GenAI SDK standard approach)
                        if hasattr(chunk, 'function_calls') and chunk.function_calls:
                            self.logger.info(f"Turn {current_turn}: Found {len(chunk.function_calls)} function calls via direct access")
                            for function_call in chunk.function_calls:
                                self.logger.info(f"Turn {current_turn}: Direct function call - {function_call.name}")
                                collected_content["function_calls"].append(function_call)
                        
                        # Handle usage metadata
                        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                            current_usage = self._process_usage_metadata(chunk.usage_metadata)
                            if accumulated_usage is None:
                                accumulated_usage = current_usage
                            else:
                                accumulated_usage['total_tokens'] += current_usage.get('total_tokens', 0)
                                accumulated_usage['prompt_tokens'] += current_usage.get('prompt_tokens', 0)
                                accumulated_usage['completion_tokens'] += current_usage.get('completion_tokens', 0)
                    
                    self.logger.info(f"Turn {current_turn}: Stream ended. Processed {chunk_count} chunks. Function calls collected: {len(collected_content['function_calls'])}, Text collected: {len(collected_content['text'])} chars")
                    
                    # Process collected function calls manually
                    if collected_content["function_calls"]:
                        self.logger.info(f"Turn {current_turn}: Processing {len(collected_content['function_calls'])} collected function calls")
                        for function_call in collected_content["function_calls"]:
                            function_name = function_call.name
                            function_args = dict(function_call.args) if function_call.args else {}
                            
                            self.logger.info(f"Function name: {function_name}")
                            
                            # Handle structure function
                            if input.structure_type and function_name == input.structure_type.__name__.lower():
                                final_response = input.structure_type(**function_args)
                                yield {"llm_response": final_response}
                                is_finished = True
                                break
                            
                            # Handle other tool functions manually
                            elif function_name in input.callable_functions:
                                role, function_response = await input.callable_functions[function_name](**function_args)
                                self.logger.debug(f"Function {function_name} response: {function_response}")
                                
                                # Add function response to conversation
                                contents.append(f"Function {function_name} result: {function_response}")
                                self.logger.info(f"Turn {current_turn}: Added function response, starting new turn")
                                break  # Break to start new turn
                            else:
                                raise ValueError(f"Function {function_name} not found in available functions")
                    
                    # Handle text-only response
                    elif collected_content["text"] and not input.structure_type:
                        self.logger.info(f"Turn {current_turn}: Text-only response, finishing")
                        yield {"llm_response": collected_content["text"]}
                        is_finished = True
                    
                    # Only increment turn if we're not finished and continuing
                    if not is_finished:
                        current_turn += 1
                        self.logger.info(f"Turn {current_turn}: Continuing, incremented turn to {current_turn}")
                    
                except Exception as e:
                    self.logger.error(f"Error in Gemini stream_with_tools turn {current_turn}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    yield {"llm_response": f"An error occurred: {str(e)}", "usage": accumulated_usage}
                    return
            
            # Final usage yield
            self.logger.info(f"Stream completed: is_finished={is_finished}, current_turn={current_turn}, max_turns={input.max_turns}")
            yield {"llm_response": None, "usage": accumulated_usage}
            
            if current_turn >= input.max_turns:
                self.logger.warning(f"Maximum turns reached: {current_turn} >= {input.max_turns}")
                yield {"llm_response": "Maximum number of function calling turns reached", "usage": accumulated_usage}
                
        except Exception as e:
            self.logger.error(f"Error in Gemini stream_with_tools: {str(e)}")
            self.logger.error(traceback.format_exc())
            yield {"llm_response": f"An error occurred: {str(e)}", "usage": None}
    
    def chat_completion(self, input: ILLMInput) -> Union[ILLMOutput, str]:
        """
        Process a chat completion message using the Gemini API.
        
        Args:
            input: The LLM input containing system prompt, user message, and options
            
        Returns:
            Dict containing the LLM response and usage information
        """
        try:
            if input.structure_type:
                # Create structure function for forced structured response
                structure_function = self._create_structure_function(input.structure_type)
                
                contents = [f"{input.system_prompt}\n\nUser: {input.user_message}"]
                contents[0] += f"\n\nYou MUST use the {input.structure_type.__name__.lower()} function to format your response."
                
                # Create tools for structured response
                tools = [Tool(function_declarations=[structure_function])]
                
                response = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=self._create_generation_config(input.structure_type, self._should_use_response_schema(input), tools)
                )
                
                # Get usage info
                usage = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = self._process_usage_metadata(response.usage_metadata)
                
                # Process function call response
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'function_calls') and candidate.function_calls:
                        function_call = candidate.function_calls[0]
                        function_args = dict(function_call.args) if function_call.args else {}
                        final_structure = input.structure_type(**function_args)
                        return {"llm_response": final_structure, "usage": usage}
                    else:
                        raise ValueError("Model did not provide structured response")
            
            # Simple completion without structure
            contents = [f"{input.system_prompt}\n\nUser: {input.user_message}"]
            
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=self._create_generation_config(input.structure_type, self._should_use_response_schema(input))
            )
            
            # Extract usage if available
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = self._process_usage_metadata(response.usage_metadata)
            
            # Extract response text
            if hasattr(response, 'text') and response.text:
                return {"llm_response": response.text, "usage": usage}
            else:
                return {"llm_response": "No response generated", "usage": usage}
                
        except Exception as e:
            self.logger.error(f"Error in Gemini chat_completion: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"llm_response": f"An error occurred: {str(e)}", "usage": None}

    async def stream_completion(
        self,
        input: ILLMInput
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream completion responses with optional structure support (Gemini implementation)"""
        try:
            # Track usage and complete content
            usage = None
            complete_content = ""
            
            # Create contents for Gemini
            contents = [f"{input.system_prompt}\n\nUser: {input.user_message}"]
            
            if input.structure_type:
                # Create structure function
                structure_function = self._create_structure_function(input.structure_type)
                
                # Add instruction to use structure function
                contents[0] += f"\n\nYou MUST use the {input.structure_type.__name__.lower()} function to format your response."
                
                # Create tools for structured response
                tools = [Tool(function_declarations=[structure_function])]
                
                # Create the stream
                stream = self._client.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=self._create_generation_config(input.structure_type, self._should_use_response_schema(input), tools)
                )

                collected_function_call = {"name": "", "args": {}}
                
                # Process the stream
                for chunk in stream:
                    # Handle usage metadata
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        usage = self._process_usage_metadata(chunk.usage_metadata)
                    
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        
                        # Handle function call
                        if hasattr(candidate, 'function_calls') and candidate.function_calls:
                            function_call = candidate.function_calls[0]
                            collected_function_call["name"] = function_call.name
                            collected_function_call["args"] = dict(function_call.args) if function_call.args else {}
                            
                            try:
                                # Create structured response
                                yield {"llm_response": input.structure_type(**collected_function_call["args"])}
                            except (TypeError, ValueError) as e:
                                self.logger.warning(f"Failed to create structured response: {str(e)}")
                                continue
                
                # Final yield with usage
                if collected_function_call["args"]:
                    try:
                        yield {"llm_response": input.structure_type(**collected_function_call["args"]), "usage": usage}
                    except (TypeError, ValueError):
                        pass
            else:
                # Simple unstructured completion
                stream = self._client.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=self._create_generation_config(input.structure_type, self._should_use_response_schema(input))
                )

                for chunk in stream:
                    # Handle usage metadata
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        usage = self._process_usage_metadata(chunk.usage_metadata)
                    
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        
                        # Handle text content
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        content = part.text
                                        complete_content += content
                                        yield {"llm_response": content}
                
                # Final yield with usage
                yield {"llm_response": complete_content, "usage": usage}

        except Exception as e:
            self.logger.error(f"Error in Gemini stream_completion: {str(e)}")
            self.logger.error(traceback.format_exc())
            yield {"llm_response": f"An error occurred: {str(e)}", "usage": None}
    
    def _is_json_complete(self, json_str: str) -> tuple[bool, str]:
        """
        Check if JSON string is complete and properly balanced.
        Simplified version for Gemini client.
        """
        try:
            json.loads(json_str)
            return True, json_str
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed_json = json_str.strip()
            if not fixed_json.endswith('}'):
                fixed_json += '}'
            try:
                json.loads(fixed_json)
                return True, fixed_json
            except json.JSONDecodeError:
                return False, json_str
    
    def _process_usage_metadata(self, raw_usage_metadata) -> Dict[str, Any]:
        """
        Process usage metadata from Gemini model response into standardized format.
        
        Args:
            raw_usage_metadata: Raw usage metadata from Gemini response
            
        Returns:
            Standardized usage metadata dictionary with OpenAI-compatible keys
        """
        try:
            if not raw_usage_metadata:
                return {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            
            # Map Gemini usage to standard OpenAI format
            prompt_tokens = getattr(raw_usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(raw_usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(raw_usage_metadata, 'total_token_count', 0)
            
            return {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens, 
                'total_tokens': total_tokens
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing Gemini usage metadata: {str(e)}")
            return {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }