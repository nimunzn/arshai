"""
Google Gemini implementation of the LLM interface using google-genai SDK.
Supports both API key and service account authentication.
"""

import os
import json
import logging
import asyncio
import functools
from typing import Dict, Any, Optional, TypeVar, Type, Union, AsyncGenerator
from google.oauth2 import service_account
import google.genai as genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from google.genai import types

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
           GOOGLE_PROJECT_ID, GOOGLE_LOCATION environment variables
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Authentication configuration
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.service_account_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_PATH")
        self.project_id = os.environ.get("GOOGLE_PROJECT_ID")
        self.location = os.environ.get("GOOGLE_LOCATION")
        
        # Gemini-specific configuration
        self.thinking_budget = int(os.environ.get("GOOGLE_THINKING_BUDGET", 
                                                 0 if "flash" in config.model.lower() else 128))
        
        self.logger.info(f"Initializing Gemini client with model: {self.config.model}")
        self.logger.info(f"Using thinking budget: {self.thinking_budget}")
        
        # Initialize the client
        self._client = self._initialize_client()
    
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
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                
                client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                    credentials=credentials
                )
                
                # Test the client with a simple call
                self._test_client_connection(client)
                return client
                
            except FileNotFoundError:
                self.logger.error(f"Service account file not found: {self.service_account_path}")
                raise ValueError(f"Service account file not found: {self.service_account_path}")
            except Exception as e:
                self.logger.error(f"Service account authentication failed: {str(e)}")
                raise ValueError(f"Service account authentication failed: {str(e)}")
        
        else:
            # No valid authentication method found
            error_msg = (
                "No valid authentication method found for Gemini. Please set either:\n"
                "1. GOOGLE_API_KEY for API key authentication, or\n"
                "2. GOOGLE_SERVICE_ACCOUNT_PATH, GOOGLE_PROJECT_ID, and GOOGLE_LOCATION "
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
    
    def _create_structure_function(self, structure_type: Type[T]) -> Dict:
        """
        Create a function definition from the structure type for Gemini function calling.
        
        Args:
            structure_type: Pydantic model class for structured output
            
        Returns:
            Function definition compatible with Gemini
        """
        schema = structure_type.model_json_schema()
        
        return {
            "name": structure_type.__name__.lower(),
            "description": structure_type.__doc__ or f"Create a {structure_type.__name__} response",
            "parameters": schema
        }
    
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
        Process a chat with tools message using the Gemini API.
        
        Args:
            input: The LLM input containing system prompt, user message, tools, and options
            
        Returns:
            Dict containing the LLM response and usage information
        """
        try:
            # Prepare content for Gemini
            system_content = input.system_prompt
            user_content = input.user_message
            
            # Combine system and user content for Gemini
            combined_content = f"{system_content}\n\nUser: {user_content}"
            
            # Prepare tools for Gemini
            tools = []
            if input.tools_list:
                # Convert tools to Gemini format
                for tool in input.tools_list:
                    tools.append(tool)
            
            # Add structure function if provided
            if input.structure_type:
                structure_function = self._create_structure_function(input.structure_type)
                tools.append(structure_function)
                combined_content += f"\n\nYou MUST use the {input.structure_type.__name__.lower()} function to format your response."
            
            current_turn = 0
            final_response = None
            accumulated_usage = None
            conversation_history = [combined_content]
            
            while current_turn < input.max_turns:
                self.logger.info(f"🔄 GEMINI CHAT TURN: {current_turn}/{input.max_turns}")
                self.logger.info(f"   📚 Conversation History Length: {len(conversation_history)}")
                self.logger.info(f"   📋 Tools Available: {len(tools)} tools")
                self.logger.info(f"   📋 Tool Names: {[tool.get('name', 'unnamed') for tool in tools]}")
                
                try:
                    # Prepare generation config
                    generation_config = GenerateContentConfig(
                        max_output_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        thinking_config=ThinkingConfig(thinking_budget=self.thinking_budget)
                    )
                    
                    # Convert tool functions to the format expected by google-genai
                    if tools and input.callable_functions:
                        self.logger.info(f"🔧 SETTING UP TOOLS FOR GEMINI:")
                        
                        # Create Python callable functions list for automatic function calling
                        python_tools = []
                        for tool_def in tools:
                            tool_name = tool_def.get('name')
                            if tool_name in input.callable_functions:
                                original_func = input.callable_functions[tool_name]
                                
                                # Check if function is async and wrap it if needed
                                if asyncio.iscoroutinefunction(original_func):
                                    self.logger.info(f"   🔄 Wrapping async function: {tool_name}")
                                    
                                    def make_sync_wrapper(async_func):
                                        @functools.wraps(async_func)
                                        def sync_wrapper(*args, **kwargs):
                                            try:
                                                # Try to get existing event loop
                                                loop = asyncio.get_running_loop()
                                                # If we're already in an async context, we need to use a thread
                                                import concurrent.futures
                                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                                                    return future.result()
                                            except RuntimeError:
                                                # No event loop running, we can create one
                                                return asyncio.run(async_func(*args, **kwargs))
                                        return sync_wrapper
                                    
                                    wrapped_func = make_sync_wrapper(original_func)
                                    python_tools.append(wrapped_func)
                                    self.logger.info(f"   📋 Adding wrapped async function: {tool_name}")
                                else:
                                    python_tools.append(original_func)
                                    self.logger.info(f"   📋 Adding sync function: {tool_name}")
                        
                        if python_tools:
                            self.logger.info(f"   ✅ Using {len(python_tools)} Python tools with automatic function calling")
                            generation_config.tools = python_tools
                        else:
                            self.logger.warning(f"   ⚠️ No matching callable functions found for tools")
                    
                    # Generate content with tools
                    response = self._client.models.generate_content(
                        model=self.config.model,
                        contents=conversation_history,
                        config=generation_config
                    )
                    
                    # Process usage metadata
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        current_usage = self._process_usage_metadata(response.usage_metadata)
                        if accumulated_usage is None:
                            accumulated_usage = current_usage
                        else:
                            # Accumulate usage metrics
                            accumulated_usage['total_token_count'] += current_usage.get('total_token_count', 0)
                            accumulated_usage['prompt_token_count'] += current_usage.get('prompt_token_count', 0)
                            accumulated_usage['candidates_token_count'] += current_usage.get('candidates_token_count', 0)
                    
                    # Process response
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        
                        # Check for text response first (automatic function calling handles tools internally)
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    # With automatic function calling, we primarily expect text responses
                                    # after tools have been executed internally by the SDK
                                    if hasattr(part, 'text') and part.text:
                                        self.logger.info(f"💬 TEXT RESPONSE RECEIVED:")
                                        self.logger.info(f"   Content: {part.text}")
                                        self.logger.info(f"   Structure Type Expected: {input.structure_type.__name__ if input.structure_type else 'None'}")
                                        
                                        if not input.structure_type:
                                            self.logger.info(f"   ✅ Returning unstructured text response")
                                            return {"llm_response": part.text, "usage": accumulated_usage}
                                        else:
                                            # Try to parse as structured response
                                            self.logger.info(f"   🔍 Attempting to parse as structured response")
                                            try:
                                                final_response = self._parse_to_structure(part.text, input.structure_type)
                                                self.logger.info(f"   ✅ Structured parsing successful: {final_response}")
                                                break
                                            except ValueError as e:
                                                # Continue if parsing fails
                                                self.logger.warning(f"   ⚠️ Structured parsing failed: {str(e)}")
                                                self.logger.info(f"   📝 Adding to conversation history and continuing")
                                                conversation_history.append(part.text)
                    
                    current_turn += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in Gemini chat_with_tools turn {current_turn}: {str(e)}")
                    return {"llm_response": f"An error occurred: {str(e)}", "usage": accumulated_usage}
            
            # Return final response or max turns message
            if final_response is not None:
                return {"llm_response": final_response, "usage": accumulated_usage}
            else:
                return {"llm_response": "Maximum number of function calling turns reached", "usage": accumulated_usage}
                
        except Exception as e:
            self.logger.error(f"Error in Gemini chat_with_tools: {str(e)}")
            return {"llm_response": f"An error occurred: {str(e)}", "usage": None}
    
    def chat_completion(self, input: ILLMInput) -> Union[ILLMOutput, str]:
        """
        Process a chat completion message using the Gemini API.
        
        Args:
            input: The LLM input containing system prompt, user message, and options
            
        Returns:
            Dict containing the LLM response and usage information
        """
        try:
            # Combine system and user content for Gemini
            combined_content = f"{input.system_prompt}\n\nUser: {input.user_message}"
            
            # Add structure instruction if needed
            if input.structure_type:
                schema = input.structure_type.model_json_schema()
                combined_content += f"\n\nYou MUST respond with valid JSON that matches this exact schema: {schema}"
            
            # Generate content
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=[combined_content],
                config=GenerateContentConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    thinking_config=ThinkingConfig(thinking_budget=self.thinking_budget)
                )
            )
            
            # Process usage metadata
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = self._process_usage_metadata(response.usage_metadata)
            
            # Extract response text
            if hasattr(response, 'text') and response.text:
                response_text = response.text
                
                # Handle structured response
                if input.structure_type:
                    try:
                        # Extract JSON from response if it contains other text
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_content = json_match.group()
                        else:
                            json_content = response_text
                        
                        final_response = self._parse_to_structure(json_content, input.structure_type)
                        return {"llm_response": final_response, "usage": usage}
                    except ValueError as e:
                        self.logger.error(f"Failed to parse structured response: {str(e)}")
                        self.logger.error(f"Raw response was: {response_text}")
                        return {"llm_response": f"Failed to generate structured response: {str(e)}", "usage": usage}
                
                return {"llm_response": response_text, "usage": usage}
            
            else:
                return {"llm_response": "No response generated", "usage": usage}
                
        except Exception as e:
            self.logger.error(f"Error in Gemini chat_completion: {str(e)}")
            return {"llm_response": f"An error occurred: {str(e)}", "usage": None}
    
    def _process_usage_metadata(self, raw_usage_metadata) -> Dict[str, Any]:
        """
        Process usage metadata from Gemini model response into clean readable dict.
        Based on the implementation from the bank statement verifier.
        
        Args:
            raw_usage_metadata: Raw usage metadata from Gemini response
            
        Returns:
            Processed usage metadata dictionary
        """
        try:
            if not raw_usage_metadata:
                return {
                    'total_token_count': 0,
                    'prompt_token_count': 0,
                    'candidates_token_count': 0,
                    'cached_content_token_count': 0,
                    'thoughts_token_count': 0,
                    'model': self.config.model
                }
            
            # Extract all fields from the raw metadata, handling None values properly
            processed_metadata = {
                'total_token_count': getattr(raw_usage_metadata, 'total_token_count', 0),
                'prompt_token_count': getattr(raw_usage_metadata, 'prompt_token_count', 0),
                'candidates_token_count': getattr(raw_usage_metadata, 'candidates_token_count', 0),
                'cached_content_token_count': getattr(raw_usage_metadata, 'cached_content_token_count', 0) or 0,
                'thoughts_token_count': getattr(raw_usage_metadata, 'thoughts_token_count', 0) or 0,
                'model': self.config.model
            }
            
            # Process prompt tokens details if available
            prompt_tokens_details = getattr(raw_usage_metadata, 'prompt_tokens_details', None)
            if prompt_tokens_details:
                processed_metadata['prompt_tokens_details'] = []
                for detail in prompt_tokens_details:
                    processed_metadata['prompt_tokens_details'].append({
                        'modality': str(detail.modality) if hasattr(detail, 'modality') else 'UNKNOWN',
                        'token_count': getattr(detail, 'token_count', 0)
                    })
            
            # Process candidates tokens details if available
            candidates_tokens_details = getattr(raw_usage_metadata, 'candidates_tokens_details', None)
            if candidates_tokens_details:
                processed_metadata['candidates_tokens_details'] = []
                for detail in candidates_tokens_details:
                    processed_metadata['candidates_tokens_details'].append({
                        'modality': str(detail.modality) if hasattr(detail, 'modality') else 'UNKNOWN',
                        'token_count': getattr(detail, 'token_count', 0)
                    })
            
            return processed_metadata
            
        except Exception as e:
            self.logger.warning(f"Error processing Gemini usage metadata: {str(e)}")
            return {
                'total_token_count': 0,
                'prompt_token_count': 0,
                'candidates_token_count': 0,
                'cached_content_token_count': 0,
                'thoughts_token_count': 0,
                'model': self.config.model,
                'error': f"Failed to process usage metadata: {str(e)}"
            }