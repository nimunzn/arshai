from openai import AzureOpenAI
import json
import os
from typing import List, Dict, Callable, Any, Optional, TypeVar, Type, Union, Generic, AsyncGenerator
from arshai.core.interfaces.illm import ILLM, ILLMConfig, ILLMInput, ILLMOutput
import logging
import traceback
T = TypeVar('T')

class AzureClient(ILLM):
    """Azure OpenAI implementation of the LLM interface"""
    
    def __init__(self, config: ILLMConfig, azure_deployment: str = None, api_version: str = None):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            config: Configuration for the LLM
            azure_deployment: Optional deployment name (if not provided, will be read from env)
            api_version: Optional API version (if not provided, will be read from env)
        """
        self.config = config
        # Get deployment and API version from params or environment variables
        self.azure_deployment = azure_deployment or os.environ.get("AZURE_DEPLOYMENT")
        self.api_version = api_version or os.environ.get("AZURE_API_VERSION")
        
        if not self.azure_deployment:
            raise ValueError("Azure deployment is required. Set AZURE_DEPLOYMENT environment variable.")
        
        if not self.api_version:
            raise ValueError("Azure API version is required. Set AZURE_API_VERSION environment variable.")
            
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        # Initialize the client
        self._client = self._initialize_client()
    
    def _initialize_client(self) -> Any:
        """Initialize the Azure OpenAI client using environment variables or provided values."""
        # AzureOpenAI client automatically reads AZURE_OPENAI_API_KEY from environment
        return AzureOpenAI(
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
        )
    
    def _create_structure_function(self, structure_type: Type[T]) -> Dict:
        """Create a function definition from the structure type"""
        
        return {
            "name": structure_type.__name__.lower(),
            "description": structure_type.__doc__ or f"Create a {structure_type.__name__} response",
            "parameters": structure_type.model_json_schema()
        }
    
    async def chat_with_tools(
        self,
        input:ILLMInput
    ) ->  Union[ILLMOutput, str]:
        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]
        
        # Add structure function if structure_type is provided
        if input.structure_type:
            structure_function = self._create_structure_function(input.structure_type)
            all_tools = [structure_function] + input.tools_list
            messages[0]["content"] += f"""\nYou MUST ALWAYS use the {input.structure_type.__name__.lower()} tool/function to format your response.
                                            Your response ALWAYS MUST be retunrned using the tool, independently of what is the message or response are.
                                            You MUST ALWAYS CALLING TOOLS FOR RETURNING RESPONSE
                                            The response Must be in JSON format
                                            """
                                            
            response_format = {"type": "json_object"}
        else:
            all_tools = input.tools_list
            response_format = None
        
        current_turn = 0
        final_response = None
        
        # Track accumulated usage metrics
        accumulated_usage = None
        
        while current_turn < input.max_turns:
            self.logger.info(f"Current turn: {current_turn}")
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    functions=all_tools,
                    function_call="auto",
                    temperature=self.config.temperature,
                    response_format=response_format
                )
                
                # Accumulate usage info
                if hasattr(response, 'usage'):
                    current_usage = response.usage
                    self.logger.info(f"Current usage: {current_usage}")
                    if accumulated_usage is None:
                        accumulated_usage = current_usage
                    else:
                        # Sum the usage metrics
                        accumulated_usage.prompt_tokens += current_usage.prompt_tokens
                        accumulated_usage.completion_tokens += current_usage.completion_tokens
                        accumulated_usage.total_tokens += current_usage.total_tokens
                
                message = response.choices[0].message

                self.logger.info(f"Message: {message}")
                
                # If no function call and no structure type required, return content
                if not message.function_call and message.content:
                    if not input.structure_type: 
                        return {"llm_response": message.content, "usage": accumulated_usage}
                    else:
                        structure_response = json.loads(message.content)
                        final_response = input.structure_type(**structure_response)
                        break
                
                # Handle function calls
                if message.function_call:
                    function_name = message.function_call.name
                    function_args = json.loads(message.function_call.arguments)
                    self.logger.info(f"Function name: {function_name}")
                    # If it's the structure function, validate and create the structured response
                    
                    if input.structure_type and function_name == input.structure_type.__name__.lower():
                        final_response = input.structure_type(**function_args)
                        break

                    # Handle other tool functions
                    if function_name in input.callable_functions:
                        # Use aexecute for async tool execution
                        role, function_response = await input.callable_functions[function_name](**function_args)
                        self.logger.debug(f"Function response: {function_response}")
                        
                        # Function responses are now in proper content format, use directly
                        messages.append({
                            "role": role,
                            "name": function_name,
                            "content": function_response
                        })
                    else:
                        raise ValueError(f"Function {function_name} not found in available functions")
                
                # Add assistant's message to conversation
                if message.content:
                    messages.append({"role": "assistant", "content": message.content})
                
                current_turn += 1
                
            except Exception as e:
                self.logger.error(f"Error in chat_with_tools: {str(e)}")
                self.logger.error(traceback.format_exc())
                return {"llm_response": f"An error occurred: {str(e)}", "usage": None}
        
        if final_response is None:
            if input.structure_type:
                try:
                    # Make one final attempt to get structured response
                    final_message = {
                        "role": "system",
                        "content": f"You must provide a final response using the {input.structure_type.__name__.lower()} function."
                    }
                    messages.append(final_message)
                    
                    response = self._client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        functions=[self._create_structure_function(input.structure_type)],
                        function_call={"name": input.structure_type.__name__.lower()},
                        temperature=self.config.temperature
                    )
                    
                    # Accumulate usage info from final call
                    if hasattr(response, 'usage'):
                        current_usage = response.usage
                        if accumulated_usage is None:
                            accumulated_usage = current_usage
                        else:
                            # Sum the usage metrics
                            accumulated_usage.prompt_tokens += current_usage.prompt_tokens
                            accumulated_usage.completion_tokens += current_usage.completion_tokens
                            accumulated_usage.total_tokens += current_usage.total_tokens
                    
                    message = response.choices[0].message
                    if message.function_call:
                        function_args = json.loads(message.function_call.arguments)
                        final_response = input.structure_type(**function_args)
                    else:
                        return {"llm_response": "Failed to generate structured response", "usage": accumulated_usage}
                except Exception as e:
                    return {"llm_response": f"Failed to generate structured response: {str(e)}", "usage": None}
            else:
                return {"llm_response": "Maximum number of function calling turns reached", "usage": accumulated_usage}
        
        # Return structured response with usage
        return {"llm_response": final_response, "usage": accumulated_usage}
    
    def chat_completion(
        self,
        input:ILLMInput
    ) -> Union[ILLMOutput, str]:
        """Process a chat completion request"""
        try:
            if input.structure_type:
                # Create structure function
                structure_function = self._create_structure_function(input.structure_type)
                
                messages = [
                    {"role": "system", "content": input.system_prompt},
                    {"role": "user", "content": input.user_message}
                ]
                
                messages[0]["content"] += f"\nYou MUST use the {input.structure_type.__name__.lower()} function to format your response IN JSON FORMAT"
                
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    functions=[structure_function],
                    function_call={"name": input.structure_type.__name__.lower()},
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"}
                )
                
                # Get usage info
                usage = None
                if hasattr(response, 'usage'):
                    usage = response.usage
                
                message = response.choices[0].message
                if message.function_call:
                    function_args = json.loads(message.function_call.arguments)
                    final_structure = input.structure_type(**function_args)
                    return {"llm_response": final_structure, "usage": usage}
                else:
                    raise ValueError("Model did not provide structured response")
            

            messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
            ]
                
            # Simple completion without structure
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature
            )
            
            # Extract usage if available
            usage = None
            if hasattr(response, 'usage'):
                usage = response.usage
            
            # If we found no match, just return the raw completion
            return {"llm_response": response.choices[0].message.content, "usage": usage}
            
        except Exception as e:
            self.logger.error(f"Error in chat_completion: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"llm_response": f"An error occurred: {str(e)}", "usage": None}

    async def stream_with_tools(
        self,
        input: ILLMInput
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            messages = [
                {"role": "system", "content": input.system_prompt},
                {"role": "user", "content": input.user_message}
            ]
            
            # Add structure function if structure_type is provided
            if input.structure_type:
                structure_function = self._create_structure_function(input.structure_type)
                all_tools = [structure_function] + input.tools_list
                messages[0]["content"] += f"""\nYou MUST ALWAYS use the {input.structure_type.__name__.lower()} tool/function to format your response.
                                                Your response ALWAYS MUST be retunrned using the tool, independently of what is the message or response are.
                                                You MUST ALWAYS CALLING TOOLS FOR RETURNING RESPONSE
                                                The response Must be in JSON format
                                                """
                response_format = {"type": "json_object"}
            else:
                all_tools = input.tools_list
                response_format = None
            
            current_turn = 0
            is_finished = False
            has_previous_delta = False
            # Track accumulated usage
            accumulated_usage = None
                
            while current_turn < input.max_turns:
                if is_finished:
                    break
                
                self.logger.info(f"Current turn: {current_turn}")
       
                collected_message = {"content": "", "function_call": None}
                s = 0
                for chunk in self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    functions=all_tools,
                    function_call="auto",
                    temperature=self.config.temperature,
                    response_format=response_format,
                    stream=True,
                    stream_options={"include_usage": True}
                ):
                    
                    s += 1
                    
                    # Handle the case where choices is empty but usage data is present (final usage chunk)
                    if not chunk.choices and chunk.usage is not None:
                        self.logger.info(f"Received final usage data: {chunk.usage}")
                        accumulated_usage = chunk.usage
                        # Check if we have a complete structured response and this is the end of streaming
                    if (input.structure_type and 
                        collected_message.get("function_call") and 
                        collected_message["function_call"].get("name") == input.structure_type.__name__.lower()):
                        # We have completed a structured response - end the conversation
                        is_finished = True                
                    # Skip chunks without choices
                    if not chunk.choices:
                        continue
                        
                    delta = chunk.choices[0].delta

                    # Check if all attributes are None, excluding private/special attributes
                    if all(getattr(delta, attr) is None for attr in vars(delta) if not attr.startswith('_')):
                        if has_previous_delta:
                            # Only finish if we're not expecting more tool calls
                            # Continue processing if we have partial function calls or if streaming just started
                            if collected_message.get("function_call") is None and collected_message.get("content", "").strip():
                                is_finished = True
                            else:
                                # We might be in a function call sequence or still building content
                                has_previous_delta = True
                        else:
                            has_previous_delta = True

                    # Handle content streaming
                    if hasattr(delta, 'content') and delta.content is not None:
                        collected_message["content"] += delta.content
                        has_previous_delta = True
                        if not input.structure_type:
                            yield {"llm_response": collected_message["content"]}
                        else:
                            # Check if JSON is complete and fix if necessary
                            is_complete, fixed_json = self._is_json_complete(collected_message["content"])
                            if is_complete:
                                try:
                                    final_response = json.loads(fixed_json)
                                    yield {"llm_response": input.structure_type(**final_response)}
                                except json.JSONDecodeError:
                                    continue
                            else:
                                continue  # Wait for more chunks if JSON is incomplete

                    # Handle function call streaming
                    if hasattr(delta, 'function_call'):
                        if delta.function_call is not None:
                            if delta.function_call.name or (collected_message["function_call"] and collected_message["function_call"].get("name")):
                                if collected_message["function_call"] is None:
                                    collected_message["function_call"] = {"name": "", "arguments": ""}
                                if delta.function_call.name:
                                    self.logger.info(f"Function name: {delta.function_call.name}")
                                    collected_message["function_call"]["name"] += delta.function_call.name
                                if delta.function_call.arguments:
                                    collected_message["function_call"]["arguments"] += delta.function_call.arguments
                                    
                                    # Try to parse and stream structured response
                                    if input.structure_type and collected_message["function_call"]["name"] == input.structure_type.__name__.lower():
                                        
                                        # Use the safe parsing function
                                        has_previous_delta = True
                                        is_complete, fixed_json = self._is_json_complete(collected_message["function_call"]["arguments"])
                                        if is_complete:
                                            try:
                                                final_response = json.loads(fixed_json)
                                                yield {"llm_response": input.structure_type(**final_response)}
                                            except json.JSONDecodeError:
                                                continue          

                                    elif collected_message["function_call"]["name"] != input.structure_type.__name__.lower():
                                        function_name = collected_message["function_call"]["name"]
                                        args_str = collected_message["function_call"]["arguments"].strip()

                                        if args_str.startswith("{") and args_str.endswith("}"):
                                            function_args = json.loads(collected_message["function_call"]["arguments"])
                                            if function_name in input.callable_functions:
                                                # Use aexecute for async tool execution
                                                role, function_response = await input.callable_functions[function_name](**function_args)
                                                self.logger.debug(f"Function {function_name} response: {function_response}")
                                                
                                                # Function responses are now in proper content format, use directly
                                                messages.append({
                                                    "role": role,
                                                     "name": function_name,
                                                    "content": function_response
                                                })
                                                
                                                # Reset collected message for next turn
                                                collected_message = {"content": "", "function_call": None}
                                                current_turn += 1
                                                break  # Break out of the streaming loop to start a new turn

            yield {"llm_response": None, "usage": accumulated_usage}

            if current_turn >= input.max_turns:
                yield {"llm_response": "Maximum number of function calling turns reached", "usage": accumulated_usage}
        except Exception as e:
            self.logger.error(f"Error in stream_with_tools: {str(e)}")
            self.logger.error(traceback.format_exc())
            yield {"llm_response": f"An error occurred: {str(e)}", "usage": None}

    async def stream_completion(
        self,
        input: ILLMInput
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream completion responses with optional structure support"""
        try:
            # Track full response for usage extraction
            full_response = None
            
            # For tracking complete content
            complete_content = ""
            
            # Create messages array properly
            messages = [
                {"role": "system", "content": input.system_prompt},
                {"role": "user", "content": input.user_message}
            ]
            
            if input.structure_type:
                # Create structure function
                structure_function = self._create_structure_function(input.structure_type)
                
                # Add instruction to use structure function
                system_message = {
                    "role": "system",
                    "content": f"You MUST use the {input.structure_type.__name__.lower()} function to format your response."
                }
                all_messages = messages + [system_message]
                
                # Create the stream
                stream = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=all_messages,
                    functions=[structure_function],
                    function_call={"name": input.structure_type.__name__.lower()},
                    temperature=self.config.temperature,
                    stream=True,
                    stream_options={"include_usage": True}
                )

                collected_message = {"function_call": {"name": "", "arguments": ""}}
                
                # Process the stream
                for chunk in stream:
                    # Handle the case where choices is empty but usage data is present
                    if not chunk.choices and chunk.usage is not None:
                        self.logger.info(f"Received final usage data: {chunk.usage}")
                        usage = chunk.usage
                        continue
                        
                    # Save full response object for usage extraction
                    full_response = chunk
                    
                    # Skip chunks without choices
                    if not chunk.choices:
                        continue
                        
                    delta = chunk.choices[0].delta
                    
                    # Handle function call streaming
                    if hasattr(delta, 'function_call'):
                        if delta.function_call.name:
                            collected_message["function_call"]["name"] += delta.function_call.name
                        if delta.function_call.arguments:
                            collected_message["function_call"]["arguments"] += delta.function_call.arguments
                            try:
                                # Try to parse and create structured response
                                function_args = json.loads(collected_message["function_call"]["arguments"])
                                yield {"llm_response": input.structure_type(**function_args)}
                            except (json.JSONDecodeError, TypeError, ValueError):
                                # Continue collecting if we can't parse yet
                                continue
                
                # Final yield with usage
                if collected_message["function_call"]["arguments"]:
                    try:
                        function_args = json.loads(collected_message["function_call"]["arguments"])
                        yield {"llm_response": input.structure_type(**function_args), "usage": usage}
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
            else:
                # Simple unstructured completion
                stream = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    stream=True,
                    stream_options={"include_usage": True}
                )

                for chunk in stream:
                    # Handle the case where choices is empty but usage data is present
                    if not chunk.choices and chunk.usage is not None:
                        self.logger.info(f"Received final usage data: {chunk.usage}")
                        usage = chunk.usage
                        continue
                    
                    # Save full response object for usage extraction
                    full_response = chunk
                    
                    # Skip chunks without choices
                    if not chunk.choices:
                        continue
                        
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        complete_content += content
                        yield {"llm_response": content}
                
                # Final yield with usage
                yield {"llm_response": complete_content, "usage": usage}

        except Exception as e:
            self.logger.error(f"Error in stream_completion: {str(e)}")
            self.logger.error(traceback.format_exc())
            yield {"llm_response": f"An error occurred: {str(e)}", "usage": None}

    def _is_json_complete(self, json_str: str) -> tuple[bool, str]:
        """
        Check if JSON string is complete and properly balanced.
        Checks curly braces, double quotes.
        Returns a tuple of (is_complete, fixed_json_string)
        If quotes are unbalanced, adds missing quotes before closing braces.
        """
        # Check curly braces balance
        open_brace_count = json_str.count('{')
        close_brace_count = json_str.count('}')
        # self.logger.info(f"Open brace count: {open_brace_count}")
        # self.logger.info(f"Close brace count: {close_brace_count}")
        
        # Check double quotes balance
        double_quote_count = json_str.count('"')
        # self.logger.info(f"Double quote count: {double_quote_count}")
                
        # Fix unbalanced quotes and handle braces
        fixed_json = json_str
        
        # Fix double quotes if unbalanced
        if double_quote_count % 2 != 0:
            # self.logger.info("Double quotes are not balanced, adding closing quote")
            fixed_json += '"'
            
        # Handle curly braces
        if open_brace_count == close_brace_count:
            # self.logger.info("Json str is complete")
            return True, fixed_json
        elif open_brace_count > close_brace_count:
            # Add missing closing braces
            missing_braces = open_brace_count - close_brace_count
            # self.logger.info(f"Missing braces: {missing_braces}")
            return True, fixed_json + ('}' * missing_braces)
        else:
            self.logger.info("Json str is not complete")
            return False, fixed_json  # More closing than opening braces - invalid JSON