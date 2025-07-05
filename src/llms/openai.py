from openai import OpenAI
import json
import os
import logging
from typing import List, Dict, Callable, Any, Optional, TypeVar, Type, Union, Generic, AsyncGenerator
from arshai.core.interfaces import ILLM, ILLMConfig, ILLMInput, ILLMOutput
from datetime import datetime

T = TypeVar('T')

class OpenAIClient(ILLM):
    """OpenAI implementation of the LLM interface"""
    
    def __init__(self, config: ILLMConfig):
        """
        Initialize the OpenAI client with configuration.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing OpenAI client with model: {self.config.model}")
        
        # Initialize the client
        self._client = self._initialize_client()
    
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
            
        return OpenAI()
    
    def _create_structure_function(self, structure_type: Type[T]) -> Dict:
        """Create a function definition from the structure type"""
        
        return {
            "name": structure_type.__name__.lower(),
            "description": structure_type.__doc__ or f"Create a {structure_type.__name__} response",
            "parameters": structure_type.model_json_schema()
        }
    
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
        
        # Check double quotes balance
        double_quote_count = json_str.count('"')
                
        # Fix unbalanced quotes and handle braces
        fixed_json = json_str
        
        # Fix double quotes if unbalanced
        if double_quote_count % 2 != 0:
            fixed_json += '"'
            
        # Handle curly braces
        if open_brace_count == close_brace_count:
            return True, fixed_json
        elif open_brace_count > close_brace_count:
            # Add missing closing braces
            missing_braces = open_brace_count - close_brace_count
            return True, fixed_json + ('}' * missing_braces)
        else:
            self.logger.info("Json str is not complete")
            return False, fixed_json  # More closing than opening braces - invalid JSON
    
    async def chat_with_tools(
        self,
        input:ILLMInput
    ) -> Union[ILLMOutput, str]:
        """
        Process a chat with tools message using the OpenAI API.
        
        Args:
            input: The LLM input containing system prompt, user message, tools, and options
            
        Returns:
            Dict containing the LLM response and usage information
        """
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
                        tools=all_tools,
                        tool_choice="auto",
                        temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens if self.config.max_tokens else None,
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

                    self.logger.info(f"Message received")
                    
                    # If no tool call and no structure type required, return content
                    if not message.tool_calls and message.content:
                        if not input.structure_type: 
                            return {"llm_response": message.content, "usage": accumulated_usage}
                        else:
                            try:
                                structure_response = json.loads(message.content)
                                final_response = input.structure_type(**structure_response)
                                break
                            except (json.JSONDecodeError, TypeError, ValueError) as e:
                                self.logger.error(f"Error parsing structure response: {e}")
                            return {"llm_response": f"Error parsing structured response: {str(e)}", "usage": accumulated_usage}
                    
                    # Handle tool calls
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            function_name = tool_call.function.name
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            self.logger.info(f"Function name: {function_name}")
                            
                            # If it's the structure function, validate and create the structured response
                            if input.structure_type and function_name == input.structure_type.__name__.lower():
                                final_response = input.structure_type(**function_args)
                                break

                            # Handle other tool functions
                            if function_name in input.callable_functions:
                                # Use aexecute for async tool execution
                                function_response = await input.callable_functions[function_name](**function_args)
                                self.logger.debug(f"Function response: {function_response}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": str(function_response)
                                })
                            else:
                                raise ValueError(f"Function {function_name} not found in available functions")
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Error parsing tool arguments: {e}")
                            return {"llm_response": f"Error parsing tool arguments: {str(e)}", "usage": accumulated_usage}
                    
                    # Add assistant's message to conversation
                    if message.content:
                        messages.append({"role": "assistant", "content": message.content})
                    
                    current_turn += 1
                    
                except Exception as e:
                    import traceback
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
                            tools=[self._create_structure_function(input.structure_type)],
                            tool_choice={"type": "function", "function": {"name": input.structure_type.__name__.lower()}},
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens if self.config.max_tokens else None
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
                        if message.tool_calls:
                            function_args = json.loads(message.tool_calls[0].function.arguments)
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
        """
        Process a chat completion message using the OpenAI API.
        
        Args:
            input: The LLM input containing system prompt, user message, and options
            
        Returns:
            Dict containing the LLM response and usage information
        """
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
                    tools=[structure_function],
                    tool_choice={"type": "function", "function": {"name": input.structure_type.__name__.lower()}},
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens if self.config.max_tokens else None,
                    response_format={"type": "json_object"}
                )
                
                # Get usage info
                usage = None
                if hasattr(response, 'usage'):
                    usage = response.usage
                
                message = response.choices[0].message
                if message.tool_calls:
                    function_args = json.loads(message.tool_calls[0].function.arguments)
                    final_structure = input.structure_type(**function_args)
                    return {"llm_response": final_structure, "usage": usage}
                else:
                    raise ValueError("Model did not provide structured response")
            
            # Simple completion without structure
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": input.system_prompt},
                    {"role": "user", "content": input.user_message}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens if self.config.max_tokens else None
            )
            
            # Extract usage if available
            usage = None
            if hasattr(response, 'usage'):
                usage = response.usage
            
            return {"llm_response": response.choices[0].message.content, "usage": usage}
            
        except Exception as e:
            self.logger.error(f"Error in chat_completion: {str(e)}")
            return {"llm_response": f"An error occurred: {str(e)}", "usage": None}

    async def stream_with_tools(
        self,
        input: ILLMInput
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat with tools responses from the OpenAI API.
        
        Args:
            input: The LLM input containing system prompt, user message, tools, and options
            
        Yields:
            Dict containing the streamed LLM response chunks and usage information
        """
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
                                        You MUST ALWAYS CALLING TOOLS FOR RETURNING RESPONSE"""
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
   
            collected_message = {"content": "", "tool_calls": []}
            
            for chunk in self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=all_tools,
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens if self.config.max_tokens else None,
                response_format=response_format,
                stream=True
            ):
                # Handle usage data if available
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if accumulated_usage is None:
                        accumulated_usage = chunk.usage
                    else:
                        # Sum the usage metrics
                        accumulated_usage.prompt_tokens += chunk.usage.prompt_tokens
                        accumulated_usage.completion_tokens += chunk.usage.completion_tokens
                        accumulated_usage.total_tokens += chunk.usage.total_tokens
                
                # Skip chunks without choices
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # Check if all attributes are None, excluding private/special attributes
                if all(getattr(delta, attr) is None for attr in vars(delta) if not attr.startswith('_')):
                    self.logger.info(f"All attributes are None, excluding private/special attributes")
                    if has_previous_delta:
                        is_finished = True
                    else:
                        has_previous_delta = True
                    continue

                # Handle content streaming
                if hasattr(delta, 'content') and delta.content is not None:
                    collected_message["content"] += delta.content
                    has_previous_delta = True
                    if not input.structure_type:
                        yield {"llm_response": delta.content, "usage": None}
                    else:
                        # Check if JSON is complete and fix if necessary
                        is_complete, fixed_json = self._is_json_complete(collected_message["content"])
                        if is_complete:
                            try:
                                final_response = json.loads(fixed_json)
                                yield {"llm_response": input.structure_type(**final_response), "usage": None}
                            except (json.JSONDecodeError, TypeError, ValueError):
                                continue
                        else:
                            continue  # Wait for more chunks if JSON is incomplete

                # Handle tool calls streaming
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for i, tool_delta in enumerate(delta.tool_calls):
                        has_previous_delta = True
                        
                        # Initialize or get current tool call
                        if i >= len(collected_message["tool_calls"]):
                            collected_message["tool_calls"].append({
                                "id": tool_delta.id or "",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        current_tool_call = collected_message["tool_calls"][i]
                        
                        # Update tool call with new delta information
                        if tool_delta.id:
                            current_tool_call["id"] = tool_delta.id
                            
                        if hasattr(tool_delta, 'function'):
                            if tool_delta.function.name:
                                current_tool_call["function"]["name"] = tool_delta.function.name
                                self.logger.info(f"Function name: {tool_delta.function.name}")
                                
                            if tool_delta.function.arguments:
                                current_tool_call["function"]["arguments"] += tool_delta.function.arguments
                                
                                # Try to parse and stream structured response
                                if (input.structure_type and 
                                    current_tool_call["function"]["name"] == input.structure_type.__name__.lower()):
                                    
                                    # Use the safe parsing function
                                    has_previous_delta = True
                                    is_complete, fixed_json = self._is_json_complete(current_tool_call["function"]["arguments"])
                                    if is_complete:
                                        try:
                                            final_response = json.loads(fixed_json)
                                            yield {"llm_response": input.structure_type(**final_response), "usage": None}
                                        except (json.JSONDecodeError, TypeError, ValueError):
                                            continue          

                                # Process tool execution
                                elif current_tool_call["function"]["name"]:
                                    function_name = current_tool_call["function"]["name"]
                                    args_str = current_tool_call["function"]["arguments"].strip()

                                    if args_str.startswith("{") and args_str.endswith("}"):
                                        try:
                                            function_args = json.loads(args_str)
                                            if function_name in input.callable_functions:
                                                # Use aexecute for async tool execution
                                                function_response = await input.callable_functions[function_name](**function_args)
                                                self.logger.info(f"Function {function_name} response: {function_response}")
                                                # Add function response to messages
                                            messages.extend([
                                                {"role": "tool", "tool_call_id": current_tool_call["id"], "name": function_name, "content": str(function_response)},
                                                {"role": "system", "content": f"You MUST NOT use and call the {function_name} tool AGAIN as it has already been used"}
                                            ])
                                            current_turn += 1
                                            break
                                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                                            self.logger.error(f"Error parsing tool arguments: {e}")
                                            continue
            
            # At this point, streaming is complete for current turn
            if is_finished:
                if not input.structure_type:
                    yield {"llm_response": collected_message["content"], "usage": accumulated_usage}
                elif collected_message["tool_calls"]:
                    for tool_call in collected_message["tool_calls"]:
                        if tool_call["function"]["name"] == input.structure_type.__name__.lower():
                            try:
                                args_str = tool_call["function"]["arguments"].strip()
                                if args_str.startswith("{") and args_str.endswith("}"):
                                    function_args = json.loads(args_str)
                                    yield {"llm_response": input.structure_type(**function_args), "usage": accumulated_usage}
                            except (json.JSONDecodeError, TypeError, ValueError) as e:
                                self.logger.error(f"Error parsing final tool arguments: {e}")
                                pass

        # Final yield with the accumulated usage if max turns reached
        if current_turn >= input.max_turns:
            yield {"llm_response": "Maximum number of function calling turns reached", "usage": accumulated_usage}

    async def stream_completion(
        self,
        input: ILLMInput
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completion responses from the OpenAI API.
        
        Args:
            input: The LLM input containing system prompt, user message, and options
            
        Yields:
            Dict containing the streamed LLM response chunks and usage information
        """
        try:
            # Track usage
            accumulated_usage = None
            
            # For tracking complete content
            complete_content = ""
            
            if input.structure_type:
                # Create structure function
                structure_function = self._create_structure_function(input.structure_type)
                
                # Prepare messages
                messages = [
                    {"role": "system", "content": f"{input.system_prompt}\nYou MUST use the {input.structure_type.__name__.lower()} function to format your response."},
                    {"role": "user", "content": input.user_message}
                ]
                
                # Create the stream
                stream = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=[structure_function],
                    tool_choice={"type": "function", "function": {"name": input.structure_type.__name__.lower()}},
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens if self.config.max_tokens else None,
                    stream=True
                )

                collected_message = {"tool_calls": [{"function": {"name": "", "arguments": ""}}]}
                
                # Process the stream
                for chunk in stream:
                    # Handle usage data if available
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        if accumulated_usage is None:
                            accumulated_usage = chunk.usage
                        else:
                            # Sum the usage metrics
                            accumulated_usage.prompt_tokens += chunk.usage.prompt_tokens
                            accumulated_usage.completion_tokens += chunk.usage.completion_tokens
                            accumulated_usage.total_tokens += chunk.usage.total_tokens
                    
                    # Skip chunks without choices
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    
                    # Handle tool call streaming
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for i, tool_delta in enumerate(delta.tool_calls):
                            # Initialize tool call if needed
                            if i >= len(collected_message["tool_calls"]):
                                collected_message["tool_calls"].append({
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            current_tool_call = collected_message["tool_calls"][i]
                            
                            if hasattr(tool_delta, 'function'):
                                if tool_delta.function.name:
                                    current_tool_call["function"]["name"] += tool_delta.function.name
                                    
                                if tool_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += tool_delta.function.arguments
                                    try:
                                        # Try to parse and create structured response
                                        function_args = json.loads(current_tool_call["function"]["arguments"])
                                        yield {"llm_response": input.structure_type(**function_args), "usage": None}
                                    except (json.JSONDecodeError, TypeError, ValueError):
                                        # Continue collecting if we can't parse yet
                                        continue
                
                # Stream is complete, yield final response with usage
                if collected_message["tool_calls"][0]["function"]["arguments"]:
                    try:
                        args_str = collected_message["tool_calls"][0]["function"]["arguments"]
                        is_complete, fixed_json = self._is_json_complete(args_str)
                        if is_complete:
                            function_args = json.loads(fixed_json)
                            yield {"llm_response": input.structure_type(**function_args), "usage": accumulated_usage}
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        self.logger.error(f"Error parsing final tool arguments: {e}")
            else:
                # Simple unstructured completion
                messages = [
                    {"role": "system", "content": input.system_prompt},
                    {"role": "user", "content": input.user_message}
                ]
                
                stream = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens if self.config.max_tokens else None,
                    stream=True
                )

                for chunk in stream:
                    # Handle usage data if available
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        if accumulated_usage is None:
                            accumulated_usage = chunk.usage
                        else:
                            # Sum the usage metrics
                            accumulated_usage.prompt_tokens += chunk.usage.prompt_tokens
                            accumulated_usage.completion_tokens += chunk.usage.completion_tokens
                            accumulated_usage.total_tokens += chunk.usage.total_tokens
                    
                    # Skip chunks without choices
                    if not chunk.choices:
                        continue
                    
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        complete_content += content
                        yield {"llm_response": content, "usage": None}
                
                # Stream is complete, yield final complete content with usage
                yield {"llm_response": complete_content, "usage": accumulated_usage}

        except Exception as e:
            self.logger.error(f"Error in stream_completion: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            yield {"llm_response": f"An error occurred: {str(e)}", "usage": None}