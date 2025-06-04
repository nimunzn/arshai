# Agent System

The agent system forms the core of Arshai's AI capabilities. It provides a flexible framework for building conversational AI agents that can understand user requests, access tools, and provide structured responses.

## Architecture Overview

The agent system follows a modular architecture with interface-first design where components are cleanly separated:

```
┌───────────────────────────────────────────────────────────────┐
│                         Agent System                           │
└───────────────────────────────────────────────────────────────┘
                              │
       ┌─────────────────────┼────────────────────┐
       │                     │                    │
       ▼                     ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    Agent    │      │  Memory     │      │   Tools     │
│ Implementations    │ Management  │      │ Integration │
└───────┬─────┘      └─────────────┘      └───────┬─────┘
        │                                          │
        │                                          │
        ▼                                          ▼
┌─────────────┐                           ┌─────────────┐
│    LLM      │                           │   Tool      │
│ Integration │                           │ Definitions │
└─────────────┘                           └─────────────┘
```

## Core Components

### Agents

Agents are the central components that process user input and generate responses. Each agent:

1. Receives input messages with conversation context
2. Processes the input using an LLM with appropriate prompting
3. Can access and use specialized tools
4. Maintains and updates conversation memory
5. Returns structured responses

#### Agent Interface

All agents implement the `IAgent` protocol defined in `seedwork/interfaces/iagent.py`:

```python
class IAgent(Protocol):
    """Interface for conversational agents."""
    
    def process_message(
        self,
        input: IAgentInput,
    ) -> IAgentOutput:
        """
        Process incoming message and generate response
        
        Args:
            input: IAgentInput containing message and context
            
        Returns:
            IAgentOutput containing the response
        """
        ...
        
    async def aprocess_message(
        self,
        input: IAgentInput,
    ) -> IAgentOutput:
        """Async version of process_message"""
        ...
```

#### Agent Configuration

Agents are configured using the `IAgentConfig` class:

```python
class IAgentConfig(IDTO):
    """Configuration for an agent"""
    task_context: str = Field(description="Conversation context and agent instructions")
    tools: List[Any] = Field(default_factory=list, description="List of tools available to the agent")
    memory_config: Optional[dict] = Field(default=None, description="Configuration for agent memory")
    output_structure: Optional[Any] = Field(default=None, description="Structure for agent output")
```

#### Agent Input/Output

Agents receive input and produce output through standardized structures:

```python
class IAgentInput(IDTO):
    """Input to an agent"""
    message: str = Field(description="The user input message")
    conversation_id: str = Field(description="Unique identifier for the conversation")
    stream: bool = Field(default=False, description="Whether to stream the response")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class IAgentOutput(IDTO):
    """Output from an agent"""
    agent_message: str = Field(description="The agent's response message")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls made by the agent")
    memory: Optional[Dict[str, Any]] = Field(default=None, description="Memory updates from the agent")
    structured_output: Optional[Dict[str, Any]] = Field(default=None, description="Structured output if requested")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
```

### Agent Implementations

Arshai includes several built-in agent implementations:

#### OperatorAgent

The primary conversational agent that can use tools and has sophisticated memory management:

```python
class OperatorAgent(IAgent):
    """
    Agent responsible for handling user interactions and managing conversation flow
    
    Features:
    - Dynamic memory management
    - Tool usage
    - Structured output
    - Streaming support
    - Human-like conversation
    """
```

#### RAGAgent

An agent specialized in retrieval-augmented generation:

```python
class RAGAgent(IAgent):
    """
    Agent specialized in retrieval-augmented generation
    
    Features:
    - Document retrieval
    - Context augmentation
    - Dynamic memory management
    - Tool usage
    """
```

#### Custom Agents

Custom agents can be created by implementing the `IAgent` protocol:

```python
class CustomAgent(IAgent):
    """Custom agent implementation."""
    
    def __init__(self, config: IAgentConfig, settings):
        # Initialize with configuration and settings
        self.config = config
        self.settings = settings
        self.llm = settings.create_llm()
        self.memory_manager = settings.create_memory_manager()
        # Custom initialization...
        
    def process_message(self, input: IAgentInput) -> IAgentOutput:
        # Custom message processing logic
        # ...
        return IAgentOutput(agent_message="Response from custom agent")
        
    async def aprocess_message(self, input: IAgentInput) -> IAgentOutput:
        # Async implementation
        return await self.process_message(input)
```

## Prompt Management

Agents use structured prompting systems to guide the LLM's behavior. A typical prompt structure includes:

1. **Task Context**: The specific role and task of the agent
2. **Memory Context**: Working memory of the conversation
3. **Tool Definitions**: Information about available tools
4. **Response Format**: Structure for the expected output
5. **Guardrails**: Safety and compliance guidelines

### Prompt Structure Example

```python
def _prepare_system_prompt(self) -> str:
    """Prepare the system prompt for the LLM."""
    
    prompt = f"""
    You are a conversational AI Operator Agent who speaks with users naturally.
    
    ### YOUR TASK, IDENTITY AND ROLE:
    {self.task_context}
    
    ### CURRENT TIME:
    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """
    
    # Add working memory if available
    if self.working_memory:
        prompt += "\n\n### CONVERSATION CONTEXT:\n"
        prompt += self.working_memory.get_formatted_memory()
    
    # Add tool usage instructions if tools are available
    if self.tools:
        prompt += "\n\n### TOOLS:\n"
        prompt += "You have access to the following tools:\n"
        for tool in self.tools:
            func_def = tool.function_definition
            prompt += f"- {func_def['name']}: {func_def['description']}\n"
        
        prompt += "\nTo use a tool, respond in the following format:\n"
        prompt += "```json\n{\"tool\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}\n```\n"
    
    # Add output structure if specified
    if self.output_structure:
        prompt += "\n\n### RESPONSE FORMAT:\n"
        prompt += "Your response should follow this structure:\n"
        prompt += json.dumps(self.output_structure.model_schema(), indent=2)
    
    # Add guardrails
    prompt += "\n\n### GUIDELINES:\n"
    prompt += "- Be helpful, respectful, and accurate\n"
    prompt += "- If you don't know something, admit it rather than making up information\n"
    prompt += "- Ensure your responses are safe, ethical, and unbiased\n"
    
    return prompt
```

## Tool Integration

Agents can use tools to extend their capabilities beyond conversation and access external functionalities.

### Tool Interface

All tools implement the `ITool` protocol defined in `seedwork/interfaces/itool.py`:

```python
class ITool(Protocol):
    """Interface for tools that can be used by agents"""
    
    @property
    def function_definition(self) -> Dict[str, Any]:
        """
        Get the function definition for the tool.
        
        Returns:
            Dictionary containing the function definition in the format:
            {
                "name": "tool_name",
                "description": "What the tool does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "Description of param1"
                        }
                    },
                    "required": ["param1"]
                }
            }
        """
        ...
    
    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with provided arguments.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        ...
```

### Built-in Tools

Arshai includes several built-in tools:

#### WebSearchTool

```python
class WebSearchTool(ITool):
    """Tool for searching the web for information"""
    
    function_definition = {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
    
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the web search tool"""
        # Implementation...
```

#### KnowledgeBaseTool

```python
class KnowledgeBaseTool(ITool):
    """Tool for querying a knowledge base"""
    
    function_definition = {
        "name": "query_knowledge_base",
        "description": "Query the knowledge base for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "collection": {
                    "type": "string",
                    "description": "The collection to search in"
                }
            },
            "required": ["query"]
        }
    }
    
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the knowledge base tool"""
        # Implementation...
```

### Tool Registration and Usage

Tools are registered with agents during initialization:

```python
# Create tools
web_search_tool = WebSearchTool(settings)
knowledge_base_tool = KnowledgeBaseTool(settings, collection="company_docs")

# Create agent with tools
agent_config = IAgentConfig(
    task_context="You are a helpful assistant",
    tools=[web_search_tool, knowledge_base_tool]
)
agent = settings.create_agent("operator", agent_config)
```

### Tool Execution Flow

When a tool is executed, the typical flow is:

1. Agent receives user input
2. Agent decides to use a tool based on LLM reasoning
3. Tool is executed with parameters determined by the LLM
4. Tool execution result is returned to the agent
5. Agent incorporates the tool result into its response

## Memory Management

Agents use memory systems to maintain context across conversation turns.

### Memory Interface

Memory management is handled by implementing the `IMemoryManager` interface:

```python
class IMemoryManager(Protocol):
    """Interface for memory management"""
    
    def retrieve_working_memory(self, conversation_id: str) -> IWorkingMemory:
        """
        Retrieve working memory for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Working memory for the conversation
        """
        ...
    
    def store_memory(self, conversation_id: str, memory_update: Dict[str, Any]) -> None:
        """
        Store memory for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            memory_update: Memory data to store
        """
        ...
```

### Working Memory

Working memory contains the conversation history and context:

```python
class IWorkingMemory(Protocol):
    """Interface for working memory"""
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to memory"""
        ...
    
    def add_agent_message(self, message: str) -> None:
        """Add an agent message to memory"""
        ...
    
    def add_system_note(self, note: str) -> None:
        """Add a system note to memory"""
        ...
    
    def get_formatted_memory(self) -> str:
        """Get formatted memory for prompt inclusion"""
        ...
```

### Memory Integration with Agents

Agents use memory during message processing:

```python
def process_message(self, input: IAgentInput) -> IAgentOutput:
    """Process a message and generate a response."""
    
    # Retrieve conversation memory
    working_memory = self.memory_manager.retrieve_working_memory(input.conversation_id)
    
    # Add user message to memory
    working_memory.add_user_message(input.message)
    
    # Prepare system prompt with memory context
    system_prompt = self._prepare_system_prompt(working_memory)
    
    # Get LLM response
    llm_response = self.llm.chat_with_tools(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input.message}
        ],
        tools=[tool.function_definition for tool in self.tools]
    )
    
    # Add agent response to memory
    working_memory.add_agent_message(llm_response.message)
    
    # Store updated memory
    self.memory_manager.store_memory(
        input.conversation_id, 
        {"history": working_memory.get_raw_history()}
    )
    
    # Return agent output
    return IAgentOutput(
        agent_message=llm_response.message,
        tool_calls=llm_response.tool_calls,
        memory={"history": working_memory.get_raw_history()}
    )
```

## Factory Pattern and Settings

Agents can be created using the `AgentFactory` through the settings:

```python
# Register an agent type
AgentFactory.register("custom_agent", CustomAgent)

# Create the agent through settings
agent = settings.create_agent(
    "custom_agent", 
    IAgentConfig(
        task_context="You are a helpful assistant",
        tools=[]
    )
)
```

## Agent Integration in Workflows

Agents are typically wrapped in workflow nodes:

```python
class AgentNode(BaseNode):
    """A node that wraps an agent in a workflow."""
    
    def __init__(self, node_id: str, name: str, agent: IAgent, settings):
        super().__init__(node_id, name, settings)
        self.agent = agent
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through the agent."""
        
        # Extract message and conversation ID from input data
        message = input_data.get("message", "")
        conversation_id = input_data.get("conversation_id", "default")
        
        # Process through agent
        agent_input = IAgentInput(
            message=message,
            conversation_id=conversation_id
        )
        
        agent_output = self.agent.process_message(agent_input)
        
        # Return agent output as node result
        return {
            "message": agent_output.agent_message,
            "tool_calls": agent_output.tool_calls,
            "memory": agent_output.memory
        }
```

## Best Practices for Agent Development

When developing custom agents, follow these best practices:

1. **Focus on a Single Responsibility**: Each agent should have a clear, focused purpose.
2. **Implement Both Sync and Async Methods**: Support both `process_message` and `aprocess_message`.
3. **Use Appropriate Memory**: Choose the right memory implementation for your use case.
4. **Validate Tools Before Use**: Ensure tools are properly defined and executable.
5. **Implement Proper Error Handling**: Handle LLM errors, tool execution failures, and other issues gracefully.
6. **Follow Interface-First Design**: Adhere to the established interfaces for compatibility.
7. **Include Appropriate Logging**: Log important events and errors for debugging.
8. **Handle Streaming Properly**: If supporting streaming, ensure it's handled correctly.
9. **Test Edge Cases**: Test with various inputs, including edge cases.
10. **Document Behavior**: Clearly document the agent's behavior and configuration options. 