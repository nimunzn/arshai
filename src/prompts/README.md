# Prompts Module

## Overview
The Prompts module provides a structured collection of prompt templates and functions for effective LLM interaction. It includes specialized prompts for specific tasks, generic templates, and utilities for prompt construction to ensure consistent and high-quality interactions with language models.

## Architecture
```
prompts/
├── general.py                # General-purpose prompt templates
├── guardrails.py             # Safety and guidance prompts
├── working_memory.py         # Working memory management prompts
├── human_like_conversation.py # Prompts for natural conversation
└── __pycache__/              # Python bytecode files
```

## Implementation Guide

### Using Prompt Functions

The prompts module uses simple functions that return formatted prompt strings:

```python
from src.prompts.general import TOOL_USAGE_PROMPT, STRUCTURED_OUTPUT_PROMPT
from src.prompts.human_like_conversation import HUMAN_LIKE_CONVERSATION_PROMPT

# Get a tool usage prompt
tool_prompt = TOOL_USAGE_PROMPT()

# Get a structured output prompt with a specific response structure
json_structure = {
    "question": "string",
    "answer": "string",
    "confidence": "number"
}
structured_prompt = STRUCTURED_OUTPUT_PROMPT(json_structure)

# Get a human-like conversation prompt
conversation_prompt = HUMAN_LIKE_CONVERSATION_PROMPT()

# Use the prompts with an LLM
messages = [
    {"role": "system", "content": conversation_prompt + "\n" + tool_prompt},
    {"role": "user", "content": "How do I reset my password?"}
]

# Send to LLM
response = llm.generate(messages=messages)
```

### Using Working Memory Prompts

The working memory prompts provide advanced memory management for conversational agents:

```python
from src.prompts.working_memory import MEMORY_PROMPT

# Current working memory state (would be maintained by the agent)
current_memory = """
USER PROFILE:
John Doe, 35 years old, software engineer interested in AI and machine learning.

AGENT PROFILE:
Technical support assistant specialized in software and AI topics.

CONVERSATION STORY:
User initiated conversation asking about machine learning frameworks. Assistant provided overview of TensorFlow and PyTorch with comparisons.

CURRENT CONVERSATION HISTORY:
User asked about differences between TensorFlow and PyTorch. 
Assistant explained TensorFlow is more production-focused while PyTorch is research-oriented with dynamic computation graphs.

DIALOGUE PLANNING AND GOALS:
Provide technical but accessible information on ML frameworks.
Understand user's experience level to tailor explanations appropriately.

CONVERSATION MOOD:
Professional and curious. User is engaged and asking technical questions.
"""

# Generate a memory-enhanced prompt
memory_prompt = MEMORY_PROMPT(current_memory)

# Use this with an LLM
messages = [
    {"role": "system", "content": memory_prompt},
    {"role": "user", "content": "What about JAX? Is it better than PyTorch for research?"}
]

# Generate response with memory context
response = llm.generate(messages=messages)
```

## Integration Points

### With Agents
Prompts enhance agent capabilities by providing structured interaction patterns:

```python
from src.agents.conversation import ConversationAgent
from src.prompts.human_like_conversation import HUMAN_LIKE_CONVERSATION_PROMPT
from src.prompts.guardrails import SAFETY_GUARDRAILS_PROMPT

# Create a combined prompt for the agent
agent_prompt = HUMAN_LIKE_CONVERSATION_PROMPT() + "\n\n" + SAFETY_GUARDRAILS_PROMPT()

# Initialize agent with the prompt
agent = ConversationAgent(
    system_prompt=agent_prompt,
    llm=llm_client
)

# Process user message
response = agent.process_message("Can you help me with my project?")
```

### With Memory Systems
Prompts help structure and utilize memory effectively:

```python
from src.memory.memory_manager import MemoryManager
from src.prompts.working_memory import MEMORY_PROMPT, WORKING_MEMORY_STRUCTURE_OUTPUT_DEFINITION

# Initialize memory manager
memory_manager = MemoryManager()

# Get current memory for a conversation
current_memory = memory_manager.get_memory("conversation-123")

# Generate memory-enhanced prompt
memory_prompt = MEMORY_PROMPT(current_memory)

# Process user message with memory context
messages = [
    {"role": "system", "content": memory_prompt},
    {"role": "user", "content": "What did we talk about last time?"}
]

# Generate response
response = llm.generate(messages=messages)

# Update memory
updated_memory = llm.generate(
    messages=[
        {"role": "system", "content": WORKING_MEMORY_STRUCTURE_OUTPUT_DEFINITION},
        {"role": "user", "content": f"Update the working memory with this new interaction:\nUser: What did we talk about last time?\nAssistant: {response.content}\n\nCurrent memory:\n{current_memory}"}
    ]
).content

# Store updated memory
memory_manager.store_memory("conversation-123", updated_memory)
```

## Configuration

Since the prompts are implemented as functions, they can be configured directly by passing parameters:

```python
from src.prompts.general import STRUCTURED_OUTPUT_PROMPT

# Configure a structured output prompt with specific requirements
custom_structure = {
    "analysis": {
        "sentiment": "string",
        "key_points": ["string"],
        "action_items": ["string"]
    },
    "recommendation": "string"
}

custom_prompt = STRUCTURED_OUTPUT_PROMPT(custom_structure)
```

### Working with Environment Variables

Some prompts may reference environment variables for configuration:

```python
import os
from src.prompts.guardrails import SAFETY_GUARDRAILS_PROMPT

# Set environment variables to configure prompts
os.environ["SAFETY_LEVEL"] = "high"

# Get prompt that uses environment configuration
safety_prompt = SAFETY_GUARDRAILS_PROMPT()
``` 