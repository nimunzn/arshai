# Callbacks Module

## Overview
The Callbacks module provides a system for tracking, monitoring, and responding to events that occur during the execution of the Arshai framework. It enables logging, analytics, usage tracking, and integration with external systems through an event-driven architecture.

## Architecture
```
callbacks/
├── chat_history.py          # Chat history tracking callbacks
├── accounting.py            # Usage and cost tracking callbacks
└── __pycache__/             # Python bytecode files
```

## Implementation Guide

### Using the ChatHistoryCallbackHandler

The `ChatHistoryCallbackHandler` provides functionality for tracking and managing conversation history:

```python
from src.callbacks.chat_history import ChatHistoryCallbackHandler

# Initialize the chat history handler
chat_history = ChatHistoryCallbackHandler(
    conversation_id="existing-conversation-id",  # Optional
    correlation_id="correlation-id",
    request_id="request-id",
    user_data={
        "user_id": "user-123",
        "details": {
            "given_name": "John",
            "family_name": "Doe"
        },
        "org_id": "org-456"
    },
    realm="main",
    agent_title="Customer Support Agent",
    is_anonymous=False
)

# Create a new conversation
conversation_id = await chat_history.create_conversation(conversation_state="normal")

# Send messages to the conversation
user_message_id = await chat_history.send_message(
    conversation_id=conversation_id,
    message_text="How can I reset my password?",
    sender="end_user",
    parent_message_id=None,
    message_time="2023-05-15T14:30:00Z",
    metadata=[],
    conversation_state="normal"
)

assistant_message_id = await chat_history.send_message(
    conversation_id=conversation_id,
    message_text="You can reset your password by going to the Account Settings page and clicking on 'Reset Password'.",
    sender="assistant",
    parent_message_id=user_message_id,
    message_time="2023-05-15T14:30:05Z",
    metadata=[],
    conversation_state="normal"
)

# Rename a conversation
await chat_history.rename_conversation(
    conversation_id=conversation_id,
    new_name="Password Reset Help"
)

# Get conversation details
conversation_details = await chat_history.get_conversation_details(
    conversation_id=conversation_id
)
```

### Using the AccountingCallbackHandler

The `AccountingCallbackHandler` provides functionality for tracking usage metrics and costs:

```python
from src.callbacks.accounting import AccountingCallbackHandler

# Initialize the accounting handler
accounting = AccountingCallbackHandler(
    correlation_id="correlation-id",
    request_id="request-id",
    user_id="user-123"
)

# Set the model name (typically done by the LLM client)
accounting.model_name = "gpt-4"

# Log usage after a request completes
await accounting.call_accounting(
    output_tokens=150,
    prompt_tokens=100,
    agent_slug="customer-support"
)
```

## Integration Points

### With LLMs
Callbacks can be used to track LLM usage and performance:

```python
from src.llms.openai import OpenAIClient
from src.callbacks.accounting import AccountingCallbackHandler
from src.callbacks.chat_history import ChatHistoryCallbackHandler

# Create callback handlers
accounting = AccountingCallbackHandler(
    correlation_id="correlation-id",
    request_id="request-id",
    user_id="user-123"
)

chat_history = ChatHistoryCallbackHandler(
    conversation_id="conversation-id",
    correlation_id="correlation-id",
    request_id="request-id",
    user_data={...}  # User data dictionary
)

# Create LLM client
llm = OpenAIClient(
    api_key="your-api-key",
    organization="your-org-id"
)

# Execute LLM request and update accounting
response = llm.generate(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    model="gpt-4"
)

# Record usage in accounting
await accounting.call_accounting(
    output_tokens=response.usage.completion_tokens,
    prompt_tokens=response.usage.prompt_tokens,
    agent_slug="qa-agent"
)

# Record in chat history
await chat_history.send_message(
    conversation_id="conversation-id",
    message_text=response.content,
    sender="assistant",
    parent_message_id="user-message-id",
    message_time=response.created_at,
    metadata=[]
)
```

### With Agents
Callbacks can be used to monitor agent interactions:

```python
from src.agents.conversation import ConversationAgent
from src.callbacks.chat_history import ChatHistoryCallbackHandler

# Initialize chat history handler
chat_history = ChatHistoryCallbackHandler(
    user_data={...},  # User data dictionary
    conversation_id="conversation-id"
)

# Create conversation agent
agent = ConversationAgent(
    name="Support Agent",
    llm=llm_client,
    callbacks=[chat_history]
)

# Process user message
response = await agent.process_message("How can I help you today?")

# Chat history is automatically updated by the agent
```

## Configuration
Configure callbacks via direct initialization with the appropriate parameters:

```python
from src.config import Settings
from src.callbacks.chat_history import ChatHistoryCallbackHandler

# Get settings
settings = Settings()

# Create chat history with configuration based on settings
chat_history = ChatHistoryCallbackHandler(
    conversation_id=settings.conversation_id,
    correlation_id=settings.correlation_id,
    request_id=settings.request_id,
    user_data=settings.get_user_data(),
    realm=settings.realm,
    agent_title=settings.agent_title,
    is_anonymous=settings.is_anonymous
)
``` 