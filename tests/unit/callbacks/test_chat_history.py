"""Unit tests for chat history callback handler."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import uuid

from arshai.callbacks.chat_history import ChatHistoryCallbackHandler


@pytest.fixture
def user_data():
    """Return mock user data for testing."""
    return {
        'user_id': 'test-user-123',
        'org_id': '27820ae0-d693-42cc-acf0-14064f8a393a',
        'details': {
            'given_name': 'Test',
            'family_name': 'User'
        },
        'user_metadata': {}
    }


@pytest.fixture
def chat_history_handler(user_data):
    """Create a chat history handler with mocked client."""
    # Create the handler with test data
    handler = ChatHistoryCallbackHandler(
        message_time=datetime.now().isoformat(),
        conversation_id='test-convo-123',
        correlation_id='test-corr-123',
        request_id='test-req-123',
        parent_message='test-parent-123',
        user_data=user_data,
        realm='test',
        agent_title='Test Agent',
        is_anonymous=False
    )
    
    # Mock the chat history client
    handler.chat_history_client = MagicMock()
    handler.chat_history_client.add_message = AsyncMock()
    handler.chat_history_client.create_conversation = AsyncMock(return_value='test-convo-123')
    handler.chat_history_client.rename_conversation = AsyncMock(return_value={'success': True})
    handler.chat_history_client.get_conversation = AsyncMock(return_value={'name': 'Test Conversation'})
    handler.chat_history_client.get_conversation_state = AsyncMock(return_value='normal')
    handler.chat_history_client.get_messages = AsyncMock(return_value={'items': [{'text': 'Test message'}]})
    
    return handler


@pytest.mark.asyncio
async def test_send_message(chat_history_handler):
    """Test sending a message."""
    message_time = datetime.now().isoformat()
    message_id = await chat_history_handler.send_message(
        conversation_id='test-convo-123',
        message_text='Hello world',
        sender='end_user',
        parent_message_id='test-parent-123',
        message_time=message_time
    )
    
    # Check that the client was called
    chat_history_handler.chat_history_client.add_message.assert_called_once()
    
    # Verify the message format
    call_args = chat_history_handler.chat_history_client.add_message.call_args[0][0]
    assert call_args['conversation_id'] == 'test-convo-123'
    assert call_args['state'] == 'normal'
    assert call_args['messages'][0]['text'] == 'Hello world'
    assert call_args['messages'][0]['responder'] == 'end_user'
    assert call_args['messages'][0]['is_visible'] == True
    
    # Verify UUID format for message ID
    assert uuid.UUID(message_id, version=4)


@pytest.mark.asyncio
async def test_create_conversation(chat_history_handler):
    """Test creating a conversation."""
    conversation_id = await chat_history_handler.create_conversation()
    
    # Check that the client was called correctly
    chat_history_handler.chat_history_client.create_conversation.assert_called_once()
    
    # Verify that the arguments were passed
    call_kwargs = chat_history_handler.chat_history_client.create_conversation.call_args[1]
    assert call_kwargs['correlation_id'] == 'test-corr-123'
    assert call_kwargs['first_name'] == 'Test'
    assert call_kwargs['last_name'] == 'User'
    assert call_kwargs['is_anonymous'] == False
    
    # Verify returned conversation ID
    assert conversation_id == 'test-convo-123'


@pytest.mark.asyncio
async def test_rename_conversation(chat_history_handler):
    """Test renaming a conversation."""
    result = await chat_history_handler.rename_conversation('test-convo-123', 'New Conversation Name')
    
    # Check that the client was called correctly
    chat_history_handler.chat_history_client.rename_conversation.assert_called_once_with(
        'test-convo-123', 'New Conversation Name'
    )
    
    # Verify result
    assert result['success'] == True


@pytest.mark.asyncio
async def test_get_conversation_details(chat_history_handler):
    """Test retrieving conversation details."""
    result = await chat_history_handler.get_conversation_details('test-convo-123')
    
    # Check that the client was called correctly
    chat_history_handler.chat_history_client.get_conversation.assert_called_once_with('test-convo-123')
    
    # Verify result
    assert result['name'] == 'Test Conversation'


@pytest.mark.asyncio
async def test_get_conversation_state(chat_history_handler):
    """Test retrieving conversation state."""
    state = await chat_history_handler.get_conversation_state('test-convo-123')
    
    # Check that the client was called correctly
    chat_history_handler.chat_history_client.get_conversation_state.assert_called_once_with('test-convo-123')
    
    # Verify state
    assert state == 'normal'


@pytest.mark.asyncio
async def test_get_latest_message(chat_history_handler):
    """Test retrieving the latest message."""
    message = await chat_history_handler.get_latest_message('test-convo-123')
    
    # Check that the client was called correctly
    chat_history_handler.chat_history_client.get_messages.assert_called_once_with('test-convo-123')
    
    # Verify message
    assert message['text'] == 'Test message'


@pytest.mark.asyncio
async def test_error_handling(chat_history_handler):
    """Test error handling in the callback functions."""
    # Mock an exception
    chat_history_handler.chat_history_client.add_message.side_effect = Exception("Test error")
    
    # The method should not raise but return message_id
    message_id = await chat_history_handler.send_message(
        conversation_id='test-convo-123',
        message_text='Test message',
        sender='end_user',
        parent_message_id='test-parent-123',
        message_time=datetime.now().isoformat()
    )
    
    # Should still return a valid UUID
    assert uuid.UUID(message_id, version=4) 