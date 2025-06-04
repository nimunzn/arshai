"""Unit tests for accounting callback handler."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from arshai.callbacks.accounting import AccountingCallbackHandler


@pytest.fixture
def accounting_handler():
    """Create an accounting handler with mocked client."""
    # Here we need to patch the entire AccountingClient, not just a method
    # This will prevent actual HTTP requests
    with patch('src.callbacks.accounting.accounting_client', autospec=True) as mock_client:
        handler = AccountingCallbackHandler(
            correlation_id='test-corr-123',
            request_id='test-req-123',
            user_id='test-user-123'
        )
        handler.model_name = 'gpt-4'
        return handler, mock_client


def test_initialization(accounting_handler):
    """Test correct initialization of the handler."""
    handler, _ = accounting_handler
    
    assert handler._correlation_id == 'test-corr-123'
    assert handler._request_id == 'test-req-123'
    assert handler._user_id == 'test-user-123'
    assert handler.model_name == 'gpt-4'


@pytest.mark.asyncio
async def test_call_accounting(accounting_handler):
    """Test calling the accounting service."""
    handler, mock_client = accounting_handler
    
    # Call the accounting method
    await handler.call_accounting(
        output_tokens=500,
        prompt_tokens=200,
        agent_slug='test-agent'
    )
    
    # Verify the accounting client was called with correct parameters
    mock_client.usage_log.assert_called_once_with(
        request_id='test-req-123',
        correlation_id='test-corr-123',
        incoming_used_tokens=200,
        outgoing_used_tokens=500,
        agent_slug='test-agent',
        user_id='test-user-123'
    ) 