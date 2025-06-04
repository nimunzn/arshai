"""Unit tests for SearxNG web search client."""

import pytest
import json
import os
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, List, Any

from arshai.core.interfaces import IWebSearchResult
from arshai.web_search.searxng import SearxNGClient, SearxNGConfig


@pytest.fixture
def searx_config():
    """Create a basic SearxNG configuration."""
    return SearxNGConfig(
        host="https://searx.example.com",
        language="en",
        timeout=10,
        verify_ssl=True,
        default_engines=["google", "bing"],
        default_categories=["general"]
    )


@pytest.fixture
def searx_client(searx_config):
    """Create a SearxNG client with the test configuration."""
    return SearxNGClient(searx_config)


@pytest.fixture
def mock_search_response():
    """Create a mock search response."""
    return {
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "This is test content 1",
                "engines": ["google"],
                "category": "general"
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "This is test content 2",
                "engines": ["bing"],
                "category": "general"
            },
            {
                "title": "Test Result 3",
                "url": "https://example.com/3",
                "content": "This is test content 3",
                "engines": ["google", "bing"],
                "category": "general"
            }
        ],
        "query": "test query",
        "number_of_results": 3
    }


def test_initialization():
    """Test client initialization with different configurations."""
    # Test with explicit host
    config = SearxNGConfig(
        host="https://searx.example.com",
        language="en"
    )
    client = SearxNGClient(config)
    assert client.base_url == "https://searx.example.com"
    
    # Test with trailing slash in host
    config = SearxNGConfig(
        host="https://searx.example.com/",
        language="en"
    )
    client = SearxNGClient(config)
    assert client.base_url == "https://searx.example.com"
    
    # Test with environment variable
    with patch.dict('os.environ', {'SEARX_INSTANCE': 'https://searx.env.com'}):
        config = SearxNGConfig(
            language="en"
        )
        client = SearxNGClient(config)
        assert client.base_url == "https://searx.env.com"
    
    # Test with missing host raises error
    with patch.dict('os.environ', {}, clear=True):
        config = SearxNGConfig(
            language="en"
        )
        with pytest.raises(ValueError, match="SearxNG instance URL not provided"):
            SearxNGClient(config)


def test_prepare_params(searx_client):
    """Test parameter preparation for search."""
    # Test with default values
    params = searx_client._prepare_params("test query")
    assert params["q"] == "test query"
    assert params["format"] == "json"
    assert params["engines"] == "google,bing"
    assert params["categories"] == "general"
    assert params["language"] == "en"
    
    # Test with custom engines and categories
    params = searx_client._prepare_params(
        "test query",
        engines=["duckduckgo"],
        categories=["science"]
    )
    assert params["engines"] == "duckduckgo"
    assert params["categories"] == "science"
    
    # Test with additional parameters
    params = searx_client._prepare_params(
        "test query",
        time_range="day",
        safesearch=1
    )
    assert params["time_range"] == "day"
    assert params["safesearch"] == 1


def test_parse_results(searx_client, mock_search_response):
    """Test parsing of search results."""
    results = searx_client._parse_results(mock_search_response, num_results=2)
    
    # Verify number of results limited correctly
    assert len(results) == 2
    
    # Verify result objects
    assert isinstance(results[0], IWebSearchResult)
    assert results[0].title == "Test Result 1"
    assert results[0].url == "https://example.com/1"
    assert results[0].content == "This is test content 1"
    assert results[0].engines == ["google"]
    assert results[0].category == "general"
    
    # Test with invalid result that would raise exception
    bad_response = {
        "results": [
            {"title": "Good Result", "url": "https://example.com"},
            {"invalid": "Missing required fields"}
        ]
    }
    
    # Should skip invalid results but not fail
    results = searx_client._parse_results(bad_response, num_results=2)
    assert len(results) == 1
    assert results[0].title == "Good Result"


def test_search(searx_client, mock_search_response):
    """Test synchronous search function."""
    with patch('requests.get') as mock_get:
        # Mock the requests.get method
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response
        mock_get.return_value = mock_response
        
        # Perform search
        results = searx_client.search("test query", num_results=2)
        
        # Verify requests.get was called with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        
        # Check URL
        assert args[0] == "https://searx.example.com/search"
        
        # Check request parameters
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["params"]["engines"] == "google,bing"
        assert kwargs["timeout"] == 10
        assert kwargs["verify"] is True
        
        # Check results
        assert len(results) == 2
        assert results[0].title == "Test Result 1"
        assert results[1].title == "Test Result 2"


def test_search_error_handling(searx_client):
    """Test error handling in synchronous search."""
    with patch('requests.get') as mock_get:
        # Mock a request that raises an exception
        mock_get.side_effect = Exception("Connection error")
        
        # Search should return empty results rather than raising
        results = searx_client.search("test query")
        assert results == []


@pytest.mark.asyncio
async def test_asearch(searx_client, mock_search_response):
    """Test asynchronous search function."""
    # Create mock for the context manager structure used in async with
    mock_client_session = MagicMock()
    mock_cm = MagicMock()
    mock_response = MagicMock()
    
    # Set up the response
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=mock_search_response)
    
    # Set up the context manager structure
    mock_client_session.return_value.__aenter__.return_value = mock_cm
    mock_cm.get.return_value.__aenter__.return_value = mock_response
    
    # Patch module object
    with patch('src.web_search.searxng.aiohttp.ClientSession', mock_client_session):
        # Perform async search
        results = await searx_client.asearch("test query", num_results=3)
        
        # Verify get was called with correct parameters
        mock_cm.get.assert_called_once()
        args, kwargs = mock_cm.get.call_args
        
        # Check URL
        assert args[0] == "https://searx.example.com/search"
        
        # Check request parameters
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["params"]["engines"] == "google,bing"
        assert kwargs["timeout"] == 10
        assert kwargs["ssl"] is True
        
        # Check results parsing
        assert len(results) == 3
        assert results[0].title == "Test Result 1"
        assert results[2].title == "Test Result 3"


@pytest.mark.asyncio
async def test_asearch_error_handling(searx_client):
    """Test error handling in asynchronous search."""
    # Test with get exception
    mock_client_session = MagicMock()
    mock_cm = MagicMock()
    
    # Set up the exception
    mock_cm.get.side_effect = Exception("Async connection error")
    mock_client_session.return_value.__aenter__.return_value = mock_cm
    
    # Patch module object
    with patch('src.web_search.searxng.aiohttp.ClientSession', mock_client_session):
        # Should return empty list on error
        results = await searx_client.asearch("test query")
        assert results == []
    
    # Test with non-200 status code
    mock_client_session = MagicMock()
    mock_cm = MagicMock()
    mock_response = MagicMock()
    
    # Set up the response with error status
    mock_response.status = 500
    mock_client_session.return_value.__aenter__.return_value = mock_cm
    mock_cm.get.return_value.__aenter__.return_value = mock_response
    
    # Patch module object
    with patch('src.web_search.searxng.aiohttp.ClientSession', mock_client_session):
        # Should return empty list on error status
        results = await searx_client.asearch("test query")
        assert results == [] 