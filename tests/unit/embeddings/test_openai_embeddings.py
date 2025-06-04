"""Unit tests for OpenAI embeddings."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import os

from arshai.core.interfaces import EmbeddingConfig
from arshai.embeddings.openai_embeddings import OpenAIEmbedding


@pytest.fixture
def embedding_config():
    """Create a basic embedding configuration for OpenAI."""
    return EmbeddingConfig(
        model_name="text-embedding-3-small",
        batch_size=10
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock_client = MagicMock()
    
    # Mock the embeddings.create method
    mock_embeddings = MagicMock()
    mock_client.embeddings.create = mock_embeddings
    
    return mock_client


@pytest.fixture
def mock_async_openai_client():
    """Create a mock async OpenAI client for testing."""
    mock_client = MagicMock()
    
    # Mock the embeddings.create method as an async mock
    mock_embeddings = MagicMock()
    mock_client.embeddings.create = mock_embeddings
    
    return mock_client


@pytest.fixture
def openai_embedding(mock_openai_client, mock_async_openai_client, embedding_config):
    """Create an OpenAI embedding instance with mocked clients."""
    with patch("src.embeddings.openai_embeddings.OpenAI", return_value=mock_openai_client):
        with patch("src.embeddings.openai_embeddings.AsyncOpenAI", return_value=mock_async_openai_client):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                embedding = OpenAIEmbedding(embedding_config)
                embedding.client = mock_openai_client
                embedding.async_client = mock_async_openai_client
                return embedding


def test_initialization():
    """Test initialization with different parameters."""
    # Test with environment variables
    with patch("src.embeddings.openai_embeddings.OpenAI") as mock_openai:
        with patch("src.embeddings.openai_embeddings.AsyncOpenAI") as mock_async_openai:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                config = EmbeddingConfig(
                    model_name="text-embedding-3-small",
                    batch_size=32
                )
                embedding = OpenAIEmbedding(config)
                
                # Verify clients were initialized with correct API key
                mock_openai.assert_called_once_with(api_key="test-key")
                mock_async_openai.assert_called_once_with(api_key="test-key")
                
                # Verify dimension is set correctly
                assert embedding.dimension == 1536
    
    # Test with alternative environment variable
    with patch("src.embeddings.openai_embeddings.OpenAI") as mock_openai:
        with patch("src.embeddings.openai_embeddings.AsyncOpenAI") as mock_async_openai:
            with patch.dict("os.environ", {"EMBEDDINGS_OPENAI_API_KEY": "test-key-2"}, clear=True):
                config = EmbeddingConfig(
                    model_name="text-embedding-3-large"
                )
                embedding = OpenAIEmbedding(config)
                
                # Verify clients were initialized with correct API key
                mock_openai.assert_called_once_with(api_key="test-key-2")
                mock_async_openai.assert_called_once_with(api_key="test-key-2")
                
                # Verify dimension is set correctly for this model
                assert embedding.dimension == 3072
    
    # Test with missing API key
    with patch.dict("os.environ", {}, clear=True):
        config = EmbeddingConfig(
            model_name="text-embedding-3-small"
        )
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAIEmbedding(config)
    
    # Test with missing OpenAI package
    with patch("src.embeddings.openai_embeddings.HAS_OPENAI", False):
        config = EmbeddingConfig(
            model_name="text-embedding-3-small"
        )
        with pytest.raises(ImportError, match="The openai package is required"):
            OpenAIEmbedding(config)


def test_embed_documents(openai_embedding, mock_openai_client):
    """Test embedding multiple documents."""
    # Set up mock response
    mock_data_1 = MagicMock()
    mock_data_1.embedding = [0.1, 0.2, 0.3]
    
    mock_data_2 = MagicMock()
    mock_data_2.embedding = [0.4, 0.5, 0.6]
    
    mock_response = MagicMock()
    mock_response.data = [mock_data_1, mock_data_2]
    mock_openai_client.embeddings.create.return_value = mock_response
    
    # Test with multiple documents
    result = openai_embedding.embed_documents(["Document 1", "Document 2"])
    
    # Verify client was called with correct parameters
    mock_openai_client.embeddings.create.assert_called_once()
    args = mock_openai_client.embeddings.create.call_args[1]
    assert args["model"] == "text-embedding-3-small"
    assert args["input"] == ["Document 1", "Document 2"]
    assert args["encoding_format"] == "float"
    
    # Verify result format
    assert "dense" in result
    assert len(result["dense"]) == 2
    assert result["dense"][0] == [0.1, 0.2, 0.3]
    assert result["dense"][1] == [0.4, 0.5, 0.6]


def test_embed_documents_batching(openai_embedding, mock_openai_client):
    """Test batching of documents for embedding."""
    # Create a larger set of documents than the batch size
    documents = [f"Document {i}" for i in range(15)]  # Batch size is 10 from fixture
    
    # Set up mock responses for both batches
    mock_data_batch1 = [MagicMock() for _ in range(10)]
    for i, data in enumerate(mock_data_batch1):
        data.embedding = [i * 0.1, i * 0.2, i * 0.3]
    
    mock_data_batch2 = [MagicMock() for _ in range(5)]
    for i, data in enumerate(mock_data_batch2):
        data.embedding = [(i + 10) * 0.1, (i + 10) * 0.2, (i + 10) * 0.3]
    
    mock_response1 = MagicMock()
    mock_response1.data = mock_data_batch1
    
    mock_response2 = MagicMock()
    mock_response2.data = mock_data_batch2
    
    mock_openai_client.embeddings.create.side_effect = [mock_response1, mock_response2]
    
    # Test with documents requiring batching
    result = openai_embedding.embed_documents(documents)
    
    # Verify client was called twice with correct batches
    assert mock_openai_client.embeddings.create.call_count == 2
    
    # First batch
    first_call_args = mock_openai_client.embeddings.create.call_args_list[0][1]
    assert len(first_call_args["input"]) == 10
    assert first_call_args["input"][0] == "Document 0"
    
    # Second batch
    second_call_args = mock_openai_client.embeddings.create.call_args_list[1][1]
    assert len(second_call_args["input"]) == 5
    assert second_call_args["input"][0] == "Document 10"
    
    # Verify result combines both batches
    assert len(result["dense"]) == 15


def test_embed_documents_empty(openai_embedding, mock_openai_client):
    """Test embedding an empty list of documents."""
    result = openai_embedding.embed_documents([])
    
    # Verify client was not called
    mock_openai_client.embeddings.create.assert_not_called()
    
    # Verify result is empty list
    assert result["dense"] == []


def test_embed_documents_error(openai_embedding, mock_openai_client):
    """Test error handling when embedding documents."""
    # Set up mock to raise an exception
    mock_openai_client.embeddings.create.side_effect = Exception("API error")
    
    # Test with error from API
    with pytest.raises(Exception, match="API error"):
        openai_embedding.embed_documents(["Document 1"])


def test_embed_document(openai_embedding, mock_openai_client):
    """Test embedding a single document."""
    # Set up mock response
    mock_data = MagicMock()
    mock_data.embedding = [0.1, 0.2, 0.3]
    
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    mock_openai_client.embeddings.create.return_value = mock_response
    
    # Test with a single document
    result = openai_embedding.embed_document("Single document")
    
    # Verify client was called with correct parameters
    mock_openai_client.embeddings.create.assert_called_once()
    args = mock_openai_client.embeddings.create.call_args[1]
    assert args["input"] == ["Single document"]
    
    # Verify result format
    assert "dense" in result
    assert result["dense"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_aembed_documents(openai_embedding, mock_async_openai_client):
    """Test asynchronous embedding of multiple documents."""
    # Set up mock response
    mock_data_1 = MagicMock()
    mock_data_1.embedding = [0.1, 0.2, 0.3]
    
    mock_data_2 = MagicMock()
    mock_data_2.embedding = [0.4, 0.5, 0.6]
    
    mock_response = MagicMock()
    mock_response.data = [mock_data_1, mock_data_2]
    
    # Configure the async mock to be awaitable
    async_mock = AsyncMock()
    async_mock.return_value = mock_response
    mock_async_openai_client.embeddings.create = async_mock
    
    # Test with multiple documents
    result = await openai_embedding.aembed_documents(["Document 1", "Document 2"])
    
    # Verify embeddings client was called with correct parameters
    mock_async_openai_client.embeddings.create.assert_called_once()
    args = mock_async_openai_client.embeddings.create.call_args[1]
    assert args["model"] == "text-embedding-3-small"
    assert args["input"] == ["Document 1", "Document 2"]
    
    # Verify result format (should match sync version)
    assert len(result["dense"]) == 2
    assert result["dense"][0] == [0.1, 0.2, 0.3]
    assert result["dense"][1] == [0.4, 0.5, 0.6] 