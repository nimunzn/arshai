"""Unit tests for Voyage AI Reranker."""

import pytest
from unittest.mock import MagicMock, patch
import os

from arshai.core.interfaces import IRerankInput
from arshai.core.interfaces import Document
from arshai.rerankers.voyage_reranker import VoyageReranker


@pytest.fixture
def mock_voyage_client():
    """Create a mock Voyage AI client."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def voyage_reranker(mock_voyage_client):
    """Create a Voyage reranker with mocked client."""
    with patch("src.rerankers.voyage_reranker.voyageai.Client", return_value=mock_voyage_client):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-api-key"}):
            reranker = VoyageReranker(model_name="rerank-2", top_k=3)
            return reranker


@pytest.fixture
def sample_documents():
    """Create a sample list of documents for testing."""
    return [
        Document(page_content="Document 1 content", metadata={"source": "source1", "id": 1}),
        Document(page_content="Document 2 content", metadata={"source": "source2", "id": 2}),
        Document(page_content="Document 3 content", metadata={"source": "source3", "id": 3}),
        Document(page_content="Document 4 content", metadata={"source": "source4", "id": 4}),
        Document(page_content="Document 5 content", metadata={"source": "source5", "id": 5})
    ]


def test_initialization():
    """Test initialization with different parameters."""
    # Test with environment variable
    with patch("src.rerankers.voyage_reranker.voyageai.Client") as mock_voyage:
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-api-key"}):
            reranker = VoyageReranker(model_name="rerank-lite-1")
            
            # Verify client was initialized with correct API key
            mock_voyage.assert_called_once_with(api_key="test-api-key")
            
            # Verify attributes
            assert reranker.model_name == "rerank-lite-1"
            assert reranker.top_k is None
    
    # Test with explicit API key
    with patch("src.rerankers.voyage_reranker.voyageai.Client") as mock_voyage:
        reranker = VoyageReranker(
            model_name="rerank-2",
            top_k=5,
            api_key="explicit-api-key"
        )
        
        # Verify client was initialized with explicit API key
        mock_voyage.assert_called_once_with(api_key="explicit-api-key")
        
        # Verify attributes
        assert reranker.model_name == "rerank-2"
        assert reranker.top_k == 5
    
    # Test with missing API key
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Voyage API key not provided"):
            VoyageReranker()
    
    # Test with missing voyageai package
    with patch("src.rerankers.voyage_reranker.voyageai.Client", side_effect=ImportError("No module")):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-api-key"}):
            with pytest.raises(ImportError, match="Could not import voyageai python package"):
                VoyageReranker()


def test_rerank(voyage_reranker, mock_voyage_client, sample_documents):
    """Test reranking documents."""
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Set up mock response from Voyage client
    mock_result_1 = MagicMock()
    mock_result_1.index = 2  # Document 3
    mock_result_1.document = "Document 3 content"
    mock_result_1.relevance_score = 0.95
    
    mock_result_2 = MagicMock()
    mock_result_2.index = 0  # Document 1
    mock_result_2.document = "Document 1 content"
    mock_result_2.relevance_score = 0.85
    
    mock_result_3 = MagicMock()
    mock_result_3.index = 4  # Document 5
    mock_result_3.document = "Document 5 content"
    mock_result_3.relevance_score = 0.75
    
    mock_reranking = MagicMock()
    mock_reranking.results = [mock_result_1, mock_result_2, mock_result_3]
    mock_voyage_client.rerank.return_value = mock_reranking
    
    # Perform reranking
    results = voyage_reranker.rerank(rerank_input)
    
    # Verify client was called with correct parameters
    mock_voyage_client.rerank.assert_called_once()
    args = mock_voyage_client.rerank.call_args[1]
    assert args["query"] == "test query"
    assert args["documents"] == [doc.page_content for doc in sample_documents]
    assert args["model"] == "rerank-2"
    assert args["top_k"] == 3
    
    # Verify results
    assert len(results) == 3
    
    # Check first result
    assert results[0].page_content == "Document 3 content"
    assert results[0].metadata["source"] == "source3"
    assert results[0].metadata["id"] == 3
    assert results[0].metadata["relevance_score"] == 0.95
    
    # Check second result
    assert results[1].page_content == "Document 1 content"
    assert results[1].metadata["relevance_score"] == 0.85
    
    # Check third result
    assert results[2].page_content == "Document 5 content"
    assert results[2].metadata["relevance_score"] == 0.75


def test_rerank_without_top_k(voyage_reranker, mock_voyage_client, sample_documents):
    """Test reranking without top_k limit."""
    # Set top_k to None to return all results
    voyage_reranker.top_k = None
    
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Set up mock response from Voyage client
    mock_results = []
    for i in range(5):
        mock_result = MagicMock()
        mock_result.index = i
        mock_result.document = f"Document {i+1} content"
        mock_result.relevance_score = 0.9 - (i * 0.1)
        mock_results.append(mock_result)
    
    mock_reranking = MagicMock()
    mock_reranking.results = mock_results
    mock_voyage_client.rerank.return_value = mock_reranking
    
    # Perform reranking
    results = voyage_reranker.rerank(rerank_input)
    
    # Verify top_k was set to document count
    args = mock_voyage_client.rerank.call_args[1]
    assert args["top_k"] == 5
    
    # Verify all documents were reranked
    assert len(results) == 5


def test_rerank_error_handling(voyage_reranker, mock_voyage_client, sample_documents):
    """Test error handling in reranking."""
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Make the Voyage client raise an exception
    mock_voyage_client.rerank.side_effect = Exception("API error")
    
    # Perform reranking (should return original documents on error)
    results = voyage_reranker.rerank(rerank_input)
    
    # Verify the original documents were returned in same order
    assert len(results) == 5
    assert results == sample_documents


@pytest.mark.asyncio
async def test_arerank(voyage_reranker, mock_voyage_client, sample_documents):
    """Test async reranking functionality."""
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Set up mock response from Voyage client
    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.document = "Document 1 content"
    mock_result.relevance_score = 0.95
    
    mock_reranking = MagicMock()
    mock_reranking.results = [mock_result]
    mock_voyage_client.rerank.return_value = mock_reranking
    
    # Perform async reranking
    results = await voyage_reranker.arerank(rerank_input)
    
    # Verify client was called
    mock_voyage_client.rerank.assert_called_once()
    
    # Verify results (should be same as sync version)
    assert len(results) == 1
    assert results[0].metadata["relevance_score"] == 0.95 