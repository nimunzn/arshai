"""Unit tests for FlashRank Reranker."""

import pytest
from unittest.mock import MagicMock, patch
import os

from arshai.core.interfaces import IRerankInput
from arshai.core.interfaces import Document
from arshai.rerankers.flashrank_reranker import FlashRankReranker


@pytest.fixture
def mock_ranker():
    """Create a mock FlashRank Ranker object."""
    mock_ranker = MagicMock()
    return mock_ranker


@pytest.fixture
def flashrank_reranker(mock_ranker):
    """Create a FlashRank reranker with mocked ranker."""
    with patch("src.rerankers.flashrank_reranker.Ranker", return_value=mock_ranker):
        reranker = FlashRankReranker(model_name="rank-T5-flan", device="cpu", top_k=3)
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
    # Test with default model
    with patch("src.rerankers.flashrank_reranker.Ranker") as mock_ranker_cls:
        reranker = FlashRankReranker()
        
        # Verify ranker was initialized with default parameters
        mock_ranker_cls.assert_called_once_with(model_name="rank-T5-flan")
        
        # Verify attributes
        assert reranker.model_name == "rank-T5-flan"
        assert reranker.device == "cpu"
        assert reranker.top_k is None
    
    # Test with custom parameters
    with patch("src.rerankers.flashrank_reranker.Ranker") as mock_ranker_cls:
        reranker = FlashRankReranker(
            model_name="custom-model",
            device="cuda",
            top_k=5
        )
        
        # Verify ranker was initialized with custom model
        mock_ranker_cls.assert_called_once_with(model_name="custom-model")
        
        # Verify attributes
        assert reranker.model_name == "custom-model"
        assert reranker.device == "cuda"
        assert reranker.top_k == 5
    
    # Test with missing flashrank package
    with patch("src.rerankers.flashrank_reranker.Ranker", side_effect=ImportError("No module")):
        with pytest.raises(ImportError, match="Could not import flashrank python package"):
            FlashRankReranker()


def test_rerank(flashrank_reranker, mock_ranker, sample_documents):
    """Test reranking documents."""
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Set up mock response from FlashRank
    mock_results = [
        {"text": "Document 3 content", "score": 0.95, "meta": {"source": "source3", "id": 3}},
        {"text": "Document 1 content", "score": 0.85, "meta": {"source": "source1", "id": 1}},
        {"text": "Document 5 content", "score": 0.75, "meta": {"source": "source5", "id": 5}},
        {"text": "Document 2 content", "score": 0.65, "meta": {"source": "source2", "id": 2}},
        {"text": "Document 4 content", "score": 0.55, "meta": {"source": "source4", "id": 4}},
    ]
    mock_ranker.rerank.return_value = mock_results
    
    # Perform reranking
    results = flashrank_reranker.rerank(rerank_input)
    
    # Verify ranker was called with correct parameters
    mock_ranker.rerank.assert_called_once()
    rerank_request = mock_ranker.rerank.call_args[0][0]
    assert rerank_request.query == "test query"
    assert len(rerank_request.passages) == 5
    
    # Check passage structure
    assert rerank_request.passages[0]["id"] == 0
    assert rerank_request.passages[0]["text"] == "Document 1 content"
    assert rerank_request.passages[0]["meta"] == {"source": "source1", "id": 1}
    
    # Verify results (top_k=3)
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


def test_rerank_without_top_k(flashrank_reranker, mock_ranker, sample_documents):
    """Test reranking without top_k limit."""
    # Set top_k to None to return all results
    flashrank_reranker.top_k = None
    
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Set up mock response from FlashRank
    mock_results = []
    for i in range(5):
        mock_results.append({
            "text": f"Document {i+1} content", 
            "score": 0.9 - (i * 0.1), 
            "meta": {"source": f"source{i+1}", "id": i+1}
        })
    
    mock_ranker.rerank.return_value = mock_results
    
    # Perform reranking
    results = flashrank_reranker.rerank(rerank_input)
    
    # Verify all documents were returned (no top_k limit)
    assert len(results) == 5


def test_rerank_error_handling(flashrank_reranker, mock_ranker, sample_documents):
    """Test error handling in reranking."""
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Make the FlashRank ranker raise an exception
    mock_ranker.rerank.side_effect = Exception("Reranking error")
    
    # Perform reranking (should return original documents on error)
    results = flashrank_reranker.rerank(rerank_input)
    
    # Verify the original documents were returned in same order
    assert len(results) == 5
    assert results == sample_documents


@pytest.mark.asyncio
async def test_arerank(flashrank_reranker, mock_ranker, sample_documents):
    """Test async reranking functionality."""
    # Create input for reranking
    rerank_input = IRerankInput(
        query="test query",
        documents=sample_documents
    )
    
    # Set up mock response from FlashRank
    mock_results = [
        {"text": "Document 1 content", "score": 0.95, "meta": {"source": "source1", "id": 1}},
    ]
    mock_ranker.rerank.return_value = mock_results
    
    # Perform async reranking
    results = await flashrank_reranker.arerank(rerank_input)
    
    # Verify ranker was called
    mock_ranker.rerank.assert_called_once()
    
    # Verify results (should be same as sync version)
    assert len(results) == 1
    assert results[0].metadata["relevance_score"] == 0.95 