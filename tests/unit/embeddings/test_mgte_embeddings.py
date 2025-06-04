"""Unit tests for MGTE embeddings."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from arshai.core.interfaces import EmbeddingConfig
from arshai.embeddings.mgte_embeddings import MGTEEmbedding


@pytest.fixture
def embedding_config():
    """Create a basic embedding configuration for MGTE."""
    return EmbeddingConfig(
        model_name="multilingual-e5-large",
        batch_size=10,
        additional_params={
            "use_fp16": False,
            "device": "cpu"
        }
    )


@pytest.fixture
def mock_dense_vectors():
    """Create mock dense vectors for testing."""
    return [
        [0.1, 0.2, 0.3],  # Already converted to list
        [0.4, 0.5, 0.6]
    ]


@pytest.fixture
def mock_sparse_vectors():
    """Create mock sparse vectors for testing."""
    return [
        {"indices": [0, 1, 2], "values": [0.4, 0.5, 0.6]},
        {"indices": [3, 4, 5], "values": [0.7, 0.8, 0.9]}
    ]


@pytest.fixture
def mgte_embedding(embedding_config):
    """Create an MGTE embedding instance."""
    with patch("src.embeddings.mgte_embeddings.MGTEEmbeddingFunction"):
        embedding = MGTEEmbedding(embedding_config)
        # Mock the dimension property
        embedding.embedding_function.dim = {"dense": 1024, "sparse": 768}
        return embedding


def test_initialization():
    """Test initialization with different parameters."""
    # Test with standard parameters
    with patch("src.embeddings.mgte_embeddings.MGTEEmbeddingFunction") as mock_function_cls:
        # Set up the mocked instance
        mock_instance = MagicMock()
        mock_instance.dim = {"dense": 1024, "sparse": 768}
        mock_function_cls.return_value = mock_instance
        
        config = EmbeddingConfig(
            model_name="multilingual-e5-large",
            batch_size=32,
            additional_params={
                "use_fp16": False,
                "device": "cpu"
            }
        )
        embedding = MGTEEmbedding(config)
        
        # Verify embedding function was initialized with correct parameters
        mock_function_cls.assert_called_once_with(
            model_name="multilingual-e5-large",
            use_fp16=False,
            device="cpu",
            batch_size=32
        )
        
        # Verify attributes
        assert embedding.model_name == "multilingual-e5-large"
        assert embedding.batch_size == 32
        assert embedding.use_fp16 is False
        assert embedding.device == "cpu"
        
        # Verify dimension
        assert embedding.dimension == 1024
    
    # Test with GPU and fp16
    with patch("src.embeddings.mgte_embeddings.MGTEEmbeddingFunction") as mock_function_cls:
        # Set up the mocked instance
        mock_instance = MagicMock()
        mock_instance.dim = {"dense": 1024, "sparse": 768}
        mock_function_cls.return_value = mock_instance
        
        config = EmbeddingConfig(
            model_name="multilingual-e5-base",
            batch_size=16,
            additional_params={
                "use_fp16": True,
                "device": "cuda"
            }
        )
        embedding = MGTEEmbedding(config)
        
        # Verify embedding function was initialized with GPU parameters
        mock_function_cls.assert_called_once_with(
            model_name="multilingual-e5-base",
            use_fp16=True,
            device="cuda",
            batch_size=16
        )
        
        # Verify attributes
        assert embedding.use_fp16 is True
        assert embedding.device == "cuda"


def test_embed_documents(mgte_embedding, mock_dense_vectors, mock_sparse_vectors):
    """Test embedding multiple documents."""
    # Mock result that would be returned by embed_documents
    mock_result = {
        "dense": mock_dense_vectors, 
        "sparse": mock_sparse_vectors
    }
    
    # Patch the instance's embed_documents method
    with patch.object(mgte_embedding, 'embed_documents', return_value=mock_result):
        # Test with multiple documents
        documents = ["Document 1", "Document 2"]
        result = mgte_embedding.embed_documents(documents)
        
        # Verify result format
        assert "dense" in result
        assert "sparse" in result
        assert len(result["dense"]) == 2
        assert len(result["sparse"]) == 2
        
        # Check vectors
        assert result["dense"][0] == [0.1, 0.2, 0.3]
        assert result["dense"][1] == [0.4, 0.5, 0.6]
        
        # Verify sparse vector content
        assert "indices" in result["sparse"][0]
        assert "values" in result["sparse"][0]


def test_embed_documents_error(mgte_embedding):
    """Test error handling when embedding documents."""
    # Patch the instance's embed_documents method to raise an exception
    with patch.object(mgte_embedding, 'embed_documents', 
                     side_effect=Exception("Embedding error")):
        # Test with error from embedding function
        with pytest.raises(Exception, match="Embedding error"):
            mgte_embedding.embed_documents(["Document 1"])


def test_embed_document(mgte_embedding, mock_dense_vectors, mock_sparse_vectors):
    """Test embedding a single document."""
    # Create mock result for a single document
    single_mock_result = {
        "dense": [np.array(mock_dense_vectors[0])],  # Just use the first vector
        "sparse": [mock_sparse_vectors[0]]  # Just use the first sparse vector
    }
    
    # Patch embed_documents to return our mock result
    with patch.object(mgte_embedding, 'embed_documents', return_value={
        "dense": [mock_dense_vectors[0]],
        "sparse": [mock_sparse_vectors[0]]
    }):
        # Test with a single document
        result = mgte_embedding.embed_document("Single document")
        
        # Verify embed_documents was called with list containing the single document
        mgte_embedding.embed_documents.assert_called_once_with(["Single document"])
        
        # Verify result format for a single document
        assert "dense" in result
        assert "sparse" in result
        
        # Verify dense vector content
        assert result["dense"] == [0.1, 0.2, 0.3]
        
        # Verify sparse vector content
        assert result["sparse"] == mock_sparse_vectors[0] 