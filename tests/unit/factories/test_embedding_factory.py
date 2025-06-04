"""Unit tests for the embedding factory."""

import pytest
from unittest.mock import MagicMock, patch

from arshai.core.interfaces import IEmbedding, EmbeddingConfig
from arshai.factories.embedding_factory import EmbeddingFactory


class MockEmbedding(IEmbedding):
    """Mock implementation of the IEmbedding interface for testing."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    async def get_embeddings(self, texts):
        return [
            [0.1, 0.2, 0.3]  # Mock embedding vector
        ]
    
    async def get_embedding(self, text):
        return [0.1, 0.2, 0.3]  # Mock embedding vector


def test_register_new_provider():
    """Test registering a new embedding provider."""
    # Save original providers to restore after test
    original_providers = EmbeddingFactory._providers.copy()
    
    try:
        # Register new provider
        EmbeddingFactory.register("mock_provider", MockEmbedding)
        
        # Verify it was added to the registry
        assert "mock_provider" in EmbeddingFactory._providers
        assert EmbeddingFactory._providers["mock_provider"] == MockEmbedding
    finally:
        # Restore original providers
        EmbeddingFactory._providers = original_providers


def test_create_openai_embedding():
    """Test creating an OpenAI embedding instance."""
    # Create a temporary providers dictionary for testing
    with patch.dict(EmbeddingFactory._providers, clear=True) as mock_providers:
        # Create a mock provider class
        mock_openai_class = MagicMock()
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance
        
        # Add our mock to the providers dictionary
        mock_providers["openai"] = mock_openai_class
        
        # Create an embedding instance
        config = {"model": "text-embedding-ada-002", "batch_size": 32}
        embedding = EmbeddingFactory.create("openai", config)
        
        # Verify the mock class was instantiated with correct config
        mock_openai_class.assert_called_once()
        config_arg = mock_openai_class.call_args[0][0]
        assert isinstance(config_arg, EmbeddingConfig)
        assert config_arg.model_name == "text-embedding-ada-002"
        assert config_arg.batch_size == 32
        assert config_arg.additional_params == config
        
        # Verify the factory returned the instance
        assert embedding == mock_instance


def test_create_mgte_embedding():
    """Test creating an MGTE embedding instance."""
    # Create a temporary providers dictionary for testing
    with patch.dict(EmbeddingFactory._providers, clear=True) as mock_providers:
        # Create a mock provider class
        mock_mgte_class = MagicMock()
        mock_instance = MagicMock()
        mock_mgte_class.return_value = mock_instance
        
        # Add our mock to the providers dictionary
        mock_providers["mgte"] = mock_mgte_class
        
        # Create an embedding instance
        config = {"model": "multilingual-e5-large", "batch_size": 16}
        embedding = EmbeddingFactory.create("mgte", config)
        
        # Verify the mock class was instantiated with correct config
        mock_mgte_class.assert_called_once()
        config_arg = mock_mgte_class.call_args[0][0]
        assert isinstance(config_arg, EmbeddingConfig)
        assert config_arg.model_name == "multilingual-e5-large"
        assert config_arg.batch_size == 16
        assert config_arg.additional_params == config
        
        # Verify the factory returned the instance
        assert embedding == mock_instance


def test_create_custom_provider():
    """Test creating a custom provider embedding instance."""
    # Save original providers to restore after test
    original_providers = EmbeddingFactory._providers.copy()
    
    try:
        # Register mock provider
        EmbeddingFactory.register("mock_provider", MockEmbedding)
        
        # Create an embedding instance
        config = {"model": "mock-model", "batch_size": 10}
        embedding = EmbeddingFactory.create("mock_provider", config)
        
        # Verify correct instance was created
        assert isinstance(embedding, MockEmbedding)
        assert embedding.config.model_name == "mock-model"
        assert embedding.config.batch_size == 10
        assert embedding.config.additional_params == config
    finally:
        # Restore original providers
        EmbeddingFactory._providers = original_providers


def test_create_with_unsupported_provider():
    """Test creating with an unsupported provider raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingFactory.create("nonexistent_provider", {})


def test_create_with_case_insensitive_provider():
    """Test provider name is case-insensitive."""
    # Create a temporary providers dictionary for testing
    with patch.dict(EmbeddingFactory._providers, clear=True) as mock_providers:
        # Create a mock provider class
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        
        # Add our mock to the providers dictionary (lowercase key)
        mock_providers["openai"] = mock_class
        
        # Create with uppercase provider name
        config = {"model": "text-embedding-ada-002"}
        embedding = EmbeddingFactory.create("OPENAI", config)
        
        # Verify the mock class was instantiated
        mock_class.assert_called_once()
        
        # Verify the factory returned the instance
        assert embedding == mock_instance 