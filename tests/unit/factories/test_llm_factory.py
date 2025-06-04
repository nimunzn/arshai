"""Unit tests for the LLM factory."""

import os
import pytest
from unittest.mock import patch, MagicMock

from arshai.core.interfaces import ILLM, ILLMConfig
from arshai.factories.llm_factory import LLMFactory
from arshai.llms.openai import OpenAIClient
from arshai.llms.azure import AzureClient


class TestLLMFactory:
    """Tests for the LLMFactory class."""
    
    @pytest.fixture
    def llm_config(self):
        """Create a basic LLM configuration for testing."""
        return ILLMConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-api-key'})
    def test_create_openai_llm(self, llm_config):
        """Test creating an OpenAI LLM client."""
        # Setup mocks
        with patch.object(OpenAIClient, '__init__', return_value=None) as mock_init, \
             patch.object(OpenAIClient, '_initialize_client', return_value=MagicMock()):
            
            # Create LLM
            llm = LLMFactory.create("openai", llm_config)
            
            # Verify it's the right type
            assert isinstance(llm, OpenAIClient)
            
            # Verify initialization was called with config
            mock_init.assert_called_once_with(llm_config)
    
    @patch.dict('os.environ', {
        'AZURE_DEPLOYMENT': 'gpt-4o',
        'AZURE_API_VERSION': '2024-08-01-preview',
        'AZURE_OPENAI_API_KEY': 'fake-azure-api-key-for-testing',
        'AZURE_OPENAI_ENDPOINT': 'https://arshai-openai.openai.azure.com/'
    })
    def test_create_azure_llm(self, llm_config):
        """Test creating an Azure LLM client."""
        # Setup mocks
        with patch.object(ILLM, '__init__', return_value=None), \
             patch.object(AzureClient, '_initialize_client', return_value=MagicMock()):
            
            # Create LLM
            llm = LLMFactory.create("azure", llm_config)
            
            # Verify the Azure-specific attributes
            assert llm.azure_deployment == "gpt-4o"
            assert llm.api_version == "2024-08-01-preview"
    
    def test_create_unknown_provider(self, llm_config):
        """Test that creating an unknown provider raises an error."""
        with pytest.raises(ValueError) as excinfo:
            LLMFactory.create("unknown_provider", llm_config)
        
        assert "Unsupported provider" in str(excinfo.value)
    
    def test_create_missing_required_params(self, llm_config):
        """Test that missing required parameters raises an error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError) as excinfo:
                # Azure requires AZURE_DEPLOYMENT environment variable
                LLMFactory.create("azure", llm_config)
            
            assert "Azure deployment is required" in str(excinfo.value) 