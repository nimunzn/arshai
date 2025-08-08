"""
Base observability integration tests.

Tests the core observability framework functionality that's shared across all LLM clients.
"""

import pytest
from arshai.core.interfaces.illm import ILLMConfig
from arshai.observability import (
    ObservabilityConfig,
    ObservabilityManager
)


def test_observability_config_creation():
    """Test that observability configuration can be created."""
    config = ObservabilityConfig(
        service_name="test-service",
        track_token_timing=True,
        collect_metrics=True,
        log_prompts=False
    )
    
    assert config.service_name == "test-service"
    assert config.track_token_timing is True
    assert config.collect_metrics is True
    assert config.log_prompts is False


def test_observability_manager_creation():
    """Test that observability manager can be created."""
    obs_config = ObservabilityConfig(service_name="test-service")
    obs_manager = ObservabilityManager(obs_config)
    
    assert obs_manager is not None
    assert isinstance(obs_manager, ObservabilityManager)


def test_direct_observability_manager_creation():
    """Test direct observability manager creation (no helper needed)."""
    obs_config = ObservabilityConfig(service_name="test-service")
    obs_manager = ObservabilityManager(obs_config)
    
    assert obs_manager is not None
    assert isinstance(obs_manager, ObservabilityManager)


def test_observability_manager_with_custom_config():
    """Test observability manager with custom configuration."""
    obs_config = ObservabilityConfig(
        service_name="custom-service",
        track_token_timing=True,
        collect_metrics=True
    )
    obs_manager = ObservabilityManager(obs_config)
    
    assert obs_manager is not None
    assert isinstance(obs_manager, ObservabilityManager)


def test_llm_config_creation():
    """Test that LLM config can be created for testing."""
    config = ILLMConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100
    )
    
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.7
    assert config.max_tokens == 100