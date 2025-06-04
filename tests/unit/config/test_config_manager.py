"""Unit tests for the ConfigManager class."""

import os
import tempfile
import pytest
from unittest.mock import patch

from arshai.config.config_manager import ConfigManager


class TestConfigManager:
    """Tests for the ConfigManager class."""
    
    def test_init_defaults(self):
        """Test that the ConfigManager initializes with defaults."""
        config_manager = ConfigManager()
        config = config_manager.get_all()
        assert isinstance(config, dict)
        assert len(config) > 0
        assert "llm" in config
        assert "memory" in config
    
    def test_load_from_file_yaml(self):
        """Test loading configuration from a YAML file."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as temp_file:
            temp_file.write("""
            llm:
              provider: yaml_provider
              api_key: yaml_key
            """)
            temp_path = temp_file.name
        
        try:
            config_manager = ConfigManager(config_path=temp_path)
            assert config_manager.get("llm.provider") == "yaml_provider"
            assert config_manager.get("llm.api_key") == "yaml_key"
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def test_load_from_file_json(self):
        """Test loading configuration from a JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as temp_file:
            temp_file.write("""
            {
                "llm": {
                    "provider": "json_provider",
                    "api_key": "json_key"
                }
            }
            """)
            temp_path = temp_file.name
        
        try:
            config_manager = ConfigManager(config_path=temp_path)
            assert config_manager.get("llm.provider") == "json_provider"
            assert config_manager.get("llm.api_key") == "json_key"
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def test_get_with_default(self):
        """Test getting a value with a default fallback."""
        config_manager = ConfigManager()
        
        # Set a test value
        config_manager.set("a.b", "c")
        
        # Existing path
        assert config_manager.get("a.b") == "c"
        
        # Non-existing path with default
        assert config_manager.get("x.y", default="default_value") == "default_value"
        
        # Non-existing path without default
        assert config_manager.get("x.y") is None
    
    def test_set_value(self):
        """Test setting a configuration value."""
        config_manager = ConfigManager()
        
        # Set a new value
        config_manager.set("test.path", "test_value")
        assert config_manager.get("test.path") == "test_value"
        
        # Override an existing value
        config_manager.set("test.path", "new_value")
        assert config_manager.get("test.path") == "new_value"
    
    def test_update_config(self):
        """Test merging multiple configurations."""
        config_manager = ConfigManager()
        
        # Set initial config values
        config_manager.set("a", 1)
        config_manager.set("b.c", 2)
        config_manager.set("b.d", 3)
        
        # Create new config to merge
        override_config = {"b": {"c": 4, "e": 5}, "f": 6}
        
        # Update config
        config_manager._update_config(config_manager._config, override_config)
        
        # Check updated values
        assert config_manager.get("a") == 1  # Unchanged
        assert config_manager.get("b.c") == 4  # Overridden
        assert config_manager.get("b.d") == 3  # Unchanged
        assert config_manager.get("b.e") == 5  # Added
        assert config_manager.get("f") == 6  # Added
    
    def test_get_all(self):
        """Test getting the entire configuration."""
        config_manager = ConfigManager()
        
        # Set some values
        config_manager.set("a", 1)
        config_manager.set("b.c", 2)
        config_manager.set("b.d", 3)
        
        # Get full config
        config = config_manager.get_all()
        
        # Check values
        assert "a" in config
        assert config["a"] == 1
        assert "b" in config
        assert "c" in config["b"]
        assert config["b"]["c"] == 2
        assert "d" in config["b"]
        assert config["b"]["d"] == 3 