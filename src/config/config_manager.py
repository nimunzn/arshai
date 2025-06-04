"""
Configuration manager for Arshai.

This module handles loading, validating, and accessing configuration settings.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.logging import get_logger, configure_logging

logger = get_logger(__name__)

class ConfigManager:
    """
    Configuration manager for Arshai.
    
    This class handles loading configuration from different sources
    (files and defaults) and provides access to configuration values.
    
    It focuses on structural configuration (which providers to use, what strategies to follow),
    while sensitive data and connection details should be read from environment variables
    directly in component implementations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a configuration file
        """
        self._config = {}
        self._load_defaults()
        
        if config_path:
            self.load_from_file(config_path)
        
        # Configure logging based on the loaded configuration
        self._configure_logging()
            
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "memory": {
                "working_memory": {
                    "provider": "in_memory",
                    "ttl": 43200  # 12 hours
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_dir": None
            }
        }
    
    def _configure_logging(self) -> None:
        """Configure logging based on the loaded configuration."""
        logging_config = self._config.get("logging", {})
        
        configure_logging(
            level=logging_config.get("level"),
            format_str=logging_config.get("format"),
            config_file=logging_config.get("config_file"),
            log_dir=logging_config.get("log_dir")
        )
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
        """
        path = Path(config_path)
        
        if not path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            if path.suffix.lower() in ['.yml', '.yaml']:
                with open(path, 'r') as file:
                    config = yaml.safe_load(file)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as file:
                    config = json.load(file)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return
            
            # Update configuration, keeping defaults for missing values
            self._update_config(self._config, config)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _update_config(self, current: Dict[str, Any], new: Dict[str, Any]) -> None:
        """
        Recursively update configuration dictionary.
        
        Args:
            current: Current configuration dictionary
            new: New configuration dictionary to merge in
        """
        for key, value in new.items():
            if isinstance(value, dict) and key in current and isinstance(current[key], dict):
                # Recurse into nested dictionaries
                self._update_config(current[key], value)
            else:
                # Update or add value
                current[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path.
        
        Args:
            path: Dot-separated path to the configuration value (e.g., "llm.provider")
            default: Default value to return if the path is not found
            
        Returns:
            The configuration value at the specified path, or the default value
        """
        parts = path.split('.')
        current = self._config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.
        
        Returns:
            The entire configuration dictionary
        """
        return self._config.copy()
    
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value by path.
        
        Args:
            path: Dot-separated path to the configuration value (e.g., "llm.provider")
            value: Value to set
        """
        parts = path.split('.')
        current = self._config
        
        # Navigate to the parent of the target key
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value 