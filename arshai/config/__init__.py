"""
Configuration management for Arshai.

This module provides optional configuration utilities for developers who want to use
configuration files. All configuration is optional - components work without config files.
"""

from .config_manager import ConfigManager
from .config_loader import (
    load_config,
    ConfigLoader,
    EXAMPLE_CONFIG
)

# Settings pattern has been removed - use direct instantiation instead

__all__ = [
    # Configuration utilities
    "load_config",
    "ConfigLoader",
    "EXAMPLE_CONFIG",
    
    # Legacy (deprecated)
    "ConfigManager",
] 