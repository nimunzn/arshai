"""
Logging configuration for Arshai.

This module provides utilities for configuring and accessing loggers throughout the
application with a consistent format and behavior.
"""

import os
import logging
import logging.config
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default logging level
DEFAULT_LEVEL = "INFO"

# A dictionary of loggers to avoid creating multiple loggers for the same name
_loggers = {}

def configure_logging(
    level: str = None, 
    format_str: str = None, 
    config_file: str = None,
    log_dir: str = None
) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Format string for log messages
        config_file: Path to logging configuration file (json or dictConfig)
        log_dir: Directory to store log files (if None, only console logging is enabled)
    """
    # Get settings from environment if not provided
    if level is None:
        level = os.environ.get("ARSHAI_LOG_LEVEL", DEFAULT_LEVEL)
        
    if format_str is None:
        format_str = os.environ.get("ARSHAI_LOG_FORMAT", DEFAULT_FORMAT)
    
    # Convert level string to logging level constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure from file if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                logging.config.dictConfig(config)
                return
        except Exception as e:
            print(f"Error loading logging config from {config_file}: {str(e)}")
            print("Falling back to basic configuration")
    
    # Basic configuration
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)
    
    # File handler (if log_dir is provided)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path / "arshai.log")
        file_handler.setFormatter(logging.Formatter(format_str))
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure package logger
    arshai_logger = logging.getLogger("arshai")
    arshai_logger.setLevel(numeric_level)
    
    logging.info(f"Logging configured with level: {level}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    This function ensures that we don't create multiple loggers for the same name
    and that all loggers have consistent formatting.
    
    Args:
        name: Name of the logger (typically __name__ from the calling module)
        
    Returns:
        A configured logger
    """
    # If we haven't seen this logger before, store it
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
        
        # Add "arshai." prefix if not already there for consistent naming
        if not name.startswith("arshai.") and not name == "arshai":
            logger.name = f"arshai.{name}"
    
    return _loggers[name]

# Configure logging with defaults on module import
configure_logging() 