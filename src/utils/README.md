# Utils Module

## Overview
The Utils module provides common utility functions and helpers used throughout the Arshai framework. It contains logging facilities, helper functions, and other shared utilities that support the core functionality of the system.

## Architecture
```
utils/
├── logging.py               # Logging configuration and utilities
└── __init__.py              # Module exports
```

## Implementation Guide

### Using the Logging Utilities

```python
from src.utils.logging import get_logger

# Create a logger for a specific component
logger = get_logger("my_component")

# Use the logger
logger.info("Component initialized successfully")
logger.debug("Processing data...")
logger.warning("Resource usage is high")
logger.error("Failed to connect to external service", exc_info=True)
```

### Creating Utility Functions

When adding new utility functions to the module, follow these principles:

1. Keep functions focused on a single responsibility
2. Use descriptive function names that clearly indicate the purpose
3. Include proper type hints and docstrings
4. Add appropriate error handling
5. Write unit tests for each utility function

Example of adding a new utility:

```python
def retry_with_backoff(func, max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for backoff
        
    Returns:
        The result of the function call
        
    Raises:
        Exception: The last exception raised by the function
    """
    import time
    
    retries = 0
    delay = initial_delay
    
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                raise e
                
            time.sleep(delay)
            delay *= backoff_factor
```

## Integration Points

### With Logging Configuration
The logging utilities can be integrated with application-wide configuration:

```python
from src.utils.logging import configure_logging
from src.config import Settings

settings = Settings()

# Configure logging based on settings
configure_logging(
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_file=settings.log_file
)
```

### With Other Modules
Utilities are designed to be used across the entire framework:

```python
from src.utils.logging import get_logger
from src.agents import ConversationAgent

class EnhancedAgent(ConversationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the logging utility
        self.logger = get_logger(f"agent.{self.__class__.__name__}")
        
    def process_message(self, message):
        self.logger.info(f"Processing message: {message[:50]}...")
        return super().process_message(message)
```

## Configuration
Configure logging through the settings system:

```python
# in config.yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/arshai.log"
  max_file_size: 10485760  # 10MB
  backup_count: 5
``` 