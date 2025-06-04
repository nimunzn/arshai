"""
Configuration and Logging in Arshai

This example demonstrates:
1. How to configure the framework using different methods
2. How to customize and use the logging system
3. How to load configuration from different sources
4. Best practices for configuration management
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional
import yaml
import json

from arshai.config import settings, ConfigManager
from arshai.utils import get_logger
from arshai.utils.logging import configure_logging

# Get a logger for this module
logger = get_logger(__name__)


# =========================================================================
# PART 1: Basic Configuration
# =========================================================================

def demonstrate_basic_configuration():
    """Demonstrate basic configuration methods."""
    print("\n=== Basic Configuration ===")
    
    # Get default settings
    print("Loading default settings...")
    default_settings = settings.Settings()
    
    # Display some default values
    print(f"Default LLM provider: {default_settings.get_value('llm.provider', 'not set')}")
    print(f"Default LLM model: {default_settings.get_value('llm.model', 'not set')}")
    print(f"Default memory provider: {default_settings.get_value('memory.working_memory.provider', 'not set')}")
    
    # Create settings with a config file
    print("\nLoading settings from file...")
    # Note: This would normally load from an actual file
    # Here we'll just show how it would be done
    print("Example: settings = Settings('config.yaml')")


def create_sample_config():
    """Create a sample configuration file for demonstration."""
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key": "${OPENAI_API_KEY}"  # Environment variable reference
        },
        "memory": {
            "working_memory": {
                "provider": "redis",
                "url": "redis://localhost:6379/0",
                "ttl": 43200  # 12 hours
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "arshai.log"
        }
    }
    
    # Write to YAML file
    with open("sample_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Write to JSON file
    with open("sample_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created sample configuration files:")
    print("  - sample_config.yaml")
    print("  - sample_config.json")
    
    return config


# =========================================================================
# PART 2: Environment Variables
# =========================================================================

def demonstrate_environment_variables():
    """Demonstrate using environment variables for configuration."""
    print("\n=== Environment Variables ===")
    
    # Set environment variables (for demonstration)
    os.environ["OPENAI_API_KEY"] = "demo-api-key-not-real"
    os.environ["ARSHAI_LLM_PROVIDER"] = "openai"
    os.environ["ARSHAI_LLM_MODEL"] = "gpt-4-turbo"
    
    print("Set environment variables:")
    print("  OPENAI_API_KEY=demo-api-key-not-real")
    print("  ARSHAI_LLM_PROVIDER=openai")
    print("  ARSHAI_LLM_MODEL=gpt-4-turbo")
    
    # Get settings with environment variables
    # In a real application, the Settings class would automatically
    # read these environment variables and override any default values
    print("\nEnvironment variables will override configuration file values")
    print("Example priority order:")
    print("1. Environment variables")
    print("2. Configuration file")
    print("3. Default values")


def demonstrate_settings_pattern():
    """Demonstrate recommended settings pattern."""
    print("\n=== Recommended Settings Pattern ===")
    
    print("Creating application-specific settings:")
    code = """
class ApplicationSettings(Settings):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        
        # Initialize application-specific resources
        self.db_client = self._create_db_client()
        self.api_client = self._create_api_client()
    
    def _create_db_client(self):
        # Get database configuration
        db_config = self.get_value("database", {})
        # Create and return database client
        return DatabaseClient(db_config)
    
    def _create_api_client(self):
        # Get API configuration
        api_config = self.get_value("api", {})
        # Create and return API client
        return ApiClient(api_config)
    
    def get_db_client(self):
        return self.db_client
    
    def get_api_client(self):
        return self.api_client
    """
    print(code)
    
    print("\nUsing application settings:")
    usage = """
# Create application settings
settings = ApplicationSettings('config.yaml')

# Get application-specific resources
db_client = settings.get_db_client()
api_client = settings.get_api_client()

# Get standard Arshai resources
llm = settings.create_llm()
memory_manager = settings.create_memory_manager()
    """
    print(usage)


# =========================================================================
# PART 3: Logging Configuration
# =========================================================================

def configure_basic_logging():
    """Configure basic logging."""
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    logger = logging.getLogger()
    
    return logger


def demonstrate_basic_logging():
    """Demonstrate basic logging usage."""
    print("\n=== Basic Logging ===")
    
    # Configure basic logging
    logger = configure_basic_logging()
    
    # Log some messages
    print("Logging messages:")
    logger.debug("This is a DEBUG message (not shown with INFO level)")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Get a module-specific logger
    module_logger = logging.getLogger("my_module")
    module_logger.info("This is a module-specific log message")


def demonstrate_logging_with_context():
    """Demonstrate logging with context information."""
    print("\n=== Logging With Context ===")
    
    user_id = "user-123"
    conversation_id = "conv-456"
    operation = "memory_retrieval"
    
    # Log with contextual information
    logger.info(f"Starting operation {operation} for user {user_id}")
    
    try:
        # Simulate an operation
        result = {"status": "success", "data": {"memory_found": True}}
        logger.info(f"Operation {operation} completed successfully for conversation {conversation_id}")
        
        # Log detailed information at debug level
        logger.debug(f"Operation result: {json.dumps(result)}")
        
    except Exception as e:
        # Log exceptions with full details
        logger.error(f"Error during {operation} for user {user_id}: {str(e)}")
        logger.exception("Exception details")


def demonstrate_configure_logging():
    """Demonstrate configuring logging programmatically."""
    print("\n=== Configure Logging Example ===")
    
    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as log_dir:
        # Configure logging with a file handler
        configure_logging(
            level="DEBUG",
            format_str="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            log_dir=log_dir
        )
        
        logger.info(f"Logging configured with log directory: {log_dir}")
        logger.debug("This debug message should appear in both console and file")
        
        # Check that the log file was created
        log_file = os.path.join(log_dir, "arshai.log")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            print(f"Log file content preview: {log_content[:100]}...")


def demonstrate_config_based_logging():
    """Demonstrate configuring logging through ConfigManager."""
    print("\n=== Config-Based Logging Example ===")
    
    # Create a configuration with custom logging settings
    config = {
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            "log_dir": tempfile.gettempdir()
        }
    }
    
    # Write config to a temporary file
    config_file = os.path.join(tempfile.gettempdir(), "arshai_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    # Initialize ConfigManager with the config file
    # This will automatically configure logging based on the settings
    config_manager = ConfigManager(config_file)
    
    logger.debug("This debug message should appear because level is set to DEBUG")
    logger.info("Logging configured through ConfigManager")
    
    # Clean up
    os.remove(config_file)


def configure_advanced_logging():
    """Configure advanced logging with handlers."""
    print("\n=== Advanced Logging Configuration ===")
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Create file handler
    file_handler = logging.FileHandler("arshai_example.log")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    print("Configured advanced logging with:")
    print("- Console handler (INFO level)")
    print("- File handler (DEBUG level) -> arshai_example.log")
    
    return logger


def demonstrate_component_loggers():
    """Demonstrate using loggers for different components."""
    print("\n=== Component-Specific Loggers ===")
    
    # Create component loggers
    agent_logger = logging.getLogger("arshai.agents")
    llm_logger = logging.getLogger("arshai.llms")
    workflow_logger = logging.getLogger("arshai.workflows")
    
    # Log messages
    agent_logger.info("Agent system initialized")
    llm_logger.info("LLM client connected")
    workflow_logger.info("Workflow orchestrator ready")
    
    # These will inherit the configuration from the root logger
    print("Component loggers inherit configuration from the root logger")
    print("This allows for centralized control over logging levels and outputs")


def demonstrate_arshai_logger_util():
    """Demonstrate using the Arshai logger utility."""
    print("\n=== Arshai Logger Utility ===")
    
    # Get loggers using Arshai utility function
    logger = get_logger(__name__)
    component_logger = get_logger("component")
    
    # Log messages
    logger.info("Main logger message")
    component_logger.info("Component logger message")
    
    print("The get_logger utility provides consistent logger configuration")
    print("It ensures all loggers follow the same pattern and formatting")


# =========================================================================
# PART 4: Configuration Best Practices
# =========================================================================

def demonstrate_config_best_practices():
    """Demonstrate configuration best practices."""
    print("\n=== Configuration Best Practices ===")
    
    print("1. Use a layered approach to configuration:")
    print("   - Default values in code")
    print("   - Configuration files for environment-specific settings")
    print("   - Environment variables for sensitive information")
    
    print("\n2. Separate configuration concerns:")
    print("   - Infrastructure settings (database, API endpoints)")
    print("   - Application settings (LLM models, memory providers)")
    print("   - Runtime settings (logging, debugging)")
    
    print("\n3. Use centralized configuration access:")
    print("   - Create a single Settings instance")
    print("   - Pass it to components that need configuration")
    print("   - Use getters for specific configuration sections")
    
    print("\n4. Handle sensitive information properly:")
    print("   - Never hardcode API keys or credentials")
    print("   - Use environment variables for sensitive values")
    print("   - Support encryption for stored credentials")
    
    print("\n5. Validate configuration at startup:")
    print("   - Check that required settings are present")
    print("   - Validate configuration values before using them")
    print("   - Provide clear error messages for missing or invalid configuration")


# =========================================================================
# Main Example Runner
# =========================================================================

def main():
    """Run all configuration and logging examples."""
    try:
        # Configuration examples
        demonstrate_basic_configuration()
        sample_config = create_sample_config()
        demonstrate_environment_variables()
        demonstrate_settings_pattern()
        
        # Basic logging examples
        demonstrate_basic_logging()
        demonstrate_logging_with_context()
        
        # Advanced logging examples
        demonstrate_configure_logging()
        demonstrate_config_based_logging()
        demonstrate_component_loggers()
        demonstrate_arshai_logger_util()
        
        # Configuration best practices
        demonstrate_config_best_practices()
        
        print("\nAll examples completed successfully!")
        
        # Clean up sample files
        if os.path.exists("sample_config.yaml"):
            os.remove("sample_config.yaml")
        if os.path.exists("sample_config.json"):
            os.remove("sample_config.json")
            
    except Exception as e:
        print(f"Error running examples: {str(e)}")


if __name__ == "__main__":
    main() 