"""
Configuration loader utilities for Arshai framework.

Provides optional configuration loading without imposing patterns on developers.
Configuration files are optional - functions return empty dict if file doesn't exist.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict containing configuration data, or empty dict if file doesn't exist
        
    Raises:
        ImportError: If PyYAML is not installed
        ValueError: If file exists but contains invalid YAML
        
    Example:
        >>> config = load_config("app.yaml")
        >>> llm_settings = config.get("llm", {})
        >>> model = llm_settings.get("model", "gpt-4o")
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for configuration loading. "
            "Install with: pip install PyYAML"
        )
    
    config_file = Path(config_path)
    
    # Return empty dict if file doesn't exist (not an error)
    if not config_file.exists():
        logger.debug(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            
        # Handle empty files
        if content is None:
            logger.debug(f"Configuration file is empty: {config_path}")
            return {}
            
        if not isinstance(content, dict):
            raise ValueError(f"Configuration file must contain a YAML object (dict), got {type(content)}")
            
        logger.debug(f"Loaded configuration from: {config_path}")
        return content
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading configuration file {config_path}: {e}")


class ConfigLoader:
    """
    Class-based configuration loader.
    
    Provides dot-notation access to configuration values with defaults.
    Configuration files are optional - loader works with empty config if file doesn't exist.
    
    Example:
        >>> loader = ConfigLoader("app.yaml")
        >>> model = loader.get("llm.model", "gpt-4o")
        >>> temperature = loader.get("llm.temperature", 0.7)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
            
        Note:
            If config file doesn't exist, loader will work with empty configuration.
            This is not considered an error - allows for environment-only configuration.
        """
        self.config_path = config_path
        self._config = load_config(config_path)
        
        if self._config:
            logger.info(f"ConfigLoader initialized with {len(self._config)} configuration sections")
        else:
            logger.debug("ConfigLoader initialized with empty configuration")
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., "llm.model")
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> loader = ConfigLoader("app.yaml")
            >>> model = loader.get("llm.model", "gpt-4o")
            >>> nested_value = loader.get("database.redis.host", "localhost")
        """
        return self._get_nested_value(self._config, key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (top-level key)
            
        Returns:
            Dictionary containing section data, or empty dict if section doesn't exist
            
        Example:
            >>> loader = ConfigLoader("app.yaml")
            >>> llm_config = loader.get_section("llm")
            >>> database_config = loader.get_section("database")
        """
        return self._config.get(section, {})
    
    def has_key(self, key: str) -> bool:
        """
        Check if configuration key exists.
        
        Args:
            key: Configuration key using dot notation
            
        Returns:
            True if key exists, False otherwise
            
        Example:
            >>> loader = ConfigLoader("app.yaml")
            >>> if loader.has_key("llm.model"):
            ...     model = loader.get("llm.model")
        """
        try:
            self._get_nested_value(self._config, key, None)
            return True
        except KeyError:
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration data.
        
        Returns:
            Complete configuration dictionary
            
        Example:
            >>> loader = ConfigLoader("app.yaml")
            >>> all_config = loader.get_all()
            >>> print(f"Loaded sections: {list(all_config.keys())}")
        """
        return self._config.copy()
    
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any) -> Any:
        """
        Get nested value using dot notation.
        
        Args:
            data: Dictionary to search
            key: Dot-separated key path
            default: Default value if key not found
            
        Returns:
            Nested value or default
        """
        keys = key.split('.')
        current = data
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default



# Example configuration structure for documentation
EXAMPLE_CONFIG = """
# Example YAML configuration structure
# Save as app.yaml and use with load_config() or ConfigLoader()
# Developers decide what to put in config files

debug: false
log_level: "INFO"
environment: "production"
app_name: "my-ai-app"

# Configuration is completely optional and developer-controlled
# Use for application-level settings, not component configuration
"""