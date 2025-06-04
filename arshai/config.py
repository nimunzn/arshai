"""Main configuration module for Arshai framework."""

from typing import Optional, Dict, Any
import os
import yaml
from pathlib import Path

# Import the existing Settings implementation
from arshai.config.settings import Settings as BaseSettings

class Settings(BaseSettings):
    """
    Enhanced Settings class with additional convenience methods for public package use.
    
    This class extends the base Settings to provide a cleaner public API.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize Settings with optional configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            **kwargs: Additional configuration parameters to override
        """
        # Load configuration from file if provided
        config = {}
        if config_path:
            config = self._load_config_file(config_path)
        
        # Merge with kwargs
        config.update(kwargs)
        
        # Initialize base settings
        super().__init__(**config)
    
    @staticmethod
    def _load_config_file(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create Settings instance from environment variables.
        
        Looks for ARSHAI_CONFIG_PATH environment variable for config file.
        """
        config_path = os.getenv("ARSHAI_CONFIG_PATH")
        return cls(config_path=config_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export current settings as dictionary."""
        return {
            "llm": self.llm_config.model_dump() if hasattr(self, 'llm_config') else {},
            "memory": {
                "working_memory": self.working_memory_config.model_dump() 
                if hasattr(self, 'working_memory_config') else {}
            },
            "embedding": self.embedding_config.model_dump() 
            if hasattr(self, 'embedding_config') else {},
            "vector_db": self.vector_db_config.model_dump()
            if hasattr(self, 'vector_db_config') else {},
        }


# Re-export key configuration classes
from arshai.config.config_manager import ConfigManager

__all__ = ["Settings", "ConfigManager"]