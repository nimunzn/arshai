<<<<<<< HEAD
"""Configuration for Arshai observability system."""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import os
from pydantic import BaseModel, Field
from arshai.core.interfaces.idto import IDTO

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ObservabilityConfig(IDTO):
    """Configuration for LLM observability features."""
    
    # Basic observability controls (always enabled)
    trace_requests: bool = Field(default=True, description="Enable request tracing")
    collect_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    # OpenTelemetry configuration
    service_name: str = Field(default="arshai-llm", description="Service name for traces")
    service_version: str = Field(default="1.0.0", description="Service version")
    environment: str = Field(default="production", description="Environment name")
    
    # Tracing configuration
    trace_sampling_rate: float = Field(default=1.0, description="Trace sampling rate (0.0-1.0)")
    max_span_attributes: int = Field(default=128, description="Maximum span attributes")
    
    # Metrics configuration
    metric_export_interval: int = Field(default=60, description="Metric export interval in seconds")
    histogram_boundaries: list = Field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
        description="Histogram bucket boundaries for timing metrics"
    )
    
    # Token timing configuration - KEY METRICS  
    track_token_timing: bool = Field(default=True, description="Enable token-level timing metrics")
    stream_chunk_timeout: float = Field(default=30.0, description="Timeout for streaming chunks in seconds")
    
    # Privacy and security
    log_prompts: bool = Field(default=False, description="Log user prompts (privacy sensitive)")
    log_responses: bool = Field(default=False, description="Log LLM responses (privacy sensitive)")
    max_prompt_length: int = Field(default=1000, description="Maximum prompt length to log")
    max_response_length: int = Field(default=1000, description="Maximum response length to log")
    
    # Provider-specific configurations
    provider_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Provider-specific configurations"
    )
    
    # Additional attributes
    custom_attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom attributes for spans")
    
    # OTLP Exporter configuration (optional)
    otlp_endpoint: Optional[str] = Field(default=None, description="OTLP endpoint URL")
    otlp_headers: Dict[str, str] = Field(default_factory=dict, description="OTLP headers")
    otlp_timeout: int = Field(default=10, description="OTLP timeout in seconds")
    
    # Arize-specific configuration (optional)
    arize_space_id: Optional[str] = Field(default=None, description="Arize space ID")
    arize_api_key: Optional[str] = Field(default=None, description="Arize API key")
    arize_project_name: Optional[str] = Field(default=None, description="Arize project name")
    
    # Non-intrusive mode
    non_intrusive: bool = Field(default=True, description="Enable non-intrusive observability mode")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ObservabilityConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ObservabilityConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is not available. Install with: pip install pyyaml")
            
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract observability section if it exists
        observability_config = config_data.get('observability', {})
        
        # Also check for llm.observability section
        if 'llm' in config_data and 'observability' in config_data['llm']:
            observability_config.update(config_data['llm']['observability'])
        
        return cls(**observability_config)
    
    @classmethod
    def from_environment(cls) -> "ObservabilityConfig":
        """Create configuration from environment variables.
        
        Returns:
            ObservabilityConfig instance
        """
        config_dict = {}
        
        # Basic configuration (observability always enabled)
        
        if "ARSHAI_TRACE_REQUESTS" in os.environ:
            config_dict["trace_requests"] = os.environ.get("ARSHAI_TRACE_REQUESTS", "true").lower() == "true"
        
        if "ARSHAI_COLLECT_METRICS" in os.environ:
            config_dict["collect_metrics"] = os.environ.get("ARSHAI_COLLECT_METRICS", "true").lower() == "true"
        
        # Service configuration
        config_dict["service_name"] = os.environ.get("OTEL_SERVICE_NAME", os.environ.get("ARSHAI_SERVICE_NAME", "arshai-llm"))
        config_dict["service_version"] = os.environ.get("OTEL_SERVICE_VERSION", os.environ.get("ARSHAI_SERVICE_VERSION", "1.0.0"))
        config_dict["environment"] = os.environ.get("DEPLOYMENT_ENVIRONMENT", os.environ.get("ARSHAI_ENVIRONMENT", "production"))
        
        # Tracing configuration
        if "OTEL_TRACE_SAMPLING_RATE" in os.environ:
            config_dict["trace_sampling_rate"] = float(os.environ.get("OTEL_TRACE_SAMPLING_RATE", "1.0"))
        
        # OTLP configuration
        config_dict["otlp_endpoint"] = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        
        if "OTEL_EXPORTER_OTLP_HEADERS" in os.environ:
            headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
            headers = {}
            for header in headers_str.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()
            config_dict["otlp_headers"] = headers
        
        # Privacy configuration
        if "ARSHAI_LOG_PROMPTS" in os.environ:
            config_dict["log_prompts"] = os.environ.get("ARSHAI_LOG_PROMPTS", "false").lower() == "true"
        
        if "ARSHAI_LOG_RESPONSES" in os.environ:
            config_dict["log_responses"] = os.environ.get("ARSHAI_LOG_RESPONSES", "false").lower() == "true"
        
        # Token timing configuration
        if "ARSHAI_TRACK_TOKEN_TIMING" in os.environ:
            config_dict["track_token_timing"] = os.environ.get("ARSHAI_TRACK_TOKEN_TIMING", "true").lower() == "true"
        
        if "ARSHAI_ENABLE_TOKEN_COUNTING" in os.environ:
            config_dict["enable_token_counting"] = os.environ.get("ARSHAI_ENABLE_TOKEN_COUNTING", "true").lower() == "true"
        
        # Non-intrusive mode
        if "ARSHAI_NON_INTRUSIVE" in os.environ:
            config_dict["non_intrusive"] = os.environ.get("ARSHAI_NON_INTRUSIVE", "true").lower() == "true"
        
        # Arize configuration
        config_dict["arize_space_id"] = os.environ.get("ARIZE_SPACE_ID")
        config_dict["arize_api_key"] = os.environ.get("ARIZE_API_KEY") 
        config_dict["arize_project_name"] = os.environ.get("ARIZE_PROJECT_NAME")
        
        # Phoenix configuration removed - data flows through OTLP collector
        
        return cls(**config_dict)
    
    @classmethod
    def from_config_file_or_env(cls, config_path: Optional[Union[str, Path]] = None) -> "ObservabilityConfig":
        """Load configuration from file if provided, otherwise from environment.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            ObservabilityConfig instance
        """
        if config_path:
            return cls.from_yaml(config_path)
        
        # Try to find config.yaml in common locations
        possible_paths = [
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path("configs/config.yaml"),
            Path.cwd() / "config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return cls.from_yaml(path)
        
        # Fall back to environment variables
        return cls.from_environment()
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> "ObservabilityConfig":
        """Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            New ObservabilityConfig instance
        """
        current_dict = self.model_dump()
        current_dict.update(config_dict)
        return self.__class__(**current_dict)
    
    def is_token_timing_enabled(self, provider: str) -> bool:
        """Check if token timing is enabled for a specific provider.
        
        Args:
            provider: LLM provider name
            
        Returns:
            True if token timing is enabled for the provider
        """
        # Check global setting first
        if not self.track_token_timing:
            return False
        
        # Check provider-specific setting
        provider_config = self.provider_configs.get(provider, {})
        return provider_config.get("track_token_timing", True)
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider.
        
        Args:
            provider: LLM provider name
            
        Returns:
            Provider-specific configuration
        """
        return self.provider_configs.get(provider, {})
    
    def configure_for_arize(self, space_id: str, api_key: str, project_name: Optional[str] = None) -> "ObservabilityConfig":
        """Configure for Arize with automatic OTLP endpoint setup.
        
        Args:
            space_id: Arize space ID
            api_key: Arize API key
            project_name: Optional project name (defaults to service_name)
            
        Returns:
            New ObservabilityConfig configured for Arize
        """
        config_dict = self.model_dump()
        config_dict.update({
            "arize_space_id": space_id,
            "arize_api_key": api_key,
            "arize_project_name": project_name or self.service_name,
            "otlp_endpoint": "https://otlp.arize.com/v1/traces",
            "otlp_headers": {
                "space-id": space_id,
                "api-key": api_key
            }
        })
        return self.__class__(**config_dict)
    
    @classmethod
    def from_arize_env(cls) -> "ObservabilityConfig":
        """Create configuration specifically from Arize environment variables.
        
        Returns:
            ObservabilityConfig configured for Arize
        """
        space_id = os.environ.get("ARIZE_SPACE_ID")
        api_key = os.environ.get("ARIZE_API_KEY")
        project_name = os.environ.get("ARIZE_PROJECT_NAME")
        
        if not space_id or not api_key:
            raise ValueError("ARIZE_SPACE_ID and ARIZE_API_KEY environment variables are required")
        
        config = cls.from_environment()
        return config.configure_for_arize(space_id, api_key, project_name)

    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional configuration options
=======
"""LLM-friendly configuration for Arshai observability.

This module provides the main configuration class that replaces the old
ObservabilityConfig with OTEL anti-patterns.

The new PackageObservabilityConfig:
✅ Never creates or configures OTEL providers
✅ Uses package-specific environment variables  
✅ Respects parent application's OTEL setup
✅ Works with and without OTEL dependencies
✅ Provides granular feature control
"""

# Re-export the new configuration system
from .package_config import (
    PackageObservabilityConfig,
    ObservabilityLevel
)

# For backward compatibility, alias the new class
ObservabilityConfig = PackageObservabilityConfig

__all__ = [
    "PackageObservabilityConfig", 
    "ObservabilityLevel",
    "ObservabilityConfig",  # Backward compatibility alias
]
>>>>>>> e00d9333252e58887413664a96bf4e43dda19c0b
