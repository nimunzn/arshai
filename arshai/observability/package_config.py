"""Package-specific observability configuration that respects parent settings.

This module provides configuration that:
1. Never overrides global OTEL configuration
2. Reads package-specific environment variables
3. Provides sensible defaults that work with any parent setup
4. Allows runtime enable/disable of features
5. Supports multiple verbosity levels
"""

import os
import logging
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

from arshai.core.interfaces.idto import IDTO


class ObservabilityLevel(str, Enum):
    """Package observability levels."""
    OFF = "OFF"          # No telemetry at all
    ERROR = "ERROR"      # Only error telemetry
    INFO = "INFO"        # Normal telemetry  
    DEBUG = "DEBUG"      # Verbose telemetry with debug info


class SpanKind(str, Enum):
    """Phoenix-compatible span kinds.
    
    Based on Phoenix documentation:
    https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/instrument-python
    """
    CHAIN = "CHAIN"  # General logic operations, functions, or code blocks
    LLM = "LLM"  # Making LLM calls
    TOOL = "TOOL"  # Completing tool calls
    RETRIEVER = "RETRIEVER"  # Retrieving documents
    EMBEDDING = "EMBEDDING"  # Generating embeddings
    AGENT = "AGENT"  # Agent invocations - typically a top level or near top level span
    RERANKER = "RERANKER"  # Reranking retrieved context
    GUARDRAIL = "GUARDRAIL"  # Guardrail checks
    EVALUATOR = "EVALUATOR"  # Evaluators - typically only use by Phoenix when automatically tracing evaluation and experiment calls


class PackageObservabilityConfig(IDTO):
    """LLM-friendly observability configuration for Arshai package.
    
    This configuration is designed to be a well-behaved OTEL citizen:
    - Never creates or configures OTEL providers/exporters
    - Respects parent application's OTEL setup
    - Uses package-specific environment variables
    - Provides runtime control over telemetry features
    """
    
    # Core enable/disable controls
    enabled: bool = Field(
        default=True, 
        description="Master switch for all package telemetry"
    )
    
    level: ObservabilityLevel = Field(
        default=ObservabilityLevel.INFO,
        description="Observability verbosity level"
    )
    
    # Feature-specific controls
    trace_llm_calls: bool = Field(
        default=True,
        description="Enable tracing of LLM calls"
    )
    
    trace_agent_operations: bool = Field(
        default=True,
        description="Enable tracing of agent operations"
    )
    
    trace_workflow_execution: bool = Field(
        default=True,
        description="Enable tracing of workflow execution"
    )
    
    collect_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    
    # OpenInference span kind for LLM operations
    span_kind: SpanKind = Field(
        default=SpanKind.LLM,
        description="Default span kind for LLM operations (Phoenix-compatible)"
    )
    
    # Token timing controls (key LLM metrics)
    track_token_timing: bool = Field(
        default=True,
        description="Track token-level timing metrics"
    )
    
    track_cost_metrics: bool = Field(
        default=False,
        description="Track cost-related metrics (may be sensitive)"
    )
    
    # Provider-specific controls
    provider_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider-specific observability settings"
    )
    
    # Privacy and security
    log_prompts: bool = Field(
        default=False,
        description="Log LLM prompts (privacy sensitive)"
    )
    
    log_responses: bool = Field(
        default=False,
        description="Log LLM responses (privacy sensitive)"
    )
    
    max_prompt_length: int = Field(
        default=500,
        description="Maximum prompt length to log (when enabled)"
    )
    
    max_response_length: int = Field(
        default=500,
        description="Maximum response length to log (when enabled)"
    )
    
    service_namespace: Optional[str] = Field(
        default=None,
        description="Service namespace for telemetry identification"
    )
    
    # Package identification (for proper OTEL resource attributes)
    package_name: str = Field(
        default="arshai",
        description="Package name for telemetry identification"
    )
    
    package_version: str = Field(
        default="1.2.3",
        description="Package version for telemetry identification"
    )
    
    # OTLP export configuration
    otlp_endpoint: Optional[str] = Field(
        default=None,
        description="OTLP collector endpoint for exporting telemetry data (e.g., 'http://phoenix:4317')"
    )
    
    # Custom attributes (added to all spans/metrics from this package)
    custom_attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom attributes for package telemetry"
    )
    
    @classmethod
    def from_environment(cls) -> "PackageObservabilityConfig":
        """Create configuration from package-specific environment variables.
        
        Uses ARSHAI_TELEMETRY_* environment variables to avoid conflicts
        with parent application's OTEL configuration.
        
        Returns:
            PackageObservabilityConfig instance
        """
        config_dict = {}
        
        # Master enable/disable
        if "ARSHAI_TELEMETRY_ENABLED" in os.environ:
            config_dict["enabled"] = os.environ.get("ARSHAI_TELEMETRY_ENABLED", "true").lower() == "true"
        
        # Observability level
        if "ARSHAI_TELEMETRY_LEVEL" in os.environ:
            level_str = os.environ.get("ARSHAI_TELEMETRY_LEVEL", "INFO").upper()
            try:
                config_dict["level"] = ObservabilityLevel(level_str)
            except ValueError:
                logging.warning(f"Invalid ARSHAI_TELEMETRY_LEVEL: {level_str}, using INFO")
                config_dict["level"] = ObservabilityLevel.INFO
        
        # Feature-specific controls
        if "ARSHAI_TRACE_LLM_CALLS" in os.environ:
            config_dict["trace_llm_calls"] = os.environ.get("ARSHAI_TRACE_LLM_CALLS", "true").lower() == "true"
        
        if "ARSHAI_TRACE_AGENT_OPERATIONS" in os.environ:
            config_dict["trace_agent_operations"] = os.environ.get("ARSHAI_TRACE_AGENT_OPERATIONS", "true").lower() == "true"
        
        if "ARSHAI_TRACE_WORKFLOW_EXECUTION" in os.environ:
            config_dict["trace_workflow_execution"] = os.environ.get("ARSHAI_TRACE_WORKFLOW_EXECUTION", "true").lower() == "true"
        
        if "ARSHAI_COLLECT_METRICS" in os.environ:
            config_dict["collect_metrics"] = os.environ.get("ARSHAI_COLLECT_METRICS", "true").lower() == "true"
        
        # Span kind configuration
        if "ARSHAI_SPAN_KIND" in os.environ:
            span_kind_str = os.environ.get("ARSHAI_SPAN_KIND", "LLM").upper()
            try:
                config_dict["span_kind"] = SpanKind(span_kind_str)
            except ValueError:
                logging.warning(f"Invalid ARSHAI_SPAN_KIND: {span_kind_str}, using LLM")
                config_dict["span_kind"] = SpanKind.LLM
        
        # Token timing controls
        if "ARSHAI_TRACK_TOKEN_TIMING" in os.environ:
            config_dict["track_token_timing"] = os.environ.get("ARSHAI_TRACK_TOKEN_TIMING", "true").lower() == "true"
        
        if "ARSHAI_TRACK_COST_METRICS" in os.environ:
            config_dict["track_cost_metrics"] = os.environ.get("ARSHAI_TRACK_COST_METRICS", "false").lower() == "true"
        
        # Privacy controls
        if "ARSHAI_LOG_PROMPTS" in os.environ:
            config_dict["log_prompts"] = os.environ.get("ARSHAI_LOG_PROMPTS", "false").lower() == "true"
        
        if "ARSHAI_LOG_RESPONSES" in os.environ:
            config_dict["log_responses"] = os.environ.get("ARSHAI_LOG_RESPONSES", "false").lower() == "true"
        
        if "ARSHAI_MAX_PROMPT_LENGTH" in os.environ:
            try:
                config_dict["max_prompt_length"] = int(os.environ.get("ARSHAI_MAX_PROMPT_LENGTH", "500"))
            except ValueError:
                logging.warning("Invalid ARSHAI_MAX_PROMPT_LENGTH, using default 500")
        
        if "ARSHAI_MAX_RESPONSE_LENGTH" in os.environ:
            try:
                config_dict["max_response_length"] = int(os.environ.get("ARSHAI_MAX_RESPONSE_LENGTH", "500"))
            except ValueError:
                logging.warning("Invalid ARSHAI_MAX_RESPONSE_LENGTH, using default 500")
        
        # Package identification
        if "ARSHAI_PACKAGE_NAME" in os.environ:
            config_dict["package_name"] = os.environ.get("ARSHAI_PACKAGE_NAME", "arshai")
        
        if "ARSHAI_PACKAGE_VERSION" in os.environ:
            config_dict["package_version"] = os.environ.get("ARSHAI_PACKAGE_VERSION", "1.2.3")
        
        # OTLP export configuration
        if "ARSHAI_OTLP_ENDPOINT" in os.environ:
            config_dict["otlp_endpoint"] = os.environ.get("ARSHAI_OTLP_ENDPOINT")
        
        # Custom attributes
        custom_attrs = {}
        for key, value in os.environ.items():
            if key.startswith("ARSHAI_ATTR_"):
                attr_name = key[13:].lower()  # Remove ARSHAI_ATTR_ prefix
                custom_attrs[attr_name] = value
        if custom_attrs:
            config_dict["custom_attributes"] = custom_attrs
        
        return cls(**config_dict)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific observability feature is enabled.
        
        Args:
            feature: Feature name (e.g., 'trace_llm_calls', 'collect_metrics')
        
        Returns:
            True if feature is enabled based on global and feature-specific settings
        """
        if not self.enabled:
            return False
        
        if self.level == ObservabilityLevel.OFF:
            return False
        
        # Check feature-specific setting
        feature_enabled = getattr(self, feature, True)
        
        # Apply level-based filtering
        if self.level == ObservabilityLevel.ERROR:
            # Only enable error-level features
            error_features = {"trace_llm_calls"}  # Minimal set for error tracking
            return feature in error_features and feature_enabled
        
        return feature_enabled
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if observability is enabled for a specific LLM provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'google')
        
        Returns:
            True if provider observability is enabled
        """
        if not self.enabled:
            return False
        
        provider_config = self.provider_configs.get(provider, {})
        return provider_config.get("enabled", True)
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get observability configuration for a specific provider.
        
        Args:
            provider: Provider name
        
        Returns:
            Provider-specific configuration dictionary
        """
        return self.provider_configs.get(provider, {})
    
    def should_log_content(self, content_type: str) -> bool:
        """Check if content logging is enabled for the given type.
        
        Args:
            content_type: 'prompts' or 'responses'
        
        Returns:
            True if content logging is enabled and safe
        """
        if not self.is_feature_enabled("log_prompts" if content_type == "prompts" else "log_responses"):
            return False
        
        # Additional safety check based on level
        if self.level in [ObservabilityLevel.OFF, ObservabilityLevel.ERROR]:
            return False
        
        return True
    
    def get_content_length_limit(self, content_type: str) -> int:
        """Get the length limit for content logging.
        
        Args:
            content_type: 'prompts' or 'responses'
        
        Returns:
            Maximum length to log
        """
        if content_type == "prompts":
            return self.max_prompt_length
        else:
            return self.max_response_length
    
    def get_span_attributes(self) -> Dict[str, str]:
        """Get default span attributes for this package.
        
        Returns:
            Dictionary of default span attributes
        """
        attributes = {
            "arshai.package.name": self.package_name,
            "arshai.package.version": self.package_version,
            "arshai.telemetry.level": self.level.value,
        }
        
        # Add custom attributes
        for key, value in self.custom_attributes.items():
            attributes[f"arshai.custom.{key}"] = value
        
        return attributes
    
    def configure_provider(
        self, 
        provider: str, 
        enabled: Optional[bool] = None,
        **kwargs
    ) -> "PackageObservabilityConfig":
        """Create a new config with provider-specific settings.
        
        Args:
            provider: Provider name
            enabled: Whether to enable observability for this provider
            **kwargs: Additional provider-specific settings
        
        Returns:
            New PackageObservabilityConfig instance
        """
        config_dict = self.model_dump()
        
        provider_config = config_dict.get("provider_configs", {}).copy()
        provider_settings = provider_config.get(provider, {}).copy()
        
        if enabled is not None:
            provider_settings["enabled"] = enabled
        
        provider_settings.update(kwargs)
        provider_config[provider] = provider_settings
        config_dict["provider_configs"] = provider_config
        
        return self.__class__(**config_dict)
    
    def with_level(self, level: ObservabilityLevel) -> "PackageObservabilityConfig":
        """Create a new config with different observability level.
        
        Args:
            level: New observability level
        
        Returns:
            New PackageObservabilityConfig instance
        """
        config_dict = self.model_dump()
        config_dict["level"] = level
        return self.__class__(**config_dict)
    
    def disable_all(self) -> "PackageObservabilityConfig":
        """Create a new config with all observability disabled.
        
        Returns:
            New PackageObservabilityConfig instance with everything disabled
        """
        config_dict = self.model_dump()
        config_dict["enabled"] = False
        return self.__class__(**config_dict)