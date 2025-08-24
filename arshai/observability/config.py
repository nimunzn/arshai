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
