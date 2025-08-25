"""LLM-friendly telemetry abstraction layer for Arshai framework.

This module provides a well-behaved OTEL citizen implementation that:
1. Never creates TracerProvider/MeterProvider instances
2. Always uses get_tracer("arshai", version) pattern
3. Respects parent application's configuration
4. Works with and without OTEL dependencies
5. Provides no-op implementations when OTEL is unavailable
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager, asynccontextmanager
import time

# No-op implementations (always available for type hints)
class NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs):
        return NoOpSpan()

class NoOpSpan:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_attribute(self, key: str, value: Any):
        pass
    def set_status(self, status):
        pass
    def record_exception(self, exception):
        pass

class NoOpMeter:
    def create_counter(self, **kwargs):
        return NoOpInstrument()
    def create_histogram(self, **kwargs):
        return NoOpInstrument()
    def create_up_down_counter(self, **kwargs):
        return NoOpInstrument()

class NoOpInstrument:
    def add(self, value, attributes=None):
        pass
    def record(self, value, attributes=None):
        pass

# OTEL imports with proper fallback handling
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Tracer, Span, Status, StatusCode, SpanKind
    from opentelemetry.metrics import Meter
    OTEL_AVAILABLE = True
except ImportError:
    # Use no-op implementations when OTEL is not available
    OTEL_AVAILABLE = False
    
    # Create dummy types for compatibility
    Tracer = NoOpTracer
    Meter = NoOpMeter
    Span = NoOpSpan
    
    class SpanKind:
        CLIENT = "CLIENT"
    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"
    class Status:
        def __init__(self, code, description=""):
            self.code = code
            self.description = description

from .package_config import PackageObservabilityConfig


class TelemetryManager:
    """LLM-friendly telemetry manager that respects parent OTEL configuration."""
    
    def __init__(self, config: Optional[PackageObservabilityConfig] = None):
        """Initialize telemetry manager.
        
        Args:
            config: Optional observability configuration
        """
        self.config = config or PackageObservabilityConfig()
        self.logger = logging.getLogger(__name__)

        
        # Package information for proper OTEL registration
        # Use config values if provided, otherwise fall back to defaults
        self._package_name = self.config.package_name
        self._package_version = self.config.package_version
        
        # Telemetry components
        self._tracer: Optional[Union[Tracer, NoOpTracer]] = None
        self._meter: Optional[Union[Meter, NoOpMeter]] = None
        
        # Decision logic for self-managed vs parent application mode
        self._should_setup_own_otel = self._should_setup_own_otel()
        
        if self._should_setup_own_otel:
            # Self-managed mode: Set up our own OTEL configuration
            if self._try_setup_own_otel():
                self._is_enabled = True
                self._initialize_telemetry()
            else:
                self._is_enabled = False
                self._initialize_noop_telemetry()
        else:
            # Parent application mode: Use existing OTEL setup
            self._is_enabled = self._detect_otel_availability()
            if self._is_enabled:
                self._initialize_telemetry()
            else:
                self._initialize_noop_telemetry()
    
    def _detect_otel_availability(self) -> bool:
        """Detect if OTEL is available and properly configured."""
        if not OTEL_AVAILABLE:
            self.logger.info("OpenTelemetry not available - using no-op implementations")
            return False
        
        # Check if parent application has configured OTEL
        try:
            tracer_provider = trace.get_tracer_provider()
            meter_provider = metrics.get_meter_provider()
            
            # Check if we have real providers (not NoOp)
            tracer_available = hasattr(tracer_provider, 'get_tracer')
            meter_available = hasattr(meter_provider, 'get_meter')
            
            # More specific check - see if it's actually a real provider, not just NoOp
            is_real_tracer = (tracer_available and 
                            not str(type(tracer_provider)).endswith("NoOpTracerProvider'>") and
                            str(type(tracer_provider)) != "<class 'opentelemetry.trace.NoOpTracerProvider'>")
            
            is_real_meter = (meter_available and 
                           not str(type(meter_provider)).endswith("NoOpMeterProvider'>") and
                           str(type(meter_provider)) != "<class 'opentelemetry.metrics.NoOpMeterProvider'>")
            
            if is_real_tracer or is_real_meter:
                self.logger.info("Using parent application's OTEL configuration")
                return True
            else:
                self.logger.info("No parent OTEL configuration found - will try to set up own OTEL")
                return False
                
        except Exception as e:
            self.logger.warning(f"Error detecting OTEL configuration: {e}")
            return False
    
    def _should_setup_own_otel(self) -> bool:
        """Determine if we should set up our own OTEL configuration.
        
        Returns True for self-managed mode when:
        1. An explicit OTLP endpoint is provided in the configuration
        2. Environment variable ARSHAI_OTLP_ENDPOINT is set
        
        Returns False for parent application mode otherwise.
        """
        # Check if explicit OTLP endpoint is configured (self-managed mode)
        has_explicit_endpoint = (
            (self.config.otlp_endpoint is not None) or
            (os.getenv("ARSHAI_OTLP_ENDPOINT") is not None)
        )
        
        if has_explicit_endpoint:
            self.logger.info("Self-managed mode: Explicit OTLP endpoint configured")
            return True
        else:
            self.logger.info("Parent application mode: No explicit endpoint, will use parent OTEL setup")
            return False
    
    def _try_setup_own_otel(self) -> bool:
        """Try to set up our own OTEL configuration for Docker/Phoenix monitoring.
        
        This is called when no parent OTEL configuration is found.
        Sets up basic OTLP export to common monitoring endpoints.
        """
        if not OTEL_AVAILABLE:
            return False
        
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            
            # Create resource with service information
            resource_attrs = {
                "service.name": f"{self._package_name}",
                "service.version": self._package_version,
            }
            
            # Add namespace only if configured
            if self.config.service_namespace:
                resource_attrs["service.namespace"] = self.config.service_namespace
            
            resource = Resource.create(resource_attrs)
            
            # Get OTLP endpoint from config or environment
            custom_endpoint = (
                self.config.otlp_endpoint or  # Config parameter takes priority
                self.config.custom_attributes.get("otlp_endpoint") or  # Custom attributes
                os.getenv("ARSHAI_OTLP_ENDPOINT")  # Environment variable fallback
            )
            
            otlp_endpoints = []
            if custom_endpoint:
                otlp_endpoints.append(custom_endpoint)
            
            # Only use custom endpoint if provided
            if not custom_endpoint:
                self.logger.info("No OTLP endpoint configured - set ARSHAI_OTLP_ENDPOINT to enable export")
                # Still create basic providers for local observability
                otlp_endpoints = []
            
            # Set up tracing
            tracer_provider = TracerProvider(resource=resource)
            
            # Add exporters if endpoints are configured
            if otlp_endpoints:
                for endpoint in otlp_endpoints:
                    try:
                        span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                        span_processor = BatchSpanProcessor(span_exporter)
                        tracer_provider.add_span_processor(span_processor)
                        
                        self.logger.info(f"Set up OTEL tracing export to {endpoint}")
                        break
                    except Exception as e:
                        self.logger.debug(f"Failed to connect to {endpoint}: {e}")
                        continue
                else:
                    self.logger.warning("Failed to connect to any OTLP endpoints - traces will not be exported")
            
            # Set as global provider
            trace.set_tracer_provider(tracer_provider)
            
            # Set up metrics
            meter_provider = MeterProvider(resource=resource)
            
            # Add metric exporters if endpoints are configured
            if otlp_endpoints:
                try:
                    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoints[0], insecure=True)
                    metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter, export_interval_millis=10000)
                    
                    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                    self.logger.info("Set up OTEL metrics export")
                except Exception as e:
                    self.logger.debug(f"Failed to set up metrics export: {e}")
            
            # Set as global provider
            metrics.set_meter_provider(meter_provider)
            
            self._is_enabled = True
            return True
            
        except ImportError as e:
            self.logger.debug(f"OTEL SDK components not available: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to set up OTEL configuration: {e}")
            return False
    
    def _initialize_telemetry(self):
        """Initialize telemetry using existing OTEL providers."""
        try:
            # Always use get_tracer pattern - NEVER create providers
            self._tracer = trace.get_tracer(
                self._package_name
            )
            
            self._meter = metrics.get_meter(
                self._package_name
            )
            
            self.logger.info(f"Telemetry initialized for {self._package_name} v{self._package_version}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize telemetry: {e}")
            self._initialize_noop_telemetry()
    
    def _initialize_noop_telemetry(self):
        """Initialize no-op implementations."""
        self._tracer = NoOpTracer()
        self._meter = NoOpMeter()
        self._is_enabled = False
        self.logger.info("Using no-op telemetry implementations")
    
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled and functional."""
        return self._is_enabled and self.config.trace_llm_calls
    
    def is_metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._is_enabled and self.config.collect_metrics
    
    def get_tracer(self) -> Union[Tracer, NoOpTracer]:
        """Get the tracer instance."""
        return self._tracer
    
    def get_meter(self) -> Union[Meter, NoOpMeter]:
        """Get the meter instance."""
        return self._meter
    
    @contextmanager
    def create_span(
        self, 
        name: str, 
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[SpanKind] = None
    ):
        """Create a span with automatic context propagation.
        
        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Optional span kind
        """
        if not self.is_enabled():
            yield None
            return
        
        # Always try to detect parent context
        span_kwargs = {}
        if attributes:
            span_kwargs['attributes'] = attributes
        if kind:
            span_kwargs['kind'] = kind
        
        try:
            with self._tracer.start_as_current_span(name, **span_kwargs) as span:
                yield span
        except Exception as e:
            self.logger.warning(f"Error creating span '{name}': {e}")
            yield None
    
    @asynccontextmanager
    async def create_async_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[SpanKind] = None
    ):
        """Create an async span with automatic context propagation.
        
        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Optional span kind
        """
        if not self.is_enabled():
            yield None
            return
        
        span_kwargs = {}
        if attributes:
            span_kwargs['attributes'] = attributes
        if kind:
            span_kwargs['kind'] = kind
        
        span = None
        try:
            with self._tracer.start_as_current_span(name, **span_kwargs) as span:
                yield span
        except Exception as e:
            self.logger.warning(f"Error creating async span '{name}': {e}")
            # Record exception on span if we have one
            if span is not None:
                try:
                    span.record_exception(e)
                    self.set_span_status_safely(span, False, str(e))
                except Exception:
                    pass  # Ignore secondary errors
            raise  # Re-raise the original exception
    
    def create_counter(self, name: str, description: str, unit: str = "1"):
        """Create a counter metric."""
        if not self.is_metrics_enabled():
            return NoOpInstrument()
        
        try:
            return self._meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        except Exception as e:
            self.logger.warning(f"Failed to create counter '{name}': {e}")
            return NoOpInstrument()
    
    def create_histogram(self, name: str, description: str, unit: str = "1"):
        """Create a histogram metric."""
        if not self.is_metrics_enabled():
            return NoOpInstrument()
        
        try:
            return self._meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        except Exception as e:
            self.logger.warning(f"Failed to create histogram '{name}': {e}")
            return NoOpInstrument()
    
    def create_up_down_counter(self, name: str, description: str, unit: str = "1"):
        """Create an up-down counter metric."""
        if not self.is_metrics_enabled():
            return NoOpInstrument()
        
        try:
            return self._meter.create_up_down_counter(
                name=name,
                description=description,
                unit=unit
            )
        except Exception as e:
            self.logger.warning(f"Failed to create up-down counter '{name}': {e}")
            return NoOpInstrument()
    
    def set_span_attributes_safely(self, span, attributes: Dict[str, Any]):
        """Safely set span attributes with error handling."""
        if span is None or not attributes:
            return
        
        for key, value in attributes.items():
            try:
                span.set_attribute(key, value)
            except Exception as e:
                self.logger.warning(f"Failed to set span attribute '{key}': {e}")
    
    def set_span_status_safely(self, span, success: bool, error_message: Optional[str] = None):
        """Safely set span status with error handling."""
        if span is None:
            return
        
        try:
            if success:
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status("OK")
            else:
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, error_message or "Operation failed"))
                else:
                    span.set_status("ERROR")
        except Exception as e:
            self.logger.warning(f"Failed to set span status: {e}")
    
    def record_exception_safely(self, span, exception: Exception):
        """Safely record exception on span."""
        if span is None:
            return
        
        try:
            span.record_exception(exception)
        except Exception as e:
            self.logger.warning(f"Failed to record exception on span: {e}")
    
    def get_package_info(self) -> Dict[str, str]:
        """Get package information for telemetry."""
        return {
            "service.name": self._package_name,
            "service.version": self._package_version,
            "telemetry.sdk.name": "arshai-observability",
            "telemetry.sdk.version": self._package_version
        }


# Global instance for package-wide use
_default_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager(config: Optional[PackageObservabilityConfig] = None) -> TelemetryManager:
    """Get or create the default telemetry manager.
    
    Args:
        config: Optional configuration (only used for first creation)
    
    Returns:
        TelemetryManager instance
    """
    global _default_telemetry_manager
    
    if _default_telemetry_manager is None:
        _default_telemetry_manager = TelemetryManager(config)
    
    return _default_telemetry_manager


def reset_telemetry_manager():
    """Reset the global telemetry manager (useful for testing)."""
    global _default_telemetry_manager
    _default_telemetry_manager = None