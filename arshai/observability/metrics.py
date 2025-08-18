"""Non-intrusive metrics collection for LLM observability."""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
import asyncio

# OpenTelemetry imports with fallbacks
try:
    from opentelemetry import metrics
    from opentelemetry.metrics import Meter, Counter, Histogram, UpDownCounter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from .config import ObservabilityConfig


@dataclass
class TimingData:
    """Container for timing measurements and OpenInference attributes."""
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    # Token counts - using LLM client naming convention
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0
    tool_calling_tokens: int = 0
    
    # OpenInference attributes
    input_value: Optional[str] = None
    output_value: Optional[str] = None
    input_mime_type: str = "application/json"
    output_mime_type: str = "application/json"
    input_messages: Optional[List[Dict[str, Any]]] = None
    output_messages: Optional[List[Dict[str, Any]]] = None
    invocation_parameters: Optional[Dict[str, Any]] = None
    function_call: Optional[Dict[str, Any]] = None
    
    # Cost tracking
    prompt_cost: Optional[float] = None
    completion_cost: Optional[float] = None
    total_cost: Optional[float] = None
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time from start to first token in seconds."""
        if self.first_token_time is not None:
            return self.first_token_time - self.start_time
        return None
    
    @property
    def time_to_last_token(self) -> Optional[float]:
        """Time from start to last token in seconds."""
        if self.last_token_time is not None:
            return self.last_token_time - self.start_time
        return None
    
    @property
    def duration_first_to_last_token(self) -> Optional[float]:
        """Duration from first token to last token in seconds."""
        if self.first_token_time is not None and self.last_token_time is not None:
            return self.last_token_time - self.first_token_time
        return None
    
    @property
    def total_duration(self) -> float:
        """Total duration from start to completion."""
        end_time = self.last_token_time if self.last_token_time else time.time()
        return end_time - self.start_time
    
    def record_first_token(self):
        """Record the time of the first token."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
    
    def record_token(self):
        """Record a token (updates last token time)."""
        self.last_token_time = time.time()
        if self.first_token_time is None:
            self.first_token_time = self.last_token_time
    
    def update_token_counts(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0,
                           thinking_tokens: int = 0, tool_calling_tokens: int = 0):
        """Update token counts from usage data."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.thinking_tokens = thinking_tokens
        self.tool_calling_tokens = tool_calling_tokens


class MetricsCollector:
    """Non-intrusive metrics collector for LLM operations."""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = config.collect_metrics  # Always enabled, only controlled by collect_metrics
        
        # Initialize OpenTelemetry metrics if available
        self.meter: Optional[Meter] = None
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize OpenTelemetry metrics."""
        if not self.enabled or not OTEL_AVAILABLE:
            return
        
        try:
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "environment": self.config.environment,
            })
            
            # Set up metric reader and provider
            if self.config.otlp_endpoint:
                # Auto-detect protocol based on endpoint
                if '/v1/metrics' in self.config.otlp_endpoint or self.config.otlp_endpoint.endswith(':4318'):
                    # HTTP endpoint detected
                    try:
                        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HTTPMetricExporter
                        metric_exporter = HTTPMetricExporter(
                            endpoint=self.config.otlp_endpoint,
                            headers=self.config.otlp_headers,
                            timeout=self.config.otlp_timeout,
                        )
                        self.logger.info("Using OTLP HTTP metrics exporter")
                    except ImportError:
                        self.logger.warning("HTTP metrics exporter not available, falling back to gRPC")
                        metric_exporter = OTLPMetricExporter(
                            endpoint=self.config.otlp_endpoint,
                            headers=self.config.otlp_headers,
                            timeout=self.config.otlp_timeout,
                        )
                else:
                    # gRPC endpoint (default)
                    metric_exporter = OTLPMetricExporter(
                        endpoint=self.config.otlp_endpoint,
                        headers=self.config.otlp_headers,
                        timeout=self.config.otlp_timeout,
                    )
                    self.logger.info("Using OTLP gRPC metrics exporter")
                metric_reader = PeriodicExportingMetricReader(
                    exporter=metric_exporter,
                    export_interval_millis=self.config.metric_export_interval * 1000,
                )
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            else:
                meter_provider = MeterProvider(resource=resource)
            
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)
            
            # Create metrics
            self._create_metrics()
            
            self.logger.info("Metrics collector initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize metrics: {e}")
            self.enabled = False
    
    def _create_metrics(self):
        """Create OpenTelemetry metrics."""
        if not self.meter:
            return
        
        # Request counters
        self.llm_requests_total = self.meter.create_counter(
            name="llm_requests_total",
            description="Total number of LLM requests",
            unit="1"
        )
        
        self.llm_requests_failed = self.meter.create_counter(
            name="llm_requests_failed", 
            description="Total number of failed LLM requests",
            unit="1"
        )
        
        # Token counters
        self.llm_tokens_total = self.meter.create_counter(
            name="llm_tokens_total",
            description="Total number of tokens processed",
            unit="1"
        )
        
        self.llm_input_tokens = self.meter.create_counter(
            name="llm_input_tokens",
            description="Total number of input tokens",
            unit="1"
        )
        
        self.llm_output_tokens = self.meter.create_counter(
            name="llm_output_tokens", 
            description="Total number of output tokens",
            unit="1"
        )
        
        # Additional token type metrics
        self.llm_thinking_tokens = self.meter.create_counter(
            name="llm_thinking_tokens",
            description="Total number of thinking/reasoning tokens",
            unit="1"
        )
        
        self.llm_tool_calling_tokens = self.meter.create_counter(
            name="llm_tool_calling_tokens",
            description="Total number of tool calling tokens",
            unit="1"
        )
        
        # KEY METRICS - The ones specifically requested
        self.llm_time_to_first_token = self.meter.create_histogram(
            name="llm_time_to_first_token_seconds",
            description="Time from request start to first token",
            unit="s"
        )
        
        self.llm_time_to_last_token = self.meter.create_histogram(
            name="llm_time_to_last_token_seconds", 
            description="Time from request start to last token",
            unit="s"
        )
        
        self.llm_duration_first_to_last_token = self.meter.create_histogram(
            name="llm_duration_first_to_last_token_seconds",
            description="Duration from first token to last token", 
            unit="s"
        )
        
        # Additional timing metrics
        self.llm_request_duration = self.meter.create_histogram(
            name="llm_request_duration_seconds",
            description="Total LLM request duration",
            unit="s"
        )
        
        # Throughput metrics
        self.llm_tokens_per_second = self.meter.create_histogram(
            name="llm_tokens_per_second",
            description="Token generation rate",
            unit="tokens/s"
        )
        
        # Active requests gauge
        self.llm_active_requests = self.meter.create_up_down_counter(
            name="llm_active_requests",
            description="Number of active LLM requests",
            unit="1"
        )
    
    async def record_usage_data(self, timing_data: TimingData, usage_data: Dict[str, Any]):
        """Record usage data from LLM response.
        
        Args:
            timing_data: TimingData instance to update
            usage_data: Usage data from LLM response containing token counts
        """
        if usage_data:
            timing_data.update_token_counts(
                input_tokens=usage_data.get('input_tokens', 0),
                output_tokens=usage_data.get('output_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0),
                thinking_tokens=usage_data.get('thinking_tokens', 0),
                tool_calling_tokens=usage_data.get('tool_calling_tokens', 0)
            )
    
    async def record_request_start(self, attributes: Dict[str, Any]) -> TimingData:
        """Record the start of an LLM request.
        
        Args:
            attributes: Request attributes for metrics
            
        Returns:
            TimingData instance for recording timing
        """
        if not self.enabled:
            return TimingData()
        
        timing_data = TimingData()
        
        try:
            # These operations are actually synchronous but we make the method async
            # for consistency and future async metric operations
            self.llm_requests_total.add(1, attributes)
            self.llm_active_requests.add(1, attributes)
        except Exception as e:
            self.logger.warning(f"Failed to record request start: {e}")
        
        return timing_data
    
    async def record_request_end(self, 
                          attributes: Dict[str, Any], 
                          timing_data: TimingData,
                          success: bool = True):
        """Record the completion of an LLM request.
        
        Args:
            attributes: Request attributes for metrics
            timing_data: Timing measurements
            success: Whether the request was successful
        """
        if not self.enabled:
            return
        
        try:
            # Update active requests
            self.llm_active_requests.add(-1, attributes)
            
            # Record failure if needed
            if not success:
                self.llm_requests_failed.add(1, attributes)
                return
            
            # Record token counts
            if timing_data.total_tokens > 0:
                self.llm_tokens_total.add(timing_data.total_tokens, attributes)
            if timing_data.input_tokens > 0:
                self.llm_input_tokens.add(timing_data.input_tokens, attributes)
            if timing_data.output_tokens > 0:
                self.llm_output_tokens.add(timing_data.output_tokens, attributes)
            if timing_data.thinking_tokens > 0:
                self.llm_thinking_tokens.add(timing_data.thinking_tokens, attributes)
            if timing_data.tool_calling_tokens > 0:
                self.llm_tool_calling_tokens.add(timing_data.tool_calling_tokens, attributes)
            
            # Record KEY METRICS - the ones specifically requested
            if timing_data.time_to_first_token is not None:
                self.llm_time_to_first_token.record(timing_data.time_to_first_token, attributes)
            
            if timing_data.time_to_last_token is not None:
                self.llm_time_to_last_token.record(timing_data.time_to_last_token, attributes)
            
            if timing_data.duration_first_to_last_token is not None:
                self.llm_duration_first_to_last_token.record(timing_data.duration_first_to_last_token, attributes)
            
            # Total request duration
            self.llm_request_duration.record(timing_data.total_duration, attributes)
            
            # Calculate and record throughput
            if timing_data.output_tokens > 0 and timing_data.total_duration > 0:
                tokens_per_second = timing_data.output_tokens / timing_data.total_duration
                self.llm_tokens_per_second.record(tokens_per_second, attributes)
            
        except Exception as e:
            self.logger.warning(f"Failed to record request end: {e}")
    
    def create_attributes(self, provider: str, model: str, **extra_attributes) -> Dict[str, Any]:
        """Create standard attributes for metrics.
        
        Args:
            provider: LLM provider name
            model: Model name
            **extra_attributes: Additional attributes
            
        Returns:
            Dictionary of attributes
        """
        attributes = {
            "llm.provider": provider,
            "llm.model_name": model,  # Renamed to match OpenInference
        }
        
        # Add custom attributes from config
        attributes.update(self.config.custom_attributes)
        
        # Add extra attributes
        attributes.update(extra_attributes)
        
        return attributes
    
    @contextmanager
    def track_request(self, provider: str, model: str, **extra_attributes):
        """Synchronous context manager for tracking a complete LLM request.
        
        Args:
            provider: LLM provider name
            model: Model name
            **extra_attributes: Additional attributes
        """
        if not self.enabled:
            timing_data = TimingData()
            yield timing_data
            return
        
        attributes = self.create_attributes(provider, model, **extra_attributes)
        # Create timing data synchronously to avoid event loop issues
        timing_data = TimingData()
        timing_data.start_time = time.time()
        success = False
        
        try:
            yield timing_data
            success = True
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise
        finally:
            # Record metrics synchronously
            try:
                self._record_metrics_sync(attributes, timing_data, success)
            except Exception as e:
                self.logger.error(f"Failed to record metrics: {e}")
    
    @asynccontextmanager
    async def async_track_request(self, provider: str, model: str, **extra_attributes):
        """Async context manager for tracking a complete LLM request.
        
        Args:
            provider: LLM provider name
            model: Model name
            **extra_attributes: Additional attributes
        """
        if not self.enabled:
            timing_data = TimingData()
            yield timing_data
            return
        
        attributes = self.create_attributes(provider, model, **extra_attributes)
        timing_data = await self.record_request_start(attributes)
        success = False
        
        try:
            yield timing_data
            success = True
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise
        finally:
            await self.record_request_end(attributes, timing_data, success)
    
    def _record_metrics_sync(self, attributes: Dict[str, Any], timing_data: TimingData, success: bool):
        """Record metrics synchronously without async operations."""
        try:
            # Record timing metrics
            if timing_data.time_to_first_token is not None:
                self.llm_time_to_first_token.record(timing_data.time_to_first_token, attributes)
            
            if timing_data.time_to_last_token is not None:
                self.llm_time_to_last_token.record(timing_data.time_to_last_token, attributes)
            
            if timing_data.duration_first_to_last_token is not None:
                self.llm_duration_first_to_last_token.record(timing_data.duration_first_to_last_token, attributes)
            
            # Record request duration
            self.llm_request_duration.record(timing_data.total_duration, attributes)
            
            # Record token counts
            if timing_data.input_tokens > 0:
                self.llm_input_tokens.add(timing_data.input_tokens, attributes)
            if timing_data.output_tokens > 0:
                self.llm_output_tokens.add(timing_data.output_tokens, attributes)
            if timing_data.total_tokens > 0:
                self.llm_tokens_total.add(timing_data.total_tokens, attributes)
            
            # Record request counts
            self.llm_requests_total.add(1, attributes)
            if not success:
                self.llm_requests_failed.add(1, attributes)
                
        except Exception as e:
            self.logger.error(f"Failed to record sync metrics: {e}")
    
    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.enabled