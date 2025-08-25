"""Core observability manager for Arshai framework."""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager, asynccontextmanager

# OpenTelemetry imports with fallbacks
try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode, Tracer, SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from .config import ObservabilityConfig
from .metrics import MetricsCollector, TimingData


class ObservabilityManager:
    """Central manager for all observability features."""
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_collector: Optional[MetricsCollector] = None
        self.tracer: Optional[Tracer] = None
        
        self._initialize_components()
        
        self.logger.info("Observability manager initialized")
    
    def _initialize_components(self):
        """Initialize observability components."""
        # Initialize metrics collector
        if self.config.collect_metrics:
            self.metrics_collector = MetricsCollector(self.config)
        
        # Initialize tracing
        if self.config.trace_requests and OTEL_AVAILABLE:
            self._initialize_tracing()
        
    
    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing."""
        try:
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "environment": self.config.environment,
            })
            
            tracer_provider = TracerProvider(resource=resource)
            
            # Set up OTLP exporter if endpoint is configured
            if self.config.otlp_endpoint:
                # Auto-detect protocol based on endpoint
                if '/v1/traces' in self.config.otlp_endpoint or self.config.otlp_endpoint.startswith('http'):
                    # HTTP endpoint detected
                    try:
                        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
                        otlp_exporter = HTTPSpanExporter(
                            endpoint=self.config.otlp_endpoint,
                            headers=self.config.otlp_headers,
                            timeout=self.config.otlp_timeout,
                        )
                        self.logger.info("Using OTLP HTTP exporter")
                    except ImportError:
                        self.logger.warning("HTTP exporter not available, falling back to gRPC")
                        otlp_exporter = OTLPSpanExporter(
                            endpoint=self.config.otlp_endpoint,
                            headers=self.config.otlp_headers,
                            timeout=self.config.otlp_timeout,
                        )
                else:
                    # gRPC endpoint (default)
                    otlp_exporter = OTLPSpanExporter(
                        endpoint=self.config.otlp_endpoint,
                        headers=self.config.otlp_headers,
                        timeout=self.config.otlp_timeout,
                    )
                    self.logger.info("Using OTLP gRPC exporter")
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            # Check if a TracerProvider is already set to avoid the override error
            current_provider = trace.get_tracer_provider()
            if not hasattr(current_provider, 'add_span_processor'):
                # No real TracerProvider is set, safe to set ours
                trace.set_tracer_provider(tracer_provider)
            else:
                self.logger.warning("TracerProvider already set, using existing provider")
                tracer_provider = current_provider
            
            self.tracer = trace.get_tracer(__name__)
            
            self.logger.info("Tracing initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize tracing: {e}")
    
    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return True  # Always enabled now
    
    def is_token_timing_enabled(self, provider: str) -> bool:
        """Check if token timing is enabled for a specific provider."""
        return self.config.is_token_timing_enabled(provider)
    
    @asynccontextmanager
    async def observe_llm_call(self, 
                        provider: str, 
                        model: str, 
                        method_name: str = "llm_call",
                        system: Optional[str] = None,
                        **extra_attributes):
        """Non-intrusive async context manager for observing LLM calls.
        
        Args:
            provider: LLM provider name
            model: Model name  
            method_name: Method being called
            **extra_attributes: Additional attributes
        """
        if not self.is_token_timing_enabled(provider):
            # Return a no-op timing data if observability is disabled
            yield TimingData()
            return
        
        # Create span attributes
        span_attributes = {
            "llm.provider": provider,
            "llm.model_name": model,  # Renamed to match OpenInference
            "llm.method": method_name,
        }
        
        # Add system if provided
        if system:
            span_attributes["llm.system"] = system
        span_attributes.update(self.config.custom_attributes)
        
        # Add remaining extra attributes
        span_attributes.update(extra_attributes)
        
        # Start tracing if enabled
        span_context = None
        if self.tracer:
            span_context = self.tracer.start_as_current_span(
                f"llm.{method_name}",
                attributes=span_attributes
            )
        
        # Start metrics collection if enabled
        metrics_context = None
        if self.metrics_collector:
            metrics_context = self.metrics_collector.track_request(
                provider, model, **extra_attributes
            )
        
        try:
            # Use the metrics context manager if available, otherwise create timing data
            if metrics_context:
                with metrics_context as timing_data:
                    if span_context:
                        with span_context as span:
                            try:
                                yield timing_data
                                self._update_span_with_timing(span, timing_data)
                                span.set_status(Status(StatusCode.OK))
                            except Exception as e:
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                span.record_exception(e)
                                raise
                    else:
                        yield timing_data
            else:
                # No metrics, just tracing or plain timing data
                timing_data = TimingData()
                if span_context:
                    with span_context as span:
                        try:
                            yield timing_data
                            self._update_span_with_timing(span, timing_data)
                            span.set_status(Status(StatusCode.OK))
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise
                else:
                    yield timing_data
                    
        except Exception as e:
            self.logger.error(f"Error in LLM call observation: {e}")
            raise
    
    @asynccontextmanager
    async def observe_streaming_llm_call(self,
                                       provider: str,
                                       model: str, 
                                       method_name: str = "stream_llm_call",
                                       system: Optional[str] = None,
                                       **extra_attributes):
        """Non-intrusive async context manager for observing streaming LLM calls.
        
        Args:
            provider: LLM provider name
            model: Model name
            method_name: Method being called
            **extra_attributes: Additional attributes
        """
        if not self.is_token_timing_enabled(provider):
            # Return a no-op timing data if observability is disabled
            yield TimingData()
            return
        
        # Create span attributes
        span_attributes = {
            "llm.provider": provider,
            "llm.model_name": model,  # Renamed to match OpenInference
            "llm.method": method_name,
            "llm.streaming": True,
        }
        
        # Add system if provided
        if system:
            span_attributes["llm.system"] = system
        span_attributes.update(self.config.custom_attributes)
        
        # Add remaining extra attributes
        span_attributes.update(extra_attributes)
        
        # Start tracing if enabled
        span_context = None
        if self.tracer:
            span_context = self.tracer.start_as_current_span(
                f"llm.{method_name}",
                attributes=span_attributes
            )
        
        # Start metrics collection if enabled
        metrics_context = None
        if self.metrics_collector:
            metrics_context = self.metrics_collector.track_request(
                provider, model, **extra_attributes
            )
        
        try:
            # Use the metrics context manager if available, otherwise create timing data
            if metrics_context:
                with metrics_context as timing_data:
                    if span_context:
                        with span_context as span:
                            try:
                                yield timing_data
                                self._update_span_with_timing(span, timing_data)
                                span.set_status(Status(StatusCode.OK))
                            except Exception as e:
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                span.record_exception(e)
                                raise
                    else:
                        yield timing_data
            else:
                # No metrics, just tracing or plain timing data
                timing_data = TimingData()
                if span_context:
                    with span_context as span:
                        try:
                            yield timing_data
                            self._update_span_with_timing(span, timing_data)
                            span.set_status(Status(StatusCode.OK))
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise
                else:
                    yield timing_data
                    
        except Exception as e:
            self.logger.error(f"Error in streaming LLM call observation: {e}")
            raise
    
    def _update_span_with_timing(self, span: Span, timing_data: TimingData):
        """Update span with timing data."""
        if not span:
            return
        
        try:
            if timing_data.time_to_first_token is not None:
                span.set_attribute("llm.time_to_first_token", timing_data.time_to_first_token)
            
            if timing_data.time_to_last_token is not None:
                span.set_attribute("llm.time_to_last_token", timing_data.time_to_last_token)
            
            span.set_attribute("llm.total_duration", timing_data.total_duration)
            
            # Token counts - using LLM client naming convention
            if timing_data.input_tokens > 0:
                span.set_attribute("llm.usage.input_tokens", timing_data.input_tokens)
            if timing_data.output_tokens > 0:
                span.set_attribute("llm.usage.output_tokens", timing_data.output_tokens)
            if timing_data.total_tokens > 0:
                span.set_attribute("llm.usage.total_tokens", timing_data.total_tokens)
                # OpenInference standard cumulative tokens parameter
                span.set_attribute("llm.token_count.total", timing_data.total_tokens)
            if timing_data.thinking_tokens > 0:
                span.set_attribute("llm.usage.thinking_tokens", timing_data.thinking_tokens)
            if timing_data.tool_calling_tokens > 0:
                span.set_attribute("llm.usage.tool_calling_tokens", timing_data.tool_calling_tokens)

            # Additional OpenInference attributes
            if hasattr(timing_data, 'input_value') and timing_data.input_value:
                span.set_attribute("input.value", timing_data.input_value)
            
            if hasattr(timing_data, 'output_value') and timing_data.output_value:
                span.set_attribute("output.value", timing_data.output_value)
            
            if hasattr(timing_data, 'input_mime_type'):
                span.set_attribute("input.mime_type", timing_data.input_mime_type)
            
            if hasattr(timing_data, 'output_mime_type'):
                span.set_attribute("output.mime_type", timing_data.output_mime_type)
            
            if hasattr(timing_data, 'input_messages') and timing_data.input_messages:
                span.set_attribute("llm.input_messages", str(timing_data.input_messages))
            
            if hasattr(timing_data, 'output_messages') and timing_data.output_messages:
                span.set_attribute("llm.output_messages", str(timing_data.output_messages))
            
            if hasattr(timing_data, 'invocation_parameters') and timing_data.invocation_parameters:
                span.set_attribute("llm.invocation_parameters", str(timing_data.invocation_parameters))
            
            if hasattr(timing_data, 'function_call') and timing_data.function_call:
                span.set_attribute("llm.function_call", str(timing_data.function_call))
            
            # Cost tracking
            if hasattr(timing_data, 'prompt_cost') and timing_data.prompt_cost is not None:
                span.set_attribute("llm.cost.prompt", timing_data.prompt_cost)
            
            if hasattr(timing_data, 'completion_cost') and timing_data.completion_cost is not None:
                span.set_attribute("llm.cost.completion", timing_data.completion_cost)
            
            if hasattr(timing_data, 'total_cost') and timing_data.total_cost is not None:
                span.set_attribute("llm.cost.total", timing_data.total_cost)
                
        except Exception as e:
            self.logger.warning(f"Failed to update span with timing: {e}")
    
    async def record_usage_data(self, timing_data: TimingData, usage_data: Dict[str, Any]):
        """Record usage data from LLM response.
        
        Args:
            timing_data: TimingData instance to update
            usage_data: Usage data from LLM response with fields:
                       input_tokens, output_tokens, total_tokens, thinking_tokens, tool_calling_tokens
        """
        if self.metrics_collector:
            await self.metrics_collector.record_usage_data(timing_data, usage_data)
    
    
    def get_config(self) -> ObservabilityConfig:
        """Get the current configuration."""
        return self.config
    
    def shutdown(self):
        """Shutdown observability components."""
        try:
            # Shutdown tracer provider
            if OTEL_AVAILABLE:
                tracer_provider = trace.get_tracer_provider()
                if hasattr(tracer_provider, 'shutdown'):
                    tracer_provider.shutdown()
            
            
            self.logger.info("Observability manager shutdown completed")
        except Exception as e:
            self.logger.warning(f"Error during shutdown: {e}")