"""LLM-friendly observability implementation for Arshai package.

This module provides the main observability interface that:
1. Uses get_tracer("arshai", version) pattern (NEVER creates providers)
2. Respects parent context and configuration
3. Provides no-op fallbacks when OTEL is unavailable
4. Focuses on LLM-specific metrics and tracing
5. Maintains backward compatibility with existing LLM clients
"""

import logging
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import time

# Safe OTEL imports with fallbacks
try:
    from opentelemetry.trace import SpanKind, Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create dummy constants for compatibility
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
from .telemetry_manager import get_telemetry_manager, TelemetryManager
from .metrics import TimingData

# OpenInference SpanAttributes for compatibility
try:
    from openinference.semconv.trace import SpanAttributes
    OPENINFERENCE_AVAILABLE = True
    
    # Add any missing attributes as fallbacks
    if not hasattr(SpanAttributes, 'EMBEDDING_TEXT'):
        SpanAttributes.EMBEDDING_TEXT = "embedding.text"
    if not hasattr(SpanAttributes, 'EMBEDDING_VECTOR'):
        SpanAttributes.EMBEDDING_VECTOR = "embedding.vector"
    if not hasattr(SpanAttributes, 'RERANKER_INPUT_DOCUMENTS'):
        SpanAttributes.RERANKER_INPUT_DOCUMENTS = "reranker.input_documents" 
    if not hasattr(SpanAttributes, 'RERANKER_OUTPUT_DOCUMENTS'):
        SpanAttributes.RERANKER_OUTPUT_DOCUMENTS = "reranker.output_documents"
    if not hasattr(SpanAttributes, 'RERANKER_MODEL_NAME'):
        SpanAttributes.RERANKER_MODEL_NAME = "reranker.model_name"
    if not hasattr(SpanAttributes, 'RERANKER_QUERY'):
        SpanAttributes.RERANKER_QUERY = "reranker.query"
    if not hasattr(SpanAttributes, 'RERANKER_TOP_K'):
        SpanAttributes.RERANKER_TOP_K = "reranker.top_k"
    if not hasattr(SpanAttributes, 'DOCUMENT_ID'):
        SpanAttributes.DOCUMENT_ID = "document.id"
    if not hasattr(SpanAttributes, 'DOCUMENT_SCORE'):
        SpanAttributes.DOCUMENT_SCORE = "document.score"
    if not hasattr(SpanAttributes, 'DOCUMENT_CONTENT'):
        SpanAttributes.DOCUMENT_CONTENT = "document.content"
    if not hasattr(SpanAttributes, 'DOCUMENT_METADATA'):
        SpanAttributes.DOCUMENT_METADATA = "document.metadata"
        
except ImportError:
    # Fallback SpanAttributes when OpenInference is not available
    OPENINFERENCE_AVAILABLE = False
    class SpanAttributes:
        # LLM Span Attributes  
        LLM_PROMPT_TEMPLATE_VARIABLES = "llm.prompt_template.variables"
        LLM_PROMPT_TEMPLATE = "llm.prompt_template"
        LLM_PROMPT_TEMPLATE_VERSION = "llm.prompt_template.version"
        LLM_TOKEN_COUNT_PROMPT = "llm.usage.prompt_tokens"
        LLM_TOKEN_COUNT_COMPLETION = "llm.usage.completion_tokens"
        LLM_TOKEN_COUNT_TOTAL = "llm.usage.total_tokens"
        LLM_FUNCTION_CALL = "llm.function_call"
        LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
        LLM_INPUT_MESSAGES = "llm.input_messages"
        LLM_OUTPUT_MESSAGES = "llm.output_messages"
        LLM_MODEL_NAME = "llm.request.model"
        
        # Embedding Span Attributes
        EMBEDDING_MODEL_NAME = "embedding.model_name"
        EMBEDDING_TEXT = "embedding.text"
        EMBEDDING_VECTOR = "embedding.vector"
        EMBEDDING_EMBEDDINGS = "embedding.embeddings"
        
        # Retriever Span Attributes
        RETRIEVAL_DOCUMENTS = "retrieval.documents"
        DOCUMENT_ID = "document.id"
        DOCUMENT_SCORE = "document.score"
        DOCUMENT_CONTENT = "document.content"
        DOCUMENT_METADATA = "document.metadata"
        
        # Tool Span Attributes
        TOOL_NAME = "tool.name"
        TOOL_DESCRIPTION = "tool.description"
        TOOL_PARAMETERS = "tool.parameters"
        
        # Reranker Span Attributes  
        RERANKER_INPUT_DOCUMENTS = "reranker.input_documents"
        RERANKER_OUTPUT_DOCUMENTS = "reranker.output_documents"
        RERANKER_MODEL_NAME = "reranker.model_name"
        RERANKER_QUERY = "reranker.query"
        RERANKER_TOP_K = "reranker.top_k"
        
        # General Span Attributes
        INPUT_VALUE = "input.value"
        INPUT_MIME_TYPE = "input.mime_type"
        OUTPUT_VALUE = "output.value"
        OUTPUT_MIME_TYPE = "output.mime_type"
        METADATA = "metadata"
        TAGS = "tags"
        TAG_TAGS = "tags"
        OPENINFERENCE_SPAN_KIND = "openinference.span.kind"


class ArshaiObservability:
    """Comprehensive observability manager for all Arshai operations that respects parent OTEL configuration.
    
    Supports multiple span types following OpenInference/Arize conventions:
    - LLM spans for language model operations
    - EMBEDDING spans for text embedding operations
    - RETRIEVER spans for document retrieval operations
    - TOOL spans for function/tool calls
    - RERANKER spans for document reranking
    - AGENT spans for agent operations
    - CHAIN spans for workflow operations
    """
    
    def __init__(self, config: Optional[PackageObservabilityConfig] = None):
        """Initialize LLM observability.
        
        Args:
            config: Optional package-specific configuration
        """
        self.config = config or PackageObservabilityConfig.from_environment()
        self.logger = logging.getLogger(__name__)
        
        # Get or create telemetry manager (uses proper OTEL patterns)
        self.telemetry = get_telemetry_manager(self.config)
        
        # Initialize metrics if enabled
        self._metrics = {}
        if self.config.is_feature_enabled("collect_metrics"):
            self._initialize_metrics()
        
        self.logger.info(
            f"LLM observability initialized - enabled: {self.config.enabled}, "
            f"level: {self.config.level.value}"
        )
    
    def _initialize_metrics(self):
        """Initialize LLM-specific metrics using telemetry manager."""
        try:
            # Request counters
            self._metrics["llm_requests_total"] = self.telemetry.create_counter(
                name="llm_requests_total",
                description="Total LLM requests from Arshai package",
                unit="1"
            )
            
            self._metrics["llm_requests_failed"] = self.telemetry.create_counter(
                name="llm_requests_failed",
                description="Failed LLM requests from Arshai package",
                unit="1"
            )
            
            # Token metrics
            self._metrics["llm_tokens_total"] = self.telemetry.create_counter(
                name="llm_tokens_total",
                description="Total tokens processed by Arshai package",
                unit="1"
            )
            
            self._metrics["llm_input_tokens"] = self.telemetry.create_counter(
                name="llm_input_tokens",
                description="Input tokens processed by Arshai package", 
                unit="1"
            )
            
            self._metrics["llm_output_tokens"] = self.telemetry.create_counter(
                name="llm_output_tokens",
                description="Output tokens generated by Arshai package",
                unit="1"
            )
            
            # Key timing metrics
            self._metrics["llm_time_to_first_token"] = self.telemetry.create_histogram(
                name="llm_time_to_first_token_seconds",
                description="Time from request start to first token (Arshai)",
                unit="s"
            )
            
            self._metrics["llm_time_to_last_token"] = self.telemetry.create_histogram(
                name="llm_time_to_last_token_seconds",
                description="Time from request start to last token (Arshai)",
                unit="s"
            )
            
            self._metrics["llm_request_duration"] = self.telemetry.create_histogram(
                name="llm_request_duration_seconds",
                description="Total LLM request duration (Arshai)",
                unit="s"
            )
            
            # Active requests gauge
            self._metrics["llm_active_requests"] = self.telemetry.create_up_down_counter(
                name="llm_active_requests", 
                description="Active LLM requests (Arshai)",
                unit="1"
            )
            
            self.logger.info("LLM metrics initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM metrics: {e}")
    
    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return self.config.enabled and self.telemetry.is_enabled()
    
    def is_metrics_enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self.is_enabled() and self.config.is_feature_enabled("collect_metrics")
    
    def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.is_enabled() and self.config.is_feature_enabled("trace_llm_calls")
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if observability is enabled for specific provider."""
        return self.is_enabled() and self.config.is_provider_enabled(provider)
    
    @asynccontextmanager
    async def observe_llm_call(
        self,
        provider: str,
        model: str,
        method_name: str = "llm_call",
        **extra_attributes
    ):
        """Observe an LLM call with proper context propagation.
        
        This is the main entry point for LLM observability that:
        - Creates spans as children of existing context
        - Records LLM-specific metrics  
        - Handles errors gracefully
        - Works with or without parent OTEL setup
        
        Args:
            provider: LLM provider name (e.g., 'openai', 'anthropic')
            model: Model name (e.g., 'gpt-4', 'claude-3')
            method_name: Method being called (e.g., 'chat', 'stream')
            **extra_attributes: Additional span attributes
        """
        # Quick exit if not enabled for this provider
        if not self.is_provider_enabled(provider):
            yield TimingData()
            return
        
        # Create span attributes - ensure all values are OTEL-compatible types
        span_attributes = self._create_span_attributes(
            provider, model, method_name, **extra_attributes
        )
        span_attributes = self._sanitize_span_attributes(span_attributes)
        
        # Create timing data
        timing_data = TimingData()
        success = False
        
        # Start metrics tracking
        if self.is_metrics_enabled():
            # Metrics attributes need to be hashable (no dict or list values)
            metrics_attributes = self._create_metrics_attributes(span_attributes)
            self._record_request_start(metrics_attributes)
        
        # Create span with proper context propagation
        span_name = f"{provider}.{method_name}"
        
        try:
            if self.is_tracing_enabled():
                async with self.telemetry.create_async_span(
                    span_name,
                    attributes=span_attributes,
                    kind=SpanKind.CLIENT if OTEL_AVAILABLE else None
                ) as span:
                    try:
                        yield timing_data
                        success = True
                        self._update_span_with_timing(span, timing_data)
                        self.telemetry.set_span_status_safely(span, True)
                    except Exception as e:
                        success = False
                        self.telemetry.set_span_status_safely(span, False, str(e))
                        self.telemetry.record_exception_safely(span, e)
                        raise
            else:
                # No tracing, just return timing data
                try:
                    yield timing_data
                    success = True
                except Exception:
                    success = False
                    raise
                
        except Exception:
            success = False
            raise
        finally:
            # Record metrics
            if self.is_metrics_enabled():
                # Metrics attributes need to be hashable (no dict or list values)
                metrics_attributes = self._create_metrics_attributes(span_attributes)
                self._record_request_end(metrics_attributes, timing_data, success)
    
    @asynccontextmanager
    async def observe_streaming_llm_call(
        self,
        provider: str,
        model: str,
        method_name: str = "stream_llm_call",
        **extra_attributes
    ):
        """Observe a streaming LLM call.
        
        Args:
            provider: LLM provider name
            model: Model name
            method_name: Method being called
            **extra_attributes: Additional span attributes
        """
        # Add streaming indicator  
        extra_attributes["llm.streaming"] = True
        
        # Use the same observation logic
        async with self.observe_llm_call(
            provider, model, method_name, **extra_attributes
        ) as timing_data:
            yield timing_data
    
    # ============================================================================
    # EMBEDDING SPAN OBSERVABILITY
    # ============================================================================
    
    @asynccontextmanager
    async def observe_embedding_operation(
        self,
        model_name: str,
        text_input: str = "",
        operation_name: str = "embedding",
        **extra_attributes
    ):
        """Observe an embedding operation with proper OpenInference attributes.
        
        Args:
            model_name: Name of the embedding model
            text_input: Input text being embedded
            operation_name: Name of the operation (e.g., 'embed_query', 'embed_documents')
            **extra_attributes: Additional span attributes
        """
        if not self.is_enabled():
            yield TimingData()
            return
        
        span_name = f"embedding.{operation_name}"
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "EMBEDDING",
            SpanAttributes.EMBEDDING_MODEL_NAME: model_name,
            SpanAttributes.EMBEDDING_TEXT: text_input[:500] if text_input else "",  # Truncate long text
            SpanAttributes.INPUT_VALUE: text_input,
            SpanAttributes.INPUT_MIME_TYPE: "text/plain",
            **extra_attributes
        }
        
        timing_data = TimingData()
        
        try:
            async with self.telemetry.create_async_span(
                span_name,
                attributes=self._sanitize_span_attributes(span_attributes),
                kind=SpanKind.CLIENT if OTEL_AVAILABLE else None
            ) as span:
                try:
                    yield timing_data
                    self.telemetry.set_span_status_safely(span, True)
                except Exception as e:
                    self.telemetry.set_span_status_safely(span, False, str(e))
                    self.telemetry.record_exception_safely(span, e)
                    raise
        except Exception:
            raise
    
    # ============================================================================
    # RETRIEVER SPAN OBSERVABILITY  
    # ============================================================================
    
    @asynccontextmanager
    async def observe_retrieval_operation(
        self,
        query: str,
        operation_name: str = "retrieve",
        collection_name: str = "",
        **extra_attributes
    ):
        """Observe a document retrieval operation.
        
        Args:
            query: Search query
            operation_name: Name of the retrieval operation
            collection_name: Name of the document collection
            **extra_attributes: Additional span attributes
        """
        if not self.is_enabled():
            yield TimingData()
            return
        
        span_name = f"retriever.{operation_name}"
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "RETRIEVER",
            SpanAttributes.INPUT_VALUE: query,
            SpanAttributes.INPUT_MIME_TYPE: "text/plain",
            "retriever.collection_name": collection_name,
            **extra_attributes
        }
        
        timing_data = TimingData()
        
        try:
            async with self.telemetry.create_async_span(
                span_name,
                attributes=self._sanitize_span_attributes(span_attributes),
                kind=SpanKind.CLIENT if OTEL_AVAILABLE else None
            ) as span:
                try:
                    yield timing_data
                    self.telemetry.set_span_status_safely(span, True)
                except Exception as e:
                    self.telemetry.set_span_status_safely(span, False, str(e))
                    self.telemetry.record_exception_safely(span, e)
                    raise
        except Exception:
            raise
    
    # ============================================================================
    # TOOL SPAN OBSERVABILITY
    # ============================================================================
    
    @asynccontextmanager
    async def observe_tool_call(
        self,
        tool_name: str,
        tool_description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        **extra_attributes
    ):
        """Observe a tool/function call operation.
        
        Args:
            tool_name: Name of the tool being called
            tool_description: Description of what the tool does
            parameters: Tool parameters/arguments
            **extra_attributes: Additional span attributes
        """
        if not self.is_enabled():
            yield TimingData()
            return
        
        span_name = f"tool.{tool_name}"
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
            SpanAttributes.TOOL_NAME: tool_name,
            SpanAttributes.TOOL_DESCRIPTION: tool_description,
            SpanAttributes.TOOL_PARAMETERS: str(parameters or {}),
            SpanAttributes.INPUT_VALUE: str(parameters or {}),
            SpanAttributes.INPUT_MIME_TYPE: "application/json",
            **extra_attributes
        }
        
        timing_data = TimingData()
        
        try:
            async with self.telemetry.create_async_span(
                span_name,
                attributes=self._sanitize_span_attributes(span_attributes),
                kind=SpanKind.CLIENT if OTEL_AVAILABLE else None
            ) as span:
                try:
                    yield timing_data
                    self.telemetry.set_span_status_safely(span, True)
                except Exception as e:
                    self.telemetry.set_span_status_safely(span, False, str(e))
                    self.telemetry.record_exception_safely(span, e)
                    raise
        except Exception:
            raise
    
    # ============================================================================
    # RERANKER SPAN OBSERVABILITY
    # ============================================================================
    
    @asynccontextmanager
    async def observe_reranking_operation(
        self,
        query: str,
        model_name: str = "",
        top_k: int = 0,
        operation_name: str = "rerank",
        **extra_attributes
    ):
        """Observe a document reranking operation.
        
        Args:
            query: Query used for reranking
            model_name: Name of the reranking model
            top_k: Number of top documents to return
            operation_name: Name of the reranking operation
            **extra_attributes: Additional span attributes
        """
        if not self.is_enabled():
            yield TimingData()
            return
        
        span_name = f"reranker.{operation_name}"
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "RERANKER",
            SpanAttributes.RERANKER_MODEL_NAME: model_name,
            SpanAttributes.RERANKER_QUERY: query,
            SpanAttributes.RERANKER_TOP_K: top_k,
            SpanAttributes.INPUT_VALUE: query,
            SpanAttributes.INPUT_MIME_TYPE: "text/plain",
            **extra_attributes
        }
        
        timing_data = TimingData()
        
        try:
            async with self.telemetry.create_async_span(
                span_name,
                attributes=self._sanitize_span_attributes(span_attributes),
                kind=SpanKind.CLIENT if OTEL_AVAILABLE else None
            ) as span:
                try:
                    yield timing_data
                    self.telemetry.set_span_status_safely(span, True)
                except Exception as e:
                    self.telemetry.set_span_status_safely(span, False, str(e))
                    self.telemetry.record_exception_safely(span, e)
                    raise
        except Exception:
            raise
    
    # ============================================================================
    # AGENT SPAN OBSERVABILITY
    # ============================================================================
    
    @asynccontextmanager
    async def observe_agent_operation(
        self,
        agent_name: str,
        operation_name: str = "process",
        input_message: str = "",
        **extra_attributes
    ):
        """Observe an agent operation.
        
        Args:
            agent_name: Name of the agent
            operation_name: Name of the agent operation
            input_message: Input message to the agent
            **extra_attributes: Additional span attributes
        """
        if not self.is_enabled():
            yield TimingData()
            return
        
        span_name = f"agent.{agent_name}.{operation_name}"
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "AGENT",
            "agent.name": agent_name,
            "agent.operation": operation_name,
            SpanAttributes.INPUT_VALUE: input_message,
            SpanAttributes.INPUT_MIME_TYPE: "text/plain",
            **extra_attributes
        }
        
        timing_data = TimingData()
        
        try:
            async with self.telemetry.create_async_span(
                span_name,
                attributes=self._sanitize_span_attributes(span_attributes),
                kind=SpanKind.CLIENT if OTEL_AVAILABLE else None
            ) as span:
                try:
                    yield timing_data
                    self.telemetry.set_span_status_safely(span, True)
                except Exception as e:
                    self.telemetry.set_span_status_safely(span, False, str(e))
                    self.telemetry.record_exception_safely(span, e)
                    raise
        except Exception:
            raise
    
    # ============================================================================
    # CHAIN SPAN OBSERVABILITY (General Operations)
    # ============================================================================
    
    @asynccontextmanager
    async def observe_chain_operation(
        self,
        operation_name: str,
        input_data: Any = None,
        **extra_attributes
    ):
        """Observe a general chain/workflow operation.
        
        Args:
            operation_name: Name of the operation
            input_data: Input data to the operation
            **extra_attributes: Additional span attributes
        """
        if not self.is_enabled():
            yield TimingData()
            return
        
        span_name = f"chain.{operation_name}"
        input_str = str(input_data) if input_data is not None else ""
        
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN",
            "chain.operation": operation_name,
            SpanAttributes.INPUT_VALUE: input_str,
            SpanAttributes.INPUT_MIME_TYPE: "application/json" if input_data else "text/plain",
            **extra_attributes
        }
        
        timing_data = TimingData()
        
        try:
            async with self.telemetry.create_async_span(
                span_name,
                attributes=self._sanitize_span_attributes(span_attributes),
                kind=SpanKind.CLIENT if OTEL_AVAILABLE else None
            ) as span:
                try:
                    yield timing_data
                    self.telemetry.set_span_status_safely(span, True)
                except Exception as e:
                    self.telemetry.set_span_status_safely(span, False, str(e))
                    self.telemetry.record_exception_safely(span, e)
                    raise
        except Exception:
            raise
    
    def _create_span_attributes(
        self,
        provider: str,
        model: str,
        method_name: str,
        **extra_attributes
    ) -> Dict[str, Any]:
        """Create standardized span attributes for LLM calls."""
        # Standard span attributes following OpenInference and OTEL conventions
        attributes = {
            "llm.provider": provider,
            "llm.model_name": model,
            "llm.method": method_name,
            "llm.request.model": model,
            "llm.system": provider,
            # Elastic APM transaction type
            "span.type": "llm",
            # Phoenix/OpenInference compatibility
            SpanAttributes.OPENINFERENCE_SPAN_KIND: self.config.span_kind.value if hasattr(self.config.span_kind, 'value') else self.config.span_kind,
        }
        
        # OpenInference standard model name attribute
        attributes[SpanAttributes.LLM_MODEL_NAME] = model if model else ""
        
        # Add streaming indicator if present
        if extra_attributes.get("llm.streaming"):
            attributes["llm.streaming"] = True
        
        # Add package-specific attributes
        attributes.update(self.config.get_span_attributes())
        
        # Add custom attributes
        attributes.update(extra_attributes)
        
        # Clean up None values
        return {k: v for k, v in attributes.items() if v is not None}
    
    def _sanitize_span_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize span attributes to ensure OTEL compatibility.
        
        OTEL only accepts: bool, str, bytes, int, float, or sequences of those types.
        """
        sanitized = {}
        for key, value in attributes.items():
            if value is None:
                continue
            elif isinstance(value, (bool, str, bytes, int, float)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert sequences to lists of primitive types
                try:
                    sanitized[key] = [self._to_primitive_type(item) for item in value]
                except (TypeError, ValueError):
                    sanitized[key] = str(value)
            else:
                # Convert complex objects to strings
                sanitized[key] = str(value)
        
        return sanitized
    
    def _to_primitive_type(self, value: Any) -> Union[bool, str, bytes, int, float]:
        """Convert a value to a primitive OTEL-compatible type."""
        if isinstance(value, (bool, str, bytes, int, float)):
            return value
        else:
            return str(value)
    
    def _create_metrics_attributes(self, span_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Create metrics-compatible attributes (only primitive hashable types)."""
        metrics_attrs = {}
        for key, value in span_attributes.items():
            if isinstance(value, (str, int, float, bool)):
                metrics_attrs[key] = value
            else:
                # Convert complex types to strings for metrics
                metrics_attrs[key] = str(value)
        return metrics_attrs
    
    def _update_span_with_timing(self, span, timing_data: TimingData):
        """Update span with timing and usage data using OpenInference standards."""
        if span is None or not timing_data:
            return
        
        try:
            # OpenInference Standard Token Counts (with null handling)
            prompt_tokens = timing_data.input_tokens if timing_data.input_tokens > 0 else 0
            completion_tokens = timing_data.output_tokens if timing_data.output_tokens > 0 else 0
            total_tokens = timing_data.total_tokens if timing_data.total_tokens > 0 else 0
            
            self.telemetry.set_span_attributes_safely(span, {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: total_tokens,
            })
            
            # Custom timing attributes (Arshai-specific)
            timing_attrs = {}
            if timing_data.time_to_first_token is not None:
                timing_attrs["llm.time_to_first_token"] = timing_data.time_to_first_token
            else:
                timing_attrs["llm.time_to_first_token"] = 0.0
            
            if timing_data.time_to_last_token is not None:
                timing_attrs["llm.time_to_last_token"] = timing_data.time_to_last_token
            else:
                timing_attrs["llm.time_to_last_token"] = 0.0
            
            timing_attrs["llm.total_duration"] = timing_data.total_duration if timing_data.total_duration else 0.0
            
            self.telemetry.set_span_attributes_safely(span, timing_attrs)
            
            # OpenInference Input/Output Messages (with null handling)
            input_messages = []
            if hasattr(timing_data, 'input_messages') and timing_data.input_messages:
                input_messages = timing_data.input_messages
            elif hasattr(timing_data, 'input_value') and timing_data.input_value:
                # Convert simple input to message format
                input_messages = [{"role": "user", "content": str(timing_data.input_value)}]
            
            output_messages = []
            if hasattr(timing_data, 'output_messages') and timing_data.output_messages:
                output_messages = timing_data.output_messages
            elif hasattr(timing_data, 'output_value') and timing_data.output_value:
                # Convert simple output to message format
                output_messages = [{"role": "assistant", "content": str(timing_data.output_value)}]
            
            # Set OpenInference message attributes (use empty lists if no data)
            self.telemetry.set_span_attributes_safely(span, {
                SpanAttributes.LLM_INPUT_MESSAGES: str(input_messages) if input_messages else "[]",
                SpanAttributes.LLM_OUTPUT_MESSAGES: str(output_messages) if output_messages else "[]",
            })
            
            # OpenInference Function Call (with null handling)
            function_call_data = ""
            if hasattr(timing_data, 'function_call') and timing_data.function_call:
                function_call_data = str(timing_data.function_call)
            
            self.telemetry.set_span_attributes_safely(span, {
                SpanAttributes.LLM_FUNCTION_CALL: function_call_data
            })
            
            # OpenInference Invocation Parameters (with null handling)
            invocation_params = {}
            if hasattr(timing_data, 'invocation_parameters') and timing_data.invocation_parameters:
                invocation_params = timing_data.invocation_parameters
            
            self.telemetry.set_span_attributes_safely(span, {
                SpanAttributes.LLM_INVOCATION_PARAMETERS: str(invocation_params) if invocation_params else "{}"
            })
            
            # OpenInference Prompt Template (with null handling)
            prompt_template = ""
            prompt_template_vars = ""
            prompt_template_version = ""
            
            if hasattr(timing_data, 'prompt_template'):
                prompt_template = str(timing_data.prompt_template) if timing_data.prompt_template else ""
            
            if hasattr(timing_data, 'prompt_template_variables'):
                prompt_template_vars = str(timing_data.prompt_template_variables) if timing_data.prompt_template_variables else "{}"
            
            if hasattr(timing_data, 'prompt_template_version'):
                prompt_template_version = str(timing_data.prompt_template_version) if timing_data.prompt_template_version else ""
            
            self.telemetry.set_span_attributes_safely(span, {
                SpanAttributes.LLM_PROMPT_TEMPLATE: prompt_template,
                SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: prompt_template_vars,
                SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION: prompt_template_version,
            })
            
            # Content attributes (if enabled and safe) - Phoenix compatibility
            if self.config.should_log_content("prompts") and hasattr(timing_data, 'input_value') and timing_data.input_value is not None:
                content = str(timing_data.input_value)
                # Set both OpenInference and custom attributes for compatibility
                self.telemetry.set_span_attributes_safely(span, {
                    "input.value": content,        # OpenInference/Phoenix standard
                    "llm.input.content": content   # Custom attribute for other platforms
                })
            else:
                # Set empty string if not logging content
                self.telemetry.set_span_attributes_safely(span, {
                    "input.value": "",
                    "llm.input.content": ""
                })
            
            if self.config.should_log_content("responses") and hasattr(timing_data, 'output_value') and timing_data.output_value is not None:
                content = str(timing_data.output_value)
                # Set both OpenInference and custom attributes for compatibility
                self.telemetry.set_span_attributes_safely(span, {
                    "output.value": content,        # OpenInference/Phoenix standard
                    "llm.output.content": content   # Custom attribute for other platforms
                })
            else:
                # Set empty string if not logging content
                self.telemetry.set_span_attributes_safely(span, {
                    "output.value": "",
                    "llm.output.content": ""
                })
            
        except Exception as e:
            self.logger.warning(f"Failed to update span with timing data: {e}")
    
    def _record_request_start(self, attributes: Dict[str, Any]):
        """Record the start of an LLM request."""
        try:
            if "llm_requests_total" in self._metrics:
                self._metrics["llm_requests_total"].add(1, attributes)
            
            if "llm_active_requests" in self._metrics:
                self._metrics["llm_active_requests"].add(1, attributes)
                
        except Exception as e:
            self.logger.warning(f"Failed to record request start metrics: {e}")
    
    def _record_request_end(
        self,
        attributes: Dict[str, Any],
        timing_data: TimingData,
        success: bool
    ):
        """Record the completion of an LLM request."""
        try:
            # Update active requests
            if "llm_active_requests" in self._metrics:
                self._metrics["llm_active_requests"].add(-1, attributes)
            
            if not success:
                if "llm_requests_failed" in self._metrics:
                    self._metrics["llm_requests_failed"].add(1, attributes)
                return
            
            # Record timing metrics
            if "llm_time_to_first_token" in self._metrics and timing_data.time_to_first_token is not None:
                self._metrics["llm_time_to_first_token"].record(
                    timing_data.time_to_first_token, attributes
                )
            
            if "llm_time_to_last_token" in self._metrics and timing_data.time_to_last_token is not None:
                self._metrics["llm_time_to_last_token"].record(
                    timing_data.time_to_last_token, attributes
                )
            
            if "llm_request_duration" in self._metrics:
                self._metrics["llm_request_duration"].record(
                    timing_data.total_duration, attributes
                )
            
            # Record token metrics
            if "llm_input_tokens" in self._metrics and timing_data.input_tokens > 0:
                self._metrics["llm_input_tokens"].add(timing_data.input_tokens, attributes)
            
            if "llm_output_tokens" in self._metrics and timing_data.output_tokens > 0:
                self._metrics["llm_output_tokens"].add(timing_data.output_tokens, attributes)
            
            if "llm_tokens_total" in self._metrics and timing_data.total_tokens > 0:
                self._metrics["llm_tokens_total"].add(timing_data.total_tokens, attributes)
                
        except Exception as e:
            self.logger.warning(f"Failed to record request end metrics: {e}")
    
    def _determine_openinference_kind(self, extra_attributes: Dict[str, Any]) -> str:
        """Determine the OpenInference span kind based on operation characteristics.
        
        Args:
            extra_attributes: Additional attributes passed to the span
        
        Returns:
            OpenInference span kind string
        """
        # Check for tool/function usage
        if extra_attributes.get("has_tools") or extra_attributes.get("has_background_tasks"):
            return "AGENT"
        
        # Check for structured output/extraction
        if extra_attributes.get("structure_requested"):
            return "LLM"  # Could also be "EXTRACTION" but LLM is more generic
        
        # Default to LLM for standard chat/completion calls
        return "LLM"
    
    async def record_usage_data(self, timing_data: TimingData, usage_data: Dict[str, Any]):
        """Record usage data from LLM response.
        
        Args:
            timing_data: TimingData instance to update
            usage_data: Usage data from LLM response
        """
        if not usage_data:
            return
        
        try:
            # Update timing data with usage information
            timing_data.update_token_counts(
                input_tokens=usage_data.get('input_tokens', 0),
                output_tokens=usage_data.get('output_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0),
                thinking_tokens=usage_data.get('thinking_tokens', 0),
                tool_calling_tokens=usage_data.get('tool_calling_tokens', 0)
            )
            
            # Add cost information if available and enabled
            if self.config.track_cost_metrics:
                if 'cost' in usage_data:
                    timing_data.total_cost = usage_data['cost']
                if 'prompt_cost' in usage_data:
                    timing_data.prompt_cost = usage_data['prompt_cost']
                if 'completion_cost' in usage_data:
                    timing_data.completion_cost = usage_data['completion_cost']
            
        except Exception as e:
            self.logger.warning(f"Failed to record usage data: {e}")
    
    # ============================================================================
    # HELPER METHODS FOR SETTING OUTPUT DATA
    # ============================================================================
    
    def set_embedding_output(self, timing_data: TimingData, embedding_vector: list):
        """Set embedding output data on timing_data for span attributes."""
        try:
            # Store embedding vector (truncated for observability)
            vector_str = str(embedding_vector[:10]) + "..." if len(embedding_vector) > 10 else str(embedding_vector)
            timing_data.output_value = vector_str
            # Store full vector for specialized embedding attribute
            if hasattr(timing_data, 'embedding_vector'):
                timing_data.embedding_vector = embedding_vector
        except Exception as e:
            self.logger.warning(f"Failed to set embedding output: {e}")
    
    def set_retrieval_output(self, timing_data: TimingData, documents: list):
        """Set retrieval output data on timing_data for span attributes."""
        try:
            # Store document summary for output.value
            doc_count = len(documents) if documents else 0
            timing_data.output_value = f"Retrieved {doc_count} documents"
            
            # Store documents for specialized retrieval attributes
            if hasattr(timing_data, 'retrieval_documents'):
                timing_data.retrieval_documents = documents
        except Exception as e:
            self.logger.warning(f"Failed to set retrieval output: {e}")
    
    def set_tool_output(self, timing_data: TimingData, tool_result: Any):
        """Set tool output data on timing_data for span attributes.""" 
        try:
            result_str = str(tool_result)
            # Truncate long outputs
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "..."
            timing_data.output_value = result_str
        except Exception as e:
            self.logger.warning(f"Failed to set tool output: {e}")
    
    def set_reranking_output(self, timing_data: TimingData, reranked_documents: list):
        """Set reranking output data on timing_data for span attributes."""
        try:
            doc_count = len(reranked_documents) if reranked_documents else 0
            timing_data.output_value = f"Reranked {doc_count} documents"
            
            # Store reranked documents for specialized attributes
            if hasattr(timing_data, 'reranked_documents'):
                timing_data.reranked_documents = reranked_documents
        except Exception as e:
            self.logger.warning(f"Failed to set reranking output: {e}")
    
    def set_agent_output(self, timing_data: TimingData, agent_response: str):
        """Set agent output data on timing_data for span attributes."""
        try:
            timing_data.output_value = str(agent_response)
        except Exception as e:
            self.logger.warning(f"Failed to set agent output: {e}")
    
    def set_chain_output(self, timing_data: TimingData, chain_result: Any):
        """Set chain/workflow output data on timing_data for span attributes."""
        try:
            timing_data.output_value = str(chain_result)
        except Exception as e:
            self.logger.warning(f"Failed to set chain output: {e}")


# Global instance for package-wide use
_default_observability: Optional[ArshaiObservability] = None


def get_llm_observability(config: Optional[PackageObservabilityConfig] = None) -> ArshaiObservability:
    """Get or create the default observability instance (backwards compatibility).
    
    Args:
        config: Optional configuration (only used for first creation)
    
    Returns:
        ArshaiObservability instance
    """
    return get_observability(config)


def get_observability(config: Optional[PackageObservabilityConfig] = None) -> ArshaiObservability:
    """Get or create the default Arshai observability instance.
    
    Args:
        config: Optional configuration (only used for first creation)
    
    Returns:
        ArshaiObservability instance supporting all span types
    """
    global _default_observability
    
    if _default_observability is None:
        _default_observability = ArshaiObservability(config)
    
    return _default_observability


def reset_llm_observability():
    """Reset the global observability instance (backwards compatibility)."""
    reset_observability()


def reset_observability():
    """Reset the global Arshai observability instance (useful for testing)."""
    global _default_observability
    _default_observability = None


# Backwards compatibility alias
LLMObservability = ArshaiObservability