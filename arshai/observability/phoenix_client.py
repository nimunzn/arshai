"""Phoenix AI Observability client integration."""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# Phoenix imports with fallback
try:
    import phoenix as px
    from phoenix.trace import TraceProvider
    from phoenix.trace.opentelemetry import OpenTelemetryInstrumentor
    from phoenix.trace.llm import LLMInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    logging.warning("Phoenix not available. Install with: pip install arize-phoenix")

# OpenInference semantic conventions
try:
    from openinference.semconv.trace import (
        SpanAttributes,
        MessageAttributes,
        ToolCallAttributes,
        DocumentAttributes,
        EmbeddingAttributes,
    )
    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False
    logging.warning("OpenInference not available. Install with: pip install openinference-semantic-conventions")

from .config import ObservabilityConfig


class PhoenixClient:
    """Client for Phoenix AI Observability integration."""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = PHOENIX_AVAILABLE and config.phoenix_enabled
        self._trace_provider: Optional[TraceProvider] = None
        
        if self.enabled:
            self._initialize_phoenix()
    
    def _initialize_phoenix(self):
        """Initialize Phoenix AI observability."""
        try:
            # Configure Phoenix endpoint
            phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
            
            # Initialize Phoenix
            px.launch_app(
                host="0.0.0.0",
                port=6006,
                notebook_env="none",  # We're not in a notebook
                collect_stats=True,
            )
            
            # Set up OpenTelemetry integration
            if self.config.otlp_endpoint:
                # Configure OTLP export to Phoenix
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = phoenix_endpoint
                os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api-key=phoenix"
            
            # Initialize instrumentors
            if OPENINFERENCE_AVAILABLE:
                self._setup_instrumentors()
            
            self.logger.info("Phoenix AI observability initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Phoenix: {e}")
            self.enabled = False
    
    def _setup_instrumentors(self):
        """Set up OpenInference instrumentors for various LLM providers."""
        try:
            # Generic LLM instrumentor
            llm_instrumentor = LLMInstrumentor()
            llm_instrumentor.instrument()
            
            # Provider-specific instrumentors can be added here
            # Example: OpenAI, Anthropic, etc.
            
            self.logger.info("Phoenix instrumentors configured")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup instrumentors: {e}")
    
    def create_span_attributes(self, 
                              provider: str,
                              model: str,
                              messages: Optional[List[Dict[str, Any]]] = None,
                              **kwargs) -> Dict[str, Any]:
        """Create OpenInference-compliant span attributes.
        
        Args:
            provider: LLM provider name
            model: Model name
            messages: Input messages
            **kwargs: Additional attributes
            
        Returns:
            Dictionary of span attributes
        """
        if not OPENINFERENCE_AVAILABLE:
            return {}
        
        attributes = {
            SpanAttributes.LLM_PROVIDER: provider,
            SpanAttributes.LLM_MODEL_NAME: model,
        }
        
        # Add message attributes if provided
        if messages:
            attributes[SpanAttributes.LLM_INPUT_MESSAGES] = self._format_messages(messages)
        
        # Add invocation parameters
        if "temperature" in kwargs:
            attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] = {
                "temperature": kwargs["temperature"]
            }
        
        # Add token usage if available
        if "usage" in kwargs:
            usage = kwargs["usage"]
            attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = usage.get("prompt_tokens", 0)
            attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = usage.get("completion_tokens", 0)
            attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = usage.get("total_tokens", 0)
        
        return attributes
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages according to OpenInference conventions."""
        if not OPENINFERENCE_AVAILABLE:
            return messages
        
        formatted = []
        for msg in messages:
            formatted_msg = {
                MessageAttributes.MESSAGE_ROLE: msg.get("role"),
                MessageAttributes.MESSAGE_CONTENT: msg.get("content"),
            }
            
            # Add tool calls if present
            if "tool_calls" in msg:
                formatted_msg[MessageAttributes.MESSAGE_TOOL_CALLS] = [
                    {
                        ToolCallAttributes.TOOL_CALL_ID: tc.get("id"),
                        ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: tc.get("function", {}).get("name"),
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS: tc.get("function", {}).get("arguments"),
                    }
                    for tc in msg["tool_calls"]
                ]
            
            formatted.append(formatted_msg)
        
        return formatted
    
    @contextmanager
    def trace_llm_call(self, 
                      provider: str,
                      model: str,
                      operation: str = "completion",
                      **kwargs):
        """Context manager for tracing LLM calls with Phoenix.
        
        Args:
            provider: LLM provider name
            model: Model name
            operation: Type of operation (completion, chat, embedding)
            **kwargs: Additional trace attributes
        """
        if not self.enabled:
            yield
            return
        
        # Create span attributes
        attributes = self.create_span_attributes(provider, model, **kwargs)
        
        # Phoenix will automatically capture the trace through OpenTelemetry
        # The actual tracing is handled by the OpenTelemetry integration
        yield
    
    def log_evaluation(self,
                      span_id: str,
                      evaluation_name: str,
                      score: float,
                      explanation: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Log an evaluation result to Phoenix.
        
        Args:
            span_id: The span ID to attach the evaluation to
            evaluation_name: Name of the evaluation metric
            score: Evaluation score
            explanation: Optional explanation for the score
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            # Phoenix evaluation logging
            # This would typically be sent to Phoenix's evaluation endpoint
            evaluation_data = {
                "span_id": span_id,
                "name": evaluation_name,
                "score": score,
                "explanation": explanation,
                "metadata": metadata or {}
            }
            
            # Log to Phoenix (implementation depends on Phoenix SDK version)
            self.logger.debug(f"Logged evaluation: {evaluation_data}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log evaluation: {e}")
    
    def get_trace_url(self, trace_id: str) -> Optional[str]:
        """Get the Phoenix UI URL for a specific trace.
        
        Args:
            trace_id: The trace ID
            
        Returns:
            URL to view the trace in Phoenix UI
        """
        if not self.enabled:
            return None
        
        phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
        return f"{phoenix_endpoint}/traces/{trace_id}"
    
    def shutdown(self):
        """Shutdown Phoenix client."""
        if self.enabled:
            try:
                # Cleanup Phoenix resources
                self.logger.info("Phoenix client shutdown completed")
            except Exception as e:
                self.logger.warning(f"Error during Phoenix shutdown: {e}")