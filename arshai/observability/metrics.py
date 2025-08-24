"""Timing data container for LLM observability.

This module contains only the TimingData class for measuring and tracking
LLM operation timing and usage metrics.

The main metrics collection is now handled by the TelemetryManager and 
LLMObservability classes using proper OTEL patterns.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class TimingData:
    """Container for timing measurements and LLM operation metadata.
    
    This class tracks all timing-related data for LLM operations including:
    - Request timing (start, first token, last token)
    - Token usage counts (input, output, thinking, tool calling)
    - Cost tracking (if enabled)
    - OpenInference-compatible attributes for observability platforms
    """
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    
    # Token counts - using LLM client naming convention
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0
    tool_calling_tokens: int = 0
    
    # OpenInference attributes for observability platforms
    input_value: Optional[str] = None
    output_value: Optional[str] = None
    input_mime_type: str = "application/json"
    output_mime_type: str = "application/json"
    input_messages: Optional[List[Dict[str, Any]]] = None
    output_messages: Optional[List[Dict[str, Any]]] = None
    invocation_parameters: Optional[Dict[str, Any]] = None
    function_call: Optional[Dict[str, Any]] = None
    
    # Cost tracking (optional, privacy-sensitive)
    prompt_cost: Optional[float] = None
    completion_cost: Optional[float] = None
    total_cost: Optional[float] = None
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time from request start to first token in seconds.
        
        This is a key LLM performance metric indicating latency.
        """
        if self.first_token_time is not None:
            return self.first_token_time - self.start_time
        return None
    
    @property
    def time_to_last_token(self) -> Optional[float]:
        """Time from request start to last token in seconds.
        
        This indicates the total time for complete response generation.
        """
        if self.last_token_time is not None:
            return self.last_token_time - self.start_time
        return None
    
    @property
    def duration_first_to_last_token(self) -> Optional[float]:
        """Duration from first token to last token in seconds.
        
        This indicates the streaming generation speed.
        """
        if self.first_token_time is not None and self.last_token_time is not None:
            return self.last_token_time - self.first_token_time
        return None
    
    @property
    def total_duration(self) -> float:
        """Total duration from request start to completion.
        
        Uses last_token_time if available, otherwise current time.
        """
        end_time = self.last_token_time if self.last_token_time else time.time()
        return end_time - self.start_time
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens per second generation rate."""
        if self.output_tokens > 0 and self.duration_first_to_last_token is not None and self.duration_first_to_last_token > 0:
            return self.output_tokens / self.duration_first_to_last_token
        return None
    
    def record_first_token(self):
        """Record the time when the first token was received."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
    
    def record_token(self):
        """Record a token reception (updates last token time).
        
        Call this for each token received during streaming.
        """
        self.last_token_time = time.time()
        if self.first_token_time is None:
            self.first_token_time = self.last_token_time
    
    def update_token_counts(
        self, 
        input_tokens: int = 0, 
        output_tokens: int = 0, 
        total_tokens: int = 0,
        thinking_tokens: int = 0, 
        tool_calling_tokens: int = 0
    ):
        """Update token counts from LLM response usage data.
        
        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens  
            total_tokens: Total tokens used (input + output + other)
            thinking_tokens: Reasoning tokens (e.g., OpenAI o1 models)
            tool_calling_tokens: Function calling tokens
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.thinking_tokens = thinking_tokens
        self.tool_calling_tokens = tool_calling_tokens
    
    def update_cost_data(
        self,
        prompt_cost: Optional[float] = None,
        completion_cost: Optional[float] = None,
        total_cost: Optional[float] = None
    ):
        """Update cost tracking data.
        
        Args:
            prompt_cost: Cost for input/prompt tokens
            completion_cost: Cost for output/completion tokens
            total_cost: Total cost for the request
        """
        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.total_cost = total_cost
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary for export.
        
        Returns:
            Dictionary containing all timing and usage metrics
        """
        metrics = {
            'total_duration': self.total_duration,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
        }
        
        # Add optional timing metrics
        if self.time_to_first_token is not None:
            metrics['time_to_first_token'] = self.time_to_first_token
        
        if self.time_to_last_token is not None:
            metrics['time_to_last_token'] = self.time_to_last_token
        
        if self.duration_first_to_last_token is not None:
            metrics['duration_first_to_last_token'] = self.duration_first_to_last_token
        
        if self.tokens_per_second is not None:
            metrics['tokens_per_second'] = self.tokens_per_second
        
        # Add token type metrics if present
        if self.thinking_tokens > 0:
            metrics['thinking_tokens'] = self.thinking_tokens
        
        if self.tool_calling_tokens > 0:
            metrics['tool_calling_tokens'] = self.tool_calling_tokens
        
        # Add cost metrics if enabled and present
        if self.total_cost is not None:
            metrics['total_cost'] = self.total_cost
        
        if self.prompt_cost is not None:
            metrics['prompt_cost'] = self.prompt_cost
        
        if self.completion_cost is not None:
            metrics['completion_cost'] = self.completion_cost
        
        return metrics
