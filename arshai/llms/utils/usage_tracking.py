"""
Usage tracking utilities for LLM responses.

Provides standardized usage metadata processing and accumulation
across different LLM providers.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def standardize_usage_metadata(raw_usage_metadata: Any, provider: str = "generic") -> Dict[str, int]:
    """
    Process usage metadata from LLM responses into standardized format.
    
    Args:
        raw_usage_metadata: Raw usage metadata from LLM response
        provider: LLM provider name for provider-specific processing
        
    Returns:
        Standardized usage metadata dictionary with OpenAI-compatible keys
    """
    try:
        if not raw_usage_metadata:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Handle different provider formats
        if provider.lower() == "gemini":
            return _process_gemini_usage(raw_usage_metadata)
        elif provider.lower() in ["openai", "azure"]:
            return _process_openai_usage(raw_usage_metadata)
        else:
            # Generic processing - try common attribute names
            return _process_generic_usage(raw_usage_metadata)

    except Exception as e:
        logger.warning(f"Error processing {provider} usage metadata: {str(e)}")
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def accumulate_usage_safely(current_usage: Optional[Dict[str, int]], accumulated_usage: Optional[Dict[str, int]]) -> Dict[str, int]:
    """
    Safely accumulate usage metrics, handling None values.
    
    Args:
        current_usage: Current usage metrics to add
        accumulated_usage: Previously accumulated usage metrics
        
    Returns:
        Updated accumulated usage metrics
    """
    if accumulated_usage is None:
        # Initialize with safe values, handling None from current_usage
        return {
            "total_tokens": (current_usage.get("total_tokens") or 0) if current_usage else 0,
            "prompt_tokens": (current_usage.get("prompt_tokens") or 0) if current_usage else 0,
            "completion_tokens": (current_usage.get("completion_tokens") or 0) if current_usage else 0,
        }
    
    if current_usage is None:
        return accumulated_usage.copy()
    
    # Safely accumulate usage metrics, handling None values
    return {
        "total_tokens": accumulated_usage.get("total_tokens", 0) + (current_usage.get("total_tokens") or 0),
        "prompt_tokens": accumulated_usage.get("prompt_tokens", 0) + (current_usage.get("prompt_tokens") or 0),
        "completion_tokens": accumulated_usage.get("completion_tokens", 0) + (current_usage.get("completion_tokens") or 0),
    }


def _process_gemini_usage(raw_usage_metadata: Any) -> Dict[str, int]:
    """Process Gemini-specific usage metadata format."""
    prompt_tokens = getattr(raw_usage_metadata, "prompt_token_count", 0)
    completion_tokens = getattr(raw_usage_metadata, "candidates_token_count", 0)
    total_tokens = getattr(raw_usage_metadata, "total_token_count", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _process_openai_usage(raw_usage_metadata: Any) -> Dict[str, int]:
    """Process OpenAI/Azure-specific usage metadata format."""
    if isinstance(raw_usage_metadata, dict):
        return {
            "prompt_tokens": raw_usage_metadata.get("prompt_tokens", 0),
            "completion_tokens": raw_usage_metadata.get("completion_tokens", 0),
            "total_tokens": raw_usage_metadata.get("total_tokens", 0),
        }
    else:
        # Handle object-style access
        prompt_tokens = getattr(raw_usage_metadata, "prompt_tokens", 0)
        completion_tokens = getattr(raw_usage_metadata, "completion_tokens", 0)
        total_tokens = getattr(raw_usage_metadata, "total_tokens", 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


def _process_generic_usage(raw_usage_metadata: Any) -> Dict[str, int]:
    """Process generic usage metadata format."""
    # Try common attribute names
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    
    if isinstance(raw_usage_metadata, dict):
        prompt_tokens = raw_usage_metadata.get("prompt_tokens", 0) or raw_usage_metadata.get("input_tokens", 0)
        completion_tokens = raw_usage_metadata.get("completion_tokens", 0) or raw_usage_metadata.get("output_tokens", 0)
        total_tokens = raw_usage_metadata.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
    else:
        # Try object attribute access
        for attr_name in ["prompt_tokens", "input_tokens"]:
            if hasattr(raw_usage_metadata, attr_name):
                prompt_tokens = getattr(raw_usage_metadata, attr_name, 0)
                break
                
        for attr_name in ["completion_tokens", "output_tokens"]:
            if hasattr(raw_usage_metadata, attr_name):
                completion_tokens = getattr(raw_usage_metadata, attr_name, 0)
                break
                
        total_tokens = getattr(raw_usage_metadata, "total_tokens", prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }