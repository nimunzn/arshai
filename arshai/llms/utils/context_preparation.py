"""
Context preparation utilities for LLM inputs.

Provides generic instruction building that can be reused across different LLM providers.
Conversation context building remains provider-specific.
"""

from typing import Dict, Optional


def build_enhanced_instructions(
    structure_type: Optional[type] = None,
    background_tasks: Optional[Dict] = None
) -> str:
    """
    Build enhanced context instructions for LLM inputs.
    
    This creates generic instructions that work across providers for:
    - Structured output requirements
    - Background task availability
    
    Args:
        structure_type: Optional structure type for JSON output instructions
        background_tasks: Optional background tasks dictionary
        
    Returns:
        Enhanced instructions string
    """
    enhanced_instructions = ""
    
    # Add structured output instructions
    if structure_type is not None:
        enhanced_instructions += "\n\nProvide your response as structured JSON matching the expected format."
    
    # Add background tasks instructions
    if background_tasks and len(background_tasks) > 0:
        background_task_names = list(background_tasks.keys())
        enhanced_instructions += (
            f"\n\nIMPORTANT: You have access to background tasks that run independently: "
            f"{', '.join(background_task_names)}. These tasks execute in fire-and-forget mode "
            "and don't return results to the conversation. Use them when appropriate for "
            "logging, notifications, or other background operations."
        )
    
    return enhanced_instructions