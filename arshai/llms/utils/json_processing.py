"""
JSON processing utilities for LLM responses.

Provides efficient JSON completion checking and fixing for streaming
JSON responses from language models.
"""

import json
import re
from typing import Tuple


def is_json_complete(json_str: str) -> Tuple[bool, str]:
    """
    Efficient JSON completion checker using state machine approach.
    Optimized for streaming JSON parsing with minimal overhead.
    
    Args:
        json_str: JSON string to check for completeness
        
    Returns:
        Tuple of (is_complete, fixed_json_string)
    """
    json_str = json_str.strip()
    if not json_str or not json_str.startswith('{'):
        return False, json_str
        
    # Fast path: try parsing as-is first
    try:
        json.loads(json_str)
        return True, json_str
    except json.JSONDecodeError:
        pass
    
    # Single-pass state analysis
    in_string = False
    escape_next = False
    brace_depth = 0
    bracket_depth = 0
    last_structural = None
    last_non_ws = None
    has_content = False
    
    for char in json_str:
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"':
            in_string = not in_string
            has_content = True
        elif not in_string:
            if char == '{':
                brace_depth += 1
                last_structural = char
            elif char == '}':
                brace_depth -= 1
                last_structural = char
            elif char == '[':
                bracket_depth += 1
                last_structural = char
            elif char == ']':
                bracket_depth -= 1
                last_structural = char
            elif char in ',:':
                last_structural = char
            elif char == ':':
                has_content = True
                
        if not char.isspace():
            last_non_ws = char
    
    # Early rejection for malformed JSON
    if brace_depth < 0 or bracket_depth < 0:
        return False, json_str
        
    # Quick completion check
    if not (has_content and (
        (brace_depth == 0 and bracket_depth == 0) or  # Already complete
        (in_string and last_structural in ',:') or     # Incomplete string
        (not in_string and last_non_ws and last_non_ws not in '{[,:') or  # Complete value
        (brace_depth > 0 and len(json_str) > 30)      # Substantial incomplete
    )):
        return False, json_str
        
    # Progressive completion
    return fix_incomplete_json(json_str, in_string, brace_depth, bracket_depth)


def fix_incomplete_json(json_str: str, in_string: bool = None, brace_depth: int = None, bracket_depth: int = None) -> Tuple[bool, str]:
    """
    Optimized JSON completion with single validation.
    
    Args:
        json_str: JSON string to fix
        in_string: Whether currently inside a string (if known)
        brace_depth: Current brace nesting depth (if known)  
        bracket_depth: Current bracket nesting depth (if known)
        
    Returns:
        Tuple of (is_valid, fixed_json_string)
    """
    # If state not provided, calculate it
    if any(x is None for x in [in_string, brace_depth, bracket_depth]):
        in_string, brace_depth, bracket_depth = _calculate_json_state(json_str)
    
    fixed = json_str
    
    # Apply fixes in order
    if in_string:
        fixed += '"'
    fixed = fixed.rstrip()
    if fixed.endswith(','):
        fixed = fixed[:-1]
    if bracket_depth > 0:
        fixed += ']' * bracket_depth
    if brace_depth > 0:
        fixed += '}' * brace_depth
        
    # Single validation attempt
    try:
        json.loads(fixed)
        return True, fixed
    except json.JSONDecodeError:
        # Fallback: find last complete key-value pair
        pattern = r'"[^"]+"\s*:\s*(?:"[^"]*"|(?:true|false|null|\d+(?:\.\d+)?))(?:\s*,\s*)?'
        matches = list(re.finditer(pattern, json_str))
        
        if matches:
            try:
                truncated = json_str[:matches[-1].end()].rstrip().rstrip(',')
                truncated += ']' * bracket_depth + '}' * brace_depth
                json.loads(truncated)
                return True, truncated
            except (json.JSONDecodeError, re.error):
                pass
                
    return False, json_str


def _calculate_json_state(json_str: str) -> Tuple[bool, int, int]:
    """
    Calculate the current state of JSON parsing.
    
    Returns:
        Tuple of (in_string, brace_depth, bracket_depth)
    """
    in_string = False
    escape_next = False
    brace_depth = 0
    bracket_depth = 0
    
    for char in json_str:
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
    
    return in_string, brace_depth, bracket_depth