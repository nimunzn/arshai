"""
Response parsing utilities for LLM structured outputs.

Provides generic parsing and validation for structured responses
from language models.
"""

import json
from typing import Union, Type, TypeVar

T = TypeVar("T")


def parse_to_structure(content: Union[str, dict], structure_type: Type[T]) -> T:
    """
    Parse response content into the specified structure type.
    
    Args:
        content: Response content to parse (string or dict)
        structure_type: Target Pydantic model class or structure type
        
    Returns:
        Instance of the structure type
        
    Raises:
        ValueError: If parsing fails or content doesn't match structure
    """
    if isinstance(content, str):
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}")
    else:
        parsed_content = content

    try:
        return structure_type(**parsed_content)
    except Exception as e:
        raise ValueError(
            f"Failed to create {structure_type.__name__} from response: {str(e)}"
        )


def validate_structure_fields(instance: object, required_fields: list) -> bool:
    """
    Validate that a structured response has all required fields.
    
    Args:
        instance: The structured response instance
        required_fields: List of required field names
        
    Returns:
        True if all required fields are present and non-empty
    """
    for field in required_fields:
        if not hasattr(instance, field):
            return False
        value = getattr(instance, field)
        if value is None or (isinstance(value, (str, list, dict)) and len(value) == 0):
            return False
    return True


def extract_text_from_structure(instance: object, text_fields: list = None) -> str:
    """
    Extract text content from a structured response for pattern matching.
    
    Args:
        instance: The structured response instance
        text_fields: List of field names to extract text from (None for all string fields)
        
    Returns:
        Combined text content from specified fields
    """
    if text_fields is None:
        # Extract from all string and list fields
        text_parts = []
        for attr_name in dir(instance):
            if not attr_name.startswith('_'):
                value = getattr(instance, attr_name)
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    text_parts.extend(str(item) for item in value)
        return " ".join(text_parts)
    else:
        # Extract from specified fields only
        text_parts = []
        for field in text_fields:
            if hasattr(instance, field):
                value = getattr(instance, field)
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    text_parts.extend(str(item) for item in value)
                else:
                    text_parts.append(str(value))
        return " ".join(text_parts)