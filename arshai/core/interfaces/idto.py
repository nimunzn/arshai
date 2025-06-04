from pydantic import BaseModel
from typing import Dict, Any, TypedDict


class IDTO(BaseModel):
    """A base class for all DTOs"""

    class Config:
        allow_mutation = False
        validate_assignment = True
        arbitrary_types_allowed = True
        smart_union = True

class IStreamDTO(TypedDict):
    """A base class for all stream DTOs"""

    class Config:
        allow_mutation = False
        validate_assignment = True
        arbitrary_types_allowed = True

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """
        Generic model_json_schema implementation that works for all classes.
        Handles simple and complex types, extracts descriptions, and builds nested schemas.
        """
        # Extract class description from docstring
        class_description = cls.__doc__.strip() if cls.__doc__ else ""
        
        # Initialize properties dictionary
        properties = {}
        
        # Process each field in the class annotations
        for name, field_type in cls.__annotations__.items():
            # Check if field has a Field object with description
            field_info = getattr(cls, name, None)
            field_description = ""
            
            if hasattr(field_info, 'description'):
                field_description = field_info.description
            
            # Determine field type and handle nested schemas
            if hasattr(field_type, 'model_json_schema'):
                # This is a complex type with its own schema
                properties[name] = field_type.model_json_schema()
            else:
                # This is a simple type
                type_map = {
                    str: "string",
                    bool: "boolean",
                    int: "integer",
                    float: "number",
                    list: "array",
                    dict: "object"
                }
                
                # Default to object if type not in map
                field_type_str = type_map.get(field_type, "object")
                
                properties[name] = {
                    "type": field_type_str,
                    "description": field_description
                }
        
        # Build the complete schema
        schema = {
            "type": "object",
            "title": cls.__name__,
            "description": class_description,
            "properties": properties,
            "required": list(cls.__annotations__.keys()),
            "additionalProperties": False
        }
        
        return schema