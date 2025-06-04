def TOOL_USAGE_PROMPT() -> str:
    """Generate a prompt that enforces tool usage for all responses."""
    return """
    CRITICAL TOOL USAGE REQUIREMENTS:
    You MUST use the provided tools for ALL operations and responses.

    TOOL USAGE GUIDELINES:
    1. ALWAYS use tools for any action or response generation
    2. Select the appropriate tool based on the required action
    3. Provide all required parameters for each tool
    4. Process tool outputs properly in subsequent steps
    5. Chain multiple tools when needed for complex tasks
    
    NEVER perform operations without using the appropriate tools.
    """

def STRUCTURED_OUTPUT_PROMPT(response_structure: str) -> str:
    """Generate a prompt that enforces structured output format."""
    return f"""
    CRITICAL OUTPUT STRUCTURE REQUIREMENTS:
    You MUST use function calls to format your responses according to this structure:
    {response_structure}

    FUNCTION CALLING GUIDELINES:
    1. RESPONSE FORMATTING:
       - ALWAYS use the designated function - direct text responses are NOT allowed
       - Include ALL required fields with correct data types
       - Maintain proper nesting and follow the schema exactly
       
    2. VALIDATION STEPS:
       - Verify all required fields are present
       - Confirm field values match required constraints
       - Ensure the response is properly formatted
       
    NO EXCEPTIONS: Every response must follow this structure.
    """
