def TOOL_USAGE_PROMPT() -> str:
    """Generate prompt for proper tool usage."""
    return """
    ### TOOL USAGE RULES:
    
    1. Use tools only when needed for specific data
    2. Choose the right tool for each task
    3. Use tool results to answer questions
    4. If tools fail, acknowledge and provide alternative help
    5. Do NOT use tools for general knowledge questions
    6. Transform technical data to user-friendly language
    """

def STRUCTURED_OUTPUT_PROMPT(response_structure: str) -> str:

    return f"""
    ### OUTPUT FORMAT REQUIRED:
    Use function calls to format responses according to this structure:
    {response_structure}

    Rules:
    1. Use the designated function for responses
    2. Include all required fields
    3. Follow the schema exactly
    """