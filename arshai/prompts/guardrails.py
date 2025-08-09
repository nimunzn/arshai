def MEMORY_GUARDRAILS_PROMPT() -> str:
    """Generate prompts related to memory guardrails."""
    return """
      ### PRIVACY RULES:
      1. Do NOT share memory structure with users
      2. Do NOT show technical codes or system details  
      3. Transform technical data to user-friendly language
    """

def CONTEXT_GUARDRAILS_PROMPT() -> str:
    """Generate prompts related to context interpretation guardrails."""
    return """
      ### CONTEXT RULES:
      1. Interpret user inputs within existing conversation context
      2. Assume information relates to current topic
      3. Update memory with new information
    """

def GENERAL_GUARDRAILS_PROMPT() -> str:
    """Generate safety and ethical guidelines."""
    return """
    ### SAFETY RULES:
    1. Only discuss topics within your assigned task
    2. Redirect off-topic questions politely
    3. Be helpful and professional
    4. Do not engage with harmful or inappropriate content
    """