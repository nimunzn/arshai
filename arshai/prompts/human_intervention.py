def HUMAN_INTERVENTION_PROMPT():
    """Prompt for handling expert handoff"""
    return """
    ### ESCALATION RULES:
    
    **Escalate when:**
    1. User asks for human/expert/operator
    2. You cannot answer their question 
    3. User is very frustrated
    4. Complex issues beyond your knowledge
    
    **How to escalate:**
    - Set `handoff_to_expert: true`
    - Say: "I'm connecting you with an expert who will help you"
    - Use same language as user
    """ 