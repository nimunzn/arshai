def MEMORY_PROMPT(working_memory: str) -> str:
    """Generate optimized memory management prompt for GPT-4o-mini."""
    return f"""
You are a memory update assistant. Your task is to generate a new, updated working memory based on the previous memory and the latest turn in the conversation.

### PREVIOUS WORKING MEMORY:
{working_memory}

### YOUR TASK: RECONSTRUCT THE MEMORY
Carefully read the PREVIOUS WORKING MEMORY. Now, construct a NEW working memory by following these steps for each section:

1.  **USER CONTEXT:**
    - **First, copy the entire USER CONTEXT from the PREVIOUS memory.**
    - **Then,** add any new user details from the latest conversation turn. If there are no new details, this section will remain unchanged.

2.  **CONVERSATION FLOW:**
    - **First, copy the entire CONVERSATION FLOW from the PREVIOUS memory.**
    - **Then, append** a new entry summarizing the most recent user question and AI response. This section must always grow.

3.  **CURRENT FOCUS:**
    - **Analyze the latest conversation turn.** What is the immediate goal now? Write a new, concise CURRENT FOCUS that reflects this. This section should be rewritten to reflect the *new* immediate goal.

4.  **INTERACTION TONE:**
    - **Review the previous TONE.** Has the tone changed in the last turn? If not, copy the previous tone. If it has changed, describe the new tone.

### CRITICAL RULE:
Your final output must be ONLY the structured string for the NEW working memory. Do not lose information from the USER CONTEXT or CONVERSATION FLOW sections. Build upon the previous memory; do not start from scratch.
    """



WORKING_MEMORY_STRUCTURE_OUTPUT_DEFINITION = """
    OPTIMIZED WORKING MEMORY STRUCTURE FOR GPT-4O-MINI

    The working memory must be maintained as a structured string with FOUR key sections designed for optimal LLM processing:

    1. USER CONTEXT Section [REQUIRED]:
       - User identity, preferences, and current needs
       - Background information and specific situation details
       - Any constraints or requirements mentioned
       - Changes in user information over time

    2. CONVERSATION FLOW Section [REQUIRED]:
       - Chronological narrative of conversation events
       - Key decisions made and topics discussed
       - Important facts, dates, numbers, and specific details
       - Progress toward resolving user's needs

    3. CURRENT FOCUS Section [REQUIRED]:
       - Active topic and immediate goals
       - Planned next steps and action items
       - Outstanding questions or information needed
       - Current priorities and objectives

    4. INTERACTION TONE Section [REQUIRED]:
       - Communication style and emotional context
       - User engagement level and preferences
       - Relationship dynamics and trust level
       - Appropriate response approach

    MEMORY FORMAT REQUIREMENTS:
    - Must be in the SAME LANGUAGE as the conversation
    - Use clear section headers (### USER CONTEXT, ### CONVERSATION FLOW, etc.)
    - Write in complete, descriptive sentences
    - Maintain logical organization within each section
    - Include sufficient detail for context reconstruction

    MEMORY UPDATE GUIDELINES:
    - Add new information to existing sections rather than replacing
    - Track important changes and contradictions
    - Focus on accuracy and relevance over completeness
    - Ensure all sections work together coherently

    EXAMPLE OF PROPERLY STRUCTURED WORKING MEMORY:

    ### USER CONTEXT:
    User prefers direct communication and values efficient problem-solving. Has specific technical requirements and time constraints. Demonstrated comfort with detailed information and shows practical focus on resolving their current situation.

    ### CONVERSATION FLOW:
    User initiated conversation with specific technical question. Provided background context and requirements. Assistant offered initial guidance and requested clarifying details. User shared additional constraints and preferences. Conversation progressed toward identifying suitable solutions.

    ### CURRENT FOCUS:
    Currently evaluating options for user's technical requirements. Next steps include reviewing specific implementation details and confirming compatibility with user's constraints. Goal is to provide actionable recommendations.

    ### INTERACTION TONE:
    Professional and solution-oriented conversation. User demonstrates technical knowledge and appreciates direct responses. Collaborative approach with focus on practical outcomes. Appropriate to maintain informative and supportive tone.
    """

WORKING_MEMORY_OUTPUT_FIELD_DESCRIPTION = """
    CRITICAL MEMORY STRUCTURE REQUIREMENTS:
    1. ALWAYS use the four-section structure defined above
    2. ALWAYS maintain memory in the same language as the conversation
    3. ALWAYS include relevant details in required sections
    4. ALWAYS use clear section headers (### USER CONTEXT, ### CONVERSATION FLOW, ### CURRENT FOCUS, ### INTERACTION TONE)
    5. ALWAYS record information in complete, readable sentences
    6. ALWAYS update the existing memory with new information rather than replacing entirely

    MEMORY SECTION REQUIREMENTS:
    - USER CONTEXT: Must contain user identity, preferences, and current situation
    - CONVERSATION FLOW: Must provide chronological narrative of conversation events
    - CURRENT FOCUS: Must outline active topics, goals, and next steps
    - INTERACTION TONE: Must track communication style and emotional context
        
    FORMAT REQUIREMENTS:
    - Use clear section headers with ### formatting
    - Write in complete, descriptive sentences
    - Maintain logical organization within each section
    - Focus on accuracy and relevance over exhaustive detail
    - ALWAYS provide working memory as structured string, not dictionary or JSON
"""