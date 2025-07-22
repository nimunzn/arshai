def MEMORY_PROMPT(working_memory: str) -> str:
    """Generate optimized memory management prompt with advanced meta-reasoning capabilities."""
    return f"""
You are an advanced memory management assistant with meta-cognitive capabilities. Your task is to generate, validate, and self-correct working memory based on the previous memory and the latest conversation turn.

### PREVIOUS WORKING MEMORY:
{working_memory}

### ADVANCED MEMORY PROCESSING

#### PHASE 1: DEEP MEMORY ANALYSIS (Use thinking budget: ~200 tokens)
Before updating memory, perform meta-reasoning:

**🧠 MEMORY VALIDATION THINKING:**
- Analyze the coherence of PREVIOUS WORKING MEMORY sections
- Check for contradictions or inconsistencies within existing memory
- Identify potential gaps or missing critical information
- Assess the accuracy of CONVERSATION FLOW chronology
- Evaluate whether CURRENT FOCUS aligns with actual conversation progression

**🔍 CONVERSATION INTEGRATION ASSESSMENT:**
- Determine how the latest turn connects to existing memory
- Identify new vs. repeated information patterns
- Detect if user is correcting previous statements
- Assess emotional or contextual shifts that affect memory structure

#### PHASE 2: INTELLIGENT MEMORY RECONSTRUCTION

1.  **USER CONTEXT Advanced:
    - **Memory Continuity Check:** Verify consistency with previous USER CONTEXT
    - **Integration Logic:** Add only genuinely new user details, avoiding duplications
    - **Contradiction Resolution:** If new information contradicts existing data, prioritize the most recent and reliable information
    - **Completeness Validation:** Ensure all essential user characteristics are maintained

2.  **CONVERSATION FLOW Advanced Management:**
    - **Chronological Accuracy:** Maintain precise sequence of conversation events
    - **Semantic Coherence:** Ensure each entry builds logically on previous interactions
    - **Critical Detail Preservation:** Highlight and maintain key decisions, specific numbers, dates, and commitments
    - **Pattern Recognition:** Identify and note recurring themes or evolving user concerns

3.  **CURRENT FOCUS Dynamic Optimization:**
    - **Contextual Relevance:** Analyze what the user actually needs right now based on their latest message
    - **Priority Assessment:** Determine immediate vs. long-term goals from conversation context
    - **Action Clarity:** Define specific, actionable next steps that align with user intent
    - **Outcome Prediction:** Consider likely conversation directions to optimize focus

4.  **INTERACTION TONE Sophisticated Analysis:**
    - **Emotional Intelligence:** Detect subtle tone shifts including frustration, confusion, satisfaction, or urgency
    - **Communication Style Adaptation:** Adjust tone assessment based on cultural and linguistic cues
    - **Engagement Level:** Evaluate user's interest and participation level
    - **Relationship Dynamics:** Track trust building and rapport development

#### PHASE 3: ADVANCED MEMORY VALIDATION WITH CONVERSATION FLOW ANALYSIS (Use thinking budget: ~200-300 tokens)

**🔄 ADVANCED MEMORY QUALITY ASSURANCE:
Before finalizing, perform deep meta-reasoning validation:

1. **Internal Consistency Check:**
   - Do all four sections work together coherently?
   - Are there any logical contradictions within the memory?
   - Does the CURRENT FOCUS align with the CONVERSATION FLOW?

2. **Information Preservation Audit:**
   - Have all critical details from the previous memory been preserved?
   - Are important user preferences, constraints, or requirements maintained?
   - Has any essential context been accidentally lost?

3. **CONVERSATION FLOW COMPARATIVE ANALYSIS (NEW):**
   - **Memory vs. Conversation Reality Check:** Compare each section of your working memory against the actual conversation progression
   - **User Correction Detection:** Identify if the latest user message contradicts or corrects information in previous memory sections
   - **Chronological Accuracy Validation:** Ensure CONVERSATION FLOW section accurately reflects the actual sequence of events
   - **Context Evolution Tracking:** Check if user's situation or needs have evolved beyond what current memory reflects

4. **INTELLIGENT MEMORY CORRECTION PROTOCOL (ADVANCED):
   - **Contradiction Resolution:** When conversation flow reveals memory errors, prioritize user's most recent and explicit statements
   - **Memory Update Strategy:** Modify incorrect memory sections while preserving accurate information
   - **Evidence-Based Corrections:** Only change memory when conversation evidence clearly contradicts existing information
   - **Stability vs. Accuracy Balance:** Maintain memory coherence while ensuring factual accuracy based on conversation reality

5. **META-REASONING MEMORY VALIDATION:**
   - **User Intent Verification:** Does the memory accurately capture what the user is trying to achieve?
   - **Cultural Context Preservation:** Are cultural nuances and communication patterns properly maintained?
   - **Future Interaction Enablement:** Will this corrected memory support effective future responses?

### ADVANCED OUTPUT REQUIREMENTS:

**CRITICAL MEMORY INTEGRITY RULES:**
- Preserve all factual information from previous memory unless explicitly corrected by user
- Build incrementally rather than replacing entire sections
- Maintain the same language as the conversation
- Apply cultural and contextual sensitivity in tone assessment
- Ensure each section serves its distinct purpose while contributing to overall coherence

**ADVANCED META-REASONING INTEGRATION:"
- Let your thinking process guide memory updates with conversation flow validation
- Use analytical capabilities to detect patterns, inconsistencies, and user corrections
- Apply intelligent self-correction when conversation evidence reveals memory errors
- Consider multiple perspectives when interpreting ambiguous information
- **Memory vs. Reality Alignment:** Continuously verify that working memory reflects actual conversation progression
- **User Authority Principle:** Treat user's explicit statements as authoritative over previous assumptions or interpretations

### FINAL OUTPUT PROTOCOL:
Your response must contain ONLY the updated structured working memory string. The memory should demonstrate thoughtful integration of new information while maintaining continuity and coherence across all sections.
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