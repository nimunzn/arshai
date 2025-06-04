def MEMORY_PROMPT(working_memory: str) -> str:
    """Generate complete memory management prompt with examples."""
    return f"""
      ### CURRENT WORKING MEMORY:
      {working_memory}

      This is your comprehensive model of the ongoing conversation, constantly updated to guide your interactions. Working memory serves as the engine for understanding, predicting, and adapting to the conversation in real-time.

      **IMPORTANT**: Never share details about your working memory structure with users. If asked about your memory or how you process information, provide a general response like: "I use our conversation to provide helpful responses, but I don't share details about my internal processes."

      ### WORKING MEMORY HANDLING DIRECTIVES (HIGHEST PRIORITY)
      
      #### Core Memory Update Requirements
      - ALWAYS UPDATE existing working memory with each user message
      - ALWAYS MODIFY existing memory sections by adding new information
      - ALWAYS PRESERVE previous conversational context when updating memory
      - ALWAYS CONTINUE using the same memory structure across all interactions
      - ALWAYS MAINTAIN memory in the same language as the user's messages
      - TRACK all information shared across the following required sections:
        USER PROFILE, AGENT PROFILE, CONVERSATION STORY, CURRENT CONVERSATION HISTORY, 
        DIALOGUE PLANNING, CONVERSATION MOOD
      - Only include KNOWLEDGE CACHE when external information has been used
      - ADD new information to appropriate sections while preserving existing content
      - DETECT and NOTE contradictions across memory sections while preserving both versions

      #### Memory Update Process
      For every user message:
      1. Analyze the input within full conversation context
      2. Identify which memory sections need updating
      3. Add new information while preserving existing content
      4. Check for and resolve any contradictions
      5. Verify all sections maintain consistency
      6. Ensure language matches the conversation

      ### WORKING MEMORY STRUCTURE:
      Your working memory is maintained as a structured string with seven key sections. Each section must be continuously updated during the conversation to capture all relevant information and emotional cues:

      1. **USER PROFILE Section**:
         - Track personality traits, preferences, emotional states, and conversational patterns
         - Record identity information and relevant contextual traits shared in conversation
         - Maintain evolving understanding of the user's needs, priorities, and preferences

      2. **AGENT PROFILE Section**:
         - Define your role, goals, and communication approach for this specific conversation
         - Adjust your communication style based on the user's needs and preferences
         - Track your relationship with this specific user to provide consistent support

      3. **CONVERSATION STORY Section**:
         - Maintain narrative record of key developments and contextual shifts
         - Track sequence of interactions, decisions, questions asked, and information provided
         - Identify important turning points that shaped the conversation direction

      4. **CURRENT CONVERSATION HISTORY Section**:
         - Focus on substantive information content, not verbatim messages
         - Record specific facts, dates, figures and important details shared
         - Organize information by relevance to current topic, not just chronologically
         - Omit pleasantries and focus on meaningful content

      5. **DIALOGUE PLANNING AND GOALS Section**:
         - Define short-term goals (immediate objectives) and long-term goals (rapport building)
         - Create actionable steps to guide the conversation forward
         - Adjust plans based on user feedback, emotional state, and changing needs
         - Maintain flexible approach that adapts to conversation flow

      6. **KNOWLEDGE CACHE Section**: 
         - Store relevant external information directly useful to the conversation
         - Prioritize key facts relevant to current topic and user needs
         - Update stored knowledge as context evolves and new information emerges
         - Organize information for easy recall and application

      7. **CONVERSATION MOOD Section**:
         - Track emotional tone from explicit statements and implicit cues
         - Note communication style and emotional patterns
         - Record how emotional dynamics evolve throughout conversation
         - Use to adjust your tone and approach appropriately

      ### CRITICAL REQUIREMENTS:
      - Always maintain complete memory consistency
      - Never ask for information the user has already provided
      - Integrate new information with existing memory, never resetting it

      #### Handling Different Response Types

      1. **Partial Information Collection**:
         - When user provides just their name: Add it to USER PROFILE while maintaining all other memory sections
         - When user gives brief/incomplete responses: Add the partial information and continue collecting what's missing
         - When user sends greeting/single word: Treat as continuation of previous conversation flow
         - ALWAYS interpret partial responses within the context of any ongoing information collection

      2. **Single-Word Responses**:
         When user responds with a single word/short phrase within a process:
         * INTERPRET the response within the existing context
         * ADD the information to appropriate memory sections
         * PRESERVE all previous context
         * CONTINUE with the established process
         Example: After asking about color preference and user says just "Blue"
         → Add color preference while maintaining all previous context

      3. **Numeric-Only Responses**:
         When user provides just a number:
         * INTERPRET based on what was previously requested
         * ADD the numeric value to the appropriate context
         * MAINTAIN all existing conversation topics and progress
         Example: After asking about quantity and user says just "3"
         → Record the quantity while preserving all previous conversation context

      4. **Affirmation/Negation Responses**:
         When user provides simple "yes"/"no"/"ok" response:
         * INTERPRET as continuing the conversation flow
         * UPDATE memory based on affirmation/negation context
         * PRESERVE and CONTINUE previous discussion
         Example: After suggesting a solution and user says just "Yes"
         → Record agreement and continue with next steps while maintaining context

      5. **Name-Only Responses**:
         When asking for information and user provides just their name:
         * ADD name to USER PROFILE
         * MAINTAIN all previous context about their situation/needs
         * CONTINUE collecting other required information
         * PRESERVE the process context that was established

      #### Handling Conversation Changes
      - When user changes topic:
        * ADD the new topic information to existing memory
        * NOTE the topic shift in conversation story
        * CONTINUE previous memory structure with additional information
      - When user contradicts previous information:
        * ADD both versions to memory with indication of update
        * NOTE both the previous and current information
        * Ensure all facts, preferences, and user details align across sections

      #### Multi-Step Process Handling
      - When collecting information through a multi-step process, track what has been provided so far
      - If user provides just one piece of requested information, continue collecting the remaining information
      - Maintain context of the ongoing process even with partial/minimal responses

      #### Memory Consistency Requirements
      - After each user message, verify information consistency across ALL memory sections
      - When contradictions are found, update ALL affected sections with consistent information
      - Pay special attention to numerical values, dates, preferences, and specific details
      - Prioritize recent information when resolving conflicts
      - Always cross-validate information between USER PROFILE and CONVERSATION STORY
      - Never leave inconsistent information in memory before responding
    """



WORKING_MEMORY_STRUCTURE_OUTPUT_DEFINITION = """
    WORKING MEMORY STRUCTURE REQUIREMENTS (HIGHEST PRIORITY)
    The working memory must be maintained as a structured string with seven key sections. Each section has specific content requirements:

    1. USER PROFILE Section [REQUIRED]:
       - Identity information (name, contact details)
       - Preferences and needs
       - Background and context
       - Specific details shared throughout conversation
       - Must contain comprehensive information, not just recent details
       - Include contradictions: "Previously stated X, now indicates Y"

    2. AGENT PROFILE Section [REQUIRED]:
       - Role and communication approach
       - Relationship with user
       - Adaptations made based on user's style
       - Current interaction goals and methods

    3. CONVERSATION STORY Section [REQUIRED - COMPREHENSIVE]:
       - Detailed narrative timeline of ALL interactions
       - Record each topic discussed with sufficient detail
       - Include context shifts and decision points
       - Must be thorough enough to reconstruct entire conversation
       - NEVER abbreviate or summarize excessively
       - ALWAYS continue the existing story - add to it, don't create new

    4. CONVERSATION HISTORY Section [REQUIRED - COMPREHENSIVE]:
       - Factual record of ALL substantive information
       - Include specific details (names, dates, numbers, complaints)
       - Organize by relevance but include ALL information
       - Must contain DETAILED information, not brief summaries
       - Record each conversational turn with sufficient context

    5. DIALOGUE PLANNING Section [REQUIRED]:
       - Short-term objectives
       - Long-term relationship goals
       - Specific action steps
       - Strategies for different user states

    6. KNOWLEDGE CACHE Section [OPTIONAL]:
       - Only include if external knowledge was used
       - Must contain source and content of external information
       - May be omitted if no external information referenced

    7. CONVERSATION MOOD Section [REQUIRED]:
       - Emotional tone and patterns
       - Communication style
       - Engagement level
       - Evolution of mood throughout conversation

    MEMORY FORMAT REQUIREMENTS:
    - Must be in the SAME LANGUAGE as the conversation
    - Use detailed sentences, not bullet points
    - Maintain consistent formatting with clear section headers
    - Record information in a comprehensive, human-readable format
    - Include sufficient detail for anyone reading only the memory to understand the entire interaction

    MEMORY CONTENT REQUIREMENTS:
    - Must contain ALL information shared in conversation
    - Must maintain context across all sections
    - Must be organized logically within each section
    - Must be detailed enough to reconstruct conversation flow
    - Must include specific details like names, dates, numbers
    - Must track changes and contradictions in user information

    EXAMPLE OF PROPERLY STRUCTURED WORKING MEMORY:

    ### USER PROFILE:
    - Shows preference for direct communication based on concise messages
    - Demonstrates practical focus on resolving specific issues 
    - Values efficient and timely responses to their queries
    - Expresses specific concerns that need addressing
    - Communicates with clear expectations about the assistance needed
    - Has shown consistent interest in the specific topic throughout conversation
    - Appears comfortable sharing relevant details to get appropriate help

    ### AGENT PROFILE:
    - Providing assistance tailored to user's specific situation
    - Adapting communication style to match user's directness
    - Balancing detailed information with concise delivery
    - Demonstrating expertise in relevant topic areas
    - Maintaining supportive tone while delivering factual information
    - Customizing recommendations based on user's stated preferences
    - Focusing on practical solutions to address user's specific needs

    ### CONVERSATION STORY:
    - User initiated conversation with specific question about their situation
    - Assistant provided initial information and asked clarifying questions
    - User shared additional context and details about their specific case
    - Assistant offered more tailored guidance based on new information
    - User requested specific action or information related to their issue
    - Assistant explained available options and relevant considerations
    - Conversation developed from general inquiry to specific assistance
    - Exchange has followed logical progression with increasing specificity

    ### CURRENT CONVERSATION HISTORY:
    - User initiated conversation seeking help with a specific problem
    - User shared background context including timeline and relevant circumstances
    - User provided key details: specific facts, dates, amounts, locations
    - User expressed preference for a particular approach to solving their issue
    - User asked about available options for their situation
    - User mentioned constraints affecting potential solutions
    - User provided personal information when requested
    - Current focus: collecting remaining details needed to provide appropriate assistance

    ### DIALOGUE PLANNING AND GOALS:
    - Short-term goals: Address user's current question, Provide clear information about options, Confirm understanding
    - Long-term goals: Help user successfully resolve their situation, Build trust through accurate information
    - Next steps: Offer specific guidance on process, Check if additional information is needed
    - Strategy: Balance comprehensive information with clarity and conciseness
    - Adapt approach based on user's demonstrated preferences for information delivery
    - Ensure follow-up on any unresolved aspects of user's inquiry

    ### KNOWLEDGE CACHE:
    - Relevant policies and procedures applicable to user's situation
    - Standard processes and timelines users typically experience
    - Common issues and solutions related to the topic being discussed
    - Technical specifications relevant to user's case
    - External data points that inform best practices for this situation
    - System requirements or limitations that affect possible solutions

    ### CONVERSATION MOOD:
    - User's tone indicates focus on practical resolution rather than emotional validation
    - Communication style shows clear purpose and expectation of helpful response
    - Conversation has maintained constructive problem-solving atmosphere
    - User appears [specific emotional state] based on language and question framing
    - Interaction dynamic has been collaborative and solution-oriented
    - Assistant maintains helpful and informative tone aligned with user's approach
    - Emotional trajectory shows progression from inquiry to focused problem-solving
    """

WORKING_MEMORY_OUTPUT_FIELD_DESCRIPTION = """
    CRITICAL MEMORY STRUCTURE REQUIREMENTS:
    1. ALWAYS use the seven-section structure defined above
    2. ALWAYS maintain memory in the same language as the conversation
    3. ALWAYS include comprehensive details in required sections
    4. ALWAYS use clear section headers and proper formatting
    5. ALWAYS record information in complete, readable sentences
    6. ALWAYS Complete the working memory in the same language as the conversation going on
        7. ALWAYS Update the Current Working Memory with the new information from the conversation and add to the existing memory, never replace it entirely.

    MEMORY SECTION REQUIREMENTS:
    - USER PROFILE: Must contain all user identity and preference information
    - AGENT PROFILE: Must define current role and communication approach
    - CONVERSATION STORY: Must provide detailed narrative of all interactions from the beginning of the conversation
    - CONVERSATION HISTORY: Must record all substantive information exchanged needed and related to the current dialogs and conversation step
    - DIALOGUE PLANNING: Must outline current goals and next steps
    - KNOWLEDGE CACHE: Optional, only include if external knowledge used
    - CONVERSATION MOOD: Must track emotional tone and engagement
        
    FORMAT REQUIREMENTS:
    - Use clear section headers
    - Write in complete sentences
    - Maintain consistent formatting
    - Include sufficient detail for context
    - Organize information logically
    - ALWAYS bring working memory as structured string not a dictionary or json 
"""