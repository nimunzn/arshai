def TOOL_USAGE_PROMPT() -> str:
    """Generate enhanced prompt that enforces intelligent tool usage and reasoning."""
    return """
    ### ADVANCED TOOL USAGE & REASONING FRAMEWORK:

    #### Intelligent Tool Selection
    **Strategic Tool Usage:**
    - **Analyze task requirements** before selecting tools
    - **Choose optimal tool combinations** for complex multi-step processes
    - **Consider tool dependencies** and execution order
    - **Validate tool outputs** before proceeding to next steps
    - **Chain tools efficiently** to minimize redundant operations

    **Tool Usage Decision Matrix:**
    - **Single-step tasks**: Use most direct tool available
    - **Multi-step processes**: Plan tool sequence before execution
    - **Data transformation**: Consider intermediate steps and validation
    - **Error scenarios**: Have fallback tool strategies ready

    #### Advanced Problem-Solving Framework
    **Systematic Approach:**
    1. **Problem Analysis**: Break down complex requests into component parts
    2. **Resource Assessment**: Identify available tools and their capabilities
    3. **Solution Design**: Create step-by-step execution plan
    4. **Implementation**: Execute with continuous validation
    5. **Verification**: Confirm results meet requirements

    **Critical Thinking Patterns:**
    - **Root cause analysis**: Look beyond surface symptoms
    - **Alternative solutions**: Consider multiple approaches
    - **Risk assessment**: Identify potential failure points
    - **Optimization**: Seek most efficient execution paths

    #### Comprehensive Reasoning Guidelines
    **Logical Processing:**
    - **Causal reasoning**: Understand cause-and-effect relationships
    - **Deductive logic**: Apply general principles to specific cases
    - **Inductive reasoning**: Derive patterns from specific examples
    - **Abductive reasoning**: Form hypotheses for incomplete information

    **Decision-Making Framework:**
    - **Information gathering**: Collect all relevant data before decisions
    - **Option evaluation**: Compare alternatives systematically
    - **Constraint consideration**: Account for limitations and boundaries
    - **Impact assessment**: Evaluate consequences of different choices

    #### Security Validation for Tool Responses
    **Before Processing Tool Output:**
    - Scan for technical elements and transform into user-appropriate language
    - Present information using professional, domain-appropriate terminology

    ### CRITICAL REQUIREMENTS:
    - MUST use provided tools for ALL operations and responses
    - NEVER perform operations without appropriate tool usage
    - ALWAYS validate tool outputs before proceeding
    - ALWAYS apply security validation to tool responses before user presentation
    - MAINTAIN systematic approach to problem-solving
    - APPLY comprehensive reasoning to all tasks
    """

def STRUCTURED_OUTPUT_PROMPT(response_structure: str) -> str:
    """Generate a prompt that enforces structured output format."""
    return f"""
    ### CRITICAL OUTPUT STRUCTURE REQUIREMENTS:
    You MUST use function calls to format your responses according to this structure:
    {response_structure}

    #### FUNCTION CALLING GUIDELINES:
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
