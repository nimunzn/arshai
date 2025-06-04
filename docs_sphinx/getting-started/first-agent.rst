=======================
Building Your First Agent
=======================

This tutorial will guide you through creating a more sophisticated AI agent with tools, memory, and custom behavior.

Overview
========

We'll build a research assistant agent that can:

- Search the web for information
- Remember conversation context
- Provide structured responses
- Use custom tools

Setting Up the Environment
==========================

First, ensure you have the necessary dependencies:

.. code-block:: bash

   pip install arshai[all]

And set up your API keys:

.. code-block:: bash

   export OPENAI_API_KEY="your-openai-api-key"
   # Optional: for Redis memory
   export REDIS_URL="redis://localhost:6379"

Step 1: Basic Agent Configuration
=================================

Let's start with a well-configured conversational agent:

.. code-block:: python

   from arshai import Settings, IAgentConfig, IAgentInput

   def create_research_agent():
       # Initialize settings
       settings = Settings()
       
       # Define the agent's role and capabilities
       agent_config = IAgentConfig(
           task_context='''
           You are a professional research assistant with the following capabilities:
           
           1. Web search for current information
           2. Analysis and synthesis of multiple sources
           3. Clear, structured responses with citations
           4. Maintaining conversation context
           
           Always:
           - Provide accurate, well-researched information
           - Cite your sources when using web search
           - Ask clarifying questions when requests are ambiguous
           - Maintain a professional but friendly tone
           ''',
           tools=[]  # We'll add tools next
       )
       
       return settings.create_agent("conversation", agent_config), settings

   # Create the agent
   agent, settings = create_research_agent()

Step 2: Adding Web Search Tool
==============================

Now let's add web search capabilities:

.. code-block:: python

   from arshai.tools.web_search_tool import WebSearchTool

   def create_research_agent_with_tools():
       settings = Settings()
       
       # Create web search tool
       web_search = WebSearchTool(settings)
       
       agent_config = IAgentConfig(
           task_context='''
           You are a research assistant with web search capabilities.
           
           When users ask questions that require current information:
           1. Use web search to find relevant, up-to-date information
           2. Analyze and synthesize information from multiple sources
           3. Provide clear, well-structured responses
           4. Always cite your sources with URLs when using search results
           
           For general knowledge questions, you can use your training data,
           but for current events, statistics, or recent developments,
           always use web search to ensure accuracy.
           ''',
           tools=[web_search]
       )
       
       return settings.create_agent("conversation", agent_config), settings

   # Create enhanced agent
   agent, settings = create_research_agent_with_tools()

Step 3: Testing the Agent
=========================

Let's test our research agent:

.. code-block:: python

   def test_research_agent():
       agent, settings = create_research_agent_with_tools()
       
       # Test with a question requiring current information
       response, usage = agent.process_message(
           IAgentInput(
               message="What are the latest developments in artificial intelligence in 2024?",
               conversation_id="research_session_1"
           )
       )
       
       print("=== Research Agent Response ===")
       print(response)
       print(f"\\nTokens used: {usage}")
       
       # Follow up question to test memory
       response2, usage2 = agent.process_message(
           IAgentInput(
               message="Can you elaborate on the AI safety developments you mentioned?",
               conversation_id="research_session_1"  # Same session
           )
       )
       
       print("\\n=== Follow-up Response ===")
       print(response2)
       print(f"\\nTokens used: {usage2}")

   # Run the test
   test_research_agent()

Step 4: Creating Custom Tools
=============================

Let's create a custom tool for our agent:

.. code-block:: python

   from arshai.core.interfaces import ITool
   from typing import Dict, Any
   import json
   import requests

   class WeatherTool(ITool):
       """Custom tool to get weather information."""
       
       @property
       def name(self) -> str:
           return "get_weather"
       
       @property
       def description(self) -> str:
           return "Get current weather information for a specified city"
       
       @property
       def parameters(self) -> Dict[str, Any]:
           return {
               "type": "object",
               "properties": {
                   "city": {
                       "type": "string",
                       "description": "The city name to get weather for"
                   },
                   "country": {
                       "type": "string", 
                       "description": "The country code (optional, e.g., 'US', 'UK')"
                   }
               },
               "required": ["city"]
           }
       
       async def execute(self, **kwargs) -> str:
           """Execute the weather tool."""
           city = kwargs.get("city")
           country = kwargs.get("country", "")
           
           # This is a mock implementation
           # In practice, you'd call a real weather API
           weather_data = {
               "city": city,
               "country": country,
               "temperature": "22°C",
               "condition": "Partly cloudy",
               "humidity": "65%",
               "wind": "10 km/h"
           }
           
           return f"Weather in {city}: {weather_data['temperature']}, {weather_data['condition']}"

Step 5: Agent with Multiple Tools
=================================

Now let's create an agent with multiple tools:

.. code-block:: python

   def create_full_featured_agent():
       settings = Settings()
       
       # Create tools
       web_search = WebSearchTool(settings)
       weather_tool = WeatherTool()
       
       agent_config = IAgentConfig(
           task_context='''
           You are a comprehensive AI assistant with multiple capabilities:
           
           1. Web search for current information and research
           2. Weather information for any city
           3. General knowledge and problem-solving
           
           Tool Usage Guidelines:
           - Use web search for current events, news, recent developments
           - Use weather tool when users ask about weather conditions
           - For general questions, use your knowledge base
           - Always be helpful and provide complete answers
           - Cite sources when using web search results
           ''',
           tools=[web_search, weather_tool]
       )
       
       return settings.create_agent("conversation", agent_config), settings

   # Test the full-featured agent
   def test_full_agent():
       agent, settings = create_full_featured_agent()
       
       # Test weather tool
       response1, _ = agent.process_message(
           IAgentInput(
               message="What's the weather like in New York?",
               conversation_id="demo_session"
           )
       )
       print("Weather Response:")
       print(response1)
       
       # Test web search
       response2, _ = agent.process_message(
           IAgentInput(
               message="What are the latest tech news today?",
               conversation_id="demo_session"
           )
       )
       print("\\nNews Response:")
       print(response2)

   test_full_agent()

Step 6: Configuration Management
================================

For production use, manage configuration with files:

.. code-block:: yaml

   # agent_config.yaml
   llm:
     provider: openai
     model: gpt-4
     temperature: 0.7
     max_tokens: 2000

   memory:
     working_memory:
       provider: redis
       ttl: 86400  # 24 hours
       
   tools:
     web_search:
       enabled: true
       max_results: 5
     
     weather:
       enabled: true
       api_key: "${WEATHER_API_KEY}"

Load configuration in your code:

.. code-block:: python

   def create_configured_agent():
       # Load settings from file
       settings = Settings(config_path="agent_config.yaml")
       
       # Tools are automatically configured based on settings
       web_search = WebSearchTool(settings)
       weather_tool = WeatherTool()
       
       agent_config = IAgentConfig(
           task_context="Your comprehensive AI assistant...",
           tools=[web_search, weather_tool]
       )
       
       return settings.create_agent("conversation", agent_config)

Step 7: Error Handling and Logging
==================================

Add proper error handling to your agent:

.. code-block:: python

   import logging
   from arshai.utils.logging import setup_logging

   def create_robust_agent():
       # Set up logging
       setup_logging(level=logging.INFO)
       logger = logging.getLogger(__name__)
       
       try:
           settings = Settings(config_path="agent_config.yaml")
           
           # Create tools with error handling
           tools = []
           try:
               web_search = WebSearchTool(settings)
               tools.append(web_search)
               logger.info("Web search tool enabled")
           except Exception as e:
               logger.warning(f"Web search tool failed to initialize: {e}")
           
           try:
               weather_tool = WeatherTool()
               tools.append(weather_tool)
               logger.info("Weather tool enabled")
           except Exception as e:
               logger.warning(f"Weather tool failed to initialize: {e}")
           
           agent_config = IAgentConfig(
               task_context="Your AI assistant with robust error handling...",
               tools=tools
           )
           
           agent = settings.create_agent("conversation", agent_config)
           logger.info("Agent created successfully")
           return agent
           
       except Exception as e:
           logger.error(f"Failed to create agent: {e}")
           raise

Next Steps
==========

Congratulations! You've built a sophisticated AI agent. Here's what you can explore next:

1. **Workflows**: Learn to create multi-agent workflows in :doc:`../user-guide/workflows/index`
2. **Advanced Memory**: Explore persistent memory options in :doc:`../user-guide/memory/index`
3. **Plugin System**: Create reusable plugins in :doc:`../user-guide/extensions/plugins`
4. **Production Deployment**: Deploy your agent in :doc:`../deployment/production`

Complete Example
===============

Here's the complete code for a production-ready research agent:

.. code-block:: python

   import logging
   from arshai import Settings, IAgentConfig, IAgentInput
   from arshai.tools.web_search_tool import WebSearchTool
   from arshai.core.interfaces import ITool

   # Set up logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   class ResearchAgent:
       def __init__(self, config_path=None):
           self.settings = Settings(config_path=config_path)
           self.agent = self._create_agent()
       
       def _create_agent(self):
           # Create tools
           tools = []
           try:
               web_search = WebSearchTool(self.settings)
               tools.append(web_search)
           except Exception as e:
               logger.warning(f"Web search unavailable: {e}")
           
           # Configure agent
           agent_config = IAgentConfig(
               task_context='''
               You are a professional research assistant. Use web search 
               for current information and provide well-cited responses.
               ''',
               tools=tools
           )
           
           return self.settings.create_agent("conversation", agent_config)
       
       def research(self, question: str, session_id: str = None) -> str:
           """Research a question and return the response."""
           if session_id is None:
               session_id = f"research_{hash(question) % 10000}"
           
           try:
               response, usage = self.agent.process_message(
                   IAgentInput(message=question, conversation_id=session_id)
               )
               logger.info(f"Research completed. Tokens used: {usage}")
               return response
           except Exception as e:
               logger.error(f"Research failed: {e}")
               return f"I encountered an error: {e}"

   # Usage
   if __name__ == "__main__":
       assistant = ResearchAgent()
       result = assistant.research("What are the latest AI developments?")
       print(result)

This example demonstrates a complete, production-ready research agent with proper error handling, logging, and configuration management.