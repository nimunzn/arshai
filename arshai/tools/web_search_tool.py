from typing import Dict, List, Any
from arshai.core.interfaces.itool import ITool
from arshai.core.interfaces.iwebsearch import IWebSearchClient
import logging
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class WebSearchTool(ITool):
    """Tool for retrieving information from web search using search engines"""
    
    def __init__(self, settings: Settings):
        """
        Initialize the web search tool.
        
        Args:
            settings: Settings instance to get search configuration
        """
        self.settings = settings
        self.search_client: IWebSearchClient = self.settings.create_web_search()
        
        if not self.search_client:
            logger.warning("Search client could not be created. Check configuration.")

    @property
    def function_definition(self) -> Dict:
        """Get the function definition for the LLM"""
        return {
            "name": "web_search",
            "description": "Search the web for information using search engines. The query MUST be self-contained and include all necessary context without relying on conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A standalone, self-contained search query that includes all necessary context. Example of a good query: 'What are the latest developments in quantum computing in 2024?' Instead of just 'What are the latest developments?'"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }

    def execute(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute web search using search engines
        
        Args:
            query: A standalone question containing all required context
            
        Returns:
            List[Dict[str, Any]]: List of content objects in the format required by the LLM
        """
        if not self.search_client:
            logger.error("Cannot perform search: No search client available")
            return "function", [{"type": "text", "text": "No search capability available. Please check configuration."}]
            
        results = self.search_client.search(query)
        if not results:
            return "function", [{"type": "text", "text": "No results found."}]
            
        # Format results into context and URLs
        context = "\n\n".join(f"{r.title}\n{r.content}" for r in results if r.content)
        urls = "\n".join(r.url for r in results)
        
        # Combine context and URLs into a single response
        full_response = f"{context}\n\nSources:\n{urls}" if urls else context
        
        return "function", [{"type": "text", "text": full_response}]
    
    async def aexecute(self, query: str) -> List[Dict[str, Any]]:
        """
        Asynchronous execution of web search using search engines
        
        Args:
            query: A standalone question containing all required context
            
        Returns:
            List[Dict[str, Any]]: List of content objects in the format required by the LLM
        """
        if not self.search_client:
            logger.error("Cannot perform search: No search client available")
            return "function", [{"type": "text", "text": "No search capability available. Please check configuration."}]
            
        results = await self.search_client.asearch(query)
        if not results:
            return "function", [{"type": "text", "text": "No results found."}]
            
        # Format results into context and URLs
        context = "\n\n".join(f"{r.title}\n{r.content}" for r in results if r.content)
        urls = "\n".join(r.url for r in results)
        
        # Combine context and URLs into a single response
        full_response = f"{context}\n\nSources:\n{urls}" if urls else context
        
        return "function", [{"type": "text", "text": full_response}]
