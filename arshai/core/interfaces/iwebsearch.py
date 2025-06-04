from typing import List, Dict, Optional, Any, Protocol
from pydantic import Field
from .idto import IDTO

class IWebSearchResult(IDTO):
    """
    Represents a single search result.
    """
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: Optional[str] = Field(default=None, description="Content snippet or description")
    engines: List[str] = Field(default_factory=list, description="Search engines that returned this result")
    category: str = Field(default="general", description="Category of the search result")

class IWebSearchConfig(IDTO):
    """
    Base configuration for web search clients.
    This is a generalized configuration that can be extended for specific search engines.
    """
    host: Optional[str] = Field(default=None, description="Search engine host URL")
    timeout: int = Field(default=10, description="Request timeout in seconds")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")
    language: str = Field(default="all", description="Search language code")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional search engine specific parameters")

class IWebSearchClient(Protocol):
    """
    Interface defining the contract for web search clients.
    Any web search client implementation must conform to this interface.
    """
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[IWebSearchResult]:
        """
        Perform synchronous web search.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        ...
        
    async def asearch(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[IWebSearchResult]:
        """
        Perform asynchronous web search.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        ... 