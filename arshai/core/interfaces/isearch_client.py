from typing import Protocol, List, Dict, Optional, Any
from pydantic import BaseModel

class SearchResult(BaseModel):
    """Base model for search results"""
    title: str
    url: str
    content: Optional[str] = None
    engines: List[str] = []
    category: str = "general"

class ISearchClient(Protocol):
    """Interface for search clients"""
    
    async def asearch(
        self,
        query: str,
        num_results: int = 10,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[SearchResult]:
        """
        Perform asynchronous search
        
        Args:
            query: Search query string
            num_results: Number of results to return
            engines: List of search engines to use
            categories: List of categories to search in
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        ...
        
    def search(
        self,
        query: str,
        num_results: int = 10,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[SearchResult]:
        """
        Perform synchronous search
        
        Args:
            query: Search query string
            num_results: Number of results to return
            engines: List of search engines to use
            categories: List of categories to search in
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        ... 