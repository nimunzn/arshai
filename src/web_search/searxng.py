import json
import os
from typing import List, Dict, Optional, Any
import aiohttp
import requests
import logging
from pydantic import Field
from arshai.core.interfaces import IWebSearchConfig, IWebSearchClient, IWebSearchResult
#TODO ADD SEARCH MODIFIERS

logger = logging.getLogger(__name__)

class SearxNGConfig(IWebSearchConfig):
    """Configuration for SearxNG client"""
    default_engines: List[str] = Field(
        default=["google", "bing", "duckduckgo"],
        description="Default search engines to use"
    )
    default_categories: List[str] = Field(
        default=["general"],
        description="Default categories to search in"
    )


class SearxNGClient(IWebSearchClient):
    """SearxNG search client implementation"""
    
    def __init__(self, config: SearxNGConfig):
        """Initialize SearxNG client with configuration"""
        self.config = config
        # Get host from config or environment variable
        host = os.getenv("SEARX_INSTANCE")
        if not host:
            raise ValueError("SearxNG instance URL not provided. Set it in config or SEARX_INSTANCE environment variable.")
        self.base_url = host.rstrip('/')
        
    def _prepare_params(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare search parameters"""
        logger.info(f"Search Language: {self.config.get('language')}")
        params = {
            'q': query,
            'format': 'json',
            'engines': ','.join(engines or self.config.default_engines),
            'categories': ','.join(categories or self.config.default_categories),
            'language': self.config.language,
            **kwargs
        }
        logger.info(f"Preparing search parameters: {params}")
        return params
        
    def _parse_results(self, raw_results: Dict[str, Any], num_results: int) -> List[IWebSearchResult]:
        """Parse raw search results into SearchResult objects"""
        results = []
        for result in raw_results.get('results', [])[:num_results]:
            try:
                search_result = IWebSearchResult(
                    title=result['title'],
                    url=result['url'],
                    content=result.get('content'),
                    engines=result.get('engines', []),
                    category=result.get('category', 'general')
                )
                results.append(search_result)
            except Exception as e:
                logger.error(f"Error parsing search result: {e}")
                continue
        return results

    async def asearch(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[IWebSearchResult]:
        """Perform asynchronous search"""
        engines = kwargs.pop('engines', None)
        categories = kwargs.pop('categories', None)
        params = self._prepare_params(query, engines, categories, **kwargs)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/search",
                    params=params,
                    timeout=self.config.timeout,
                    ssl=self.config.verify_ssl
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Search failed with status {response.status}")
                    
                    results = await response.json()
                    logger.info(f"Search results: {results}")
                    return self._parse_results(results, num_results)
                    
        except Exception as e:
            logger.error(f"Async search error: {str(e)}")
            return []

    def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[IWebSearchResult]:
        """Perform synchronous search"""
        engines = kwargs.pop('engines', None)
        categories = kwargs.pop('categories', None)
        params = self._prepare_params(query, engines, categories, **kwargs)
        
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
            
            results = response.json()
            return self._parse_results(results, num_results)
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [] 
        