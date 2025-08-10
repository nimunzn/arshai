import json
import os
import asyncio
import atexit
from typing import List, Dict, Optional, Any
import aiohttp
import requests
import logging
from pydantic import Field
from arshai.core.interfaces.iwebsearch import IWebSearchConfig, IWebSearchClient, IWebSearchResult
#TODO ADD SEARCH MODIFIERS

logger = logging.getLogger(__name__)


class SearxNGClient(IWebSearchClient):
    """SearxNG search client implementation"""
    
    # Class-level session for connection reuse
    _shared_session: Optional[aiohttp.ClientSession] = None
    _session_lock = asyncio.Lock()
    _cleanup_registered = False
    
    def __init__(self, config: dict):
        """Initialize SearxNG client with configuration"""
        self.config = config
        # Get host from config or environment variable
        host = os.getenv("SEARX_INSTANCE")
        if not host:
            raise ValueError("SearxNG instance URL not provided. Set it in config or SEARX_INSTANCE environment variable.")
        self.base_url = host.rstrip('/')
        
        # Connection pool configuration from environment or defaults
        self._max_connections = int(os.getenv("ARSHAI_MAX_CONNECTIONS", "100"))
        self._max_connections_per_host = int(os.getenv("ARSHAI_MAX_CONNECTIONS_PER_HOST", "10"))
        self._connection_timeout = int(os.getenv("ARSHAI_CONNECTION_TIMEOUT", "30"))
        
        # Register cleanup on first instance creation
        if not SearxNGClient._cleanup_registered:
            atexit.register(self._cleanup_session_sync)
            SearxNGClient._cleanup_registered = True
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared aiohttp session with connection pooling."""
        async with self._session_lock:
            if SearxNGClient._shared_session is None or SearxNGClient._shared_session.closed:
                # Create connector with connection limits
                connector = aiohttp.TCPConnector(
                    limit=self._max_connections,              # Total connection limit
                    limit_per_host=self._max_connections_per_host,  # Per-host limit
                    ttl_dns_cache=300,                       # DNS cache TTL (5 minutes)
                    use_dns_cache=True,                      # Enable DNS caching
                    keepalive_timeout=60,                    # Keep connections alive for 60s
                    enable_cleanup_closed=True,              # Cleanup closed connections
                    force_close=False,                       # Reuse connections when possible
                )
                
                # Create timeout configuration
                timeout = aiohttp.ClientTimeout(
                    total=self._connection_timeout,
                    connect=10,                              # Connection timeout
                    sock_read=10,                            # Socket read timeout
                    sock_connect=10                          # Socket connect timeout
                )
                
                SearxNGClient._shared_session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={'User-Agent': 'Arshai/1.0'}    # Identify the client
                )
                logger.info(f"Created new aiohttp session with connection limits: "
                          f"max={self._max_connections}, per_host={self._max_connections_per_host}")
            
            return SearxNGClient._shared_session
    
    @classmethod
    async def cleanup_session(cls) -> None:
        """Cleanup the shared session. Call this when shutting down the application."""
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None
            logger.info("Cleaned up aiohttp session")
    
    def _cleanup_session_sync(self) -> None:
        """Synchronous cleanup for atexit handler"""
        if SearxNGClient._shared_session and not SearxNGClient._shared_session.closed:
            try:
                asyncio.run(SearxNGClient.cleanup_session())
            except RuntimeError:
                # Event loop might be closed
                pass
        
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
            'engines': ','.join(engines or self.config.get('default_engines', [])),
            'categories': ','.join(categories or self.config.get('default_categories', [])),
            'language': self.config.get('language'),
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
        """Perform asynchronous search with connection pooling."""
        engines = kwargs.pop('engines', None)
        categories = kwargs.pop('categories', None)
        params = self._prepare_params(query, engines, categories, **kwargs)
        
        try:
            # Get the shared session with connection pooling
            session = await self._get_session()
            
            # Use individual timeout for this request if specified in config
            request_timeout = self.config.get('timeout', None)
            
            # Perform the search request
            async with session.get(
                f"{self.base_url}/search",
                params=params,
                timeout=aiohttp.ClientTimeout(total=request_timeout) if request_timeout else None,
                ssl=self.config.get('verify_ssl', True)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Search failed with status {response.status}")
                
                results = await response.json()
                logger.info(f"Search results: {results}")
                return self._parse_results(results, num_results)
                
        except asyncio.TimeoutError:
            logger.error(f"Search timeout for query: {query}")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during search: {str(e)}")
            return []
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
            response = requests.get(  # nosec B113 - timeout is properly configured
                f"{self.base_url}/search",
                params=params,
                timeout=self.config.get('timeout', 10),
                verify=self.config.get('verify_ssl', True)
            )
            response.raise_for_status()
            
            results = response.json()
            return self._parse_results(results, num_results)
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [] 
        