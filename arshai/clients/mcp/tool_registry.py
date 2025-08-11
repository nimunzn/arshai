"""
MCP Tool Registry with Dynamic Discovery and Caching

Phase 2: Tool Management Modernization
- Dynamic tool discovery with event-driven updates
- TTL-based caching for performance optimization  
- Lazy loading for efficient resource utilization
- Real-time tool availability monitoring
"""

import asyncio
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
from cachetools import TTLCache

try:
    from asyncio import Event as AsyncEvent
except ImportError:
    # Fallback for older Python versions
    class AsyncEvent:
        def __init__(self):
            self._event = asyncio.Event()
        
        async def wait(self):
            await self._event.wait()
        
        def set(self):
            self._event.set()
        
        def clear(self):
            self._event.clear()

from .server_manager import MCPServerManager
from .exceptions import MCPError, ToolNotFoundError
from .observability import MCPObservabilityManager

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Enhanced tool specification for registry management."""
    name: str
    description: str
    server_name: str
    server_url: str
    input_schema: Dict[str, Any]
    
    # Phase 2: Registry metadata
    discovered_at: datetime
    last_verified: datetime
    availability_score: float  # 0.0-1.0 based on recent success rate
    tags: List[str]
    category: str
    
    # Performance metadata
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['discovered_at'] = self.discovered_at.isoformat()
        data['last_verified'] = self.last_verified.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolSpec':
        """Create from dictionary (deserialization)."""
        # Convert ISO strings back to datetime objects
        data['discovered_at'] = datetime.fromisoformat(data['discovered_at'])
        data['last_verified'] = datetime.fromisoformat(data['last_verified'])
        return cls(**data)
    
    @classmethod
    def from_mcp_tool_data(cls, tool_data: Dict[str, Any]) -> 'ToolSpec':
        """Create from raw MCP server tool discovery data."""
        now = datetime.utcnow()
        
        # Extract category from description or default
        category = "general"
        description_lower = tool_data.get('description', '').lower()
        if any(keyword in description_lower for keyword in ['file', 'read', 'write']):
            category = "filesystem"
        elif any(keyword in description_lower for keyword in ['web', 'http', 'api']):
            category = "web"
        elif any(keyword in description_lower for keyword in ['data', 'search', 'query']):
            category = "data"
        
        # Extract tags from description
        tags = []
        if 'file' in description_lower:
            tags.append('file-operations')
        if 'web' in description_lower or 'http' in description_lower:
            tags.append('web-integration')
        if 'search' in description_lower:
            tags.append('search')
        
        return cls(
            name=tool_data['name'],
            description=tool_data['description'],
            server_name=tool_data['server_name'],
            server_url=tool_data.get('server_url', ''),
            input_schema=tool_data.get('inputSchema', {}),
            discovered_at=now,
            last_verified=now,
            availability_score=1.0,  # Start optimistic
            tags=tags,
            category=category
        )


class AsyncEventBus:
    """Simple async event bus for tool registry updates."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    async def publish(self, event_type: str, **event_data):
        """Publish event to all subscribers."""
        if event_type in self._subscribers:
            tasks = []
            for callback in self._subscribers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(**event_data))
                else:
                    # Run sync callbacks in thread pool
                    tasks.append(asyncio.get_event_loop().run_in_executor(None, callback, event_data))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class ServerWatcher:
    """Monitors individual server for tool changes."""
    
    def __init__(self, server_name: str, server_manager: MCPServerManager, event_bus: AsyncEventBus):
        self.server_name = server_name
        self.server_manager = server_manager
        self.event_bus = event_bus
        self._last_tools: Set[str] = set()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = AsyncEvent()
    
    async def start_monitoring(self, check_interval: int = 30):
        """Start monitoring server for tool changes."""
        if self._monitoring_task and not self._monitoring_task.done():
            return  # Already monitoring
        
        self._monitoring_task = asyncio.create_task(
            self._monitor_loop(check_interval)
        )
        logger.info(f"📡 Started monitoring server '{self.server_name}' for tool changes")
    
    async def stop_monitoring(self):
        """Stop monitoring the server."""
        self._stop_event.set()
        if self._monitoring_task:
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._monitoring_task.cancel()
        logger.info(f"🛑 Stopped monitoring server '{self.server_name}'")
    
    async def _monitor_loop(self, check_interval: int):
        """Main monitoring loop."""
        while True:
            try:
                # Check if we should stop
                if self._stop_event._event.is_set():
                    break
                
                # Get current tools from server
                try:
                    current_tools = await self.server_manager.get_all_available_tools()
                    server_tools = set(
                        tool['name'] for tool in current_tools 
                        if tool.get('server_name') == self.server_name
                    )
                    
                    # Detect changes
                    added_tools = server_tools - self._last_tools
                    removed_tools = self._last_tools - server_tools
                    
                    if added_tools or removed_tools:
                        logger.info(f"🔄 Tool changes detected on server '{self.server_name}':")
                        if added_tools:
                            logger.info(f"  ✅ Added: {list(added_tools)}")
                            await self.event_bus.publish('tools_added', 
                                server_name=self.server_name,
                                tools=list(added_tools)
                            )
                        
                        if removed_tools:
                            logger.info(f"  ❌ Removed: {list(removed_tools)}")
                            await self.event_bus.publish('tools_removed',
                                server_name=self.server_name, 
                                tools=list(removed_tools)
                            )
                    
                    self._last_tools = server_tools
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error monitoring server '{self.server_name}': {e}")
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in server monitor '{self.server_name}': {e}")
                await asyncio.sleep(check_interval)  # Continue monitoring


class MCPToolRegistry:
    """
    Phase 2: Advanced tool registry with dynamic discovery and caching.
    
    Features:
    - TTL-based caching for performance optimization
    - Dynamic tool discovery with real-time updates
    - Event-driven architecture for tool changes
    - Lazy loading for efficient resource utilization
    - Performance metrics and availability tracking
    """
    
    def __init__(self, 
                 cache_ttl: int = 300,  # 5 minutes
                 cache_maxsize: int = 10000,
                 monitoring_interval: int = 30):
        """
        Initialize the tool registry.
        
        Args:
            cache_ttl: Time-to-live for cached tools in seconds
            cache_maxsize: Maximum number of tools to cache
            monitoring_interval: Seconds between server monitoring checks
        """
        # Phase 2: TTL Cache for tool discovery
        self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        
        # Event-driven updates
        self.event_bus = AsyncEventBus()
        
        # Server monitoring
        self.server_watchers: Dict[str, ServerWatcher] = {}
        self.server_manager: Optional[MCPServerManager] = None
        self.monitoring_interval = monitoring_interval
        
        # Performance tracking
        self.tool_metrics: Dict[str, Dict[str, Any]] = {}
        self._initialization_time: Optional[datetime] = None
        
        # Observability integration
        self.observability: Optional[MCPObservabilityManager] = None
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info(f"🗂️ MCP Tool Registry initialized (TTL: {cache_ttl}s, Max: {cache_maxsize})")
    
    def _setup_event_handlers(self):
        """Setup internal event handlers for cache invalidation."""
        self.event_bus.subscribe('tools_added', self._handle_tools_added)
        self.event_bus.subscribe('tools_removed', self._handle_tools_removed)
        self.event_bus.subscribe('tool_execution_complete', self._handle_execution_metrics)
    
    async def _handle_tools_added(self, server_name: str, tools: List[str]):
        """Handle tools added event."""
        logger.debug(f"📥 Handling added tools from '{server_name}': {tools}")
        # Invalidate cache for affected tools to force rediscovery
        for tool_name in tools:
            cache_key = f"{server_name}:{tool_name}"
            self.cache.pop(cache_key, None)
    
    async def _handle_tools_removed(self, server_name: str, tools: List[str]):
        """Handle tools removed event."""
        logger.debug(f"📤 Handling removed tools from '{server_name}': {tools}")
        # Remove from cache and metrics
        for tool_name in tools:
            cache_key = f"{server_name}:{tool_name}"
            self.cache.pop(cache_key, None)
            self.tool_metrics.pop(cache_key, None)
    
    async def _handle_execution_metrics(self, tool_name: str, server_name: str, 
                                       duration_ms: float, success: bool):
        """Handle tool execution metrics for performance tracking."""
        cache_key = f"{server_name}:{tool_name}"
        
        if cache_key not in self.tool_metrics:
            self.tool_metrics[cache_key] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_duration_ms': 0.0,
                'avg_latency_ms': 0.0,
                'success_rate': 1.0
            }
        
        metrics = self.tool_metrics[cache_key]
        metrics['total_executions'] += 1
        metrics['total_duration_ms'] += duration_ms
        
        if success:
            metrics['successful_executions'] += 1
        
        metrics['avg_latency_ms'] = metrics['total_duration_ms'] / metrics['total_executions']
        metrics['success_rate'] = metrics['successful_executions'] / metrics['total_executions']
        
        # Update cached tool spec if available
        if cached_tool := self.cache.get(cache_key):
            if isinstance(cached_tool, ToolSpec):
                cached_tool.avg_latency_ms = metrics['avg_latency_ms']
                cached_tool.success_rate = metrics['success_rate']
                cached_tool.usage_count = metrics['total_executions']
    
    def set_server_manager(self, server_manager: MCPServerManager):
        """Set the server manager for tool discovery."""
        self.server_manager = server_manager
        if hasattr(server_manager, 'observability'):
            self.observability = server_manager.observability
    
    async def initialize(self):
        """Initialize the registry and start server monitoring."""
        if not self.server_manager:
            raise MCPError("Server manager not set. Call set_server_manager() first.")
        
        self._initialization_time = datetime.utcnow()
        
        # Start monitoring all connected servers
        connected_servers = self.server_manager.get_connected_servers()
        for server_name in connected_servers:
            await self._start_server_monitoring(server_name)
        
        logger.info(f"🚀 Tool registry initialized with {len(connected_servers)} servers")
    
    async def _start_server_monitoring(self, server_name: str):
        """Start monitoring a specific server."""
        if server_name not in self.server_watchers:
            watcher = ServerWatcher(server_name, self.server_manager, self.event_bus)
            self.server_watchers[server_name] = watcher
            await watcher.start_monitoring(self.monitoring_interval)
    
    async def get_tool(self, tool_name: str, server_name: Optional[str] = None) -> Optional[ToolSpec]:
        """
        Get tool specification with lazy loading and caching.
        
        Args:
            tool_name: Name of the tool to retrieve
            server_name: Optional server name for faster lookup
            
        Returns:
            ToolSpec if found, None otherwise
        """
        start_time = time.time()
        
        try:
            # Try cache first (with server-specific key if provided)
            cache_key = f"{server_name}:{tool_name}" if server_name else tool_name
            
            if cached_tool := self.cache.get(cache_key):
                if self.observability:
                    self.observability.record_cache_metrics("tool_registry", "hit")
                logger.debug(f"🎯 Cache hit for tool '{tool_name}' (server: {server_name})")
                return cached_tool
            
            # Cache miss - discover tool
            if self.observability:
                self.observability.record_cache_metrics("tool_registry", "miss")
            
            tool_spec = await self._discover_tool(tool_name, server_name)
            if tool_spec:
                # Cache the discovered tool
                final_cache_key = f"{tool_spec.server_name}:{tool_name}"
                self.cache[final_cache_key] = tool_spec
                
                logger.debug(f"📦 Cached tool '{tool_name}' from server '{tool_spec.server_name}'")
            
            return tool_spec
            
        finally:
            # Record discovery latency
            discovery_time = (time.time() - start_time) * 1000
            if self.observability:
                self.observability.record_performance_metrics(
                    "tool_discovery", discovery_time
                )
    
    async def _discover_tool(self, tool_name: str, preferred_server: Optional[str] = None) -> Optional[ToolSpec]:
        """Discover tool from MCP servers."""
        if not self.server_manager:
            return None
        
        try:
            # Get all available tools from all servers
            all_tools = await self.server_manager.get_all_available_tools()
            
            # Filter by tool name (and optionally by server)
            matching_tools = [
                tool for tool in all_tools 
                if tool['name'] == tool_name and (
                    preferred_server is None or tool.get('server_name') == preferred_server
                )
            ]
            
            if not matching_tools:
                return None
            
            # Prefer specified server, otherwise take first match
            tool_data = matching_tools[0]
            if preferred_server:
                server_matches = [t for t in matching_tools if t.get('server_name') == preferred_server]
                if server_matches:
                    tool_data = server_matches[0]
            
            # Create ToolSpec from discovered data
            tool_spec = ToolSpec.from_mcp_tool_data(tool_data)
            
            # Apply any existing metrics
            cache_key = f"{tool_spec.server_name}:{tool_name}"
            if cache_key in self.tool_metrics:
                metrics = self.tool_metrics[cache_key]
                tool_spec.avg_latency_ms = metrics['avg_latency_ms']
                tool_spec.success_rate = metrics['success_rate']
                tool_spec.usage_count = metrics['total_executions']
            
            logger.debug(f"🔍 Discovered tool '{tool_name}' on server '{tool_spec.server_name}'")
            return tool_spec
            
        except Exception as e:
            logger.error(f"❌ Error discovering tool '{tool_name}': {e}")
            return None
    
    async def get_tools_by_category(self, category: str) -> List[ToolSpec]:
        """Get all tools in a specific category."""
        # This requires full discovery since we need to check categories
        # TODO: Implement more efficient category indexing
        all_tools = await self.get_all_tools()
        return [tool for tool in all_tools if tool.category == category]
    
    async def get_tools_by_tags(self, tags: List[str]) -> List[ToolSpec]:
        """Get all tools matching any of the specified tags."""
        all_tools = await self.get_all_tools()
        return [
            tool for tool in all_tools 
            if any(tag in tool.tags for tag in tags)
        ]
    
    async def get_all_tools(self, force_refresh: bool = False) -> List[ToolSpec]:
        """
        Get all available tools from all servers.
        
        Args:
            force_refresh: If True, bypass cache and discover fresh
            
        Returns:
            List of all available tool specifications
        """
        if not self.server_manager:
            return []
        
        try:
            # Get all tools from server manager
            all_tool_data = await self.server_manager.get_all_available_tools()
            
            tools = []
            for tool_data in all_tool_data:
                tool_name = tool_data['name']
                server_name = tool_data.get('server_name', '')
                cache_key = f"{server_name}:{tool_name}"
                
                if not force_refresh and (cached_tool := self.cache.get(cache_key)):
                    tools.append(cached_tool)
                else:
                    # Create new ToolSpec
                    tool_spec = ToolSpec.from_mcp_tool_data(tool_data)
                    
                    # Apply metrics if available
                    if cache_key in self.tool_metrics:
                        metrics = self.tool_metrics[cache_key]
                        tool_spec.avg_latency_ms = metrics['avg_latency_ms']
                        tool_spec.success_rate = metrics['success_rate']
                        tool_spec.usage_count = metrics['total_executions']
                    
                    # Cache the tool
                    self.cache[cache_key] = tool_spec
                    tools.append(tool_spec)
            
            logger.info(f"📋 Retrieved {len(tools)} tools from registry")
            return tools
            
        except Exception as e:
            logger.error(f"❌ Error getting all tools: {e}")
            return []
    
    async def invalidate_cache(self, tool_name: Optional[str] = None, server_name: Optional[str] = None):
        """
        Invalidate registry cache.
        
        Args:
            tool_name: Specific tool to invalidate (optional)
            server_name: Specific server to invalidate (optional)
        """
        if tool_name and server_name:
            # Invalidate specific tool
            cache_key = f"{server_name}:{tool_name}"
            self.cache.pop(cache_key, None)
            logger.debug(f"🗑️ Invalidated cache for tool '{tool_name}' on server '{server_name}'")
        elif server_name:
            # Invalidate all tools for server
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{server_name}:")]
            for key in keys_to_remove:
                self.cache.pop(key, None)
            logger.debug(f"🗑️ Invalidated cache for server '{server_name}' ({len(keys_to_remove)} tools)")
        else:
            # Invalidate entire cache
            cache_size = len(self.cache)
            self.cache.clear()
            logger.info(f"🗑️ Invalidated entire tool registry cache ({cache_size} tools)")
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        stats = {
            'cache_stats': {
                'current_size': len(self.cache),
                'max_size': self.cache.maxsize,
                'ttl_seconds': self.cache.ttl,
                'hit_rate': getattr(self.cache, 'hits', 0) / max(1, getattr(self.cache, 'hits', 0) + getattr(self.cache, 'misses', 0))
            },
            'monitoring': {
                'monitored_servers': len(self.server_watchers),
                'monitoring_interval': self.monitoring_interval,
                'initialization_time': self._initialization_time.isoformat() if self._initialization_time else None
            },
            'tool_metrics': {
                'total_unique_tools': len(self.tool_metrics),
                'total_executions': sum(m['total_executions'] for m in self.tool_metrics.values()),
                'average_success_rate': sum(m['success_rate'] for m in self.tool_metrics.values()) / max(1, len(self.tool_metrics))
            }
        }
        
        # Add performance statistics
        if self.tool_metrics:
            avg_latencies = [m['avg_latency_ms'] for m in self.tool_metrics.values() if m['avg_latency_ms'] > 0]
            if avg_latencies:
                stats['tool_metrics']['average_latency_ms'] = sum(avg_latencies) / len(avg_latencies)
                stats['tool_metrics']['min_latency_ms'] = min(avg_latencies)
                stats['tool_metrics']['max_latency_ms'] = max(avg_latencies)
        
        return stats
    
    async def cleanup(self):
        """Clean up registry resources."""
        logger.info("🧹 Cleaning up tool registry...")
        
        # Stop all server watchers
        cleanup_tasks = []
        for watcher in self.server_watchers.values():
            cleanup_tasks.append(watcher.stop_monitoring())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear caches and data
        self.cache.clear()
        self.tool_metrics.clear()
        self.server_watchers.clear()
        
        logger.info("✅ Tool registry cleanup completed")
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return (f"MCPToolRegistry(cached_tools={len(self.cache)}, "
                f"monitored_servers={len(self.server_watchers)}, "
                f"ttl={self.cache.ttl}s)")


# Registry Factory Functions
def create_tool_registry(
    server_manager: MCPServerManager,
    cache_ttl: int = 300,
    cache_maxsize: int = 10000,
    monitoring_interval: int = 30
) -> MCPToolRegistry:
    """
    Factory function to create a properly configured tool registry.
    
    Args:
        server_manager: MCP server manager instance
        cache_ttl: Cache time-to-live in seconds
        cache_maxsize: Maximum cache size
        monitoring_interval: Server monitoring interval in seconds
        
    Returns:
        Configured MCPToolRegistry instance
    """
    registry = MCPToolRegistry(
        cache_ttl=cache_ttl,
        cache_maxsize=cache_maxsize,
        monitoring_interval=monitoring_interval
    )
    
    registry.set_server_manager(server_manager)
    return registry


async def initialize_tool_registry(server_manager: MCPServerManager) -> MCPToolRegistry:
    """
    Convenience function to create and initialize a tool registry.
    
    Args:
        server_manager: MCP server manager instance
        
    Returns:
        Initialized MCPToolRegistry instance
    """
    registry = create_tool_registry(server_manager)
    await registry.initialize()
    return registry