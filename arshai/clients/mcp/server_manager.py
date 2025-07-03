"""
MCP Server Manager

Manages multiple MCP servers and provides a unified interface for tool discovery
and execution across all configured servers.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from arshai.core.interfaces.isetting import ISetting

from .config import MCPConfig, MCPServerConfig
from .base_client import BaseMCPClient
from .exceptions import MCPError, MCPConnectionError, MCPConfigurationError

logger = logging.getLogger(__name__)


class MCPServerManager:
    """
    Manages multiple MCP servers and provides unified access to their tools.
    
    This manager handles:
    - Configuration loading from Settings
    - Connection management for multiple servers
    - Tool discovery across all servers
    - Server health monitoring
    - Graceful error handling and fallback
    """
    
    def __init__(self, settings: ISetting):
        """
        Initialize the MCP server manager.
        
        Args:
            settings: ISetting instance for reading configuration
        """
        self.settings = settings
        self.config: Optional[MCPConfig] = None
        self.clients: Dict[str, BaseMCPClient] = {}
        self._connected_servers: Set[str] = set()
        self._failed_servers: Set[str] = set()
        
    async def initialize(self) -> None:
        """
        Initialize the manager by loading configuration and connecting to servers.
        
        Raises:
            MCPConfigurationError: If configuration is invalid
        """
        try:
            # Load MCP configuration from settings
            self.config = MCPConfig.from_settings(self.settings)
            
            if not self.config.enabled:
                logger.info("MCP is disabled in configuration")
                return
            
            logger.info(f"Initializing MCP manager with {len(self.config.servers)} servers")
            
            # Create clients for each configured server
            for server_config in self.config.servers:
                client = BaseMCPClient(server_config)
                self.clients[server_config.name] = client
                logger.debug(f"Created MCP client for server '{server_config.name}'")
            
            # Attempt to connect to all servers (non-blocking for individual failures)
            await self._connect_all_servers()
            
            if self._connected_servers:
                logger.info(f"MCP manager initialized successfully. Connected to: {list(self._connected_servers)}")
                if self._failed_servers:
                    logger.warning(f"Failed to connect to: {list(self._failed_servers)}")
            else:
                logger.warning("MCP manager initialized but no servers are available")
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP server manager: {e}")
            raise MCPConfigurationError(f"MCP initialization failed: {e}")
    
    async def _connect_all_servers(self) -> None:
        """Connect to all configured MCP servers concurrently."""
        if not self.clients:
            return
        
        # Create connection tasks for all servers
        connection_tasks = {}
        for server_name, client in self.clients.items():
            connection_tasks[server_name] = asyncio.create_task(
                self._connect_server_safe(server_name, client)
            )
        
        # Wait for all connection attempts to complete
        await asyncio.gather(*connection_tasks.values(), return_exceptions=True)
    
    async def _connect_server_safe(self, server_name: str, client: BaseMCPClient) -> None:
        """
        Safely attempt to connect to a single server.
        
        Args:
            server_name: Name of the server
            client: MCP client for the server
        """
        try:
            await client.connect()
            self._connected_servers.add(server_name)
            logger.info(f"Successfully connected to MCP server '{server_name}'")
        except Exception as e:
            self._failed_servers.add(server_name)
            logger.warning(f"Failed to connect to MCP server '{server_name}': {e}")
    
    async def get_all_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools from all connected MCP servers.
        
        Returns:
            List of tools from all servers, with server information included
        """
        if not self.is_enabled():
            return []
        
        all_tools = []
        
        # Collect tools from all connected servers
        for server_name in self._connected_servers:
            try:
                client = self.clients[server_name]
                server_tools = await client.get_available_tools()
                
                # Add server context to each tool
                for tool in server_tools:
                    tool['server_name'] = server_name
                    tool['server_url'] = client.server_url
                
                all_tools.extend(server_tools)
                logger.debug(f"Retrieved {len(server_tools)} tools from server '{server_name}'")
                
            except Exception as e:
                logger.warning(f"Failed to get tools from server '{server_name}': {e}")
                # Mark server as failed and remove from connected set
                self._connected_servers.discard(server_name)
                self._failed_servers.add(server_name)
        
        logger.info(f"Retrieved total of {len(all_tools)} tools from {len(self._connected_servers)} servers")
        return all_tools
    
    async def call_tool(self, tool_name: str, server_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific tool on a specific server.
        
        Args:
            tool_name: Name of the tool to call
            server_name: Name of the server hosting the tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            MCPError: If server not found or tool call fails
        """
        if not self.is_enabled():
            raise MCPError("MCP is not enabled")
        
        if server_name not in self.clients:
            raise MCPError(f"Unknown MCP server: '{server_name}'")
        
        if server_name not in self._connected_servers:
            # Try to reconnect
            await self._reconnect_server(server_name)
            if server_name not in self._connected_servers:
                raise MCPConnectionError(f"MCP server '{server_name}' is not connected", server_name)
        
        try:
            client = self.clients[server_name]
            result = await client.call_tool(tool_name, arguments)
            logger.debug(f"Successfully called tool '{tool_name}' on server '{server_name}'")
            return result
        except Exception as e:
            # Mark server as potentially problematic
            logger.warning(f"Tool call failed for '{tool_name}' on server '{server_name}': {e}")
            raise
    
    async def _reconnect_server(self, server_name: str) -> None:
        """
        Attempt to reconnect to a specific server.
        
        Args:
            server_name: Name of the server to reconnect
        """
        if server_name not in self.clients:
            return
        
        try:
            client = self.clients[server_name]
            await client.reconnect()
            self._connected_servers.add(server_name)
            self._failed_servers.discard(server_name)
            logger.info(f"Successfully reconnected to MCP server '{server_name}'")
        except Exception as e:
            logger.warning(f"Failed to reconnect to MCP server '{server_name}': {e}")
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all servers.
        
        Returns:
            Dictionary with server health status
        """
        health_status = {}
        
        for server_name, client in self.clients.items():
            try:
                if server_name in self._connected_servers:
                    is_healthy = await client.ping()
                    health_status[server_name] = {
                        'status': 'healthy' if is_healthy else 'unhealthy',
                        'connected': True,
                        'url': client.server_url
                    }
                else:
                    health_status[server_name] = {
                        'status': 'disconnected',
                        'connected': False,
                        'url': client.server_url
                    }
            except Exception as e:
                health_status[server_name] = {
                    'status': 'error',
                    'connected': False,
                    'url': client.server_url,
                    'error': str(e)
                }
        
        return health_status
    
    def is_enabled(self) -> bool:
        """Check if MCP is enabled and has connected servers."""
        return (self.config is not None and 
                self.config.enabled and 
                len(self._connected_servers) > 0)
    
    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server names."""
        return list(self._connected_servers)
    
    def get_failed_servers(self) -> List[str]:
        """Get list of servers that failed to connect."""
        return list(self._failed_servers)
    
    async def cleanup(self) -> None:
        """Clean up all connections and resources."""
        logger.info("Cleaning up MCP server manager")
        
        # Disconnect all clients
        disconnect_tasks = []
        for server_name, client in self.clients.items():
            if server_name in self._connected_servers:
                disconnect_tasks.append(client.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        # Clear state
        self._connected_servers.clear()
        self._failed_servers.clear()
        self.clients.clear()
        
        logger.info("MCP server manager cleanup completed")
    
    def __repr__(self) -> str:
        """String representation of the manager."""
        if not self.config:
            return "MCPServerManager(uninitialized)"
        
        return (f"MCPServerManager(enabled={self.config.enabled}, "
                f"connected={len(self._connected_servers)}, "
                f"failed={len(self._failed_servers)})")