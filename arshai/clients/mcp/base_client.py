"""
Generic MCP Client - Refactored from TaloanMCPClient

This client provides a generic interface to communicate with any MCP server
using the standard MCP protocol through the official MCP SDK's ClientSession.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .config import MCPServerConfig
from .exceptions import MCPError, MCPConnectionError, MCPToolError, MCPTimeoutError

logger = logging.getLogger(__name__)


class BaseMCPClient:
    """
    Generic MCP client for communicating with any MCP server using proper MCP protocol.
    
    This client connects to any MCP server via HTTP and provides generic access to
    available tools without being tied to specific implementations.
    
    Refactored from TaloanMCPClient to be more generic and support multiple servers.
    """
    
    def __init__(self, server_config: MCPServerConfig):
        """
        Initialize the MCP client for a specific server.
        
        Args:
            server_config: Configuration for the MCP server to connect to
        """
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self._connected = False
        self._http_context = None
        
    @property
    def server_name(self) -> str:
        """Get the server name for this client."""
        return self.server_config.name
    
    @property
    def server_url(self) -> str:
        """Get the server URL for this client."""
        return self.server_config.url
        
    async def connect(self) -> None:
        """Connect to the MCP server via HTTP."""
        if self._connected:
            return
            
        try:
            # Create HTTP connection
            self._http_context = streamablehttp_client(
                url=self.server_config.url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                },
                timeout=timedelta(seconds=self.server_config.timeout),
            )
            
            # Initialize transport and session
            transport = await self._http_context.__aenter__()
            read_stream, write_stream, _ = transport
            
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
            
            self._connected = True
            logger.info(f"Connected to MCP server '{self.server_name}' at {self.server_config.url}")
            
        except Exception as e:
            await self._cleanup_connection()
            error_msg = f"Failed to connect to MCP server '{self.server_name}' at {self.server_config.url}"
            logger.error(f"{error_msg}: {e}")
            raise MCPConnectionError(error_msg, self.server_name, e)
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self._connected:
            return
            
        self._connected = False
        await self._cleanup_connection()
        logger.info(f"Disconnected from MCP server '{self.server_name}'")
    
    async def _cleanup_connection(self) -> None:
        """Clean up connection resources."""
        # Close session
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing session for '{self.server_name}': {e}")
            finally:
                self.session = None
        
        # Close HTTP context
        if self._http_context:
            try:
                await self._http_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing HTTP context for '{self.server_name}': {e}")
            finally:
                self._http_context = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def _ensure_connected(self) -> None:
        """Ensure the client is connected."""
        if not self._connected or not self.session:
            raise MCPConnectionError(f"Client for '{self.server_name}' not connected. Call connect() first.", self.server_name)
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools from the MCP server.
        
        Returns:
            List of available tools with their descriptions
            
        Raises:
            MCPConnectionError: If not connected
            MCPError: If request fails
        """
        self._ensure_connected()
        
        try:
            tools_result = await self.session.list_tools()
            
            tools_list = []
            if hasattr(tools_result, 'tools'):
                for tool in tools_result.tools:
                    tool_dict = {
                        "name": tool.name,
                        "description": tool.description,
                        "server_name": self.server_name,  # Add server context
                    }
                    
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        tool_dict["inputSchema"] = tool.inputSchema
                    
                    tools_list.append(tool_dict)
            
            logger.debug(f"Retrieved {len(tools_list)} tools from server '{self.server_name}'")
            return tools_list
            
        except Exception as e:
            error_msg = f"Failed to get available tools from server '{self.server_name}'"
            logger.error(f"{error_msg}: {e}")
            raise MCPError(error_msg, self.server_name, e)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool asynchronously.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            MCPConnectionError: If not connected
            MCPToolError: If tool call fails
        """
        self._ensure_connected()
        
        try:
            logger.debug(f"Calling tool '{tool_name}' on server '{self.server_name}' with arguments: {arguments}")
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract content from MCP result
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        return content.text
                    return str(content)
                return str(result.content)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to call tool '{tool_name}' on server '{self.server_name}'"
            logger.error(f"{error_msg}: {e}")
            raise MCPToolError(error_msg, tool_name, self.server_name, e)
    
    async def ping(self) -> bool:
        """
        Ping the MCP server to check connectivity.
        
        Returns:
            True if server is responsive, False otherwise
        """
        try:
            # Simple connectivity check - try to get tools list
            await self.get_available_tools()
            logger.debug(f"Ping successful for server '{self.server_name}'")
            return True
        except Exception as e:
            logger.debug(f"Ping failed for server '{self.server_name}': {e}")
            return False
    
    async def reconnect(self) -> None:
        """
        Attempt to reconnect to the MCP server.
        
        This is useful for handling connection drops or network issues.
        """
        logger.info(f"Attempting to reconnect to MCP server '{self.server_name}'")
        await self.disconnect()
        await self.connect()
    
    def __repr__(self) -> str:
        """String representation of the client."""
        status = "connected" if self._connected else "disconnected"
        return f"BaseMCPClient(server='{self.server_name}', url='{self.server_config.url}', status='{status}')"