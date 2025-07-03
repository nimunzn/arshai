"""
MCP (Model Context Protocol) Infrastructure Package

This package provides generic infrastructure for connecting to multiple MCP servers
and managing their tools dynamically.

Components:
- config: MCP server configuration management
- base_client: Generic MCP client implementation 
- server_manager: Multi-server management
- exceptions: MCP-specific exceptions
"""

from .config import MCPConfig, MCPServerConfig
from .exceptions import MCPError, MCPConnectionError, MCPToolError

__all__ = [
    'MCPConfig',
    'MCPServerConfig', 
    'MCPError',
    'MCPConnectionError',
    'MCPToolError'
]