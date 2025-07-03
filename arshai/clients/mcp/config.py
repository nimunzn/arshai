"""
MCP Server Configuration Management

Handles configuration for multiple MCP servers using the project's Settings system.
Reads configuration from config.yaml through the ISetting interface.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from arshai.core.interfaces.isetting import ISetting
from .exceptions import MCPConfigurationError


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    url: str
    timeout: int = 30
    max_retries: int = 3
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name or not self.name.strip():
            raise MCPConfigurationError("Server name is required and cannot be empty")
        
        if not self.url or not self.url.strip():
            raise MCPConfigurationError(f"URL is required for MCP server '{self.name}'")
        
        if not self.url.startswith(('http://', 'https://')):
            raise MCPConfigurationError(f"Invalid URL format for MCP server '{self.name}': {self.url}")
        
        if self.timeout <= 0:
            raise MCPConfigurationError(f"Timeout must be positive for server '{self.name}'")
        
        if self.max_retries < 0:
            raise MCPConfigurationError(f"Max retries cannot be negative for server '{self.name}'")


@dataclass 
class MCPConfig:
    """Global MCP configuration for all servers."""
    enabled: bool
    servers: List[MCPServerConfig]
    connection_timeout: int = 30
    default_max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.enabled and not self.servers:
            raise MCPConfigurationError("At least one MCP server must be configured when MCP is enabled")
        
        # Check for duplicate server names
        if self.servers:
            server_names = [server.name for server in self.servers]
            if len(server_names) != len(set(server_names)):
                duplicates = [name for name in server_names if server_names.count(name) > 1]
                raise MCPConfigurationError(f"Duplicate server names found: {', '.join(set(duplicates))}")
    
    @classmethod
    def from_settings(cls, settings: ISetting) -> 'MCPConfig':
        """
        Load MCP configuration from the project's Settings system.
        
        Reads configuration from config.yaml using the ISetting interface.
        
        Args:
            settings: ISetting instance (e.g., from workflow or agent)
            
        Returns:
            MCPConfig instance with servers from config.yaml
            
        Raises:
            MCPConfigurationError: If configuration is invalid
        """
        # Get MCP configuration section from config.yaml
        mcp_config = settings.get("mcp", {})
        
        # Check if MCP is enabled
        enabled = mcp_config.get("enabled", False)
        
        # Get global MCP settings
        connection_timeout = mcp_config.get("connection_timeout", 30)
        default_max_retries = mcp_config.get("default_max_retries", 3)
        
        servers = []
        
        if enabled:
            # Get servers configuration
            servers_config = mcp_config.get("servers", [])
            
            for server_config in servers_config:
                if not isinstance(server_config, dict):
                    raise MCPConfigurationError(f"Server configuration must be a dictionary, got: {type(server_config)}")
                
                try:
                    # Use server-specific timeout/retries or fall back to global defaults
                    server_timeout = server_config.get("timeout", connection_timeout)
                    server_max_retries = server_config.get("max_retries", default_max_retries)
                    
                    servers.append(MCPServerConfig(
                        name=server_config["name"],
                        url=server_config["url"],
                        timeout=server_timeout,
                        max_retries=server_max_retries,
                        description=server_config.get("description")
                    ))
                except KeyError as e:
                    raise MCPConfigurationError(f"Missing required field in server configuration: {e}")
                except Exception as e:
                    raise MCPConfigurationError(f"Invalid server configuration: {e}")
            
            # Fallback to environment variable if no servers in config but MCP is enabled
            if not servers:
                fallback_url = os.getenv('MCP_SERVER_URL')
                if fallback_url:
                    servers.append(MCPServerConfig(
                        name='default',
                        url=fallback_url,
                        timeout=connection_timeout,
                        max_retries=default_max_retries,
                        description='Default MCP server from environment variable'
                    ))
        
        return cls(
            enabled=enabled,
            servers=servers,
            connection_timeout=connection_timeout,
            default_max_retries=default_max_retries
        )
    
    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get server configuration by name."""
        for server in self.servers:
            if server.name == name:
                return server
        return None
    
    def get_server_names(self) -> List[str]:
        """Get list of all configured server names."""
        return [server.name for server in self.servers]
    
    def add_server(self, server: MCPServerConfig) -> None:
        """Add a new server configuration."""
        # Check for duplicate names
        if self.get_server(server.name):
            raise MCPConfigurationError(f"Server with name '{server.name}' already exists")
        
        self.servers.append(server)
    
    def remove_server(self, name: str) -> bool:
        """Remove server configuration by name. Returns True if server was found and removed."""
        for i, server in enumerate(self.servers):
            if server.name == name:
                del self.servers[i]
                return True
        return False