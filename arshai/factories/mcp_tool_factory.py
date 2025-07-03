"""
MCP Tool Factory

Creates ITool instances from MCP servers using the project's Settings system.
This is the main entry point for adding MCP tools to workflows.
"""

import asyncio
import logging
from typing import List, Optional
from arshai.core.interfaces.isetting import ISetting
from arshai.core.interfaces.itool import ITool

from arshai.clients.mcp.server_manager import MCPServerManager
from arshai.clients.mcp.config import MCPConfig
from arshai.clients.mcp.exceptions import MCPError, MCPConfigurationError
from arshai.tools.mcp_dynamic_tool import MCPDynamicTool

logger = logging.getLogger(__name__)


class MCPToolFactory:
    """
    Factory for creating ITool instances from MCP servers.
    
    This factory provides a simple interface for workflows to get all MCP tools
    without needing to manage individual server connections or tool discovery.
    """
    
    def __init__(self, settings: ISetting):
        """
        Initialize the MCP tool factory.
        
        Args:
            settings: ISetting instance for reading configuration
        """
        self.settings = settings
        self.server_manager: Optional[MCPServerManager] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the factory and server manager.
        
        This must be called before creating tools.
        """
        if self._initialized:
            return
        
        try:
            # Check if MCP is enabled in configuration
            mcp_config = MCPConfig.from_settings(self.settings)
            
            if not mcp_config.enabled:
                logger.info("MCP is disabled in configuration, skipping initialization")
                self._initialized = True
                return
            
            # Initialize server manager
            self.server_manager = MCPServerManager(self.settings)
            await self.server_manager.initialize()
            
            self._initialized = True
            logger.info("MCP tool factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tool factory: {e}")
            # Set as initialized to prevent retry loops, but server_manager will be None
            self._initialized = True
            raise MCPConfigurationError(f"MCP tool factory initialization failed: {e}")
    
    async def create_all_tools(self) -> List[ITool]:
        """
        Create ITool instances for all available tools from all MCP servers.
        
        This is the main method that workflows should call to get MCP tools.
        
        Returns:
            List of ITool instances, one for each tool from all connected servers
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.server_manager or not self.server_manager.is_enabled():
            logger.info("MCP is not enabled or no servers available, returning empty tool list")
            return []
        
        try:
            # Discover all tools from all connected servers
            all_tool_specs = await self.server_manager.get_all_available_tools()
            
            if not all_tool_specs:
                logger.warning("No tools found on any MCP servers")
                return []
            
            # Create individual ITool instances for each discovered tool
            tools = []
            for tool_spec in all_tool_specs:
                try:
                    tool = MCPDynamicTool(tool_spec, self.server_manager)
                    tools.append(tool)
                    logger.debug(f"Created tool: {tool.name} from server: {tool.server_name}")
                except Exception as e:
                    logger.warning(f"Failed to create tool from spec {tool_spec.get('name', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Successfully created {len(tools)} MCP tools from {len(self.server_manager.get_connected_servers())} servers")
            
            # Log server and tool summary
            connected_servers = self.server_manager.get_connected_servers()
            failed_servers = self.server_manager.get_failed_servers()
            
            if connected_servers:
                logger.info(f"Connected MCP servers: {connected_servers}")
            if failed_servers:
                logger.warning(f"Failed MCP servers: {failed_servers}")
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to create MCP tools: {e}")
            return []
    
    async def get_server_health(self) -> dict:
        """
        Get health status of all configured MCP servers.
        
        Returns:
            Dictionary with server health information
        """
        if not self.server_manager:
            return {}
        
        return await self.server_manager.health_check()
    
    def is_enabled(self) -> bool:
        """
        Check if MCP is enabled and has available servers.
        
        Returns:
            True if MCP is enabled and has connected servers
        """
        return (self._initialized and 
                self.server_manager is not None and 
                self.server_manager.is_enabled())
    
    async def cleanup(self) -> None:
        """Clean up resources used by the factory."""
        if self.server_manager:
            await self.server_manager.cleanup()
            self.server_manager = None
        self._initialized = False
        logger.info("MCP tool factory cleaned up")
    
    @classmethod
    def create_all_tools_from_config(cls, settings: ISetting) -> List[ITool]:
        """
        Convenience method to create all MCP tools in one call (synchronous).
        
        This is the simplest way for workflows to get MCP tools.
        
        Args:
            settings: ISetting instance for reading configuration
            
        Returns:
            List of ITool instances for all available MCP tools
        """
        return sync_create_all_mcp_tools(settings)
    
    def __repr__(self) -> str:
        """String representation of the factory."""
        if not self._initialized:
            return "MCPToolFactory(uninitialized)"
        
        if not self.server_manager:
            return "MCPToolFactory(disabled)"
        
        connected = len(self.server_manager.get_connected_servers())
        failed = len(self.server_manager.get_failed_servers())
        return f"MCPToolFactory(connected_servers={connected}, failed_servers={failed})"


def sync_create_all_mcp_tools(settings: ISetting) -> List[ITool]:
    """
    Synchronous wrapper for creating all MCP tools.
    
    This function handles the asyncio event loop management for synchronous contexts.
    
    Args:
        settings: ISetting instance for reading configuration
        
    Returns:
        List of ITool instances for all available MCP tools
    """
    try:
        # Handle event loop management
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, use thread executor
            logger.info("Creating MCP tools from within async context using thread executor")
            
            import concurrent.futures
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    async def create_tools():
                        factory = MCPToolFactory(settings)
                        return await factory.create_all_tools()
                    return new_loop.run_until_complete(create_tools())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                result = future.result(timeout=60)  # 60 second timeout
                return result
                
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            async def create_tools():
                factory = MCPToolFactory(settings)
                return await factory.create_all_tools()
            
            return asyncio.run(create_tools())
            
    except Exception as e:
        logger.error(f"Failed to create MCP tools synchronously: {e}")
        return []