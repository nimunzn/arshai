#!/usr/bin/env python3
"""
Example: Using MCP Server Manager with YAML Configuration

This example demonstrates how to use the MCP server manager to connect to 
MCP servers configured in a YAML file, following the direct instantiation pattern.
"""

import asyncio
import logging
from arshai.clients.mcp.server_manager import MCPServerManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Demonstrate MCP server manager usage with YAML configuration."""
    
    # Method 1: Load directly from YAML file (recommended)
    print("=== Method 1: Loading from config.yaml ===")
    try:
        # This expects a config.yaml file with mcp section like:
        # mcp:
        #   enabled: true
        #   connection_timeout: 30
        #   default_max_retries: 3
        #   servers:
        #     - name: "taloan"
        #       url: "https://taloan-mcp-baadbaan.rahkar.team/mcp/"
        #       timeout: 30
        #       max_retries: 3
        #       description: "Taloan MCP server for user and loan information"
        
        from arshai.config import load_config
        config_dict = load_config("config.yaml")
        manager = MCPServerManager(config_dict)
        await manager.initialize()
        
        # List available tools from all connected servers
        tools = await manager.get_all_available_tools()
        print(f"Available tools: {len(tools)}")
        
        # Test connection status
        status = await manager.health_check()
        print(f"Server status: {status}")
        
    except FileNotFoundError:
        print("config.yaml not found, trying method 2...")
    except Exception as e:
        print(f"Error with method 1: {e}")
    
    # Method 2: Create configuration programmatically
    print("\n=== Method 2: Programmatic configuration ===")
    try:
        # Create config dictionary
        config_dict = {
            "mcp": {
                "enabled": True,
                "connection_timeout": 30,
                "default_max_retries": 3,
                "servers": [
                    {
                        "name": "example_server",
                        "url": "http://localhost:8001/mcp",
                        "timeout": 30,
                        "max_retries": 3,
                        "description": "Example MCP server"
                    }
                ]
            }
        }
        
        # Create manager with config dictionary
        manager = MCPServerManager(config_dict)
        
        # Initialize (this will attempt to connect)
        await manager.initialize()
        
        # Check which servers connected successfully
        connected_servers = manager.get_connected_servers()
        print(f"Connected servers: {connected_servers}")
        
        failed_servers = manager.get_failed_servers() 
        if failed_servers:
            print(f"Failed servers: {failed_servers}")
        
    except Exception as e:
        print(f"Error with method 2: {e}")
    
    # Method 3: Handle disabled MCP
    print("\n=== Method 3: Disabled MCP configuration ===")
    try:
        disabled_config = {
            "mcp": {
                "enabled": False,
                "servers": []
            }
        }
        
        manager = MCPServerManager(disabled_config)
        await manager.initialize()
        
        print("MCP is disabled - no servers to connect to")
        
    except Exception as e:
        print(f"Error with method 3: {e}")

async def demonstrate_tool_execution():
    """Demonstrate executing tools through the MCP manager."""
    try:
        # Load from config file
        manager = MCPServerManager.from_config_file("config.yaml")
        await manager.initialize()
        
        if not manager.get_connected_servers():
            print("No servers connected - cannot demonstrate tool execution")
            return
        
        # List available tools
        tools = await manager.list_all_tools()
        print(f"Available tools: {list(tools.keys())}")
        
        # Execute a tool (example - adjust based on actual available tools)
        # This is just an example - actual tool names depend on your MCP server
        if "get_user_info" in tools:
            result = await manager.execute_tool("get_user_info", {
                "user_id": "12345"
            })
            print(f"Tool result: {result}")
        
    except Exception as e:
        print(f"Error in tool execution demo: {e}")

if __name__ == "__main__":
    print("MCP Server Manager Example")
    print("This example shows how to use MCP with YAML configuration files")
    print("following the direct instantiation pattern.\n")
    
    # Run the main examples
    asyncio.run(main())
    
    print("\n" + "="*50)
    print("Tool Execution Example")
    asyncio.run(demonstrate_tool_execution())