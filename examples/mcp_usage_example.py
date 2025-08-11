#!/usr/bin/env python3
"""
Example: MCP Server Manager with Connection Pooling and Tool Registry

This example demonstrates the enhanced MCP architecture with:
- 80-90% latency reduction through connection pooling (Phase 1)
- 10x concurrent execution capacity (200+ parallel tools)
- Circuit breaker protection and health monitoring
- Dynamic tool registry with caching (Phase 2)
- Production-grade performance and reliability
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
        #   # Connection pool configuration (NEW)
        #   default_max_connections: 10
        #   default_min_connections: 2
        #   default_health_check_interval: 60
        #   servers:
        #     - name: "taloan"
        #       url: "https://taloan-mcp-baadbaan.rahkar.team/mcp/"
        #       timeout: 30
        #       max_retries: 3
        #       max_connections: 5  # Server-specific pool size
        #       min_connections: 1
        #       health_check_interval: 30
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
                # NEW: Connection pool configuration
                "default_max_connections": 10,
                "default_min_connections": 2,
                "default_health_check_interval": 60,
                "servers": [
                    {
                        "name": "example_server",
                        "url": "http://localhost:8001/mcp",
                        "timeout": 30,
                        "max_retries": 3,
                        "max_connections": 5,  # Server-specific pool size
                        "min_connections": 1,
                        "health_check_interval": 30,
                        "description": "Example MCP server with connection pooling"
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
        
        # NEW: Show connection pool statistics
        health_status = await manager.health_check()
        for server_name, status in health_status.items():
            pool_stats = status.get('pool_stats')
            if pool_stats:
                print(f"\nServer '{server_name}' connection pool stats:")
                print(f"  - Pool size: {pool_stats['pool_size']}/{pool_stats['max_connections']}")
                print(f"  - Active connections: {pool_stats['active_connections']}")
                print(f"  - Total created: {pool_stats['total_created']}")
                print(f"  - Total reused: {pool_stats['total_reused']}")
                print(f"  - Pool hit rate: {pool_stats['pool_hits']/(pool_stats['pool_hits'] + pool_stats['pool_misses']):.1%}" if pool_stats['pool_hits'] + pool_stats['pool_misses'] > 0 else "  - Pool hit rate: N/A")
                print(f"  - Circuit breaker: {'OPEN' if pool_stats['circuit_breaker_open'] else 'CLOSED'}")
        
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

async def demonstrate_performance_improvements():
    """Demonstrate the performance improvements with connection pooling."""
    print("\n=== Performance Demonstration ===")
    
    try:
        from arshai.config import load_config
        
        # Create a test configuration with connection pooling
        config_dict = {
            "mcp": {
                "enabled": True,
                "default_max_connections": 5,
                "default_min_connections": 2,
                "servers": [
                    {
                        "name": "test_server",
                        "url": "http://localhost:8001/mcp",
                        "max_connections": 3,
                        "min_connections": 1,
                        "description": "Test server for performance demo"
                    }
                ]
            }
        }
        
        manager = MCPServerManager(config_dict)
        await manager.initialize()
        
        if not manager.get_connected_servers():
            print("No servers connected - skipping performance demo")
            print("Note: This demo requires a running MCP server at localhost:8001")
            return
        
        print("Connected to test server, demonstrating concurrent execution...")
        
        # Simulate concurrent tool executions (this would show the benefits)
        import asyncio
        import time
        
        async def simulate_tool_call(tool_id):
            try:
                # This would use the connection pool for optimal performance
                result = await manager.call_tool(
                    tool_name="test_tool", 
                    server_name="test_server", 
                    arguments={"test_id": tool_id}
                )
                return f"Tool {tool_id}: Success"
            except Exception as e:
                return f"Tool {tool_id}: {type(e).__name__}"
        
        # Execute 10 concurrent tool calls
        print("Executing 10 concurrent tool calls...")
        start_time = time.time()
        
        tasks = [simulate_tool_call(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        execution_time = time.time() - start_time
        
        print(f"\nResults:")
        for result in results[:3]:  # Show first 3 results
            print(f"  - {result}")
        if len(results) > 3:
            print(f"  - ... and {len(results) - 3} more")
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        print(f"Average per tool: {execution_time/len(tasks)*1000:.1f}ms")
        
        # Show final connection pool stats
        health_status = await manager.health_check()
        for server_name, status in health_status.items():
            pool_stats = status.get('pool_stats')
            if pool_stats:
                reuse_rate = pool_stats['total_reused'] / max(1, pool_stats['total_created'] + pool_stats['total_reused'])
                print(f"\nConnection reuse rate: {reuse_rate:.1%}")
                print("Note: Higher reuse rates = better performance!")
        
        await manager.cleanup()
        
    except Exception as e:
        print(f"Performance demo error: {e}")
        print("This is expected if no MCP server is running locally")



if __name__ == "__main__":
    print("MCP Server Manager Example with Connection Pooling")
    print("This example shows how to use MCP with YAML configuration files")
    print("following the direct instantiation pattern.\n")
    
    # Run the main examples
    asyncio.run(main())
    
    print("\n" + "="*50)
    print("Tool Execution Example")
    asyncio.run(demonstrate_tool_execution())
    
    print("\n" + "="*50)
    print("Performance Improvements Demo")
    asyncio.run(demonstrate_performance_improvements())