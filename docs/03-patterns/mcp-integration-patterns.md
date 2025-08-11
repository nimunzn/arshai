# MCP Integration Patterns

**Model Context Protocol (MCP) Integration Patterns for Arshai Applications**

This guide provides architectural patterns and best practices for integrating MCP (Model Context Protocol) servers with Arshai applications. These patterns ensure optimal performance, reliability, and maintainability in production environments.

## Overview

MCP integration in Arshai follows clean architecture principles with three distinct layers:
- **Application Layer**: MCPToolFactory and workflow integration
- **Domain Layer**: MCPServerManager, MCPToolRegistry, business logic
- **Infrastructure Layer**: Connection pools, protocol clients, external systems

## Core Integration Patterns

### 1. Factory-Based Tool Creation Pattern

The primary pattern for integrating MCP tools into Arshai applications.

```python
from arshai.factories.mcp_tool_factory import MCPToolFactory

class ApplicationService:
    def __init__(self, config_path: str):
        self.mcp_factory = MCPToolFactory(config_path)
        self.tools = []
    
    async def initialize(self):
        """Initialize MCP integration with connection pooling."""
        await self.mcp_factory.initialize()
        self.tools = await self.mcp_factory.create_all_tools()
    
    async def get_specialized_tools(self):
        """Get tools by category for specialized workflows."""
        fs_tools = await self.mcp_factory.get_tools_by_category("filesystem")
        web_tools = await self.mcp_factory.get_tools_by_category("web")
        return {"filesystem": fs_tools, "web": web_tools}
    
    async def cleanup(self):
        """Proper resource cleanup."""
        await self.mcp_factory.cleanup()
```

**Benefits:**
- ✅ Connection pooling with 80-90% latency reduction
- ✅ Automatic tool discovery and caching
- ✅ Circuit breaker protection
- ✅ Clean separation of concerns

**When to Use:**
- Production applications requiring MCP tool integration
- Applications with multiple MCP servers
- Systems needing high-performance tool execution

### 2. Selective Tool Loading Pattern

Optimize performance by loading only required tools instead of all available tools.

```python
from arshai.factories.mcp_tool_factory import MCPToolFactory

class OptimizedWorkflowService:
    def __init__(self, config_path: str):
        self.mcp_factory = MCPToolFactory(config_path)
    
    async def create_document_workflow(self):
        """Create workflow with only document-related tools."""
        await self.mcp_factory.initialize()
        
        # Load specific tool categories
        file_tools = await self.mcp_factory.get_tools_by_category("filesystem")
        search_tools = await self.mcp_factory.get_tools_by_tags(["search", "data-processing"])
        
        # Combine tools for specific workflow
        workflow_tools = file_tools + search_tools
        
        # Create agent with optimized tool set
        agent_config = IAgentConfig(
            task_context="Document processing and analysis",
            tools=workflow_tools,
            max_turns=5
        )
        
        return await settings.create_agent("conversation", agent_config)
    
    async def create_web_workflow(self):
        """Create workflow with only web-related tools."""
        await self.mcp_factory.initialize()
        
        web_tools = await self.mcp_factory.get_tools_by_category("web")
        api_tools = await self.mcp_factory.get_tools_by_tags(["web-integration", "api"])
        
        workflow_tools = web_tools + api_tools
        
        agent_config = IAgentConfig(
            task_context="Web research and API interactions",
            tools=workflow_tools,
            max_turns=3
        )
        
        return await settings.create_agent("conversation", agent_config)
```

**Benefits:**
- ✅ Reduced memory footprint
- ✅ Faster initialization times
- ✅ Improved cache hit rates
- ✅ Specialized tool sets per workflow

**When to Use:**
- Applications with distinct workflow types
- Memory-constrained environments
- Services with specific tool requirements

### 3. Health-Aware Connection Pattern

Implement robust error handling and health monitoring for production reliability.

```python
from arshai.factories.mcp_tool_factory import MCPToolFactory
from arshai.clients.mcp.exceptions import MCPConnectionError, MCPError

class ResilientMCPService:
    def __init__(self, config_path: str):
        self.mcp_factory = MCPToolFactory(config_path)
        self.health_check_interval = 60  # seconds
        self.max_retries = 3
    
    async def initialize_with_health_monitoring(self):
        """Initialize with comprehensive health monitoring."""
        await self.mcp_factory.initialize()
        
        # Start background health monitoring
        asyncio.create_task(self._monitor_server_health())
    
    async def execute_tool_with_fallback(self, tool_name: str, **kwargs):
        """Execute tool with fallback and retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Get tool by name
                tool = await self.mcp_factory.get_tool(tool_name)
                if not tool:
                    raise MCPError(f"Tool '{tool_name}' not found")
                
                # Execute with connection pooling
                return await tool.aexecute(**kwargs)
                
            except MCPConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
            
            except MCPError as e:
                logger.error(f"MCP tool execution failed: {e}")
                raise
        
        raise MCPConnectionError(f"Failed to execute '{tool_name}' after {self.max_retries} attempts")
    
    async def get_server_health_summary(self):
        """Get comprehensive health summary."""
        health_status = await self.mcp_factory.get_server_health()
        stats = await self.mcp_factory.get_registry_stats()
        
        summary = {
            "servers": {},
            "overall_health": "healthy",
            "performance_metrics": {
                "cache_hit_rate": stats['registry']['cache_stats']['hit_rate'],
                "average_latency": stats['registry']['tool_metrics']['average_latency_ms']
            }
        }
        
        unhealthy_servers = 0
        for server_name, status in health_status.items():
            is_healthy = status['status'] == 'healthy'
            if not is_healthy:
                unhealthy_servers += 1
            
            pool_stats = status.get('pool_stats', {})
            summary["servers"][server_name] = {
                "status": status['status'],
                "connected": status['connected'],
                "pool_utilization": (
                    pool_stats.get('active_connections', 0) / 
                    pool_stats.get('max_connections', 1)
                ) if pool_stats else 0,
                "connection_reuse_rate": (
                    pool_stats.get('total_reused', 0) / 
                    max(1, pool_stats.get('total_created', 1))
                ) if pool_stats else 0
            }
        
        # Set overall health status
        if unhealthy_servers == 0:
            summary["overall_health"] = "healthy"
        elif unhealthy_servers < len(health_status) / 2:
            summary["overall_health"] = "degraded"
        else:
            summary["overall_health"] = "unhealthy"
        
        return summary
    
    async def _monitor_server_health(self):
        """Background health monitoring task."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                health_summary = await self.get_server_health_summary()
                
                if health_summary["overall_health"] != "healthy":
                    logger.warning(f"MCP health degraded: {health_summary}")
                    
                    # Trigger reconnection for failed servers
                    await self._attempt_server_recovery()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _attempt_server_recovery(self):
        """Attempt to recover failed servers."""
        health_status = await self.mcp_factory.get_server_health()
        
        for server_name, status in health_status.items():
            if status['status'] != 'healthy':
                try:
                    # Trigger server reconnection
                    await self.mcp_factory.reconnect_server(server_name)
                    logger.info(f"Successfully recovered server '{server_name}'")
                except Exception as e:
                    logger.error(f"Failed to recover server '{server_name}': {e}")
```

**Benefits:**
- ✅ Production-ready error handling
- ✅ Automatic server recovery
- ✅ Comprehensive health monitoring
- ✅ Exponential backoff retry logic

**When to Use:**
- Production applications requiring high availability
- Services with critical MCP dependencies
- Systems needing automatic recovery

### 4. Multi-Server Load Balancing Pattern

Distribute tool execution across multiple MCP servers for optimal performance.

```python
from arshai.factories.mcp_tool_factory import MCPToolFactory
import random
from typing import List, Dict, Any

class LoadBalancedMCPService:
    def __init__(self, config_path: str):
        self.mcp_factory = MCPToolFactory(config_path)
        self.server_weights = {}  # Server performance weights
        self.load_stats = {}      # Track server load
    
    async def initialize(self):
        """Initialize with load balancing capabilities."""
        await self.mcp_factory.initialize()
        
        # Initialize server weights based on configuration
        connected_servers = self.mcp_factory.get_connected_servers()
        for server in connected_servers:
            self.server_weights[server] = 1.0  # Equal weight initially
            self.load_stats[server] = {"requests": 0, "failures": 0, "avg_latency": 0}
    
    async def execute_tool_with_load_balancing(self, tool_name: str, **kwargs):
        """Execute tool with intelligent server selection."""
        # Get all servers that have this tool
        available_servers = await self._get_servers_with_tool(tool_name)
        
        if not available_servers:
            raise MCPError(f"Tool '{tool_name}' not available on any server")
        
        # Select optimal server based on load and performance
        selected_server = await self._select_optimal_server(available_servers)
        
        # Execute tool on selected server
        start_time = time.time()
        try:
            tool = await self.mcp_factory.get_tool(tool_name, selected_server)
            result = await tool.aexecute(**kwargs)
            
            # Update success stats
            execution_time = (time.time() - start_time) * 1000
            await self._update_server_stats(selected_server, execution_time, success=True)
            
            return result
            
        except Exception as e:
            # Update failure stats
            execution_time = (time.time() - start_time) * 1000
            await self._update_server_stats(selected_server, execution_time, success=False)
            raise
    
    async def _get_servers_with_tool(self, tool_name: str) -> List[str]:
        """Find all servers that provide the specified tool."""
        all_tools = await self.mcp_factory.get_all_tools()
        servers_with_tool = []
        
        for tool_spec in all_tools:
            if tool_spec['name'] == tool_name:
                servers_with_tool.append(tool_spec['server_name'])
        
        return list(set(servers_with_tool))  # Remove duplicates
    
    async def _select_optimal_server(self, available_servers: List[str]) -> str:
        """Select the optimal server based on load and performance."""
        if len(available_servers) == 1:
            return available_servers[0]
        
        # Calculate server scores based on multiple factors
        server_scores = {}
        
        for server in available_servers:
            stats = self.load_stats.get(server, {"requests": 0, "failures": 0, "avg_latency": 50})
            weight = self.server_weights.get(server, 1.0)
            
            # Calculate composite score (lower is better)
            failure_rate = stats["failures"] / max(1, stats["requests"])
            latency_factor = stats["avg_latency"] / 50.0  # Normalize to 50ms baseline
            load_factor = stats["requests"] / 100.0       # Normalize to 100 requests baseline
            
            score = (
                failure_rate * 0.4 +      # 40% weight on reliability
                latency_factor * 0.4 +    # 40% weight on performance  
                load_factor * 0.2         # 20% weight on current load
            ) / weight
            
            server_scores[server] = score
        
        # Select server with lowest score (best performance)
        optimal_server = min(server_scores.items(), key=lambda x: x[1])[0]
        return optimal_server
    
    async def _update_server_stats(self, server: str, execution_time: float, success: bool):
        """Update server performance statistics."""
        if server not in self.load_stats:
            self.load_stats[server] = {"requests": 0, "failures": 0, "avg_latency": 50}
        
        stats = self.load_stats[server]
        stats["requests"] += 1
        
        if not success:
            stats["failures"] += 1
            # Reduce server weight for poor performance
            self.server_weights[server] = max(0.1, self.server_weights[server] * 0.9)
        else:
            # Update rolling average latency
            current_avg = stats["avg_latency"]
            stats["avg_latency"] = (current_avg * 0.8) + (execution_time * 0.2)
            
            # Increase server weight for good performance
            if execution_time < current_avg:
                self.server_weights[server] = min(2.0, self.server_weights[server] * 1.1)
    
    async def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics."""
        return {
            "server_weights": self.server_weights,
            "load_stats": self.load_stats,
            "total_requests": sum(stats["requests"] for stats in self.load_stats.values()),
            "overall_failure_rate": sum(stats["failures"] for stats in self.load_stats.values()) / 
                                   max(1, sum(stats["requests"] for stats in self.load_stats.values()))
        }
```

**Benefits:**
- ✅ Optimal server utilization
- ✅ Automatic performance adaptation
- ✅ High availability through redundancy
- ✅ Load-aware request distribution

**When to Use:**
- Multi-server MCP environments
- High-throughput applications
- Systems requiring maximum availability

### 5. Configuration-Driven Pattern

Flexible MCP integration that adapts to different environments through configuration.

```python
from arshai.factories.mcp_tool_factory import MCPToolFactory
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MCPEnvironmentConfig:
    """Environment-specific MCP configuration."""
    environment: str  # dev, staging, prod
    connection_pool_size: int
    cache_ttl: int
    health_check_interval: int
    retry_attempts: int
    timeout: int
    
    @classmethod
    def for_environment(cls, env: str):
        """Factory method for environment-specific configs."""
        configs = {
            "dev": cls(
                environment="dev",
                connection_pool_size=2,
                cache_ttl=60,
                health_check_interval=120,
                retry_attempts=1,
                timeout=10
            ),
            "staging": cls(
                environment="staging", 
                connection_pool_size=5,
                cache_ttl=300,
                health_check_interval=60,
                retry_attempts=2,
                timeout=20
            ),
            "prod": cls(
                environment="prod",
                connection_pool_size=10,
                cache_ttl=600,
                health_check_interval=30,
                retry_attempts=3,
                timeout=30
            )
        }
        return configs.get(env, configs["dev"])

class ConfigurableMCPService:
    def __init__(self, base_config_path: str, environment: str = "dev"):
        self.env_config = MCPEnvironmentConfig.for_environment(environment)
        self.base_config_path = base_config_path
        self.mcp_factory: Optional[MCPToolFactory] = None
    
    async def initialize(self):
        """Initialize with environment-specific configuration."""
        # Load base configuration
        base_config = await self._load_base_config()
        
        # Apply environment-specific overrides
        adjusted_config = self._apply_environment_config(base_config)
        
        # Create factory with adjusted configuration
        self.mcp_factory = MCPToolFactory(adjusted_config)
        await self.mcp_factory.initialize()
    
    def _apply_environment_config(self, base_config: Dict) -> Dict:
        """Apply environment-specific configuration overrides."""
        config = base_config.copy()
        
        # Update global MCP settings
        mcp_config = config.setdefault("mcp", {})
        mcp_config["connection_timeout"] = self.env_config.timeout
        mcp_config["default_max_retries"] = self.env_config.retry_attempts
        mcp_config["default_max_connections"] = self.env_config.connection_pool_size
        mcp_config["default_health_check_interval"] = self.env_config.health_check_interval
        
        # Update server-specific settings
        for server in mcp_config.get("servers", []):
            server["max_connections"] = min(
                server.get("max_connections", self.env_config.connection_pool_size),
                self.env_config.connection_pool_size
            )
            server["timeout"] = self.env_config.timeout
            server["max_retries"] = self.env_config.retry_attempts
            server["health_check_interval"] = self.env_config.health_check_interval
        
        return config
    
    async def _load_base_config(self) -> Dict:
        """Load base configuration from file."""
        from arshai.config import load_config
        return load_config(self.base_config_path)
    
    async def create_environment_optimized_workflow(self):
        """Create workflow optimized for current environment."""
        if not self.mcp_factory:
            await self.initialize()
        
        if self.env_config.environment == "dev":
            # Development: Load all tools for testing
            tools = await self.mcp_factory.create_all_tools()
        elif self.env_config.environment == "staging":
            # Staging: Selective loading for performance testing
            tools = await self._get_staging_tools()
        else:  # production
            # Production: Highly optimized tool selection
            tools = await self._get_production_tools()
        
        return tools
    
    async def _get_staging_tools(self) -> List:
        """Get tools optimized for staging environment."""
        # Load core tools by category
        fs_tools = await self.mcp_factory.get_tools_by_category("filesystem")
        web_tools = await self.mcp_factory.get_tools_by_category("web")
        
        # Limit to essential tools for testing
        return (fs_tools[:3] if len(fs_tools) > 3 else fs_tools) + \
               (web_tools[:2] if len(web_tools) > 2 else web_tools)
    
    async def _get_production_tools(self) -> List:
        """Get tools optimized for production environment."""
        # Load only high-performance, well-tested tools
        performance_tags = ["high-performance", "production-ready", "core"]
        return await self.mcp_factory.get_tools_by_tags(performance_tags)
    
    async def get_environment_info(self) -> Dict:
        """Get current environment configuration information."""
        stats = await self.mcp_factory.get_registry_stats() if self.mcp_factory else {}
        
        return {
            "environment": self.env_config.environment,
            "configuration": {
                "connection_pool_size": self.env_config.connection_pool_size,
                "cache_ttl": self.env_config.cache_ttl,
                "health_check_interval": self.env_config.health_check_interval,
                "retry_attempts": self.env_config.retry_attempts,
                "timeout": self.env_config.timeout
            },
            "runtime_stats": stats
        }
```

**Benefits:**
- ✅ Environment-specific optimization
- ✅ Flexible configuration management
- ✅ Easy deployment across environments
- ✅ Performance tuning per environment

**When to Use:**
- Multi-environment deployments
- Applications with varying performance requirements
- Systems needing different configurations per stage

## Integration Best Practices

### 1. Connection Pool Configuration

```yaml
# Production-optimized configuration
mcp:
  enabled: true
  connection_timeout: 30
  default_max_retries: 3
  default_max_connections: 10
  default_min_connections: 2
  default_health_check_interval: 60
  
  servers:
    - name: "high_traffic_server"
      url: "http://api.example.com/mcp"
      max_connections: 20      # High capacity
      min_connections: 5       # Always ready
      health_check_interval: 30
    
    - name: "low_traffic_server"
      url: "http://internal.example.com/mcp"
      max_connections: 5       # Conservative
      min_connections: 1       # Minimal overhead
      health_check_interval: 60
```

### 2. Error Handling Strategy

```python
class RobustMCPIntegration:
    async def execute_with_comprehensive_error_handling(self, tool_name: str, **kwargs):
        """Execute tool with comprehensive error handling."""
        try:
            tool = await self.mcp_factory.get_tool(tool_name)
            return await tool.aexecute(**kwargs)
            
        except MCPConnectionError as e:
            # Server connectivity issues
            logger.error(f"MCP connection error: {e}")
            await self._handle_connection_error(e)
            raise
            
        except MCPToolError as e:
            # Tool-specific errors
            logger.error(f"MCP tool error: {e}")
            await self._handle_tool_error(e)
            raise
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected MCP error: {e}")
            await self._handle_unexpected_error(e)
            raise
    
    async def _handle_connection_error(self, error):
        """Handle connection errors with retry logic."""
        # Implement exponential backoff, circuit breaker, etc.
        pass
    
    async def _handle_tool_error(self, error):
        """Handle tool-specific errors."""
        # Log tool errors, update tool registry, etc.
        pass
    
    async def _handle_unexpected_error(self, error):
        """Handle unexpected errors."""
        # Alert monitoring, failsafe mechanisms, etc.
        pass
```

### 3. Performance Monitoring Integration

```python
class MonitoredMCPService:
    def __init__(self, config_path: str, metrics_client):
        self.mcp_factory = MCPToolFactory(config_path)
        self.metrics = metrics_client
    
    async def execute_with_metrics(self, tool_name: str, **kwargs):
        """Execute tool with comprehensive metrics collection."""
        start_time = time.time()
        
        try:
            result = await self.mcp_factory.get_tool(tool_name)
            execution_time = (time.time() - start_time) * 1000
            
            # Record success metrics
            self.metrics.histogram("mcp_tool_execution_duration_ms", execution_time, 
                                 tags={"tool": tool_name, "status": "success"})
            self.metrics.increment("mcp_tool_executions_total", 
                                 tags={"tool": tool_name, "status": "success"})
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Record error metrics
            self.metrics.histogram("mcp_tool_execution_duration_ms", execution_time,
                                 tags={"tool": tool_name, "status": "error"})
            self.metrics.increment("mcp_tool_executions_total",
                                 tags={"tool": tool_name, "status": "error", "error_type": type(e).__name__})
            
            raise
    
    async def collect_pool_metrics(self):
        """Collect and emit connection pool metrics."""
        stats = await self.mcp_factory.get_registry_stats()
        
        for server_name, server_stats in stats.get('servers', {}).items():
            pool_stats = server_stats.get('pool_stats', {})
            
            self.metrics.gauge("mcp_pool_active_connections", 
                             pool_stats.get('active_connections', 0),
                             tags={"server": server_name})
            self.metrics.gauge("mcp_pool_utilization", 
                             pool_stats.get('active_connections', 0) / pool_stats.get('max_connections', 1),
                             tags={"server": server_name})
```

## Common Anti-Patterns to Avoid

### ❌ Direct Client Creation Anti-Pattern

```python
# DON'T DO THIS - Creates connection anti-pattern
from arshai.clients.mcp.base_client import BaseMCPClient

async def bad_mcp_usage():
    # This creates a new connection for every operation (50-100ms overhead)
    client = BaseMCPClient(server_config)
    await client.connect()
    result = await client.call_tool("tool_name", {})
    await client.disconnect()
    return result
```

### ✅ Use Factory Pattern Instead

```python
# DO THIS - Uses connection pooling (5-10ms overhead)
from arshai.factories.mcp_tool_factory import MCPToolFactory

async def good_mcp_usage():
    factory = MCPToolFactory("config.yaml")
    await factory.initialize()
    
    tool = await factory.get_tool("tool_name")
    result = await tool.aexecute()  # Uses connection pool automatically
    
    await factory.cleanup()
    return result
```

### ❌ Blocking Operations Anti-Pattern

```python
# DON'T DO THIS - Blocks event loop
def blocking_mcp_usage():
    factory = MCPToolFactory("config.yaml")
    asyncio.run(factory.initialize())  # Blocking in async context
    
    tools = asyncio.run(factory.create_all_tools())  # More blocking
    return tools
```

### ✅ Proper Async Usage

```python
# DO THIS - Proper async handling
async def async_mcp_usage():
    factory = MCPToolFactory("config.yaml")
    await factory.initialize()
    
    tools = await factory.create_all_tools()
    return tools
```

## Testing Patterns

### Unit Testing MCP Integration

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from arshai.factories.mcp_tool_factory import MCPToolFactory

class TestMCPIntegration:
    @pytest.fixture
    async def mcp_service(self):
        """Create test MCP service with mocked dependencies."""
        service = MCPService("test_config.yaml")
        service.mcp_factory = AsyncMock(spec=MCPToolFactory)
        return service
    
    async def test_tool_execution_success(self, mcp_service):
        """Test successful tool execution."""
        # Mock tool behavior
        mock_tool = AsyncMock()
        mock_tool.aexecute.return_value = "test_result"
        mcp_service.mcp_factory.get_tool.return_value = mock_tool
        
        # Execute
        result = await mcp_service.execute_tool("test_tool", param="value")
        
        # Verify
        assert result == "test_result"
        mcp_service.mcp_factory.get_tool.assert_called_once_with("test_tool")
        mock_tool.aexecute.assert_called_once_with(param="value")
    
    async def test_connection_error_handling(self, mcp_service):
        """Test connection error handling."""
        from arshai.clients.mcp.exceptions import MCPConnectionError
        
        # Mock connection error
        mcp_service.mcp_factory.get_tool.side_effect = MCPConnectionError("Connection failed", "test_server")
        
        # Execute and verify exception
        with pytest.raises(MCPConnectionError):
            await mcp_service.execute_tool("test_tool")
```

### Integration Testing

```python
class TestMCPIntegrationReal:
    @pytest.fixture
    async def real_mcp_factory(self):
        """Create real MCP factory for integration tests."""
        factory = MCPToolFactory("test_config.yaml")
        await factory.initialize()
        yield factory
        await factory.cleanup()
    
    async def test_real_connection_pooling(self, real_mcp_factory):
        """Test actual connection pooling performance."""
        tools = await real_mcp_factory.create_all_tools()
        
        if not tools:
            pytest.skip("No MCP servers available for integration test")
        
        tool = tools[0]
        
        # Execute multiple times to test connection reuse
        start_time = time.time()
        results = []
        
        for i in range(10):
            result = await tool.aexecute(test_param=f"test_{i}")
            results.append(result)
        
        duration = time.time() - start_time
        avg_duration = (duration / 10) * 1000  # Convert to ms
        
        # Verify connection pooling performance
        assert avg_duration < 50, f"Average execution time {avg_duration:.1f}ms too high, connection pooling may not be working"
        assert len(results) == 10, "Not all tool executions completed"
```

## Migration Guide

### From Direct Client Usage to Factory Pattern

1. **Identify Direct Client Usage:**
```python
# Old pattern
client = BaseMCPClient(config)
await client.connect()
result = await client.call_tool("tool", {})
await client.disconnect()
```

2. **Replace with Factory Pattern:**
```python
# New pattern
factory = MCPToolFactory("config.yaml")
await factory.initialize()
tool = await factory.get_tool("tool")
result = await tool.aexecute()
await factory.cleanup()
```

3. **Update Configuration:**
```yaml
# Add connection pool settings to existing config
mcp:
  enabled: true
  # Add these new settings:
  default_max_connections: 10
  default_min_connections: 2
  default_health_check_interval: 60
  
  servers:
    - name: "existing_server"
      # Add pool settings:
      max_connections: 5
      min_connections: 1
```

4. **Test Performance Improvement:**
```python
# Measure before/after performance
import time

async def measure_performance():
    start = time.time()
    
    # Execute 100 tool calls
    for i in range(100):
        result = await tool.aexecute()
    
    duration = time.time() - start
    avg_latency = (duration / 100) * 1000
    
    print(f"Average latency: {avg_latency:.1f}ms")
    # Should see 80-90% improvement with connection pooling
```

## Conclusion

These MCP integration patterns provide a solid foundation for building high-performance, reliable applications with Arshai's MCP capabilities. The patterns emphasize:

- **Performance**: Connection pooling for 80-90% latency reduction
- **Reliability**: Circuit breaker protection and health monitoring
- **Scalability**: Load balancing and selective tool loading
- **Maintainability**: Clean architecture and configuration-driven design

Choose the appropriate pattern based on your application's specific requirements, and always follow the best practices to ensure optimal performance and reliability in production environments.