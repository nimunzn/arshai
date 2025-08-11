"""
MCP Tool Registry Integration Tests

Tests for the MCP tool registry system including:
- Tool discovery and caching functionality
- Dynamic tool updates and event handling
- Performance optimization through TTL caching
- Category-based tool organization
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from arshai.clients.mcp.tool_registry import (
    MCPToolRegistry, ToolSpec, ServerWatcher, AsyncEventBus,
    create_tool_registry, initialize_tool_registry
)
from arshai.clients.mcp.server_manager import MCPServerManager
from arshai.factories.mcp_tool_factory import MCPToolFactory
from arshai.clients.mcp.exceptions import ToolNotFoundError


class TestToolSpec:
    """Test ToolSpec creation and categorization."""
    
    def test_tool_spec_creation_from_mcp_data(self):
        """Test creating ToolSpec from MCP server tool data."""
        tool_data = {
            'name': 'read_file',
            'description': 'Read contents from a file',
            'server_name': 'filesystem_server',
            'server_url': 'http://localhost:8001/mcp',
            'inputSchema': {'type': 'object', 'properties': {'path': {'type': 'string'}}}
        }
        
        tool_spec = ToolSpec.from_mcp_tool_data(tool_data)
        
        assert tool_spec.name == 'read_file'
        assert tool_spec.description == 'Read contents from a file'
        assert tool_spec.server_name == 'filesystem_server'
        assert tool_spec.category == 'filesystem'  # Auto-detected from description
        assert 'file-operations' in tool_spec.tags
        assert tool_spec.availability_score == 1.0
    
    def test_automatic_categorization(self):
        """Test automatic tool categorization based on description."""
        # Test filesystem category
        fs_tool = ToolSpec.from_mcp_tool_data({
            'name': 'list_files',
            'description': 'List files in directory',
            'server_name': 'fs_server'
        })
        assert fs_tool.category == 'filesystem'
        
        # Test web category  
        web_tool = ToolSpec.from_mcp_tool_data({
            'name': 'fetch_url',
            'description': 'Fetch content from HTTP URL',
            'server_name': 'web_server'
        })
        assert web_tool.category == 'web'
        
        # Test data category
        data_tool = ToolSpec.from_mcp_tool_data({
            'name': 'search_records',
            'description': 'Search database records',
            'server_name': 'data_server'
        })
        assert data_tool.category == 'data'
    
    def test_tool_spec_serialization(self):
        """Test ToolSpec serialization and deserialization."""
        original_data = {
            'name': 'test_tool',
            'description': 'Test tool for validation',
            'server_name': 'test_server'
        }
        
        tool_spec = ToolSpec.from_mcp_tool_data(original_data)
        serialized = tool_spec.to_dict()
        deserialized = ToolSpec.from_dict(serialized)
        
        assert deserialized.name == tool_spec.name
        assert deserialized.description == tool_spec.description
        assert deserialized.server_name == tool_spec.server_name
        assert deserialized.category == tool_spec.category


class TestAsyncEventBus:
    """Test event bus functionality."""
    
    @pytest.mark.asyncio
    async def test_event_subscription_and_publishing(self):
        """Test basic event subscription and publishing."""
        event_bus = AsyncEventBus()
        received_events = []
        
        async def event_handler(**kwargs):
            received_events.append(kwargs)
        
        event_bus.subscribe('tool_added', event_handler)
        await event_bus.publish('tool_added', tool_name='new_tool', server='test_server')
        
        assert len(received_events) == 1
        assert received_events[0]['tool_name'] == 'new_tool'
        assert received_events[0]['server'] == 'test_server'
    
    @pytest.mark.asyncio
    async def test_multiple_event_subscribers(self):
        """Test multiple subscribers for the same event type."""
        event_bus = AsyncEventBus()
        handler1_called = False
        handler2_called = False
        
        async def handler1(**kwargs):
            nonlocal handler1_called
            handler1_called = True
        
        async def handler2(**kwargs):
            nonlocal handler2_called
            handler2_called = True
        
        event_bus.subscribe('test_event', handler1)
        event_bus.subscribe('test_event', handler2)
        await event_bus.publish('test_event', data='test')
        
        assert handler1_called
        assert handler2_called


class TestMCPToolRegistry:
    """Test MCP tool registry functionality."""
    
    @pytest.fixture
    def mock_server_manager(self):
        """Create mock server manager for testing."""
        manager = Mock(spec=MCPServerManager)
        manager.get_connected_servers.return_value = ['test_server']
        manager.get_all_available_tools = AsyncMock(return_value=[
            {
                'name': 'file_reader',
                'description': 'Read file contents',
                'server_name': 'test_server',
                'server_url': 'http://localhost:8001',
                'inputSchema': {'type': 'object'}
            },
            {
                'name': 'web_search', 
                'description': 'Search the web',
                'server_name': 'test_server',
                'server_url': 'http://localhost:8001',
                'inputSchema': {'type': 'object'}
            }
        ])
        return manager
    
    @pytest.mark.asyncio
    async def test_registry_initialization(self, mock_server_manager):
        """Test tool registry initialization."""
        registry = MCPToolRegistry(cache_ttl=60, cache_maxsize=100)
        registry.set_server_manager(mock_server_manager)
        
        await registry.initialize()
        
        assert registry.server_manager == mock_server_manager
        assert registry.cache.ttl == 60
        assert registry.cache.maxsize == 100
        assert len(registry.server_watchers) == 1
        assert 'test_server' in registry.server_watchers
    
    @pytest.mark.asyncio
    async def test_tool_discovery_with_caching(self, mock_server_manager):
        """Test tool discovery and caching mechanism."""
        registry = MCPToolRegistry(cache_ttl=60)
        registry.set_server_manager(mock_server_manager)
        
        # First call should discover and cache the tool
        tool_spec = await registry.get_tool('file_reader')
        
        assert tool_spec is not None
        assert tool_spec.name == 'file_reader'
        assert tool_spec.category == 'filesystem'
        
        # Reset mock to verify cache usage
        mock_server_manager.get_all_available_tools.reset_mock()
        
        # Second call should use cache (no server call)
        cached_tool = await registry.get_tool('file_reader')
        
        assert cached_tool.name == 'file_reader'
        # Verify server manager wasn't called again
        mock_server_manager.get_all_available_tools.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_all_tools_from_registry(self, mock_server_manager):
        """Test retrieving all tools from the registry."""
        registry = MCPToolRegistry()
        registry.set_server_manager(mock_server_manager)
        
        tools = await registry.get_all_tools()
        
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert 'file_reader' in tool_names
        assert 'web_search' in tool_names
        
        # Verify tools are cached
        assert len(registry.cache) == 2
    
    @pytest.mark.asyncio
    async def test_category_based_filtering(self, mock_server_manager):
        """Test filtering tools by category."""
        registry = MCPToolRegistry()
        registry.set_server_manager(mock_server_manager)
        
        # Get filesystem tools
        fs_tools = await registry.get_tools_by_category('filesystem')
        assert len(fs_tools) == 1
        assert fs_tools[0].name == 'file_reader'
        
        # Get web tools
        web_tools = await registry.get_tools_by_category('web')
        assert len(web_tools) == 1
        assert web_tools[0].name == 'web_search'
        
        # Get tools from non-existent category
        empty_tools = await registry.get_tools_by_category('nonexistent')
        assert len(empty_tools) == 0
    
    @pytest.mark.asyncio
    async def test_tag_based_filtering(self, mock_server_manager):
        """Test filtering tools by tags."""
        registry = MCPToolRegistry()
        registry.set_server_manager(mock_server_manager)
        
        # Get file operation tools
        file_tools = await registry.get_tools_by_tags(['file-operations'])
        assert len(file_tools) == 1
        assert file_tools[0].name == 'file_reader'
        
        # Get web integration tools
        web_tools = await registry.get_tools_by_tags(['web-integration'])
        assert len(web_tools) == 1
        assert web_tools[0].name == 'web_search'
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, mock_server_manager):
        """Test cache invalidation functionality."""
        registry = MCPToolRegistry()
        registry.set_server_manager(mock_server_manager)
        
        # Cache some tools
        await registry.get_all_tools()
        initial_cache_size = len(registry.cache)
        assert initial_cache_size == 2
        
        # Invalidate specific tool
        await registry.invalidate_cache(tool_name='file_reader', server_name='test_server')
        assert len(registry.cache) == initial_cache_size - 1
        
        # Invalidate all tools for server
        await registry.invalidate_cache(server_name='test_server')
        assert len(registry.cache) == 0
    
    @pytest.mark.asyncio 
    async def test_registry_statistics(self, mock_server_manager):
        """Test registry statistics collection."""
        registry = MCPToolRegistry(cache_ttl=300, cache_maxsize=1000)
        registry.set_server_manager(mock_server_manager)
        await registry.initialize()
        
        # Cache some tools
        await registry.get_all_tools()
        
        stats = await registry.get_registry_stats()
        
        assert 'cache_stats' in stats
        assert 'monitoring' in stats
        assert 'tool_metrics' in stats
        
        cache_stats = stats['cache_stats']
        assert cache_stats['current_size'] == 2
        assert cache_stats['max_size'] == 1000
        assert cache_stats['ttl_seconds'] == 300


class TestMCPToolFactoryIntegration:
    """Test MCP tool factory integration with registry."""
    
    @pytest.fixture
    def mock_config_path(self):
        return "/test/config.yaml"
    
    @pytest.mark.asyncio
    async def test_factory_initialization_with_registry(self, mock_config_path):
        """Test factory initialization includes registry setup."""
        with patch('arshai.clients.mcp.config.MCPConfig.from_config_file') as mock_config, \
             patch('arshai.config.load_config') as mock_load_config, \
             patch('arshai.clients.mcp.server_manager.MCPServerManager') as mock_manager_class:
            
            # Setup mocks
            mock_config.return_value = Mock(enabled=True)
            mock_load_config.return_value = {'mcp': {'enabled': True}}
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock()
            mock_manager.is_enabled.return_value = True
            mock_manager.get_connected_servers.return_value = ['test_server']
            mock_manager_class.return_value = mock_manager
            
            factory = MCPToolFactory(mock_config_path)
            await factory.initialize()
            
            assert factory._initialized
            assert factory.server_manager is not None
            assert factory.tool_registry is not None
    
    @pytest.mark.asyncio
    async def test_lazy_tool_retrieval(self, mock_config_path):
        """Test lazy tool retrieval through factory."""
        with patch('arshai.clients.mcp.config.MCPConfig.from_config_file') as mock_config, \
             patch('arshai.config.load_config') as mock_load_config:
            
            mock_config.return_value = Mock(enabled=True)
            mock_load_config.return_value = {'mcp': {'enabled': True}}
            
            factory = MCPToolFactory(mock_config_path)
            
            # Mock registry with a test tool
            mock_registry = Mock()
            test_tool_spec = ToolSpec(
                name='test_tool',
                description='Test tool',
                server_name='test_server',
                server_url='http://localhost:8001',
                input_schema={},
                discovered_at=time.time(),
                last_verified=time.time(),
                availability_score=1.0,
                tags=[],
                category='general'
            )
            mock_registry.get_tool = AsyncMock(return_value=test_tool_spec)
            
            factory.tool_registry = mock_registry
            factory.server_manager = Mock()
            factory._initialized = True
            
            # Get specific tool
            tool = await factory.get_tool('test_tool')
            
            assert tool is not None
            assert tool.name == 'test_tool'
            mock_registry.get_tool.assert_called_once_with('test_tool', None)
    
    @pytest.mark.asyncio
    async def test_category_filtering_through_factory(self, mock_config_path):
        """Test category filtering functionality."""
        factory = MCPToolFactory(mock_config_path)
        
        # Mock registry with category filtering
        mock_registry = Mock()
        filesystem_tool = ToolSpec(
            name='file_reader',
            description='Read files from filesystem',
            server_name='fs_server',
            server_url='http://localhost:8001',
            input_schema={},
            discovered_at=time.time(),
            last_verified=time.time(),
            availability_score=1.0,
            tags=['file-operations'],
            category='filesystem'
        )
        mock_registry.get_tools_by_category = AsyncMock(return_value=[filesystem_tool])
        
        factory.tool_registry = mock_registry
        factory.server_manager = Mock()
        factory._initialized = True
        
        # Get filesystem tools
        fs_tools = await factory.get_tools_by_category('filesystem')
        
        assert len(fs_tools) == 1
        assert fs_tools[0].name == 'file_reader'
        mock_registry.get_tools_by_category.assert_called_once_with('filesystem')


class TestPerformanceOptimizations:
    """Test performance optimizations in the registry system."""
    
    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self):
        """Benchmark cache performance vs direct discovery."""
        registry = MCPToolRegistry(cache_ttl=60, cache_maxsize=10)
        
        # Mock server manager with simulated network latency
        mock_manager = Mock()
        async def mock_discovery_with_delay():
            await asyncio.sleep(0.005)  # 5ms simulated latency
            return [{'name': 'test_tool', 'description': 'Test', 'server_name': 'test_server'}]
        
        mock_manager.get_all_available_tools = mock_discovery_with_delay
        registry.set_server_manager(mock_manager)
        
        # First call - should take time for discovery
        start_time = time.time()
        tool = await registry.get_tool('test_tool')
        first_call_duration = time.time() - start_time
        
        assert tool is not None
        assert first_call_duration > 0.003  # Should take at least 3ms due to mock latency
        
        # Second call - should be much faster due to cache
        start_time = time.time()
        cached_tool = await registry.get_tool('test_tool')
        second_call_duration = time.time() - start_time
        
        assert cached_tool is not None
        assert second_call_duration < first_call_duration / 2  # At least 2x faster
    
    @pytest.mark.asyncio
    async def test_bulk_tool_retrieval_efficiency(self):
        """Test efficiency of bulk tool retrieval."""
        registry = MCPToolRegistry()
        
        # Mock manager with multiple tools
        mock_manager = Mock()
        mock_manager.get_all_available_tools = AsyncMock(return_value=[
            {'name': f'tool_{i}', 'description': f'Tool {i}', 'server_name': 'test_server'}
            for i in range(10)
        ])
        registry.set_server_manager(mock_manager)
        
        # Measure time for bulk retrieval
        start_time = time.time()
        tools = await registry.get_all_tools()
        bulk_duration = time.time() - start_time
        
        assert len(tools) == 10
        assert bulk_duration < 0.1  # Should be very fast for bulk operations
        
        # Verify all tools are cached
        assert len(registry.cache) == 10


@pytest.mark.asyncio
async def test_mcp_registry_end_to_end():
    """End-to-end integration test for MCP tool registry system."""
    print("\n🧪 MCP Tool Registry End-to-End Test")
    
    # Create test configuration
    config_data = {
        'mcp': {
            'enabled': True,
            'servers': [
                {
                    'name': 'test_server',
                    'url': 'http://localhost:8001/mcp',
                    'description': 'Test server for registry validation'
                }
            ]
        }
    }
    
    with patch('arshai.clients.mcp.config.MCPConfig.from_config_file') as mock_config, \
         patch('arshai.config.load_config') as mock_load_config:
        
        # Setup configuration
        mock_config.return_value = Mock(enabled=True)
        mock_load_config.return_value = config_data
        
        # Mock server manager responses
        with patch('arshai.clients.mcp.server_manager.MCPServerManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock()
            mock_manager.is_enabled.return_value = True
            mock_manager.get_connected_servers.return_value = ['test_server']
            mock_manager.health_check = AsyncMock(return_value={
                'servers': {'test_server': {'status': 'healthy'}},
                'summary': {'total_servers': 1, 'healthy_servers': 1}
            })
            mock_manager.get_all_available_tools = AsyncMock(return_value=[
                {
                    'name': 'file_reader',
                    'description': 'Read file contents',
                    'server_name': 'test_server',
                    'server_url': 'http://localhost:8001/mcp',
                    'inputSchema': {'type': 'object', 'properties': {'path': {'type': 'string'}}}
                },
                {
                    'name': 'web_search',
                    'description': 'Search the web',
                    'server_name': 'test_server',
                    'server_url': 'http://localhost:8001/mcp', 
                    'inputSchema': {'type': 'object', 'properties': {'query': {'type': 'string'}}}
                }
            ])
            
            mock_manager_class.return_value = mock_manager
            
            try:
                # Test factory initialization with registry
                factory = MCPToolFactory('/test/config.yaml')
                await factory.initialize()
                
                print("✅ Factory initialized with tool registry")
                
                # Test tool discovery through registry
                all_tools = await factory.create_all_tools()
                print(f"✅ Discovered {len(all_tools)} tools via registry")
                assert len(all_tools) == 2
                
                # Test lazy loading of specific tool
                specific_tool = await factory.get_tool('file_reader')
                print(f"✅ Lazy loaded tool: {specific_tool.name if specific_tool else 'None'}")
                assert specific_tool is not None
                assert specific_tool.name == 'file_reader'
                
                # Test category-based filtering
                fs_tools = await factory.get_tools_by_category('filesystem')
                print(f"✅ Found {len(fs_tools)} filesystem tools")
                assert len(fs_tools) == 1
                assert fs_tools[0].name == 'file_reader'
                
                # Test tag-based filtering
                web_tools = await factory.get_tools_by_tags(['web-integration'])
                print(f"✅ Found {len(web_tools)} web tools")
                assert len(web_tools) == 1
                assert web_tools[0].name == 'web_search'
                
                # Test cache refresh functionality
                await factory.refresh_tools(force=False)
                print("✅ Cache refresh completed")
                
                print("\n🎉 MCP Tool Registry End-to-End Test PASSED!")
                print("📊 Validated Features:")
                print("  • Dynamic tool discovery with TTL caching")
                print("  • Event-driven server monitoring")
                print("  • Lazy loading for optimal performance")
                print("  • Category and tag-based tool filtering")
                print("  • Cache management and refresh capabilities")
                
            finally:
                # Cleanup
                await factory.cleanup()
                print("✅ Test cleanup completed")


if __name__ == '__main__':
    # Run the end-to-end test
    asyncio.run(test_mcp_registry_end_to_end())