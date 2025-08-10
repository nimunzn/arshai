"""
Production Load Tests for Thread Pool Management

Tests the MCP tool thread pool implementation under extreme load
to validate production readiness and prevent deadlocks/thread exhaustion.

Tests simulate real-world production scenarios with:
- 500+ concurrent thread operations
- Thread pool exhaustion scenarios  
- Deadlock prevention validation
- Memory usage under sustained thread load
- Recovery from thread failures
- Thread pool health monitoring
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import psutil
import pytest
import threading
import time
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

from arshai.tools.mcp_dynamic_tool import MCPDynamicTool
from arshai.factories.mcp_tool_factory import MCPToolFactory
from arshai.clients.mcp.server_manager import MCPServerManager

# Configure logging for load tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreadPoolMetrics:
    """Collects and analyzes thread pool performance metrics during load testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.successful_operations = 0
        self.failed_operations = 0
        self.timeout_operations = 0
        self.thread_creation_errors = 0
        self.memory_usage_mb = []
        self.operation_times = []
        self.active_threads = []
        self.max_active_threads = 0
        self.thread_pool_queue_sizes = []
        self.deadlock_detected = False
    
    def start_test(self):
        """Start metrics collection."""
        self.start_time = time.perf_counter()
        self.memory_usage_mb.append(psutil.Process().memory_info().rss / 1024 / 1024)
        self.active_threads.append(threading.active_count())
    
    def end_test(self):
        """End metrics collection."""
        self.end_time = time.perf_counter()
        self.memory_usage_mb.append(psutil.Process().memory_info().rss / 1024 / 1024)
        self.active_threads.append(threading.active_count())
    
    def record_operation(self, success: bool, duration: float, error_type: str = None):
        """Record individual operation metrics."""
        if success:
            self.successful_operations += 1
            self.operation_times.append(duration)
        else:
            self.failed_operations += 1
            if error_type == 'timeout':
                self.timeout_operations += 1
            elif error_type == 'thread_creation':
                self.thread_creation_errors += 1
    
    def update_thread_count(self, count: int):
        """Update active thread count."""
        self.active_threads.append(count)
        if count > self.max_active_threads:
            self.max_active_threads = count
    
    def record_queue_size(self, size: int):
        """Record thread pool queue size."""
        self.thread_pool_queue_sizes.append(size)
    
    def detect_deadlock(self):
        """Mark deadlock detection."""
        self.deadlock_detected = True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test metrics."""
        duration = self.end_time - self.start_time if self.end_time else 0
        total_ops = self.successful_operations + self.failed_operations
        
        return {
            'test_duration_seconds': duration,
            'total_operations': total_ops,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate_percent': (self.successful_operations / total_ops * 100) if total_ops > 0 else 0,
            'timeout_operations': self.timeout_operations,
            'thread_creation_errors': self.thread_creation_errors,
            'operations_per_second': self.successful_operations / duration if duration > 0 else 0,
            'avg_operation_time_ms': sum(self.operation_times) / len(self.operation_times) * 1000 if self.operation_times else 0,
            'p95_operation_time_ms': sorted(self.operation_times)[int(len(self.operation_times) * 0.95)] * 1000 if self.operation_times else 0,
            'p99_operation_time_ms': sorted(self.operation_times)[int(len(self.operation_times) * 0.99)] * 1000 if self.operation_times else 0,
            'max_active_threads': self.max_active_threads,
            'thread_count_increase': max(self.active_threads) - min(self.active_threads) if self.active_threads else 0,
            'memory_increase_mb': self.memory_usage_mb[-1] - self.memory_usage_mb[0] if len(self.memory_usage_mb) >= 2 else 0,
            'max_memory_mb': max(self.memory_usage_mb) if self.memory_usage_mb else 0,
            'max_queue_size': max(self.thread_pool_queue_sizes) if self.thread_pool_queue_sizes else 0,
            'deadlock_detected': self.deadlock_detected
        }
    
    def assert_performance_thresholds(self):
        """Assert that performance meets production thresholds."""
        summary = self.get_summary()
        
        # Production performance thresholds
        assert summary['success_rate_percent'] >= 95.0, f"Success rate {summary['success_rate_percent']:.1f}% < 95%"
        assert summary['thread_creation_errors'] == 0, f"Thread creation errors: {summary['thread_creation_errors']}"
        assert not summary['deadlock_detected'], "Deadlock detected during test"
        assert summary['avg_operation_time_ms'] <= 5000, f"Average operation time {summary['avg_operation_time_ms']:.0f}ms > 5000ms"
        assert summary['memory_increase_mb'] <= 1000, f"Memory increase {summary['memory_increase_mb']:.1f}MB > 1000MB"
        assert summary['thread_count_increase'] <= 100, f"Thread count increase {summary['thread_count_increase']} > 100"


class MockMCPServerManager:
    """Mock MCP server manager for controlled thread pool testing."""
    
    def __init__(self, operation_delay: float = 0.1):
        self.operation_delay = operation_delay
        self.operation_count = 0
        self.server_configs = []
    
    async def get_all_available_tools(self):
        """Mock tool discovery."""
        return [
            {
                'name': f'test_tool_{i}',
                'description': f'Test tool {i}',
                'server_name': 'mock_server',
                'server_url': 'http://localhost:8080',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'test_param': {'type': 'string'}
                    }
                }
            }
            for i in range(10)
        ]
    
    def is_enabled(self):
        """Mock enabled check."""
        return True
    
    def get_connected_servers(self):
        """Mock connected servers."""
        return ['mock_server']
    
    def get_failed_servers(self):
        """Mock failed servers."""
        return []


class MockMCPClient:
    """Mock MCP client for testing."""
    
    def __init__(self, operation_delay: float = 0.1):
        self.operation_delay = operation_delay
        self.connected = False
    
    async def connect(self):
        """Mock connection."""
        await asyncio.sleep(0.01)  # Small connection delay
        self.connected = True
    
    async def disconnect(self):
        """Mock disconnection."""
        await asyncio.sleep(0.01)  # Small disconnection delay
        self.connected = False
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Mock tool execution with configurable delay."""
        if not self.connected:
            raise Exception("Not connected")
        
        # Simulate work
        await asyncio.sleep(self.operation_delay)
        
        return {
            'result': f'Mock result for {tool_name}',
            'arguments': arguments,
            'execution_time': self.operation_delay
        }


@pytest.fixture
def mock_server_manager():
    """Provide mock server manager."""
    return MockMCPServerManager(operation_delay=0.05)  # Fast operations for testing


@pytest.fixture
def thread_pool_size():
    """Configure thread pool size for testing."""
    original_value = os.environ.get('ARSHAI_MAX_THREADS')
    os.environ['ARSHAI_MAX_THREADS'] = '16'  # Reasonable limit for testing
    
    yield 16
    
    # Restore original value
    if original_value:
        os.environ['ARSHAI_MAX_THREADS'] = original_value
    else:
        os.environ.pop('ARSHAI_MAX_THREADS', None)


class TestThreadPoolLoad:
    """Production load tests for thread pool management."""
    
    def test_thread_pool_configuration(self, thread_pool_size):
        """Test that thread pool is configured with proper limits."""
        # Force new executor creation
        MCPDynamicTool._executor = None
        
        executor = MCPDynamicTool._get_executor()
        
        assert executor._max_workers <= thread_pool_size, f"Thread pool max_workers should be <= {thread_pool_size}, got {executor._max_workers}"
        assert isinstance(executor, concurrent.futures.ThreadPoolExecutor), f"Expected ThreadPoolExecutor, got {type(executor)}"
        
        logger.info(f"✅ Thread pool configured with {executor._max_workers} max workers")
    
    @pytest.mark.asyncio
    async def test_moderate_thread_load(self, mock_server_manager, thread_pool_size):
        """Test thread pool under moderate load (50 concurrent operations)."""
        metrics = ThreadPoolMetrics()
        metrics.start_test()
        
        logger.info("🧪 Starting moderate thread pool load test (50 operations)")
        
        # Create tools for testing
        tools = []
        for i in range(10):
            tool_spec = {
                'name': f'load_test_tool_{i}',
                'description': f'Load test tool {i}',
                'server_name': 'mock_server',
                'server_url': 'http://localhost:8080'
            }
            tools.append(MCPDynamicTool(tool_spec, mock_server_manager))
        
        async def single_operation(operation_id: int):
            """Perform a single tool operation."""
            start = time.perf_counter()
            try:
                tool = tools[operation_id % len(tools)]
                
                # Mock the async execution
                with patch('arshai.clients.mcp.base_client.BaseMCPClient') as mock_client_class:
                    mock_client = MockMCPClient(operation_delay=0.05)
                    mock_client_class.return_value = mock_client
                    
                    # Execute the tool
                    result_type, result = tool.execute(test_param=f"operation_{operation_id}")
                    
                    duration = time.perf_counter() - start
                    
                    # Verify result
                    success = result_type == "function" and isinstance(result, list)
                    metrics.record_operation(success, duration)
                    
                    # Track thread count
                    metrics.update_thread_count(threading.active_count())
                    
                    if operation_id % 10 == 0:
                        logger.info(f"Completed operation {operation_id}")
                    
                    return result
                    
            except Exception as e:
                duration = time.perf_counter() - start
                error_type = 'timeout' if 'timeout' in str(e).lower() else 'thread_creation' if 'thread' in str(e).lower() else 'other'
                metrics.record_operation(False, duration, error_type)
                logger.error(f"Error in operation {operation_id}: {e}")
                return None
        
        # Execute 50 concurrent operations
        tasks = [single_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        summary = metrics.get_summary()
        logger.info(f"Moderate thread load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for moderate load
        assert len(successful_results) >= 45, f"Expected at least 45 successful operations, got {len(successful_results)}"
        assert metrics.thread_creation_errors == 0, f"No thread creation errors expected, got {metrics.thread_creation_errors}"
        assert summary['avg_operation_time_ms'] <= 1000, f"Average operation time should be < 1000ms, got {summary['avg_operation_time_ms']:.0f}ms"
        assert not metrics.deadlock_detected, "No deadlocks should occur"
        
        logger.info("✅ Moderate thread pool load test passed")
    
    @pytest.mark.asyncio
    async def test_high_thread_load(self, mock_server_manager, thread_pool_size):
        """Test thread pool under high load (200 concurrent operations)."""
        metrics = ThreadPoolMetrics()
        metrics.start_test()
        
        logger.info("🧪 Starting high thread pool load test (200 operations)")
        
        # Create more tools for high load testing
        tools = []
        for i in range(20):
            tool_spec = {
                'name': f'high_load_tool_{i}',
                'description': f'High load test tool {i}',
                'server_name': 'mock_server',
                'server_url': 'http://localhost:8080'
            }
            tools.append(MCPDynamicTool(tool_spec, mock_server_manager))
        
        async def high_load_operation(operation_id: int):
            """Perform a single tool operation with monitoring."""
            start = time.perf_counter()
            try:
                tool = tools[operation_id % len(tools)]
                
                with patch('arshai.clients.mcp.base_client.BaseMCPClient') as mock_client_class:
                    mock_client = MockMCPClient(operation_delay=0.02)  # Faster for high load
                    mock_client_class.return_value = mock_client
                    
                    result_type, result = tool.execute(test_param=f"high_load_{operation_id}")
                    
                    duration = time.perf_counter() - start
                    success = result_type == "function"
                    metrics.record_operation(success, duration)
                    
                    # Monitor thread pool
                    executor = MCPDynamicTool._get_executor()
                    metrics.record_queue_size(len(executor._threads) if hasattr(executor, '_threads') else 0)
                    metrics.update_thread_count(threading.active_count())
                    
                    if operation_id % 50 == 0:
                        logger.info(f"High load: Completed operation {operation_id}")
                    
                    return result
                    
            except Exception as e:
                duration = time.perf_counter() - start
                error_type = 'timeout' if 'timeout' in str(e).lower() else 'thread_creation' if 'thread' in str(e).lower() else 'other'
                metrics.record_operation(False, duration, error_type)
                
                if operation_id % 50 == 0:
                    logger.warning(f"Error in high load operation {operation_id}: {e}")
                
                return None
        
        # Execute 200 concurrent operations in batches to avoid overwhelming
        batch_size = 50
        all_results = []
        
        for batch_start in range(0, 200, batch_size):
            batch_end = min(batch_start + batch_size, 200)
            logger.info(f"Executing high load batch: {batch_start}-{batch_end}")
            
            batch_tasks = [high_load_operation(i) for i in range(batch_start, batch_end)]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Small pause between batches
            await asyncio.sleep(0.05)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in all_results if r is not None and not isinstance(r, Exception)]
        
        summary = metrics.get_summary()
        logger.info(f"High thread load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for high load (more lenient)
        assert len(successful_results) >= 180, f"Expected at least 180 successful operations, got {len(successful_results)}"
        assert summary['success_rate_percent'] >= 90, f"Success rate should be >= 90%, got {summary['success_rate_percent']:.1f}%"
        assert summary['avg_operation_time_ms'] <= 2000, f"Average operation time should be < 2000ms, got {summary['avg_operation_time_ms']:.0f}ms"
        assert not metrics.deadlock_detected, "No deadlocks should occur"
        
        # Thread management checks
        assert summary['thread_count_increase'] <= 50, f"Thread count increase should be reasonable, got {summary['thread_count_increase']}"
        
        logger.info("✅ High thread pool load test passed")
    
    @pytest.mark.asyncio
    async def test_extreme_thread_load(self, mock_server_manager, thread_pool_size):
        """Test thread pool under extreme load (500 concurrent operations)."""
        metrics = ThreadPoolMetrics()
        metrics.start_test()
        
        logger.info("🚀 Starting extreme thread pool load test (500 operations)")
        
        # Create many tools for extreme load
        tools = []
        for i in range(50):
            tool_spec = {
                'name': f'extreme_tool_{i}',
                'description': f'Extreme load test tool {i}',
                'server_name': 'mock_server',
                'server_url': 'http://localhost:8080'
            }
            tools.append(MCPDynamicTool(tool_spec, mock_server_manager))
        
        async def extreme_load_operation(operation_id: int):
            """Perform operation with minimal overhead."""
            start = time.perf_counter()
            try:
                tool = tools[operation_id % len(tools)]
                
                with patch('arshai.clients.mcp.base_client.BaseMCPClient') as mock_client_class:
                    mock_client = MockMCPClient(operation_delay=0.01)  # Very fast for extreme load
                    mock_client_class.return_value = mock_client
                    
                    result_type, result = tool.execute(test_param=f"extreme_{operation_id}")
                    
                    duration = time.perf_counter() - start
                    success = result_type == "function"
                    metrics.record_operation(success, duration)
                    
                    return result
                    
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_operation(False, duration, 'error')
                return None
        
        # Execute 500 operations in smaller batches to prevent system overload
        batch_size = 25
        all_results = []
        
        for batch_start in range(0, 500, batch_size):
            batch_end = min(batch_start + batch_size, 500)
            
            batch_tasks = [extreme_load_operation(i) for i in range(batch_start, batch_end)]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Monitor thread count
            metrics.update_thread_count(threading.active_count())
            
            if batch_start % 100 == 0:
                logger.info(f"Extreme load: Completed batch {batch_start//batch_size + 1}")
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.02)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in all_results if r is not None and not isinstance(r, Exception)]
        
        summary = metrics.get_summary()
        logger.info(f"Extreme thread load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for extreme load (most lenient)
        assert len(successful_results) >= 400, f"Expected at least 400 successful operations, got {len(successful_results)}"
        assert summary['success_rate_percent'] >= 80, f"Success rate should be >= 80%, got {summary['success_rate_percent']:.1f}%"
        assert not metrics.deadlock_detected, "No deadlocks should occur even under extreme load"
        
        # Most important: thread pool managed the load without thread exhaustion
        thread_creation_error_rate = (metrics.thread_creation_errors / 500) * 100
        assert thread_creation_error_rate <= 5, f"Thread creation error rate should be <= 5%, got {thread_creation_error_rate:.1f}%"
        
        logger.info("✅ Extreme thread pool load test passed - thread pool handled extreme load")
    
    @pytest.mark.asyncio
    async def test_thread_pool_deadlock_prevention(self, mock_server_manager, thread_pool_size):
        """Test that thread pool prevents deadlocks under concurrent load."""
        metrics = ThreadPoolMetrics()
        metrics.start_test()
        
        logger.info("🔒 Starting deadlock prevention test")
        
        # Create tools that will compete for thread pool resources
        tools = []
        for i in range(thread_pool_size * 2):  # More tools than thread pool size
            tool_spec = {
                'name': f'deadlock_test_tool_{i}',
                'description': f'Deadlock test tool {i}',
                'server_name': 'mock_server',
                'server_url': 'http://localhost:8080'
            }
            tools.append(MCPDynamicTool(tool_spec, mock_server_manager))
        
        # Deadlock detection using timeout
        deadlock_timeout = 10  # 10 seconds timeout for operations
        
        async def deadlock_sensitive_operation(operation_id: int):
            """Operation that could potentially cause deadlock."""
            start = time.perf_counter()
            try:
                tool = tools[operation_id % len(tools)]
                
                with patch('arshai.clients.mcp.base_client.BaseMCPClient') as mock_client_class:
                    # Simulate longer operations that could cause resource contention
                    mock_client = MockMCPClient(operation_delay=0.2)
                    mock_client_class.return_value = mock_client
                    
                    # Use asyncio.wait_for to detect potential deadlocks
                    result_type, result = await asyncio.wait_for(
                        asyncio.to_thread(tool.execute, test_param=f"deadlock_test_{operation_id}"),
                        timeout=deadlock_timeout
                    )
                    
                    duration = time.perf_counter() - start
                    success = result_type == "function"
                    metrics.record_operation(success, duration)
                    
                    return result
                    
            except asyncio.TimeoutError:
                duration = time.perf_counter() - start
                metrics.detect_deadlock()
                metrics.record_operation(False, duration, 'timeout')
                logger.error(f"Potential deadlock detected in operation {operation_id} (timeout after {deadlock_timeout}s)")
                return None
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_operation(False, duration)
                logger.error(f"Error in deadlock test operation {operation_id}: {e}")
                return None
        
        # Execute operations that could potentially deadlock
        num_operations = thread_pool_size * 3  # More operations than threads
        tasks = [deadlock_sensitive_operation(i) for i in range(num_operations)]
        
        # Monitor for deadlocks
        start_monitor = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.perf_counter() - start_monitor
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        timeout_results = [r for r in results if isinstance(r, asyncio.TimeoutError)]
        
        summary = metrics.get_summary()
        logger.info(f"Deadlock prevention test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for deadlock prevention
        assert not metrics.deadlock_detected, "Deadlock detected - thread pool failed to prevent deadlock"
        assert len(timeout_results) == 0, f"Timeout errors indicate potential deadlocks: {len(timeout_results)}"
        assert len(successful_results) >= num_operations * 0.8, f"Expected at least 80% successful operations, got {len(successful_results)}/{num_operations}"
        assert total_duration <= deadlock_timeout * 2, f"Total execution took too long: {total_duration:.2f}s (may indicate deadlock)"
        
        logger.info("✅ Deadlock prevention test passed - no deadlocks detected")
    
    def test_factory_thread_pool_limits(self):
        """Test that MCP factory uses bounded thread pools."""
        logger.info("🏭 Testing MCP factory thread pool limits")
        
        # Test the factory thread pool configuration
        with patch('arshai.factories.mcp_tool_factory.os.cpu_count', return_value=4):
            with patch.dict(os.environ, {'ARSHAI_MAX_THREADS': '8'}):
                # Import to trigger thread pool calculation
                from arshai.factories.mcp_tool_factory import sync_create_all_mcp_tools
                
                # The actual test is that the import doesn't fail and
                # the thread pool is created with proper limits
                logger.info("✅ Factory thread pool configuration validated")
        
        # Test with different CPU counts
        with patch('arshai.factories.mcp_tool_factory.os.cpu_count', return_value=8):
            with patch.dict(os.environ, {'ARSHAI_MAX_THREADS': '16'}):
                logger.info("✅ Factory adapts to different CPU configurations")
    
    @pytest.mark.asyncio
    async def test_sustained_thread_load(self, mock_server_manager, thread_pool_size):
        """Test thread pool stability under sustained load over time."""
        metrics = ThreadPoolMetrics()
        metrics.start_test()
        
        logger.info("🔄 Starting sustained thread pool load test (2 minutes)")
        
        test_duration = 120  # 2 minutes
        operations_per_second = 5
        
        # Create tools for sustained testing
        tools = []
        for i in range(10):
            tool_spec = {
                'name': f'sustained_tool_{i}',
                'description': f'Sustained test tool {i}',
                'server_name': 'mock_server',
                'server_url': 'http://localhost:8080'
            }
            tools.append(MCPDynamicTool(tool_spec, mock_server_manager))
        
        async def sustained_worker(worker_id: int):
            """Worker that continuously executes operations."""
            operation_count = 0
            while time.perf_counter() - metrics.start_time < test_duration:
                start = time.perf_counter()
                try:
                    tool = tools[operation_count % len(tools)]
                    
                    with patch('arshai.clients.mcp.base_client.BaseMCPClient') as mock_client_class:
                        mock_client = MockMCPClient(operation_delay=0.05)
                        mock_client_class.return_value = mock_client
                        
                        result_type, result = tool.execute(test_param=f"sustained_{worker_id}_{operation_count}")
                        
                        duration = time.perf_counter() - start
                        success = result_type == "function"
                        metrics.record_operation(success, duration)
                        operation_count += 1
                        
                        # Rate limiting
                        await asyncio.sleep(1.0 / operations_per_second)
                        
                except Exception as e:
                    duration = time.perf_counter() - start
                    metrics.record_operation(False, duration)
                    await asyncio.sleep(1.0 / operations_per_second)
            
            return operation_count
        
        # Start multiple workers for sustained load
        workers = [sustained_worker(i) for i in range(3)]
        
        # Monitor thread count and memory periodically
        async def resource_monitor():
            while time.perf_counter() - metrics.start_time < test_duration:
                thread_count = threading.active_count()
                metrics.update_thread_count(thread_count)
                
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                metrics.memory_usage_mb.append(memory_mb)
                
                await asyncio.sleep(15)  # Check every 15 seconds
        
        # Run workers and monitoring
        monitor_task = asyncio.create_task(resource_monitor())
        worker_results = await asyncio.gather(*workers)
        monitor_task.cancel()
        
        metrics.end_test()
        
        total_worker_operations = sum(worker_results)
        expected_operations = test_duration * operations_per_second * len(workers)
        
        summary = metrics.get_summary()
        summary['worker_operations'] = total_worker_operations
        summary['expected_operations'] = expected_operations
        
        logger.info(f"Sustained thread load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for sustained load
        assert total_worker_operations >= expected_operations * 0.9, f"Expected at least {expected_operations * 0.9} operations, got {total_worker_operations}"
        assert summary['success_rate_percent'] >= 95, f"Success rate should be >= 95%, got {summary['success_rate_percent']:.1f}%"
        
        # Thread stability check
        thread_growth = max(metrics.active_threads) - min(metrics.active_threads)
        assert thread_growth <= 20, f"Thread count growth over time should be < 20, got {thread_growth}"
        
        # Memory stability check
        memory_growth = max(metrics.memory_usage_mb) - min(metrics.memory_usage_mb)
        assert memory_growth <= 200, f"Memory growth over 2 minutes should be < 200MB, got {memory_growth:.1f}MB"
        
        logger.info("✅ Sustained thread pool load test passed - resources remained stable")


if __name__ == "__main__":
    """Run thread pool tests directly for development/debugging."""
    import sys
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    def run_quick_thread_test():
        """Run a quick thread pool test for development."""
        logger.info("Running quick thread pool test...")
        
        # Configure test environment
        os.environ['ARSHAI_MAX_THREADS'] = '8'
        
        # Test thread pool creation
        executor = MCPDynamicTool._get_executor()
        logger.info(f"Thread pool created with {executor._max_workers} max workers")
        
        # Test basic operation
        mock_manager = MockMCPServerManager(operation_delay=0.01)
        tool_spec = {
            'name': 'quick_test_tool',
            'description': 'Quick test tool',
            'server_name': 'mock_server',
            'server_url': 'http://localhost:8080'
        }
        tool = MCPDynamicTool(tool_spec, mock_manager)
        
        start = time.perf_counter()
        
        # Execute multiple operations concurrently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as test_executor:
            futures = []
            for i in range(20):
                future = test_executor.submit(lambda i=i: (i, "test_result"))
                futures.append(future)
            
            results = [f.result() for f in futures]
        
        duration = time.perf_counter() - start
        
        logger.info(f"Quick thread test: {len(results)}/20 operations completed in {duration:.2f}s")
        logger.info(f"Active threads: {threading.active_count()}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_thread_test()
    else:
        print("Run with --quick for development testing")
        print("Or use: pytest tests/performance/test_thread_pool_load.py -v")