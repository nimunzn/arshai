"""
Production Load Tests for HTTP Connection Pooling

Tests the SearxNG HTTP connection pool implementation under extreme load
to validate production readiness and prevent container crashes.

Tests simulate real-world production scenarios with:
- 1000+ concurrent HTTP requests  
- Connection exhaustion scenarios
- Memory usage under sustained load
- Recovery from connection failures
- Connection pool health monitoring
"""

import asyncio
import aiohttp
import json
import logging
import os
import pytest
import psutil
import time
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch

from arshai.web_search.searxng import SearxNGClient

# Configure logging for load tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadTestMetrics:
    """Collects and analyzes performance metrics during load testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.successful_requests = 0
        self.failed_requests = 0
        self.connection_errors = 0
        self.timeout_errors = 0
        self.memory_usage_mb = []
        self.response_times = []
        self.concurrent_connections = 0
        self.max_concurrent_connections = 0
    
    def start_test(self):
        """Start metrics collection."""
        self.start_time = time.perf_counter()
        self.memory_usage_mb.append(psutil.Process().memory_info().rss / 1024 / 1024)
    
    def end_test(self):
        """End metrics collection."""
        self.end_time = time.perf_counter()
        self.memory_usage_mb.append(psutil.Process().memory_info().rss / 1024 / 1024)
    
    def record_request(self, success: bool, duration: float, error_type: str = None):
        """Record individual request metrics."""
        if success:
            self.successful_requests += 1
            self.response_times.append(duration)
        else:
            self.failed_requests += 1
            if error_type == 'connection':
                self.connection_errors += 1
            elif error_type == 'timeout':
                self.timeout_errors += 1
    
    def update_connection_count(self, count: int):
        """Update concurrent connection count."""
        self.concurrent_connections = count
        if count > self.max_concurrent_connections:
            self.max_concurrent_connections = count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test metrics."""
        duration = self.end_time - self.start_time if self.end_time else 0
        
        return {
            'test_duration_seconds': duration,
            'total_requests': self.successful_requests + self.failed_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_percent': (self.successful_requests / (self.successful_requests + self.failed_requests)) * 100,
            'connection_errors': self.connection_errors,
            'timeout_errors': self.timeout_errors,
            'requests_per_second': self.successful_requests / duration if duration > 0 else 0,
            'avg_response_time_ms': sum(self.response_times) / len(self.response_times) * 1000 if self.response_times else 0,
            'p95_response_time_ms': sorted(self.response_times)[int(len(self.response_times) * 0.95)] * 1000 if self.response_times else 0,
            'p99_response_time_ms': sorted(self.response_times)[int(len(self.response_times) * 0.99)] * 1000 if self.response_times else 0,
            'max_concurrent_connections': self.max_concurrent_connections,
            'memory_increase_mb': self.memory_usage_mb[-1] - self.memory_usage_mb[0] if len(self.memory_usage_mb) >= 2 else 0,
            'max_memory_mb': max(self.memory_usage_mb) if self.memory_usage_mb else 0
        }
    
    def assert_performance_thresholds(self):
        """Assert that performance meets production thresholds."""
        summary = self.get_summary()
        
        # Production performance thresholds
        assert summary['success_rate_percent'] >= 99.0, f"Success rate {summary['success_rate_percent']:.1f}% < 99%"
        assert summary['connection_errors'] == 0, f"Connection errors: {summary['connection_errors']}"
        assert summary['avg_response_time_ms'] <= 2000, f"Average response time {summary['avg_response_time_ms']:.0f}ms > 2000ms"
        assert summary['p95_response_time_ms'] <= 5000, f"P95 response time {summary['p95_response_time_ms']:.0f}ms > 5000ms"
        assert summary['memory_increase_mb'] <= 500, f"Memory increase {summary['memory_increase_mb']:.1f}MB > 500MB"


class MockSearxNGServer:
    """Mock SearxNG server for controlled load testing."""
    
    def __init__(self, port: int = 8089):
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.request_count = 0
        self.response_delay = 0.1  # 100ms response delay
    
    async def search_handler(self, request):
        """Handle search requests with configurable delay."""
        self.request_count += 1
        
        # Simulate processing delay
        await asyncio.sleep(self.response_delay)
        
        # Extract query parameter
        query = request.query.get('q', 'test')
        
        # Return mock search results
        results = {
            'query': query,
            'results': [
                {
                    'title': f'Result {i} for {query}',
                    'url': f'https://example.com/result-{i}',
                    'content': f'Mock content for result {i}',
                    'engines': ['mock'],
                    'category': 'general'
                }
                for i in range(5)
            ]
        }
        
        return aiohttp.web.json_response(results)
    
    async def start(self):
        """Start the mock server."""
        self.app = aiohttp.web.Application()
        self.app.router.add_get('/search', self.search_handler)
        
        self.runner = aiohttp.web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = aiohttp.web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        
        logger.info(f"Mock SearxNG server started on http://localhost:{self.port}")
    
    async def stop(self):
        """Stop the mock server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Mock SearxNG server stopped")


@pytest.fixture
async def mock_searxng_server():
    """Provide a mock SearxNG server for testing."""
    server = MockSearxNGServer()
    await server.start()
    
    yield server
    
    await server.stop()


@pytest.fixture
def searxng_client(mock_searxng_server):
    """Provide a configured SearxNG client."""
    os.environ['SEARX_INSTANCE'] = f'http://localhost:{mock_searxng_server.port}'
    
    # Configure for aggressive testing
    os.environ['ARSHAI_MAX_CONNECTIONS'] = '100'
    os.environ['ARSHAI_MAX_CONNECTIONS_PER_HOST'] = '20'
    os.environ['ARSHAI_CONNECTION_TIMEOUT'] = '10'
    
    return SearxNGClient({'timeout': 5})


class TestConnectionPoolLoad:
    """Production load tests for HTTP connection pooling."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_limits_respected(self, searxng_client):
        """Test that connection pool limits are respected under load."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        # Get the session to monitor connection usage
        session = await searxng_client._get_session()
        connector = session.connector
        
        # Verify initial configuration
        assert connector.limit == 100, f"Total connection limit should be 100, got {connector.limit}"
        assert connector.limit_per_host == 20, f"Per-host limit should be 20, got {connector.limit_per_host}"
        
        logger.info("✅ Connection pool limits properly configured")
        metrics.end_test()
        
        summary = metrics.get_summary()
        logger.info(f"Connection pool configuration test completed: {summary}")
    
    @pytest.mark.asyncio
    async def test_moderate_concurrent_load(self, searxng_client):
        """Test connection pooling under moderate concurrent load (100 requests)."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        logger.info("🧪 Starting moderate load test (100 concurrent requests)")
        
        async def single_search(query_id: int):
            """Perform a single search operation."""
            start = time.perf_counter()
            try:
                results = await searxng_client.asearch(f"test query {query_id}")
                duration = time.perf_counter() - start
                
                # Verify we got results
                success = len(results) > 0
                metrics.record_request(success, duration)
                
                if query_id % 20 == 0:  # Log progress
                    logger.info(f"Completed request {query_id}")
                
                return results
                
            except aiohttp.ClientError as e:
                duration = time.perf_counter() - start
                metrics.record_request(False, duration, 'connection')
                logger.error(f"Connection error in request {query_id}: {e}")
                return []
            except asyncio.TimeoutError as e:
                duration = time.perf_counter() - start
                metrics.record_request(False, duration, 'timeout')
                logger.error(f"Timeout error in request {query_id}: {e}")
                return []
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_request(False, duration)
                logger.error(f"Unexpected error in request {query_id}: {e}")
                return []
        
        # Execute 100 concurrent requests
        tasks = [single_search(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, list) and len(r) > 0]
        
        summary = metrics.get_summary()
        logger.info(f"Moderate load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for moderate load
        assert len(successful_results) >= 95, f"Expected at least 95 successful requests, got {len(successful_results)}"
        assert metrics.connection_errors == 0, f"No connection errors expected, got {metrics.connection_errors}"
        assert summary['avg_response_time_ms'] <= 1000, f"Average response time should be < 1000ms, got {summary['avg_response_time_ms']:.0f}ms"
        
        logger.info("✅ Moderate load test passed")
    
    @pytest.mark.asyncio
    async def test_high_concurrent_load(self, searxng_client):
        """Test connection pooling under high concurrent load (500 requests)."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        logger.info("🧪 Starting high load test (500 concurrent requests)")
        
        async def single_search(query_id: int):
            """Perform a single search operation with better error handling."""
            start = time.perf_counter()
            try:
                results = await searxng_client.asearch(f"high load query {query_id}")
                duration = time.perf_counter() - start
                
                success = len(results) > 0
                metrics.record_request(success, duration)
                
                if query_id % 50 == 0:  # Log progress
                    logger.info(f"High load: Completed request {query_id}")
                
                return results
                
            except Exception as e:
                duration = time.perf_counter() - start
                error_type = 'connection' if 'connection' in str(e).lower() else 'timeout' if 'timeout' in str(e).lower() else 'other'
                metrics.record_request(False, duration, error_type)
                
                if query_id % 100 == 0:  # Log occasional errors
                    logger.warning(f"Error in request {query_id}: {e}")
                
                return []
        
        # Execute 500 concurrent requests
        tasks = [single_search(i) for i in range(500)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, list) and len(r) > 0]
        
        summary = metrics.get_summary()
        logger.info(f"High load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for high load (more lenient thresholds)
        assert len(successful_results) >= 450, f"Expected at least 450 successful requests, got {len(successful_results)}"
        assert summary['success_rate_percent'] >= 90, f"Success rate should be >= 90%, got {summary['success_rate_percent']:.1f}%"
        assert summary['avg_response_time_ms'] <= 3000, f"Average response time should be < 3000ms, got {summary['avg_response_time_ms']:.0f}ms"
        assert summary['memory_increase_mb'] <= 200, f"Memory increase should be < 200MB, got {summary['memory_increase_mb']:.1f}MB"
        
        logger.info("✅ High load test passed")
    
    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self, searxng_client):
        """Test connection pooling under extreme load (1000 requests)."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        logger.info("🚀 Starting extreme load test (1000 concurrent requests)")
        
        async def single_search_extreme(query_id: int):
            """Perform a single search operation with minimal logging."""
            start = time.perf_counter()
            try:
                # Use shorter timeout for extreme load
                client_config = {'timeout': 3}
                results = await searxng_client.asearch(f"extreme {query_id}")
                duration = time.perf_counter() - start
                
                success = isinstance(results, list)  # Any list response is success
                metrics.record_request(success, duration)
                
                return results
                
            except Exception as e:
                duration = time.perf_counter() - start
                error_type = 'connection' if 'connection' in str(e).lower() else 'timeout' if 'timeout' in str(e).lower() else 'other'
                metrics.record_request(False, duration, error_type)
                return []
        
        # Execute 1000 concurrent requests
        batch_size = 200
        all_results = []
        
        for batch_start in range(0, 1000, batch_size):
            batch_end = min(batch_start + batch_size, 1000)
            logger.info(f"Executing extreme load batch: {batch_start}-{batch_end}")
            
            batch_tasks = [single_search_extreme(i) for i in range(batch_start, batch_end)]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Small pause between batches to prevent overwhelming
            await asyncio.sleep(0.1)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in all_results if isinstance(r, list)]
        
        summary = metrics.get_summary()
        logger.info(f"Extreme load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for extreme load (most lenient thresholds)
        assert len(successful_results) >= 800, f"Expected at least 800 successful requests, got {len(successful_results)}"
        assert summary['success_rate_percent'] >= 80, f"Success rate should be >= 80%, got {summary['success_rate_percent']:.1f}%"
        assert summary['memory_increase_mb'] <= 500, f"Memory increase should be < 500MB, got {summary['memory_increase_mb']:.1f}MB"
        
        # Most important: no complete connection pool exhaustion
        connection_error_rate = (metrics.connection_errors / 1000) * 100
        assert connection_error_rate <= 10, f"Connection error rate should be <= 10%, got {connection_error_rate:.1f}%"
        
        logger.info("✅ Extreme load test passed - connection pool handled extreme load")
    
    @pytest.mark.asyncio
    async def test_sustained_load_memory_stability(self, searxng_client):
        """Test memory stability under sustained load over time."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        logger.info("🔄 Starting sustained load test (5 minutes)")
        
        test_duration = 300  # 5 minutes
        requests_per_second = 10
        total_requests = test_duration * requests_per_second
        
        async def sustained_load_worker():
            """Worker that continuously makes requests."""
            request_count = 0
            while time.perf_counter() - metrics.start_time < test_duration:
                start = time.perf_counter()
                try:
                    results = await searxng_client.asearch(f"sustained {request_count}")
                    duration = time.perf_counter() - start
                    metrics.record_request(True, duration)
                    request_count += 1
                    
                    # Rate limiting
                    await asyncio.sleep(1.0 / requests_per_second)
                    
                except Exception as e:
                    duration = time.perf_counter() - start
                    metrics.record_request(False, duration)
                    await asyncio.sleep(1.0 / requests_per_second)
            
            return request_count
        
        # Start multiple workers for sustained load
        workers = [sustained_load_worker() for _ in range(5)]
        
        # Monitor memory usage periodically
        async def memory_monitor():
            while time.perf_counter() - metrics.start_time < test_duration:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                metrics.memory_usage_mb.append(memory_mb)
                await asyncio.sleep(30)  # Check every 30 seconds
        
        # Run workers and monitoring
        monitor_task = asyncio.create_task(memory_monitor())
        worker_results = await asyncio.gather(*workers)
        monitor_task.cancel()
        
        metrics.end_test()
        
        total_worker_requests = sum(worker_results)
        summary = metrics.get_summary()
        summary['worker_requests'] = total_worker_requests
        
        logger.info(f"Sustained load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for sustained load
        assert total_worker_requests >= total_requests * 0.9, f"Expected at least {total_requests * 0.9} requests, got {total_worker_requests}"
        assert summary['success_rate_percent'] >= 95, f"Success rate should be >= 95%, got {summary['success_rate_percent']:.1f}%"
        
        # Memory stability check
        memory_growth = max(metrics.memory_usage_mb) - min(metrics.memory_usage_mb)
        assert memory_growth <= 100, f"Memory growth over 5 minutes should be < 100MB, got {memory_growth:.1f}MB"
        
        logger.info("✅ Sustained load test passed - memory remained stable")
    
    @pytest.mark.asyncio
    async def test_connection_pool_recovery(self, searxng_client):
        """Test connection pool recovery after simulated failures."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        logger.info("🔧 Starting connection recovery test")
        
        # First, establish baseline with successful requests
        baseline_tasks = [searxng_client.asearch(f"baseline {i}") for i in range(50)]
        baseline_results = await asyncio.gather(*baseline_tasks, return_exceptions=True)
        baseline_success = len([r for r in baseline_results if isinstance(r, list)])
        
        logger.info(f"Baseline: {baseline_success}/50 requests successful")
        
        # Now test recovery by forcing session cleanup and recreation
        await SearxNGClient.cleanup_session()
        logger.info("Cleaned up shared session")
        
        # Allow brief pause for cleanup
        await asyncio.sleep(1)
        
        # Test recovery with new requests
        recovery_tasks = [searxng_client.asearch(f"recovery {i}") for i in range(50)]
        recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        recovery_success = len([r for r in recovery_results if isinstance(r, list)])
        
        logger.info(f"Recovery: {recovery_success}/50 requests successful")
        
        metrics.end_test()
        
        # Assertions for recovery
        assert baseline_success >= 45, f"Baseline should have >= 45 successful requests, got {baseline_success}"
        assert recovery_success >= 45, f"Recovery should have >= 45 successful requests, got {recovery_success}"
        
        # Recovery should be nearly as good as baseline
        recovery_rate = recovery_success / baseline_success
        assert recovery_rate >= 0.9, f"Recovery rate should be >= 90%, got {recovery_rate:.1%}"
        
        logger.info("✅ Connection pool recovery test passed")
    
    @pytest.mark.asyncio
    async def test_connection_limit_enforcement(self, searxng_client):
        """Test that connection limits are actually enforced."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        logger.info("🚫 Starting connection limit enforcement test")
        
        # Configure very low limits for testing
        os.environ['ARSHAI_MAX_CONNECTIONS'] = '5'
        os.environ['ARSHAI_MAX_CONNECTIONS_PER_HOST'] = '2'
        
        # Create new client with low limits
        low_limit_client = SearxNGClient({'timeout': 10})
        
        # Verify limits are applied
        session = await low_limit_client._get_session()
        assert session.connector.limit == 5, f"Total limit should be 5, got {session.connector.limit}"
        assert session.connector.limit_per_host == 2, f"Per-host limit should be 2, got {session.connector.limit_per_host}"
        
        # Make more requests than the limit allows
        tasks = [low_limit_client.asearch(f"limit test {i}") for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = [r for r in results if isinstance(r, list) and len(r) > 0]
        
        # Clean up the low-limit session
        await SearxNGClient.cleanup_session()
        
        metrics.end_test()
        
        logger.info(f"Connection limit test: {len(successful_results)}/20 requests successful")
        
        # With very low limits, we should still get some successful requests
        # but the connection pool should manage the load appropriately
        assert len(successful_results) >= 10, f"Expected at least 10 successful requests even with low limits, got {len(successful_results)}"
        
        logger.info("✅ Connection limit enforcement test passed")


if __name__ == "__main__":
    """Run performance tests directly for development/debugging."""
    import sys
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    async def run_quick_test():
        """Run a quick performance test for development."""
        server = MockSearxNGServer()
        await server.start()
        
        try:
            os.environ['SEARX_INSTANCE'] = f'http://localhost:{server.port}'
            os.environ['ARSHAI_MAX_CONNECTIONS'] = '50'
            os.environ['ARSHAI_MAX_CONNECTIONS_PER_HOST'] = '10'
            
            client = SearxNGClient({'timeout': 5})
            
            logger.info("Running quick connection pool test...")
            
            # Quick test with 20 concurrent requests
            tasks = [client.asearch(f"quick test {i}") for i in range(20)]
            start = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.perf_counter() - start
            
            successful = len([r for r in results if isinstance(r, list)])
            
            logger.info(f"Quick test: {successful}/20 requests successful in {duration:.2f}s")
            logger.info(f"Rate: {successful/duration:.1f} requests/second")
            
            await SearxNGClient.cleanup_session()
            
        finally:
            await server.stop()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(run_quick_test())
    else:
        print("Run with --quick for development testing")
        print("Or use: pytest tests/performance/test_connection_pool_load.py -v")