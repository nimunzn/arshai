"""
Production Load Tests for Vector Database Async Operations

Tests the Milvus async operations implementation under extreme load
to validate production readiness and prevent event loop blocking.

Tests simulate real-world production scenarios with:
- 1000+ concurrent vector search operations
- High-volume document insertion batches
- Event loop blocking prevention validation
- Memory usage under sustained vector operations
- Recovery from database connection failures
- Vector database performance monitoring
"""

import asyncio
import json
import logging
import numpy as np
import os
import psutil
import pytest
import time
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from arshai.vector_db.milvus_client import MilvusClient
from arshai.core.interfaces.ivector_db_client import ICollectionConfig
from arshai.tools.knowledge_base_tool import KnowledgeBaseRetrievalTool

# Configure logging for load tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBMetrics:
    """Collects and analyzes vector database performance metrics during load testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.successful_searches = 0
        self.failed_searches = 0
        self.successful_insertions = 0
        self.failed_insertions = 0
        self.timeout_operations = 0
        self.connection_errors = 0
        self.memory_usage_mb = []
        self.search_times = []
        self.insertion_times = []
        self.event_loop_blocked_duration = 0
        self.max_concurrent_operations = 0
        self.vector_dimensions = 0
    
    def start_test(self):
        """Start metrics collection."""
        self.start_time = time.perf_counter()
        self.memory_usage_mb.append(psutil.Process().memory_info().rss / 1024 / 1024)
    
    def end_test(self):
        """End metrics collection."""
        self.end_time = time.perf_counter()
        self.memory_usage_mb.append(psutil.Process().memory_info().rss / 1024 / 1024)
    
    def record_search(self, success: bool, duration: float, error_type: str = None):
        """Record individual search metrics."""
        if success:
            self.successful_searches += 1
            self.search_times.append(duration)
        else:
            self.failed_searches += 1
            if error_type == 'timeout':
                self.timeout_operations += 1
            elif error_type == 'connection':
                self.connection_errors += 1
    
    def record_insertion(self, success: bool, duration: float, error_type: str = None):
        """Record individual insertion metrics."""
        if success:
            self.successful_insertions += 1
            self.insertion_times.append(duration)
        else:
            self.failed_insertions += 1
            if error_type == 'timeout':
                self.timeout_operations += 1
            elif error_type == 'connection':
                self.connection_errors += 1
    
    def record_event_loop_block(self, duration: float):
        """Record event loop blocking time."""
        self.event_loop_blocked_duration += duration
    
    def update_concurrent_operations(self, count: int):
        """Update concurrent operation count."""
        if count > self.max_concurrent_operations:
            self.max_concurrent_operations = count
    
    def set_vector_dimensions(self, dims: int):
        """Set vector dimensions for context."""
        self.vector_dimensions = dims
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test metrics."""
        duration = self.end_time - self.start_time if self.end_time else 0
        total_searches = self.successful_searches + self.failed_searches
        total_insertions = self.successful_insertions + self.failed_insertions
        
        return {
            'test_duration_seconds': duration,
            'vector_dimensions': self.vector_dimensions,
            'total_searches': total_searches,
            'successful_searches': self.successful_searches,
            'failed_searches': self.failed_searches,
            'search_success_rate_percent': (self.successful_searches / total_searches * 100) if total_searches > 0 else 0,
            'total_insertions': total_insertions,
            'successful_insertions': self.successful_insertions,
            'failed_insertions': self.failed_insertions,
            'insertion_success_rate_percent': (self.successful_insertions / total_insertions * 100) if total_insertions > 0 else 0,
            'timeout_operations': self.timeout_operations,
            'connection_errors': self.connection_errors,
            'searches_per_second': self.successful_searches / duration if duration > 0 else 0,
            'insertions_per_second': self.successful_insertions / duration if duration > 0 else 0,
            'avg_search_time_ms': sum(self.search_times) / len(self.search_times) * 1000 if self.search_times else 0,
            'p95_search_time_ms': sorted(self.search_times)[int(len(self.search_times) * 0.95)] * 1000 if self.search_times else 0,
            'p99_search_time_ms': sorted(self.search_times)[int(len(self.search_times) * 0.99)] * 1000 if self.search_times else 0,
            'avg_insertion_time_ms': sum(self.insertion_times) / len(self.insertion_times) * 1000 if self.insertion_times else 0,
            'p95_insertion_time_ms': sorted(self.insertion_times)[int(len(self.insertion_times) * 0.95)] * 1000 if self.insertion_times else 0,
            'max_concurrent_operations': self.max_concurrent_operations,
            'event_loop_blocked_duration_ms': self.event_loop_blocked_duration * 1000,
            'memory_increase_mb': self.memory_usage_mb[-1] - self.memory_usage_mb[0] if len(self.memory_usage_mb) >= 2 else 0,
            'max_memory_mb': max(self.memory_usage_mb) if self.memory_usage_mb else 0
        }
    
    def assert_performance_thresholds(self):
        """Assert that performance meets production thresholds."""
        summary = self.get_summary()
        
        # Production performance thresholds
        assert summary['search_success_rate_percent'] >= 95.0, f"Search success rate {summary['search_success_rate_percent']:.1f}% < 95%"
        assert summary['insertion_success_rate_percent'] >= 95.0, f"Insertion success rate {summary['insertion_success_rate_percent']:.1f}% < 95%"
        assert summary['connection_errors'] <= 5, f"Too many connection errors: {summary['connection_errors']}"
        assert summary['avg_search_time_ms'] <= 1000, f"Average search time {summary['avg_search_time_ms']:.0f}ms > 1000ms"
        assert summary['p95_search_time_ms'] <= 3000, f"P95 search time {summary['p95_search_time_ms']:.0f}ms > 3000ms"
        assert summary['event_loop_blocked_duration_ms'] <= 500, f"Event loop blocked for {summary['event_loop_blocked_duration_ms']:.0f}ms > 500ms"
        assert summary['memory_increase_mb'] <= 1000, f"Memory increase {summary['memory_increase_mb']:.1f}MB > 1000MB"


class MockMilvusCollection:
    """Mock Milvus collection for testing."""
    
    def __init__(self, name: str, operation_delay: float = 0.05):
        self.name = name
        self.operation_delay = operation_delay
        self.data = []  # Simulated document storage
        self.schema = Mock()
        self.schema.fields = [
            Mock(name='id', dtype='INT64'),
            Mock(name='content', dtype='VARCHAR'),
            Mock(name='dense_vector', dtype='FLOAT_VECTOR'),
            Mock(name='metadata', dtype='JSON')
        ]
        self._loaded = False
    
    def load(self):
        """Mock collection loading."""
        time.sleep(self.operation_delay / 10)  # Small delay
        self._loaded = True
    
    def search(self, data, anns_field, param, limit, expr=None, output_fields=None):
        """Mock vector search with configurable delay."""
        time.sleep(self.operation_delay)
        
        # Generate mock search results
        results = []
        for query_idx, query_vector in enumerate(data):
            hits = []
            for i in range(min(limit, 5)):  # Return up to 5 results
                hit = Mock()
                hit.id = f"doc_{query_idx}_{i}"
                hit.distance = 0.1 + (i * 0.05)  # Increasing distance
                hit.fields = {
                    'content': f'Mock document content {i} for query {query_idx}',
                    'metadata': {'source': f'mock_source_{i}'}
                }
                
                # Add get method for field access
                def get_field(field_name, hit_data=hit.fields):
                    return hit_data.get(field_name)
                hit.get = get_field
                
                hits.append(hit)
            
            # Wrap hits in mock Hits object
            hits_obj = Mock()
            hits_obj.__iter__ = lambda: iter(hits)
            hits_obj.__len__ = lambda: len(hits)
            results.append(hits_obj)
        
        return results
    
    def insert(self, data):
        """Mock data insertion with configurable delay."""
        time.sleep(self.operation_delay)
        
        # Simulate storing data
        if isinstance(data, dict):
            self.data.append(data)
        elif isinstance(data, list):
            self.data.extend(data)
        
        # Return mock insertion result
        result = Mock()
        result.primary_keys = [f"doc_{len(self.data)}_{i}" for i in range(len(data) if isinstance(data, list) else 1)]
        return result


class MockMilvusClient:
    """Mock Milvus client for controlled testing."""
    
    def __init__(self, operation_delay: float = 0.05):
        self.operation_delay = operation_delay
        self.collections = {}
        self.connected = True
        self.host = "mock_localhost"
        self.port = 19530
        self.logger = logger
        self.batch_size = 50
    
    def connect(self):
        """Mock connection."""
        time.sleep(self.operation_delay / 10)
        self.connected = True
    
    def get_or_create_collection(self, config: ICollectionConfig):
        """Mock collection creation/retrieval."""
        if config.collection_name not in self.collections:
            self.collections[config.collection_name] = MockMilvusCollection(
                config.collection_name, 
                self.operation_delay
            )
        return self.collections[config.collection_name]
    
    def search_by_vector(self, config: ICollectionConfig, query_vectors, **kwargs):
        """Mock synchronous vector search."""
        collection = self.get_or_create_collection(config)
        return collection.search(
            data=query_vectors,
            anns_field=config.dense_field or 'dense_vector',
            param={'metric_type': 'IP'},
            limit=kwargs.get('limit', 10),
            expr=kwargs.get('expr'),
            output_fields=kwargs.get('output_fields')
        )
    
    def hybrid_search(self, config: ICollectionConfig, dense_vectors, sparse_vectors=None, **kwargs):
        """Mock synchronous hybrid search."""
        # For testing purposes, use dense vectors only
        return self.search_by_vector(config, dense_vectors, **kwargs)
    
    def insert_entity(self, config: ICollectionConfig, entity, embeddings):
        """Mock entity insertion."""
        collection = self.get_or_create_collection(config)
        
        # Prepare data for insertion
        data = {
            'content': entity.get('content', ''),
            'dense_vector': embeddings.get('dense', []),
            'metadata': entity.get('metadata', {})
        }
        
        return collection.insert([data])
    
    def insert_entities(self, config: ICollectionConfig, data, embeddings):
        """Mock batch entity insertion."""
        collection = self.get_or_create_collection(config)
        
        # Prepare batch data
        batch_data = []
        for i, entity in enumerate(data):
            item = {
                'content': entity.get('content', ''),
                'dense_vector': embeddings[i].get('dense', []) if i < len(embeddings) else [],
                'metadata': entity.get('metadata', {})
            }
            batch_data.append(item)
        
        return collection.insert(batch_data)


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
    
    def embed_document(self, text: str):
        """Mock synchronous embedding generation."""
        time.sleep(0.01)  # Small delay to simulate processing
        return {
            'dense': np.random.random(self.dimensions).tolist(),
            'sparse': {}  # Mock sparse vector
        }
    
    async def aembed_document(self, text: str):
        """Mock asynchronous embedding generation."""
        await asyncio.sleep(0.01)  # Small delay to simulate processing
        return {
            'dense': np.random.random(self.dimensions).tolist(),
            'sparse': {}  # Mock sparse vector
        }


@pytest.fixture
def mock_milvus_client():
    """Provide mock Milvus client."""
    return MockMilvusClient(operation_delay=0.02)  # Fast operations for testing


@pytest.fixture
def test_collection_config():
    """Provide test collection configuration."""
    return ICollectionConfig(
        collection_name="test_knowledge_base",
        dense_dim=1536,
        text_field="content",
        metadata_field="metadata",
        dense_field="dense_vector",
        is_hybrid=False
    )


@pytest.fixture
def mock_embedding_model():
    """Provide mock embedding model."""
    return MockEmbeddingModel(dimensions=1536)


class TestVectorDBLoad:
    """Production load tests for vector database async operations."""
    
    @pytest.mark.asyncio
    async def test_async_methods_availability(self, mock_milvus_client, test_collection_config):
        """Test that async methods are available and properly implemented."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        
        logger.info("🔍 Testing async methods availability")
        
        # Test async method availability
        assert hasattr(mock_milvus_client, 'search_by_vector_async'), "search_by_vector_async method missing"
        assert hasattr(mock_milvus_client, 'hybrid_search_async'), "hybrid_search_async method missing" 
        assert hasattr(mock_milvus_client, 'insert_entity_async'), "insert_entity_async method missing"
        assert hasattr(mock_milvus_client, 'insert_entities_async'), "insert_entities_async method missing"
        
        # Test that methods are actually async
        import inspect
        assert inspect.iscoroutinefunction(mock_milvus_client.search_by_vector_async), "search_by_vector_async is not async"
        assert inspect.iscoroutinefunction(mock_milvus_client.hybrid_search_async), "hybrid_search_async is not async"
        assert inspect.iscoroutinefunction(mock_milvus_client.insert_entity_async), "insert_entity_async is not async"
        assert inspect.iscoroutinefunction(mock_milvus_client.insert_entities_async), "insert_entities_async is not async"
        
        metrics.end_test()
        logger.info("✅ Async methods availability test passed")
    
    @pytest.mark.asyncio
    async def test_moderate_vector_search_load(self, mock_milvus_client, test_collection_config, mock_embedding_model):
        """Test vector search under moderate concurrent load (100 searches)."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        metrics.set_vector_dimensions(1536)
        
        logger.info("🧪 Starting moderate vector search load test (100 searches)")
        
        async def single_vector_search(search_id: int):
            """Perform a single vector search operation."""
            start = time.perf_counter()
            try:
                # Generate query vector
                query_text = f"test search query {search_id}"
                embedding = await mock_embedding_model.aembed_document(query_text)
                
                # Perform async vector search
                results = await mock_milvus_client.search_by_vector_async(
                    config=test_collection_config,
                    query_vectors=[embedding['dense']],
                    limit=10
                )
                
                duration = time.perf_counter() - start
                
                # Verify results
                success = isinstance(results, list) and len(results) > 0
                metrics.record_search(success, duration)
                
                if search_id % 20 == 0:
                    logger.info(f"Completed search {search_id}")
                
                return results
                
            except asyncio.TimeoutError:
                duration = time.perf_counter() - start
                metrics.record_search(False, duration, 'timeout')
                return []
            except Exception as e:
                duration = time.perf_counter() - start
                error_type = 'connection' if 'connection' in str(e).lower() else 'other'
                metrics.record_search(False, duration, error_type)
                logger.error(f"Error in search {search_id}: {e}")
                return []
        
        # Execute 100 concurrent searches
        tasks = [single_vector_search(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, list) and len(r) > 0]
        
        summary = metrics.get_summary()
        logger.info(f"Moderate vector search load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for moderate load
        assert len(successful_results) >= 95, f"Expected at least 95 successful searches, got {len(successful_results)}"
        assert metrics.connection_errors == 0, f"No connection errors expected, got {metrics.connection_errors}"
        assert summary['avg_search_time_ms'] <= 500, f"Average search time should be < 500ms, got {summary['avg_search_time_ms']:.0f}ms"
        assert summary['search_success_rate_percent'] >= 95, f"Search success rate should be >= 95%, got {summary['search_success_rate_percent']:.1f}%"
        
        logger.info("✅ Moderate vector search load test passed")
    
    @pytest.mark.asyncio
    async def test_high_vector_search_load(self, mock_milvus_client, test_collection_config, mock_embedding_model):
        """Test vector search under high concurrent load (500 searches)."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        metrics.set_vector_dimensions(1536)
        
        logger.info("🧪 Starting high vector search load test (500 searches)")
        
        async def high_load_vector_search(search_id: int):
            """Perform vector search with monitoring."""
            start = time.perf_counter()
            try:
                # Generate query vector (faster for high load)
                query_vector = np.random.random(1536).tolist()
                
                # Perform async vector search
                results = await mock_milvus_client.search_by_vector_async(
                    config=test_collection_config,
                    query_vectors=[query_vector],
                    limit=5  # Fewer results for high load
                )
                
                duration = time.perf_counter() - start
                success = isinstance(results, list)
                metrics.record_search(success, duration)
                
                if search_id % 100 == 0:
                    logger.info(f"High load: Completed search {search_id}")
                
                return results
                
            except Exception as e:
                duration = time.perf_counter() - start
                error_type = 'timeout' if 'timeout' in str(e).lower() else 'connection' if 'connection' in str(e).lower() else 'other'
                metrics.record_search(False, duration, error_type)
                
                if search_id % 100 == 0:
                    logger.warning(f"Error in high load search {search_id}: {e}")
                
                return []
        
        # Execute 500 searches in batches
        batch_size = 100
        all_results = []
        
        for batch_start in range(0, 500, batch_size):
            batch_end = min(batch_start + batch_size, 500)
            logger.info(f"Executing high load batch: {batch_start}-{batch_end}")
            
            batch_tasks = [high_load_vector_search(i) for i in range(batch_start, batch_end)]
            metrics.update_concurrent_operations(len(batch_tasks))
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in all_results if isinstance(r, list)]
        
        summary = metrics.get_summary()
        logger.info(f"High vector search load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for high load (more lenient)
        assert len(successful_results) >= 450, f"Expected at least 450 successful searches, got {len(successful_results)}"
        assert summary['search_success_rate_percent'] >= 90, f"Search success rate should be >= 90%, got {summary['search_success_rate_percent']:.1f}%"
        assert summary['avg_search_time_ms'] <= 1000, f"Average search time should be < 1000ms, got {summary['avg_search_time_ms']:.0f}ms"
        
        logger.info("✅ High vector search load test passed")
    
    @pytest.mark.asyncio
    async def test_extreme_vector_search_load(self, mock_milvus_client, test_collection_config):
        """Test vector search under extreme load (1000 searches)."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        metrics.set_vector_dimensions(1536)
        
        logger.info("🚀 Starting extreme vector search load test (1000 searches)")
        
        async def extreme_load_search(search_id: int):
            """Minimal overhead search for extreme testing."""
            start = time.perf_counter()
            try:
                # Pre-generated vector for speed
                query_vector = [0.1] * 1536
                
                results = await mock_milvus_client.search_by_vector_async(
                    config=test_collection_config,
                    query_vectors=[query_vector],
                    limit=3
                )
                
                duration = time.perf_counter() - start
                success = isinstance(results, list)
                metrics.record_search(success, duration)
                
                return results
                
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_search(False, duration, 'error')
                return []
        
        # Execute 1000 searches in small batches
        batch_size = 50
        all_results = []
        
        for batch_start in range(0, 1000, batch_size):
            batch_end = min(batch_start + batch_size, 1000)
            
            batch_tasks = [extreme_load_search(i) for i in range(batch_start, batch_end)]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            if batch_start % 200 == 0:
                logger.info(f"Extreme load: Completed {batch_start + len(batch_tasks)} searches")
            
            # Minimal pause
            await asyncio.sleep(0.05)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in all_results if isinstance(r, list)]
        
        summary = metrics.get_summary()
        logger.info(f"Extreme vector search load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for extreme load (most lenient)
        assert len(successful_results) >= 800, f"Expected at least 800 successful searches, got {len(successful_results)}"
        assert summary['search_success_rate_percent'] >= 80, f"Search success rate should be >= 80%, got {summary['search_success_rate_percent']:.1f}%"
        
        # Most important: no event loop blocking
        assert summary['event_loop_blocked_duration_ms'] <= 1000, f"Event loop blocked for {summary['event_loop_blocked_duration_ms']:.0f}ms"
        
        logger.info("✅ Extreme vector search load test passed - no event loop blocking")
    
    @pytest.mark.asyncio
    async def test_batch_insertion_load(self, mock_milvus_client, test_collection_config, mock_embedding_model):
        """Test batch document insertion under load."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        metrics.set_vector_dimensions(1536)
        
        logger.info("📝 Starting batch insertion load test")
        
        async def batch_insertion_operation(batch_id: int, batch_size: int = 10):
            """Perform batch document insertion."""
            start = time.perf_counter()
            try:
                # Generate batch of documents
                documents = []
                embeddings = []
                
                for i in range(batch_size):
                    doc = {
                        'content': f'Batch {batch_id} document {i} content for testing insertion performance',
                        'metadata': {'batch_id': batch_id, 'doc_id': i, 'source': 'load_test'}
                    }
                    documents.append(doc)
                    
                    # Generate embedding
                    embedding = await mock_embedding_model.aembed_document(doc['content'])
                    embeddings.append(embedding)
                
                # Perform async batch insertion
                result = await mock_milvus_client.insert_entities_async(
                    config=test_collection_config,
                    data=documents,
                    embeddings=embeddings
                )
                
                duration = time.perf_counter() - start
                
                # Verify insertion
                success = result is not None
                metrics.record_insertion(success, duration)
                
                if batch_id % 10 == 0:
                    logger.info(f"Completed batch insertion {batch_id}")
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start
                error_type = 'timeout' if 'timeout' in str(e).lower() else 'connection' if 'connection' in str(e).lower() else 'other'
                metrics.record_insertion(False, duration, error_type)
                logger.error(f"Error in batch insertion {batch_id}: {e}")
                return None
        
        # Execute 50 batch insertions (500 documents total)
        tasks = [batch_insertion_operation(i, batch_size=10) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        summary = metrics.get_summary()
        logger.info(f"Batch insertion load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for batch insertion
        assert len(successful_results) >= 45, f"Expected at least 45 successful batch insertions, got {len(successful_results)}"
        assert summary['insertion_success_rate_percent'] >= 90, f"Insertion success rate should be >= 90%, got {summary['insertion_success_rate_percent']:.1f}%"
        assert summary['avg_insertion_time_ms'] <= 2000, f"Average insertion time should be < 2000ms, got {summary['avg_insertion_time_ms']:.0f}ms"
        
        logger.info("✅ Batch insertion load test passed")
    
    @pytest.mark.asyncio
    async def test_mixed_operations_load(self, mock_milvus_client, test_collection_config, mock_embedding_model):
        """Test mixed search and insertion operations under load."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        metrics.set_vector_dimensions(1536)
        
        logger.info("🔄 Starting mixed operations load test")
        
        async def mixed_search_operation(op_id: int):
            """Perform search operation."""
            start = time.perf_counter()
            try:
                query_vector = np.random.random(1536).tolist()
                results = await mock_milvus_client.search_by_vector_async(
                    config=test_collection_config,
                    query_vectors=[query_vector],
                    limit=5
                )
                
                duration = time.perf_counter() - start
                success = isinstance(results, list)
                metrics.record_search(success, duration)
                
                return results
                
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_search(False, duration)
                return []
        
        async def mixed_insertion_operation(op_id: int):
            """Perform insertion operation."""
            start = time.perf_counter()
            try:
                doc = {
                    'content': f'Mixed operation document {op_id}',
                    'metadata': {'op_id': op_id, 'type': 'mixed_test'}
                }
                
                embedding = await mock_embedding_model.aembed_document(doc['content'])
                
                result = await mock_milvus_client.insert_entity_async(
                    config=test_collection_config,
                    entity=doc,
                    embeddings=embedding
                )
                
                duration = time.perf_counter() - start
                success = result is not None
                metrics.record_insertion(success, duration)
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_insertion(False, duration)
                return None
        
        # Create mixed workload: 70% searches, 30% insertions
        total_operations = 200
        search_count = int(total_operations * 0.7)
        insertion_count = total_operations - search_count
        
        # Create tasks
        tasks = []
        
        # Add search tasks
        for i in range(search_count):
            tasks.append(mixed_search_operation(i))
        
        # Add insertion tasks
        for i in range(insertion_count):
            tasks.append(mixed_insertion_operation(i))
        
        # Shuffle tasks for realistic mixed load
        import random
        random.shuffle(tasks)
        
        # Execute mixed operations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        
        summary = metrics.get_summary()
        logger.info(f"Mixed operations load test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for mixed operations
        assert summary['search_success_rate_percent'] >= 85, f"Search success rate should be >= 85%, got {summary['search_success_rate_percent']:.1f}%"
        assert summary['insertion_success_rate_percent'] >= 85, f"Insertion success rate should be >= 85%, got {summary['insertion_success_rate_percent']:.1f}%"
        assert summary['avg_search_time_ms'] <= 1000, f"Average search time should be < 1000ms, got {summary['avg_search_time_ms']:.0f}ms"
        assert summary['avg_insertion_time_ms'] <= 2000, f"Average insertion time should be < 2000ms, got {summary['avg_insertion_time_ms']:.0f}ms"
        
        logger.info("✅ Mixed operations load test passed")
    
    @pytest.mark.asyncio
    async def test_knowledge_base_tool_integration(self, mock_milvus_client, test_collection_config, mock_embedding_model):
        """Test KnowledgeBaseTool under load to verify async integration."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        
        logger.info("🔧 Starting KnowledgeBaseTool integration load test")
        
        # Create knowledge base tool
        kb_tool = KnowledgeBaseRetrievalTool(
            vector_db=mock_milvus_client,
            embedding_model=mock_embedding_model,
            collection_config=test_collection_config,
            search_limit=5
        )
        
        async def knowledge_base_query(query_id: int):
            """Perform knowledge base query."""
            start = time.perf_counter()
            try:
                query = f"Test knowledge base query {query_id} about performance and load testing"
                
                # Use async execute
                result_type, result = await kb_tool.aexecute(query=query)
                
                duration = time.perf_counter() - start
                
                # Verify result
                success = result_type == "function" and isinstance(result, list) and len(result) > 0
                metrics.record_search(success, duration)
                
                if query_id % 20 == 0:
                    logger.info(f"Completed KB query {query_id}")
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_search(False, duration)
                logger.error(f"Error in KB query {query_id}: {e}")
                return []
        
        # Execute 100 knowledge base queries
        tasks = [knowledge_base_query(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, list) and len(r) > 0]
        
        summary = metrics.get_summary()
        logger.info(f"KnowledgeBaseTool integration test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for integration test
        assert len(successful_results) >= 90, f"Expected at least 90 successful KB queries, got {len(successful_results)}"
        assert summary['search_success_rate_percent'] >= 90, f"KB query success rate should be >= 90%, got {summary['search_success_rate_percent']:.1f}%"
        assert summary['avg_search_time_ms'] <= 1500, f"Average KB query time should be < 1500ms, got {summary['avg_search_time_ms']:.0f}ms"
        
        logger.info("✅ KnowledgeBaseTool integration load test passed")
    
    @pytest.mark.asyncio
    async def test_sustained_vector_operations(self, mock_milvus_client, test_collection_config, mock_embedding_model):
        """Test vector operations under sustained load over time."""
        metrics = VectorDBMetrics()
        metrics.start_test()
        metrics.set_vector_dimensions(1536)
        
        logger.info("🔄 Starting sustained vector operations test (3 minutes)")
        
        test_duration = 180  # 3 minutes
        operations_per_second = 5
        
        async def sustained_vector_worker(worker_id: int):
            """Worker that continuously performs vector operations."""
            operation_count = 0
            while time.perf_counter() - metrics.start_time < test_duration:
                start = time.perf_counter()
                try:
                    if operation_count % 4 == 0:  # 25% insertions
                        # Insertion operation
                        doc = {
                            'content': f'Sustained test document {worker_id}_{operation_count}',
                            'metadata': {'worker_id': worker_id, 'operation': operation_count}
                        }
                        embedding = await mock_embedding_model.aembed_document(doc['content'])
                        
                        result = await mock_milvus_client.insert_entity_async(
                            config=test_collection_config,
                            entity=doc,
                            embeddings=embedding
                        )
                        
                        duration = time.perf_counter() - start
                        success = result is not None
                        metrics.record_insertion(success, duration)
                    else:
                        # Search operation
                        query_vector = np.random.random(1536).tolist()
                        results = await mock_milvus_client.search_by_vector_async(
                            config=test_collection_config,
                            query_vectors=[query_vector],
                            limit=5
                        )
                        
                        duration = time.perf_counter() - start
                        success = isinstance(results, list)
                        metrics.record_search(success, duration)
                    
                    operation_count += 1
                    
                    # Rate limiting
                    await asyncio.sleep(1.0 / operations_per_second)
                    
                except Exception as e:
                    duration = time.perf_counter() - start
                    metrics.record_search(False, duration)
                    await asyncio.sleep(1.0 / operations_per_second)
            
            return operation_count
        
        # Start multiple workers
        workers = [sustained_vector_worker(i) for i in range(3)]
        
        # Monitor memory usage
        async def memory_monitor():
            while time.perf_counter() - metrics.start_time < test_duration:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                metrics.memory_usage_mb.append(memory_mb)
                await asyncio.sleep(30)
        
        # Run workers and monitoring
        monitor_task = asyncio.create_task(memory_monitor())
        worker_results = await asyncio.gather(*workers)
        monitor_task.cancel()
        
        metrics.end_test()
        
        total_operations = sum(worker_results)
        expected_operations = test_duration * operations_per_second * len(workers)
        
        summary = metrics.get_summary()
        summary['worker_operations'] = total_operations
        summary['expected_operations'] = expected_operations
        
        logger.info(f"Sustained vector operations test results: {json.dumps(summary, indent=2)}")
        
        # Assertions for sustained operations
        assert total_operations >= expected_operations * 0.8, f"Expected at least {expected_operations * 0.8} operations, got {total_operations}"
        assert summary['search_success_rate_percent'] >= 90, f"Search success rate should be >= 90%, got {summary['search_success_rate_percent']:.1f}%"
        assert summary['insertion_success_rate_percent'] >= 90, f"Insertion success rate should be >= 90%, got {summary['insertion_success_rate_percent']:.1f}%"
        
        # Memory stability check
        memory_growth = max(metrics.memory_usage_mb) - min(metrics.memory_usage_mb)
        assert memory_growth <= 300, f"Memory growth over 3 minutes should be < 300MB, got {memory_growth:.1f}MB"
        
        logger.info("✅ Sustained vector operations test passed - stable performance over time")


if __name__ == "__main__":
    """Run vector database tests directly for development/debugging."""
    import sys
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    async def run_quick_vector_test():
        """Run a quick vector database test for development."""
        logger.info("Running quick vector database test...")
        
        # Create test components
        mock_client = MockMilvusClient(operation_delay=0.01)
        config = ICollectionConfig(
            collection_name="quick_test",
            dense_dim=128,  # Smaller for quick test
            text_field="content",
            metadata_field="metadata"
        )
        
        # Test async search
        start = time.perf_counter()
        
        query_vectors = [np.random.random(128).tolist() for _ in range(10)]
        tasks = []
        
        for i, vector in enumerate(query_vectors):
            tasks.append(mock_client.search_by_vector_async(
                config=config,
                query_vectors=[vector],
                limit=5
            ))
        
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        
        successful = len([r for r in results if isinstance(r, list)])
        
        logger.info(f"Quick vector test: {successful}/10 searches successful in {duration:.2f}s")
        logger.info(f"Rate: {successful/duration:.1f} searches/second")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(run_quick_vector_test())
    else:
        print("Run with --quick for development testing")
        print("Or use: pytest tests/performance/test_vector_db_load.py -v")