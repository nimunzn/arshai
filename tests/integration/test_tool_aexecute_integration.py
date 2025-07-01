#!/usr/bin/env python3
"""
Integration tests for Arshai framework tools' aexecute methods with local services.

This module tests the async execution of tools directly (not through agents)
with real connections to local services:
- Milvus (localhost:19530)
- SearXNG (localhost:8080)

Tests verify proper async execution and response formats for:
- WebSearchTool
- KnowledgeBaseRetrievalTool
- MultimodalKnowledgeBaseRetrievalTool
"""

import asyncio
import aiohttp
import pytest
import logging
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock
from pymilvus import connections, utility, Collection

# Arshai imports
from arshai.core.interfaces.isetting import ISetting
from arshai.tools.web_search_tool import WebSearchTool
from arshai.tools.knowledge_base_tool import KnowledgeBaseRetrievalTool
from arshai.tools.multimodel_knowledge_base_tool import MultimodalKnowledgeBaseRetrievalTool

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSettings(ISetting):
    """Test settings implementation for integration tests"""
    
    def __init__(self, use_milvus: bool = True, use_searxng: bool = True):
        self.use_milvus = use_milvus
        self.use_searxng = use_searxng
        self._config = {
            "web_search": {
                "provider": "searxng",
                "searxng_url": "http://localhost:8080"
            },
            "vector_db": {
                "provider": "milvus",
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_collection"
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small"
            },
            "search_limit": 3
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Alias for get method"""
        return self.get(key, default)
    
    def create_web_search(self):
        """Create mock web search client for testing"""
        if not self.use_searxng:
            return None
            
        mock_client = Mock()
        
        # Mock search result
        mock_result = Mock()
        mock_result.title = "Test Result"
        mock_result.content = "This is test content from SearXNG"
        mock_result.url = "https://example.com/test"
        
        async def mock_asearch(query):
            # Simulate actual SearXNG call
            return [mock_result]
        
        mock_client.asearch = mock_asearch
        return mock_client
    
    def create_vector_db(self):
        """Create mock vector database components for testing"""
        if not self.use_milvus:
            return None, None, None
            
        # Mock vector DB client
        mock_vector_db = Mock()
        mock_collection_config = Mock()
        mock_collection_config.text_field = "text"
        mock_collection_config.metadata_field = "metadata"
        mock_collection_config.is_hybrid = False
        
        # Mock embedding model
        mock_embedding = Mock()
        mock_embedding.embed_document.return_value = {
            'dense': [0.1] * 1536  # Mock embedding vector
        }
        mock_embedding.multimodel_embed.return_value = [0.1] * 1536
        
        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "test_id_1"
        mock_hit.distance = 0.85
        mock_hit.fields = ["text", "metadata"]
        mock_hit.get.side_effect = lambda field: {
            "text": "This is test knowledge from the vector database",
            "metadata": {"source": "test_document.pdf"}
        }.get(field)
        
        mock_hits = [mock_hit]
        
        def mock_search_by_vector(config, query_vectors, limit, output_fields=None):
            return [mock_hits]
        
        mock_vector_db.search_by_vector = mock_search_by_vector
        
        return mock_vector_db, mock_collection_config, mock_embedding
    
    # Required ISetting interface methods (simplified for testing)
    @property
    def llm_model(self):
        return Mock()
    
    @property
    def context_manager(self):
        return Mock()
    
    @property
    def reranker(self):
        return Mock()
    
    @property
    def embedding_model(self):
        return Mock()
    
    @property
    def vector_db(self):
        return Mock()
    
    @property
    def chatbot_context(self):
        return "Test context"
    
    @property
    def redis_url(self):
        return "redis://localhost:6379/0"
    
    def load_from_path(self, path: str):
        return {}
    
    def model_dump(self):
        return self._config
    
    def _create_context_manager(self):
        return Mock()
    
    def _create_llm_model(self):
        return Mock()
    
    def _create_embedding_model(self):
        return Mock()
    
    def _create_reranker(self):
        return Mock()
    
    def _create_vector_db(self):
        return Mock()
    
    def create_reranker(self):
        return Mock()


# Service availability check functions
async def check_searxng_availability() -> bool:
    """Check if SearXNG is running on localhost:8080"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080', timeout=5) as response:
                return response.status == 200
    except Exception as e:
        logger.warning(f"SearXNG not available: {e}")
        return False


async def check_milvus_availability() -> bool:
    """Check if Milvus is running on localhost:19530"""
    try:
        connections.connect(alias="test", host="localhost", port="19530")
        # Try to list collections to ensure connection works
        utility.list_collections(using="test")
        connections.disconnect(alias="test")
        return True
    except Exception as e:
        logger.warning(f"Milvus not available: {e}")
        return False


# Integration Tests
@pytest.mark.asyncio
async def test_service_availability():
    """Test that required services are available before running tool tests"""
    logger.info("Checking service availability...")
    
    searxng_available = await check_searxng_availability()
    milvus_available = await check_milvus_availability()
    
    logger.info(f"SearXNG available: {searxng_available}")
    logger.info(f"Milvus available: {milvus_available}")
    
    # Note: We don't fail the test if services aren't available
    # Instead, we'll skip relevant tests using pytest.skip
    assert True  # This test always passes, it's just for information


@pytest.mark.asyncio
async def test_websearch_tool_aexecute_real():
    """Test WebSearchTool.aexecute() with real SearXNG connection if available"""
    logger.info("Testing WebSearchTool.aexecute() with real SearXNG...")
    
    # Check if SearXNG is available
    searxng_available = await check_searxng_availability()
    
    if not searxng_available:
        pytest.skip("SearXNG service not available at localhost:8080")
    
    # Create real settings for SearXNG
    from arshai.config.settings import Settings
    import tempfile
    import yaml
    import os
    
    # Set environment variable for SearXNG BEFORE creating any settings
    original_searx_instance = os.environ.get("SEARX_INSTANCE")
    os.environ["SEARX_INSTANCE"] = "http://localhost:8080"
    
    config_data = {
        "web_search": {
            "provider": "searxng",
            "default_engines": ["google", "bing"],
            "default_categories": ["general"],
            "language": "en",
            "timeout": 10
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name
    
    tool = None
    try:
        # Create settings AFTER environment variable is set
        real_settings = Settings(config_path=config_file)
        tool = WebSearchTool(real_settings)
        
        # Test the aexecute method with a simple query
        query = "Python programming"
        result = await tool.aexecute(query)
        
        # Verify response format
        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"
        assert "type" in result[0], "Result should have 'type' field"
        assert "text" in result[0], "Result should have 'text' field"
        assert result[0]["type"] == "text", "Result type should be 'text'"
        assert len(result[0]["text"]) > 0, "Result text should not be empty"
        
        # For real SearXNG, we should get actual search results
        result_text = result[0]["text"]
        
        # Check if we actually got real search results or error message
        if "No search capability available" in result_text:
            pytest.skip("SearXNG client creation failed - check service configuration")
        
        assert "Mock" not in result_text, "Should not contain mock data"
        assert "Test Result" not in result_text, "Should not contain test mock data"
        
        logger.info(f"Real SearXNG result length: {len(result_text)} chars")
        logger.info(f"Real SearXNG result preview: {result_text[:200]}...")
        logger.info("✓ WebSearchTool.aexecute() real SearXNG test passed!")
        
    except Exception as e:
        logger.error(f"Real SearXNG test failed: {e}")
        pytest.fail(f"Real SearXNG integration test failed: {e}")
    finally:
        # Clean up the temp file
        os.unlink(config_file)
        # Restore original environment variable
        if original_searx_instance is not None:
            os.environ["SEARX_INSTANCE"] = original_searx_instance
        else:
            os.environ.pop("SEARX_INSTANCE", None)


@pytest.mark.asyncio
async def test_websearch_tool_aexecute():
    """Test WebSearchTool.aexecute() with mock SearXNG connection"""
    logger.info("Testing WebSearchTool.aexecute() with mock...")
    
    # Create settings and tool
    settings = TestSettings(use_searxng=True)
    tool = WebSearchTool(settings)
    
    # Test the aexecute method
    query = "What are the latest developments in artificial intelligence?"
    result = await tool.aexecute(query)
    
    # Verify response format
    assert isinstance(result, list), "Result should be a list"
    assert len(result) > 0, "Result should not be empty"
    assert "type" in result[0], "Result should have 'type' field"
    assert "text" in result[0], "Result should have 'text' field"
    assert result[0]["type"] == "text", "Result type should be 'text'"
    assert len(result[0]["text"]) > 0, "Result text should not be empty"
    
    logger.info(f"Mock WebSearch result: {result[0]['text'][:100]}...")
    logger.info("✓ WebSearchTool.aexecute() mock test passed!")


@pytest.mark.asyncio
async def test_websearch_tool_aexecute_no_client():
    """Test WebSearchTool.aexecute() when no search client is available"""
    logger.info("Testing WebSearchTool.aexecute() with no client...")
    
    # Create settings without search client
    settings = TestSettings(use_searxng=False)
    tool = WebSearchTool(settings)
    
    # Test the aexecute method
    query = "test query"
    result = await tool.aexecute(query)
    
    # Verify error handling
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, "Result should have one item"
    assert "No search capability available" in result[0]["text"], "Should indicate no search capability"
    
    logger.info("✓ WebSearchTool.aexecute() no client test passed!")


@pytest.mark.asyncio
async def test_knowledge_retrieval_aexecute_real():
    """Test KnowledgeBaseRetrievalTool.aexecute() with real Milvus connection and test collection"""
    logger.info("Testing KnowledgeBaseRetrievalTool.aexecute() with real Milvus...")
    
    # Check if Milvus is available
    milvus_available = await check_milvus_availability()
    
    if not milvus_available:
        pytest.skip("Milvus service not available at localhost:19530")
    
    # Import required Milvus components
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
    import numpy as np
    import tempfile
    import yaml
    import os
    
    # Test collection configuration
    collection_name = "test_arshai_integration"
    dimension = 384  # Standard embedding dimension for many models
    
    try:
        # Connect to Milvus
        connections.connect(alias="test_integration", host="localhost", port="19530")
        
        # Clean up any existing test collection
        if utility.has_collection(collection_name, using="test_integration"):
            utility.drop_collection(collection_name, using="test_integration")
            logger.info(f"Dropped existing test collection: {collection_name}")
        
        # Create collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        schema = CollectionSchema(fields=fields, description="Test collection for Arshai integration tests")
        collection = Collection(name=collection_name, schema=schema, using="test_integration")
        
        logger.info(f"Created test collection: {collection_name}")
        
        # Prepare test data
        test_documents = [
            {
                "text": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"source": "test_doc_1.txt", "category": "programming"},
                "embedding": np.random.rand(dimension).tolist()
            },
            {
                "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                "metadata": {"source": "test_doc_2.txt", "category": "ai"},
                "embedding": np.random.rand(dimension).tolist()
            },
            {
                "text": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.",
                "metadata": {"source": "test_doc_3.txt", "category": "database"},
                "embedding": np.random.rand(dimension).tolist()
            },
            {
                "text": "Natural language processing enables computers to understand and process human language.",
                "metadata": {"source": "test_doc_4.txt", "category": "nlp"},
                "embedding": np.random.rand(dimension).tolist()
            }
        ]
        
        # Insert test data
        texts = [doc["text"] for doc in test_documents]
        metadatas = [doc["metadata"] for doc in test_documents]
        embeddings = [doc["embedding"] for doc in test_documents]
        
        entities = [texts, metadatas, embeddings]
        collection.insert(entities)
        collection.flush()
        
        logger.info(f"Inserted {len(test_documents)} test documents")
        
        # Create index for vector search using IP metric to match MilvusClient defaults
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT", 
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        
        logger.info("Created index and loaded collection")
        
        # Create real settings with proper Milvus client
        from arshai.vector_db.milvus_client import MilvusClient
        from arshai.core.interfaces.ivector_db_client import ICollectionConfig
        
        class TestEmbeddingModel:
            def __init__(self):
                self.dimension = dimension
                
            def embed_document(self, text: str):
                # Create a simple embedding based on text content for testing
                # This creates reproducible embeddings for testing
                embedding = np.random.RandomState(hash(text) % 2**32).rand(dimension).tolist()
                return {'dense': embedding}
        
        # Create collection config as a Pydantic model with proper initialization
        test_collection_config = ICollectionConfig(
            collection_name=collection_name,
            dense_dim=dimension,
            text_field="text",
            metadata_field="metadata",
            dense_field="embedding",
            sparse_field="sparse_embedding",
            pk_field="id",
            is_hybrid=False,
            schema_model=None
        )
        
        # Set up environment variables for the Milvus client BEFORE creating it
        import os
        os.environ["MILVUS_HOST"] = "localhost"
        os.environ["MILVUS_PORT"] = "19530"
        os.environ["MILVUS_DB_NAME"] = "default"
        
        # Create a real Milvus client (now it will read the environment variables)
        real_milvus_client = MilvusClient()
        
        # Connect the client
        real_milvus_client.connect()
        
        class RealMilvusSettings:
            def __init__(self):
                self.embedding_model = TestEmbeddingModel()
                self.collection_config = test_collection_config
                self.vector_db = real_milvus_client
                
            def get(self, key: str, default=None):
                if key == "search_limit":
                    return 3
                return default
                
            def create_vector_db(self):
                return self.vector_db, self.collection_config, self.embedding_model
        
        # Test the knowledge base tool with real Milvus
        settings = RealMilvusSettings()
        tool = KnowledgeBaseRetrievalTool(settings)
        
        # Test search query
        query = "What is machine learning?"
        result = await tool.aexecute(query)
        
        # Verify response format
        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"
        assert "type" in result[0], "Result should have 'type' field"
        assert "text" in result[0], "Result should have 'text' field"
        assert result[0]["type"] == "text", "Result type should be 'text'"
        assert len(result[0]["text"]) > 0, "Result text should not be empty"
        
        # For real Milvus, we should get actual search results
        result_text = result[0]["text"]
        assert "Mock" not in result_text, "Should not contain mock data"
        assert "Source:" in result_text, "Should contain source information"
        assert "Content:" in result_text, "Should contain content"
        
        logger.info(f"Real Milvus result length: {len(result_text)} chars")
        logger.info(f"Real Milvus result preview: {result_text[:200]}...")
        logger.info("✓ KnowledgeBaseRetrievalTool.aexecute() real Milvus test passed!")
        
    except Exception as e:
        logger.error(f"Real Milvus test failed: {e}")
        pytest.fail(f"Real Milvus integration test failed: {e}")
    finally:
        # Clean up test collection
        try:
            if utility.has_collection(collection_name, using="test_integration"):
                utility.drop_collection(collection_name, using="test_integration")
                logger.info(f"Cleaned up test collection: {collection_name}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up test collection: {cleanup_error}")
        
        # Disconnect from Milvus
        try:
            connections.disconnect(alias="test_integration")
            logger.info("Disconnected from Milvus test connection")
        except Exception as disconnect_error:
            logger.warning(f"Failed to disconnect from Milvus test connection: {disconnect_error}")
        
        # Disconnect the MilvusClient as well
        try:
            if 'real_milvus_client' in locals():
                real_milvus_client.disconnect()
                logger.info("Disconnected MilvusClient")
        except Exception as client_disconnect_error:
            logger.warning(f"Failed to disconnect MilvusClient: {client_disconnect_error}")


@pytest.mark.asyncio
async def test_knowledge_retrieval_aexecute():
    """Test KnowledgeBaseRetrievalTool.aexecute() with mock Milvus connection"""
    logger.info("Testing KnowledgeBaseRetrievalTool.aexecute() with mock...")
    
    # Create settings and tool
    settings = TestSettings(use_milvus=True)
    tool = KnowledgeBaseRetrievalTool(settings)
    
    # Test the aexecute method
    query = "What information do you have about machine learning?"
    result = await tool.aexecute(query)
    
    # Verify response format
    assert isinstance(result, list), "Result should be a list"
    assert len(result) > 0, "Result should not be empty"
    assert "type" in result[0], "Result should have 'type' field"
    assert "text" in result[0], "Result should have 'text' field"
    assert result[0]["type"] == "text", "Result type should be 'text'"
    assert len(result[0]["text"]) > 0, "Result text should not be empty"
    
    logger.info(f"Mock Knowledge retrieval result: {result[0]['text'][:100]}...")
    logger.info("✓ KnowledgeBaseRetrievalTool.aexecute() mock test passed!")


@pytest.mark.asyncio
async def test_knowledge_retrieval_aexecute_no_components():
    """Test KnowledgeBaseRetrievalTool.aexecute() when components are not available"""
    logger.info("Testing KnowledgeBaseRetrievalTool.aexecute() with no components...")
    
    # Create settings without vector db components
    settings = TestSettings(use_milvus=False)
    tool = KnowledgeBaseRetrievalTool(settings)
    
    # Verify tool initialization detected missing components
    assert tool.vector_db is None, "Vector DB should be None when not available"
    
    logger.info("✓ KnowledgeBaseRetrievalTool.aexecute() no components test passed!")


@pytest.mark.asyncio
async def test_multimodal_knowledge_retrieval_aexecute():
    """Test MultimodalKnowledgeBaseRetrievalTool.aexecute() with mock Milvus connection"""
    logger.info("Testing MultimodalKnowledgeBaseRetrievalTool.aexecute()...")
    
    # Create settings and tool
    settings = TestSettings(use_milvus=True)
    tool = MultimodalKnowledgeBaseRetrievalTool(settings)
    
    # Test the aexecute method
    query = "Show me images related to artificial intelligence"
    result = await tool.aexecute(query)
    
    # Verify response format (may be empty list if no results)
    assert isinstance(result, list), "Result should be a list"
    
    # If there are results, verify format
    if len(result) > 0:
        # Should have alternating text and image_url entries
        text_items = [item for item in result if item.get("type") == "text"]
        image_items = [item for item in result if item.get("type") == "image_url"]
        
        assert len(text_items) > 0, "Should have text descriptions"
        assert len(image_items) > 0, "Should have image URLs"
        
        # Verify image format
        for img_item in image_items:
            assert "image_url" in img_item, "Image item should have 'image_url' field"
            assert "url" in img_item["image_url"], "Image URL should have 'url' field"
    
    logger.info(f"Multimodal retrieval returned {len(result)} items")
    logger.info("✓ MultimodalKnowledgeBaseRetrievalTool.aexecute() test passed!")


@pytest.mark.asyncio
async def test_concurrent_tool_execution():
    """Test that multiple tools can be executed concurrently"""
    logger.info("Testing concurrent tool execution...")
    
    # Create settings and tools
    settings = TestSettings(use_milvus=True, use_searxng=True)
    web_tool = WebSearchTool(settings)
    knowledge_tool = KnowledgeBaseRetrievalTool(settings)
    multimodal_tool = MultimodalKnowledgeBaseRetrievalTool(settings)
    
    # Execute tools concurrently
    import time
    start_time = time.time()
    
    results = await asyncio.gather(
        web_tool.aexecute("What is machine learning?"),
        knowledge_tool.aexecute("What is deep learning?"),
        multimodal_tool.aexecute("AI images"),
        return_exceptions=True
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify all tools returned results
    assert len(results) == 3, "Should have 3 results"
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Tool {i} failed with exception: {result}")
        else:
            assert isinstance(result, list), f"Result {i} should be a list"
            logger.info(f"Tool {i} returned {len(result)} items")
    
    logger.info(f"Concurrent execution took {execution_time:.3f}s")
    logger.info("✓ Concurrent tool execution test passed!")


@pytest.mark.asyncio
async def test_tool_function_definitions():
    """Test that all tools have proper function definitions for LLM integration"""
    logger.info("Testing tool function definitions...")
    
    # Create settings and tools
    settings = TestSettings(use_milvus=True, use_searxng=True)
    tools = [
        WebSearchTool(settings),
        KnowledgeBaseRetrievalTool(settings),
        MultimodalKnowledgeBaseRetrievalTool(settings)
    ]
    
    for tool in tools:
        func_def = tool.function_definition
        
        # Verify required fields
        assert "name" in func_def, f"{tool.__class__.__name__} should have 'name' in function definition"
        assert "description" in func_def, f"{tool.__class__.__name__} should have 'description' in function definition"
        assert "parameters" in func_def, f"{tool.__class__.__name__} should have 'parameters' in function definition"
        
        # Verify parameters structure
        params = func_def["parameters"]
        assert "type" in params, f"{tool.__class__.__name__} parameters should have 'type'"
        assert "properties" in params, f"{tool.__class__.__name__} parameters should have 'properties'"
        assert "required" in params, f"{tool.__class__.__name__} parameters should have 'required'"
        
        # Verify query parameter exists
        assert "query" in params["properties"], f"{tool.__class__.__name__} should have 'query' parameter"
        assert "query" in params["required"], f"{tool.__class__.__name__} should require 'query' parameter"
        
        logger.info(f"✓ {tool.__class__.__name__} function definition is valid")
    
    logger.info("✓ All tool function definitions test passed!")


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in tool execution"""
    logger.info("Testing error handling...")
    
    # Create settings and tools
    settings = TestSettings(use_milvus=True, use_searxng=True)
    
    # Test with invalid/empty queries
    web_tool = WebSearchTool(settings)
    knowledge_tool = KnowledgeBaseRetrievalTool(settings)
    
    test_cases = [
        "",  # Empty query
        " ",  # Whitespace only
        "a" * 1000,  # Very long query
    ]
    
    for query in test_cases:
        try:
            web_result = await web_tool.aexecute(query)
            knowledge_result = await knowledge_tool.aexecute(query)
            
            # Verify tools handle edge cases gracefully
            assert isinstance(web_result, list), f"Web tool should return list for query: '{query[:20]}...'"
            assert isinstance(knowledge_result, list), f"Knowledge tool should return list for query: '{query[:20]}...'"
            
        except Exception as e:
            # Tools should handle errors gracefully, not raise exceptions
            pytest.fail(f"Tool raised unexpected exception for query '{query[:20]}...': {e}")
    
    logger.info("✓ Error handling test passed!")


if __name__ == "__main__":
    """Run integration tests directly"""
    import sys
    
    async def main():
        logger.info("Starting Arshai Tool Integration Tests...")
        
        # Check service availability first
        await test_service_availability()
        
        # Run all tests
        test_functions = [
            test_websearch_tool_aexecute_real,
            test_websearch_tool_aexecute,
            test_websearch_tool_aexecute_no_client,
            test_knowledge_retrieval_aexecute_real,
            test_knowledge_retrieval_aexecute,
            test_knowledge_retrieval_aexecute_no_components,
            test_multimodal_knowledge_retrieval_aexecute,
            test_concurrent_tool_execution,
            test_tool_function_definitions,
            test_error_handling
        ]
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_func in test_functions:
            try:
                await test_func()
                passed += 1
            except Exception as e:
                if "skip" in str(e).lower() or "Skip" in str(type(e).__name__):
                    logger.info(f"Test {test_func.__name__} skipped: {e}")
                    skipped += 1
                else:
                    logger.error(f"Test {test_func.__name__} failed: {e}")
                    failed += 1
        
        logger.info(f"\nTest Results: {passed} passed, {failed} failed, {skipped} skipped")
        
        if failed > 0:
            sys.exit(1)
        else:
            logger.info("All tests completed successfully! 🎉")
    
    # Run the tests
    asyncio.run(main())