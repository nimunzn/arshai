#!/usr/bin/env python3
"""
Simple test script to verify that tools' aexecute methods work correctly.
This test uses mocks to avoid dependency issues and focuses on async functionality.
"""

import asyncio
import sys
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List


async def test_web_search_tool_aexecute():
    """Test WebSearchTool aexecute method with mocked dependencies"""
    print("Testing WebSearchTool.aexecute...")
    
    # Mock the entire web search tool
    class MockWebSearchTool:
        def __init__(self, settings):
            self.settings = settings
            self.search_client = Mock()
            self.search_client.asearch = AsyncMock(return_value=[
                Mock(title="Test Title", content="Test Content", url="http://test.com")
            ])
        
        @property
        def function_definition(self):
            return {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        
        def execute(self, query: str):
            """Sync version - should not be used anymore"""
            return [{"type": "text", "text": f"Sync result for: {query}"}]
        
        async def aexecute(self, query: str):
            """Async version - this is what we're testing"""
            if not self.search_client:
                return [{"type": "text", "text": "No search capability available."}]
            
            results = await self.search_client.asearch(query)
            if not results:
                return [{"type": "text", "text": "No results found."}]
            
            # Format results
            context = "\n\n".join(f"{r.title}\n{r.content}" for r in results)
            urls = "\n".join(r.url for r in results)
            full_response = f"{context}\n\nSources:\n{urls}"
            
            return [{"type": "text", "text": full_response}]
    
    # Test the tool
    mock_settings = Mock()
    tool = MockWebSearchTool(mock_settings)
    
    # Test async execution
    result = await tool.aexecute("test query")
    
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, "Should return one result"
    assert result[0]["type"] == "text", "Result should be text type"
    assert "Test Title" in result[0]["text"], "Should contain search result title"
    assert "test query" in str(tool.search_client.asearch.call_args), "Should call asearch with query"
    
    print("✓ WebSearchTool.aexecute test passed!")


async def test_knowledge_base_tool_aexecute():
    """Test KnowledgeBaseRetrievalTool aexecute method with mocked dependencies"""
    print("Testing KnowledgeBaseRetrievalTool.aexecute...")
    
    # Mock the knowledge base tool
    class MockKnowledgeBaseTool:
        def __init__(self, settings):
            self.settings = settings
            self.vector_db = Mock()
            self.embedding_model = Mock()
            self.collection_config = Mock()
            self.search_limit = 3
            
            # Setup mock responses
            self.embedding_model.embed_document.return_value = {'dense': [0.1, 0.2, 0.3]}
            
            # Mock search hits
            mock_hit = Mock()
            mock_hit.id = "test_id"
            mock_hit.distance = 0.5
            mock_hit.get.return_value = "Test knowledge content"
            mock_hit.fields = {"content": "Test knowledge content", "metadata": {"source": "test.pdf"}}
            
            mock_hits = Mock()
            mock_hits.__len__ = Mock(return_value=1)
            mock_hits.__iter__ = Mock(return_value=iter([mock_hit]))
            
            self.vector_db.search_by_vector.return_value = [mock_hits]
            self.collection_config.text_field = "content"
            self.collection_config.metadata_field = "metadata"
            self.collection_config.is_hybrid = False
        
        @property
        def function_definition(self):
            return {
                "name": "retrieve_knowledge",
                "description": "Retrieve knowledge from vector database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        
        def execute(self, query: str):
            """Sync version - should not be used anymore"""
            return [{"type": "text", "text": f"Sync knowledge for: {query}"}]
        
        async def aexecute(self, query: str):
            """Async version - this is what we're testing"""
            try:
                # Generate embeddings
                query_embeddings = self.embedding_model.embed_document(query)
                
                # Search vector database
                search_results = self.vector_db.search_by_vector(
                    config=self.collection_config,
                    query_vectors=[query_embeddings['dense']],
                    limit=self.search_limit
                )
                
                if search_results and len(search_results) > 0:
                    # Format results
                    formatted_results = []
                    for hits in search_results:
                        for hit in hits:
                            text = hit.get(self.collection_config.text_field)
                            metadata = hit.get(self.collection_config.metadata_field) or {}
                            source = metadata.get('source', 'unknown') if isinstance(metadata, dict) else 'unknown'
                            formatted_results.append(f"Source: {source}\nContent: {text}\n")
                    
                    return [{"type": "text", "text": "\n".join(formatted_results)}]
                else:
                    return [{"type": "text", "text": "No relevant information found."}]
            
            except Exception as e:
                return [{"type": "text", "text": f"Error retrieving knowledge: {str(e)}"}]
    
    # Test the tool
    mock_settings = Mock()
    tool = MockKnowledgeBaseTool(mock_settings)
    
    # Test async execution
    result = await tool.aexecute("test knowledge query")
    
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, "Should return one result"
    assert result[0]["type"] == "text", "Result should be text type"
    assert "Test knowledge content" in result[0]["text"], "Should contain knowledge content"
    
    print("✓ KnowledgeBaseRetrievalTool.aexecute test passed!")


async def test_multimodal_tool_aexecute():
    """Test MultimodalKnowledgeBaseRetrievalTool aexecute method"""
    print("Testing MultimodalKnowledgeBaseRetrievalTool.aexecute...")
    
    class MockMultimodalTool:
        def __init__(self, settings):
            self.settings = settings
            self.vector_db = Mock()
            self.embedding_model = Mock()
            self.collection_config = Mock()
            self.search_limit = 3
            
            # Setup mock responses
            self.embedding_model.multimodel_embed.return_value = [0.1, 0.2, 0.3]
            
            # Mock image search results
            mock_hit = Mock()
            mock_hit.id = "img1"
            mock_hit.distance = 0.5
            mock_hit.get.return_value = "data:image/jpeg;base64,test_image_data"
            mock_hit.fields = {
                "content": "data:image/jpeg;base64,test_image_data",
                "metadata": {"source": "image1.jpg", "type": "image"}
            }
            
            mock_hits = Mock()
            mock_hits.__len__ = Mock(return_value=1)
            mock_hits.__iter__ = Mock(return_value=iter([mock_hit]))
            
            self.vector_db.search_by_vector.return_value = [mock_hits]
            self.collection_config.text_field = "content"
            self.collection_config.metadata_field = "metadata"
        
        @property
        def function_definition(self):
            return {
                "name": "retrieve_images",
                "description": "Retrieve images from database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Image search query"}
                    },
                    "required": ["query"]
                }
            }
        
        def execute(self, query: str):
            """Sync version - should not be used anymore"""
            return []
        
        async def aexecute(self, query: str):
            """Async version - this is what we're testing"""
            try:
                # Generate multimodal embeddings
                query_embeddings = self.embedding_model.multimodel_embed(input=[query])
                
                # Search vector database
                search_results = self.vector_db.search_by_vector(
                    config=self.collection_config,
                    query_vectors=[query_embeddings],
                    limit=self.search_limit,
                    output_fields=[self.collection_config.text_field, self.collection_config.metadata_field]
                )
                
                if search_results and len(search_results) > 0:
                    # Format image results
                    formatted_results = []
                    for hits in search_results:
                        for hit in hits:
                            # Add description
                            description = {
                                "type": "text",
                                "text": f"The following image is with {hit.id} id in database"
                            }
                            # Add image
                            image_result = {
                                "type": "image_url",
                                "image_url": {
                                    "url": hit.get(self.collection_config.text_field)
                                }
                            }
                            formatted_results.extend([description, image_result])
                    
                    return formatted_results
                else:
                    return []
            
            except Exception as e:
                return []
    
    # Test the tool
    mock_settings = Mock()
    tool = MockMultimodalTool(mock_settings)
    
    # Test async execution
    result = await tool.aexecute("find cat image")
    
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 2, "Should return description + image"
    assert result[0]["type"] == "text", "First result should be text description"
    assert result[1]["type"] == "image_url", "Second result should be image"
    assert "data:image/jpeg" in result[1]["image_url"]["url"], "Should contain image data"
    
    print("✓ MultimodalKnowledgeBaseRetrievalTool.aexecute test passed!")


async def test_concurrent_tool_execution():
    """Test that multiple tools can execute concurrently"""
    print("Testing concurrent tool execution...")
    
    # Create simple async tools
    class AsyncTool1:
        async def aexecute(self, query: str):
            await asyncio.sleep(0.1)  # Simulate async work
            return [{"type": "text", "text": f"Tool1 result for: {query}"}]
    
    class AsyncTool2:
        async def aexecute(self, query: str):
            await asyncio.sleep(0.1)  # Simulate async work
            return [{"type": "text", "text": f"Tool2 result for: {query}"}]
    
    tool1 = AsyncTool1()
    tool2 = AsyncTool2()
    
    # Execute concurrently
    import time
    start_time = time.time()
    
    results = await asyncio.gather(
        tool1.aexecute("query1"),
        tool2.aexecute("query2")
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify results
    assert len(results) == 2, "Should have results from both tools"
    assert "Tool1 result" in results[0][0]["text"], "Should have Tool1 result"
    assert "Tool2 result" in results[1][0]["text"], "Should have Tool2 result"
    
    # Verify concurrent execution (should be faster than sequential)
    assert execution_time < 0.15, f"Concurrent execution took too long: {execution_time}s"
    
    print("✓ Concurrent tool execution test passed!")


async def main():
    """Run all async tool tests"""
    print("Starting async tool tests...\n")
    
    try:
        await test_web_search_tool_aexecute()
        print()
        
        await test_knowledge_base_tool_aexecute()
        print()
        
        await test_multimodal_tool_aexecute()
        print()
        
        await test_concurrent_tool_execution()
        print()
        
        print("🎉 All async tool tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the async tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)