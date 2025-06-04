"""Unit tests for vector database clients."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import os
import numpy as np

from arshai.core.interfaces import ICollectionConfig
from arshai.vector_db.milvus_client import MilvusClient
from pymilvus import DataType, FieldSchema, CollectionSchema, Collection


class TestCollectionConfig(ICollectionConfig):
    """Test implementation of ICollectionConfig."""
    
    def __init__(self, 
                 collection_name="test_collection",
                 pk_field="id",
                 text_field="text",
                 metadata_field="metadata",
                 dense_field="vector",
                 sparse_field="sparse_vector",
                 schema_model=None,
                 is_hybrid=False):
        self.collection_name = collection_name
        self.pk_field = pk_field
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.dense_field = dense_field
        self.sparse_field = sparse_field
        self.schema_model = schema_model
        self.is_hybrid = is_hybrid


@pytest.fixture
def mock_milvus():
    """Create a mock for pymilvus components."""
    with patch('src.vector_db.milvus_client.connections') as mock_connections, \
         patch('src.vector_db.milvus_client.utility') as mock_utility, \
         patch('src.vector_db.milvus_client.Collection') as mock_collection_cls, \
         patch('src.vector_db.milvus_client.CollectionSchema') as mock_schema_cls, \
         patch('src.vector_db.milvus_client.FieldSchema') as mock_field_schema_cls, \
         patch('src.vector_db.milvus_client.DataType', return_value=MagicMock()) as mock_data_type, \
         patch.dict(os.environ, {
             "MILVUS_HOST": "localhost",
             "MILVUS_PORT": "19530",
             "MILVUS_DB_NAME": "test_db"
         }):
        
        # Mock collection
        mock_collection = MagicMock(spec=Collection)
        mock_collection_cls.return_value = mock_collection
        mock_collection.name = "test_collection"
        mock_collection.schema = MagicMock()
        mock_collection.schema.fields = [
            MagicMock(name="id", dtype=MagicMock(spec=DataType.INT64)),
            MagicMock(name="text", dtype=MagicMock(spec=DataType.VARCHAR)),
            MagicMock(name="metadata", dtype=MagicMock(spec=DataType.JSON)),
            MagicMock(name="vector", dtype=MagicMock(spec=DataType.FLOAT_VECTOR))
        ]
        mock_collection.num_entities = 100
        mock_collection.indexes = ["index1", "index2"]
        
        # Mock utility
        mock_utility.has_collection.return_value = False
        
        yield {
            "connections": mock_connections,
            "utility": mock_utility,
            "collection_cls": mock_collection_cls,
            "schema_cls": mock_schema_cls,
            "field_schema_cls": mock_field_schema_cls,
            "data_type": mock_data_type,
            "collection": mock_collection
        }


@pytest.fixture
def milvus_client(mock_milvus):
    """Create a MilvusClient instance with mocked pymilvus."""
    client = MilvusClient()
    return client


@pytest.fixture
def collection_config():
    """Create a test collection configuration."""
    return TestCollectionConfig()


def test_milvus_client_initialization(milvus_client):
    """Test MilvusClient initializes with correct environment variables."""
    assert milvus_client.host == "localhost"
    assert milvus_client.port == "19530"
    assert milvus_client.db_name == "test_db"


def test_connect(milvus_client, mock_milvus):
    """Test connection to Milvus server."""
    milvus_client.connect()
    
    # Verify the connection was made with correct parameters
    mock_milvus["connections"].connect.assert_called_once_with(
        alias="default",
        host="localhost",
        port="19530"
    )


def test_disconnect(milvus_client, mock_milvus):
    """Test disconnection from Milvus server."""
    milvus_client.disconnect()
    
    # Verify the disconnection was called
    mock_milvus["connections"].disconnect.assert_called_once_with(alias="default")


def test_create_schema(milvus_client, collection_config, mock_milvus):
    """Test creating a collection schema."""
    schema = milvus_client.create_schema(collection_config)
    
    # Verify schema creation with correct fields
    mock_milvus["schema_cls"].assert_called_once()
    mock_milvus["field_schema_cls"].call_count >= 3  # At least pk, text, and vector fields


def test_get_or_create_collection_existing(milvus_client, collection_config, mock_milvus):
    """Test getting an existing collection."""
    # Set the collection to exist
    mock_milvus["utility"].has_collection.return_value = True
    
    collection = milvus_client.get_or_create_collection(collection_config)
    
    # Verify the collection was fetched not created
    mock_milvus["collection_cls"].assert_called_once_with(name=collection_config.collection_name)
    assert collection == mock_milvus["collection"]


def test_get_or_create_collection_new(milvus_client, collection_config, mock_milvus):
    """Test creating a new collection."""
    # Set the collection to not exist
    mock_milvus["utility"].has_collection.return_value = False
    
    collection = milvus_client.get_or_create_collection(collection_config)
    
    # Verify schema was created and collection created with it
    mock_milvus["schema_cls"].assert_called_once()
    mock_milvus["collection_cls"].assert_called_once()
    assert collection == mock_milvus["collection"]


def test_get_collection_stats(milvus_client, collection_config, mock_milvus):
    """Test getting collection statistics."""
    stats = milvus_client.get_collection_stats(collection_config)
    
    # Verify stats match the mock collection
    assert stats["name"] == collection_config.collection_name
    assert stats["num_entities"] == 100
    assert stats["schema"] == mock_milvus["collection"].schema
    assert stats["indexes"] == ["index1", "index2"]


def test_insert_entity(milvus_client, collection_config, mock_milvus):
    """Test inserting a single entity."""
    # Test data
    entity = {
        "content": "Test document content",
        "metadata": {
            "source": "test",
            "date": "2023-01-01"
        }
    }
    
    embeddings = {
        "dense": [0.1, 0.2, 0.3]
    }
    
    milvus_client.insert_entity(collection_config, entity, embeddings)
    
    # Verify insert was called on the collection
    mock_milvus["collection"].insert.assert_called_once()
    mock_milvus["collection"].flush.assert_called_once()


def test_insert_entities(milvus_client, collection_config, mock_milvus):
    """Test inserting multiple entities."""
    # Test data
    entities = [
        {
            "content": "Test document 1",
            "metadata": {"source": "test1"}
        },
        {
            "content": "Test document 2",
            "metadata": {"source": "test2"}
        }
    ]
    
    embeddings = {
        "dense": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    }
    
    milvus_client.insert_entities(collection_config, entities, embeddings)
    
    # Verify batch insert was called
    mock_milvus["collection"].insert.assert_called_once()
    mock_milvus["collection"].flush.assert_called_once()


def test_query_by_expr(milvus_client, collection_config, mock_milvus):
    """Test querying by expression."""
    # Mock query results
    mock_milvus["collection"].query.return_value = [
        {"id": 1, "text": "Result 1"},
        {"id": 2, "text": "Result 2"}
    ]
    
    results = milvus_client.query_by_expr(
        config=collection_config,
        expr="id in [1, 2]",
        output_fields=["id", "text"]
    )
    
    # Verify query was called with correct parameters
    mock_milvus["collection"].query.assert_called_once()
    call_args = mock_milvus["collection"].query.call_args[0]
    assert call_args[0] == "id in [1, 2]"
    assert call_args[1] == ["id", "text"]
    
    # Verify results
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["text"] == "Result 2"


def test_search_by_vector(milvus_client, collection_config, mock_milvus):
    """Test searching by vector."""
    # Mock search results
    mock_search_result = MagicMock()
    mock_search_result.ids = [[1, 2]]
    mock_search_result.distances = [[0.1, 0.2]]
    mock_search_result.fields_data = [[
        {"id": 1, "text": "Result 1"},
        {"id": 2, "text": "Result 2"}
    ]]
    mock_milvus["collection"].search.return_value = mock_search_result
    
    # Test query vector
    query_vector = [0.1, 0.2, 0.3]
    
    results = milvus_client.search_by_vector(
        config=collection_config,
        query_vectors=query_vector,
        limit=2
    )
    
    # Verify search was called
    mock_milvus["collection"].search.assert_called_once()
    
    # Verify results structure
    assert len(results) > 0
    assert "documents" in results[0]
    assert "similarity" in results[0]


def test_delete_entity(milvus_client, collection_config, mock_milvus):
    """Test deleting entities by expression."""
    # Mock delete results
    mock_milvus["collection"].delete.return_value = 2  # 2 entities deleted
    
    result = milvus_client.delete_entity(
        config=collection_config,
        filter_expr="id in [1, 2]"
    )
    
    # Verify delete was called with correct expression
    mock_milvus["collection"].delete.assert_called_once_with("id in [1, 2]")
    
    # Verify result
    assert result == 2 