# Clients Directory

This directory contains client implementations for various external services used by the Arshai package.

## Structure

```
/clients
├── vector_db/            # Vector database clients
│   ├── milvus_client.py  # Milvus vector database client
│   └── ...
├── web_search/           # Web search clients
│   ├── searxng.py        # SearxNG search client
│   └── ...
└── arshai/               # Arshai API clients
    └── ...
```

## Client Types

### Vector Database Clients

Vector database clients handle connections to vector databases for storing and searching embeddings. All vector database clients implement the `IVectorDBClient` interface from `seedwork.interfaces.ivector_db_client`.

Examples:
- `MilvusClient`: Client for the Milvus vector database

### Web Search Clients

Web search clients provide functionality for searching the web. All web search clients implement the `ISearchClient` interface from `seedwork.interfaces.isearch_client`.

Examples:
- `SearxNGClient`: Client for the SearxNG metasearch engine

## Usage

Clients are typically initialized with configuration objects and provide methods for interacting with external services. They are mainly used internally by the retriever components, not directly by users.

### Example: Using a Vector Database Client (for internal development)

```python
from src.clients.vector_db import MilvusClient, MilvusConfig

# Using the client directly
config = MilvusConfig(
    host="localhost",
    port="19530",
    db_name="default",
    collection_name="embeddings"
)
client = MilvusClient(config)

# Using the client
client.connect()
results = client.search_vectors(query_vectors, field_name="embedding", top_k=10)
client.disconnect()
``` 