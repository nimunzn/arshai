# Rerankers Module

## Overview
The Rerankers module provides components for reordering and refining search results based on relevance. Rerankers improve retrieval quality by applying advanced models to reorder initial search results from vector databases or other retrieval systems, optimizing for semantic relevance beyond vector similarity.

## Architecture
```
rerankers/
├── voyage_reranker.py       # Implementation of Voyage AI reranker
├── flashrank_reranker.py    # Implementation of FlashRank reranker
└── __init__.py              # Module exports
```

## Implementation Guide

### Creating a Custom Reranker

```python
from seedwork.interfaces.ireranker import BaseReranker, IRerankerResult, IRerankerConfig

class CustomReranker(BaseReranker):
    """Custom reranker implementation."""
    
    def __init__(self, config: IRerankerConfig = None):
        super().__init__(config)
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the reranker model."""
        # Import necessary libraries
        import custom_reranker_lib
        
        # Initialize the model
        model = custom_reranker_lib.Model(
            model_name=self.config.model_name,
            api_key=self.config.api_key
        )
        
        return model
    
    def rerank(self, query: str, documents: list[dict], top_k: int = None) -> list[IRerankerResult]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of document dictionaries with 'id', 'content', and optional metadata
            top_k: Number of top results to return
            
        Returns:
            List of reranked documents with relevance scores
        """
        # Prepare inputs for the model
        texts = [doc.get('content', '') for doc in documents]
        
        # Run reranking
        scores = self.model.score(query=query, documents=texts)
        
        # Create result objects with scores
        results = []
        for i, doc in enumerate(documents):
            results.append(IRerankerResult(
                id=doc.get('id'),
                content=doc.get('content'),
                metadata=doc.get('metadata', {}),
                score=scores[i]
            ))
        
        # Sort by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to top_k if specified
        if top_k and top_k < len(results):
            results = results[:top_k]
            
        return results
```

### Using Rerankers

```python
from src.factories import reranker_factory, vector_db_factory, embedding_factory

# Create components
vector_db = vector_db_factory.create_vector_db("pinecone")
embedder = embedding_factory.create_embedding_model("openai")
reranker = reranker_factory.create_reranker("voyage")

# Two-stage retrieval with reranking
def enhanced_search(query, top_k_retrieval=20, top_k_final=5):
    # Stage 1: Vector similarity search
    query_embedding = embedder.create_embedding(query)
    initial_results = vector_db.search(
        query_vector=query_embedding,
        top_k=top_k_retrieval
    )
    
    # Convert to format expected by reranker
    documents = [
        {
            'id': result.id,
            'content': result.content,
            'metadata': result.metadata
        }
        for result in initial_results
    ]
    
    # Stage 2: Rerank results
    reranked_results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=top_k_final
    )
    
    return reranked_results
```

## Integration Points

### With Retrieval Systems
Rerankers integrate with vector databases and other retrieval systems to improve search quality:

```python
from src.factories import reranker_factory
from src.vector_db import PineconeDB

class EnhancedRetriever:
    """Retriever with reranking capabilities."""
    
    def __init__(self, vector_db, reranker, initial_k=100, final_k=10):
        self.vector_db = vector_db
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k
        
    def retrieve(self, query, query_vector=None):
        """Retrieve and rerank documents."""
        # First stage retrieval
        if query_vector is None:
            # Vector search using embeddings is handled internally
            initial_results = self.vector_db.semantic_search(
                query=query,
                top_k=self.initial_k
            )
        else:
            # Use provided vector
            initial_results = self.vector_db.search(
                query_vector=query_vector,
                top_k=self.initial_k
            )
            
        # Convert to common format
        documents = [
            {
                'id': result.id,
                'content': result.content,
                'metadata': result.metadata
            }
            for result in initial_results
        ]
        
        # Second stage reranking
        final_results = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=self.final_k
        )
        
        return final_results
```

### With Agents
Rerankers help agents retrieve more relevant information:

```python
from src.factories import agent_factory, reranker_factory, vector_db_factory, embedding_factory

# Create components
agent = agent_factory.create_agent("conversation")
embedder = embedding_factory.create_embedding_model("openai")
vector_db = vector_db_factory.create_vector_db("pinecone")
reranker = reranker_factory.create_reranker("voyage")

# Agent with high-quality retrieval
def answer_with_enhanced_retrieval(question):
    # Get query embedding
    query_embedding = embedder.create_embedding(question)
    
    # First stage: Get initial results
    initial_results = vector_db.search(
        query_vector=query_embedding,
        top_k=20  # Cast a wide net initially
    )
    
    # Convert to format expected by reranker
    documents = [
        {
            'id': result.id,
            'content': result.content,
            'metadata': result.metadata
        }
        for result in initial_results
    ]
    
    # Second stage: Rerank results
    reranked_results = reranker.rerank(
        query=question,
        documents=documents,
        top_k=3  # Focus on the most relevant documents
    )
    
    # Extract content from reranked results
    context = "\n\n".join([
        f"Document: {result.content}"
        for result in reranked_results
    ])
    
    # Let the agent answer with the enhanced context
    augmented_question = (
        f"Please answer the following question based on the provided context:\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}"
    )
    
    response = agent.process_message(augmented_question)
    return response
```

## Configuration
Configure rerankers through the settings system:

```python
from src.config import Settings

settings = Settings()
reranker_config = settings.get_reranker_config("voyage")

# Create a reranker with config
reranker = reranker_factory.create_reranker(
    "voyage",
    config=reranker_config
)
```

### Reranker Configuration in YAML

```yaml
# in config.yaml
rerankers:
  voyage:
    api_key: "${VOYAGE_API_KEY}"
    model_name: "voyage-2"
    batch_size: 16
  flashrank:
    model_path: "/models/flashrank-base"
    quantize: true
    batch_size: 32
``` 