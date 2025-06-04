# Document Indexing and RAG System

Arshai provides a robust document indexing and retrieval system for building Retrieval Augmented Generation (RAG) applications. This document explains how to use this system to index, store, and retrieve information from documents.

## Architecture Overview

The document indexing system follows a clean, modular architecture:

```
┌─────────────────────────────────────────────────────────┐
│                  DocumentOrchestrator                    │
└─────────────────┬─────────────────────┬─────────────────┘
                  │                     │
          ┌───────▼────────┐    ┌───────▼────────┐
          │    Loaders     │    │   Processors   │
          └───────┬────────┘    └───────┬────────┘
                  │                     │
          ┌───────▼────────┐    ┌───────▼────────┐
          │    Documents   │    │     Chunks     │
          └───────┬────────┘    └───────┬────────┘
                  │                     │
                  │             ┌───────▼────────┐
                  │             │   Embeddings   │
                  │             └───────┬────────┘
                  │                     │
          ┌───────▼─────────────────────▼────────┐
          │              Vector DB                │
          └──────────────────────────────────────┘
```

The system consists of several key components:

1. **DocumentOrchestrator**: Coordinates the indexing and retrieval process
2. **Loaders**: Load documents from various sources and formats
3. **Processors**: Process documents into smaller chunks
4. **Embeddings**: Generate vector embeddings for text chunks
5. **Vector DB**: Store and retrieve document embeddings
6. **Rerankers**: Improve retrieval quality by reranking results

## Core Components

### Document Loaders

Document loaders are responsible for loading documents from various sources and formats. Arshai includes several built-in loaders:

```python
class PDFLoader(IFileLoader):
    """Loader for PDF documents."""
    
    def load(self, file_path: str) -> IDocument:
        """
        Load a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            IDocument containing the text and metadata
        """
        # Implementation...
```

Built-in loaders include:
- **PDFLoader**: For PDF documents
- **DocxLoader**: For Microsoft Word documents
- **TextLoader**: For plain text files
- **CSVLoader**: For CSV files
- **JSONLoader**: For JSON files
- **MarkdownLoader**: For Markdown files
- **WebPageLoader**: For web pages

### Text Processors

Text processors handle document chunking and preprocessing:

```python
class TextSplitter(ITextSplitter):
    """Split text into chunks based on configuration."""
    
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Implementation...
```

Built-in splitters include:
- **ParagraphSplitter**: Split by paragraphs
- **SentenceSplitter**: Split by sentences
- **TokenSplitter**: Split by tokens
- **CharacterSplitter**: Split by characters
- **RecursiveCharacterSplitter**: Advanced character-based splitting

### Embedding Models

Embedding models generate vector representations of text:

```python
class OpenAIEmbedding(IEmbedding):
    """OpenAI embedding model."""
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as a list of floats
        """
        # Implementation...
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings
        """
        # Implementation...
```

Supported embedding models include:
- **OpenAIEmbedding**: OpenAI's text embedding models
- **AzureEmbedding**: Azure's embedding services
- **MGTEEmbedding**: Microsoft's General-Text Embeddings
- **HuggingFaceEmbedding**: Various models from HuggingFace

### Vector Databases

Vector databases store and retrieve document embeddings:

```python
class MilvusClient(IVectorDBClient):
    """Milvus vector database client."""
    
    def create_collection(self, collection_name: str, dimension: int) -> None:
        """Create a collection in the vector database."""
        # Implementation...
    
    def insert(self, collection_name: str, embeddings: List[List[float]], 
               metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        """Insert embeddings and metadata into the collection."""
        # Implementation...
    
    def search(self, collection_name: str, query_embedding: List[float], 
              top_k: int = 10) -> List[QueryResult]:
        """Search for similar embeddings in the collection."""
        # Implementation...
```

Supported vector databases include:
- **Milvus**: High-performance vector database
- **FAISS**: Facebook AI Similarity Search
- **Chroma**: Simple, in-memory vector database
- **Pinecone**: Managed vector database service
- **Redis**: Using Redis as a vector database

### Document Orchestrator

The DocumentOrchestrator coordinates the entire process:

```python
class DocumentOrchestrator:
    """
    Orchestrator for document indexing and retrieval.
    
    The orchestrator coordinates the process of loading documents,
    processing them into chunks, generating embeddings, and storing
    them in a vector database.
    """
    
    def __init__(self, settings):
        """
        Initialize the document orchestrator.
        
        Args:
            settings: Settings object for creating components
        """
        self.settings = settings
        self.embedding = settings.create_embedding()
        self.vector_db = settings.create_vector_db()
        self.loaders = self._initialize_loaders()
        self.processors = self._initialize_processors()
    
    def index_documents(self, request: IndexingRequest) -> IndexingResult:
        """
        Index documents based on the request.
        
        Args:
            request: IndexingRequest containing indexing parameters
            
        Returns:
            IndexingResult with indexing statistics
        """
        # Implementation...
    
    def retrieve_documents(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve documents based on the request.
        
        Args:
            request: RetrievalRequest containing retrieval parameters
            
        Returns:
            RetrievalResult with retrieved documents
        """
        # Implementation...
```

## Using the Document Indexing System

### Indexing Documents

To index documents, create an IndexingRequest and use the DocumentOrchestrator:

```python
from arshai import Settings
from src.indexers import DocumentOrchestrator, IndexingRequest

# Initialize settings and orchestrator
settings = Settings()
orchestrator = DocumentOrchestrator(settings)

# Create indexing request
request = IndexingRequest(
    file_paths=["document1.pdf", "document2.docx", "document3.txt"],
    collection_name="company_documents",
    chunking_strategy="paragraph",  # or "sentence", "token", "character"
    chunk_size=1000,
    chunk_overlap=200,
    add_metadata=True,
    add_context=True  # Generate context for each chunk using LLM
)

# Execute indexing
result = orchestrator.index_documents(request)

print(f"Indexed {result.indexed_count} documents")
print(f"Created {result.chunk_count} chunks")
print(f"Failed: {result.failed_paths}")
```

### Retrieving Documents

To retrieve documents, create a RetrievalRequest:

```python
from src.indexers import RetrievalRequest

# Create retrieval request
request = RetrievalRequest(
    query="What is the company's policy on remote work?",
    collection_name="company_documents",
    top_k=5,
    rerank=True,  # Use reranker if available
    similarity_threshold=0.7
)

# Execute retrieval
result = orchestrator.retrieve_documents(request)

# Access retrieved documents
for item in result.items:
    print(f"Score: {item.score}")
    print(f"Text: {item.text}")
    print(f"Source: {item.metadata.get('source')}")
    print("---")
```

### Using Hybrid Search

For improved results, you can use hybrid search (combining vector and keyword search):

```python
# Create hybrid search request
request = RetrievalRequest(
    query="What is the company's policy on remote work?",
    collection_name="company_documents",
    top_k=10,
    hybrid_search=True,  # Enable hybrid search
    hybrid_search_weight=0.3  # Weight for keyword search (0-1)
)

# Execute hybrid retrieval
result = orchestrator.retrieve_documents(request)
```

## RAG Agent Integration

Arshai includes a specialized RAGAgent that integrates retrieval with generation:

```python
from arshai import Settings
from seedwork.interfaces.iagent import IAgentConfig, IAgentInput

# Initialize settings
settings = Settings()

# Create RAG agent configuration
rag_config = IAgentConfig(
    task_context="You are a helpful assistant that answers questions based on retrieved documents.",
    tools=[],
    rag_config={
        "collection_name": "company_documents",
        "top_k": 5,
        "rerank": True,
        "similarity_threshold": 0.7
    }
)

# Create RAG agent
rag_agent = settings.create_agent("rag", rag_config)

# Process a query
response = rag_agent.process_message(
    IAgentInput(
        message="What is our company's policy on remote work?",
        conversation_id="rag_demo"
    )
)

print(response.agent_message)
```

## Advanced Features

### Document Processing Pipeline

You can configure custom document processing pipelines:

```python
# Custom processing pipeline
request = IndexingRequest(
    file_paths=["document.pdf"],
    collection_name="custom_pipeline",
    processing_pipeline=[
        {"type": "text_extraction", "config": {"extract_tables": True}},
        {"type": "chunking", "config": {"strategy": "paragraph", "size": 800}},
        {"type": "embedding", "config": {"batch_size": 32}}
    ]
)

result = orchestrator.index_documents(request)
```

### Adding Context with LLMs

The indexing system can use LLMs to generate context for each chunk:

```python
# Add LLM-generated context
request = IndexingRequest(
    file_paths=["document.pdf"],
    collection_name="with_context",
    add_context=True,
    context_type="summary"  # or "keywords", "entities", "concepts"
)

result = orchestrator.index_documents(request)
```

### Reranking Results

To improve retrieval quality, you can use rerankers:

```python
from src.indexers import RetrievalRequest

# Create request with reranking
request = RetrievalRequest(
    query="What is our policy on remote work?",
    collection_name="company_documents",
    top_k=20,  # Retrieve more candidates
    rerank=True,  # Enable reranking
    rerank_top_k=5  # Return top 5 after reranking
)

result = orchestrator.retrieve_documents(request)
```

### Multi-Collection Search

You can search across multiple collections:

```python
# Search across multiple collections
request = RetrievalRequest(
    query="What is our policy on remote work?",
    collection_names=["company_policies", "employee_handbook", "hr_documents"],
    top_k=3,  # Top 3 from each collection
    merge_results=True  # Merge and rerank all results
)

result = orchestrator.retrieve_documents(request)
```

## Best Practices

### Chunking Strategies

Choose the right chunking strategy for your content:

1. **Paragraph chunks**: Good for structured documents with clear paragraphs
2. **Sentence chunks**: Better for capturing fine-grained information
3. **Fixed-size chunks**: Consistent chunk sizes for predictable behavior
4. **Semantic chunks**: Split at semantic boundaries for better context preservation

### Embedding Models

Considerations for embedding models:

1. **Dimensionality**: Higher dimensions capture more information but require more storage
2. **Speed vs. Quality**: Faster models may sacrifice accuracy
3. **Domain-Specific**: Some models perform better on specific domains
4. **Cost**: Consider API costs for hosted models

### Vector Database Selection

Choose the right vector database based on:

1. **Scale**: How many documents will you index?
2. **Query Speed**: How fast do you need results?
3. **Deployment**: Self-hosted vs. managed service
4. **Feature Set**: Filtering, hybrid search, metadata storage

### Performance Optimization

Tips for optimizing performance:

1. **Batch Processing**: Process documents in batches
2. **Chunk Size**: Balance between context and retrieval granularity
3. **Embedding Caching**: Cache embeddings for repeated texts
4. **Index Management**: Regularly optimize and compact indices

## Configuration

### Settings Integration

Configure the document indexing system through settings:

```yaml
# config.yaml
embedding:
  provider: openai
  model: text-embedding-3-small

vector_db:
  provider: milvus
  host: localhost
  port: 19530
  
reranker:
  provider: bge
  model: bge-reranker-large
```

```python
from arshai import Settings

# Load settings from config file
settings = Settings("config.yaml")

# Create orchestrator with settings
orchestrator = DocumentOrchestrator(settings)
```

### Component Options

All components can be configured with options:

```python
# Configure embedding model
embedding_config = {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "batch_size": 32
}

# Configure vector database
vector_db_config = {
    "provider": "milvus",
    "host": "localhost",
    "port": 19530,
    "consistency_level": "Strong"
}

# Use in settings
settings = Settings({
    "embedding": embedding_config,
    "vector_db": vector_db_config
})
```

## Examples

### Basic Document Indexing

```python
from arshai import Settings
from src.indexers import DocumentOrchestrator, IndexingRequest

# Initialize
settings = Settings()
orchestrator = DocumentOrchestrator(settings)

# Index documents
request = IndexingRequest(
    file_paths=["./documents/policies.pdf", "./documents/handbook.docx"],
    collection_name="company_docs",
    chunking_strategy="paragraph",
    chunk_size=1000,
    chunk_overlap=200
)

result = orchestrator.index_documents(request)
print(f"Indexed {result.indexed_count} documents with {result.chunk_count} chunks")
```

### Simple RAG Application

```python
from arshai import Settings
from src.indexers import DocumentOrchestrator, RetrievalRequest
from seedwork.interfaces.illm import ILLMConfig, ILLM

# Initialize components
settings = Settings()
orchestrator = DocumentOrchestrator(settings)
llm = settings.create_llm()

# Retrieve documents
request = RetrievalRequest(
    query="What is our policy on remote work?",
    collection_name="company_docs",
    top_k=5
)

result = orchestrator.retrieve_documents(request)

# Format context for LLM
context = "Here are some relevant documents:\n\n"
for i, item in enumerate(result.items):
    context += f"Document {i+1}:\n{item.text}\n\n"

# Generate answer with LLM
messages = [
    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
    {"role": "user", "content": f"{context}\n\nBased on these documents, what is the company's policy on remote work?"}
]

response = llm.chat(messages)
print(response.message)
```

### Advanced RAG with Reranking

```python
from arshai import Settings
from src.indexers import DocumentOrchestrator, RetrievalRequest

# Initialize with reranker
settings = Settings({
    "reranker": {
        "provider": "bge",
        "model": "bge-reranker-large"
    }
})
orchestrator = DocumentOrchestrator(settings)

# Retrieve with reranking
request = RetrievalRequest(
    query="What are the requirements for submitting expense reports?",
    collection_name="company_docs",
    top_k=20,
    rerank=True,
    rerank_top_k=5
)

result = orchestrator.retrieve_documents(request)

# Print results
for item in result.items:
    print(f"Score: {item.score}")
    print(f"Text: {item.text[:100]}...")
    print("---")
```

### Multi-Modal Document Indexing

```python
from arshai import Settings
from src.indexers import DocumentOrchestrator, IndexingRequest

# Initialize
settings = Settings()
orchestrator = DocumentOrchestrator(settings)

# Index with image processing
request = IndexingRequest(
    file_paths=["./documents/presentation.pdf"],
    collection_name="presentations",
    chunking_strategy="slide",  # Custom slide-based chunking
    process_images=True,  # Extract and process images
    image_captioning=True,  # Generate captions for images
    ocr_enabled=True  # Extract text from images
)

result = orchestrator.index_documents(request)
print(f"Processed {result.image_count} images")
``` 