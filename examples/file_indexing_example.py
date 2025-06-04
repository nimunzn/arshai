"""
Example usage of the document indexing system in Arshai.

This script demonstrates how to:
1. Configure settings for document indexing
2. Index various document types (PDF, DOCX, etc.)
3. Use different chunking strategies
4. Add contextual information to chunks
5. Query the indexed documents
"""

import asyncio
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

from arshai.config.settings import Settings
from src.indexers import DocumentOrchestrator, IndexingRequest, IndexingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def index_documents(settings, file_paths: List[str], collection_name: str = "documents"):
    """
    Index a list of documents using the DocumentOrchestrator.
    
    Args:
        settings: Application settings
        file_paths: List of paths to document files
        collection_name: Name of the vector database collection
    """
    logger.info(f"Indexing {len(file_paths)} documents to collection '{collection_name}'")
    
    # Create document orchestrator
    orchestrator = DocumentOrchestrator(settings)
    
    # Create indexing request
    request = IndexingRequest(
        file_paths=file_paths,
        collection_name=collection_name,
        # Additional metadata to store with documents
        metadata={
            "source": "example_indexing",
            "organization": "Arshai",
            "indexed_date": "2023-07-15"
        },
        # Chunking strategy (paragraph, character, token, sentence, semantic)
        chunking_strategy="paragraph",
        # Whether to add wider context to chunks
        add_wider_context=True,
        # Whether to add LLM-generated context
        add_context=True
    )
    
    # Execute indexing
    result = await orchestrator.aindex_documents(request)
    
    # Display results
    if result.success:
        logger.info(f"Successfully indexed {result.indexed_count} documents")
        logger.info(f"Processing time: {result.processing_time:.2f} seconds")
    else:
        logger.error(f"Indexing failed: {result.message}")
        if result.errors:
            for file_path, error in result.errors.items():
                logger.error(f"  - {file_path}: {error}")
    
    return result

async def process_file_without_indexing(settings, file_path: str):
    """
    Process a file to extract and chunk content without storing in a vector database.
    
    Args:
        settings: Application settings
        file_path: Path to document file
    """
    logger.info(f"Processing file: {file_path}")
    
    # Create document orchestrator
    orchestrator = DocumentOrchestrator(settings)
    
    # Process the file
    result = orchestrator.process_file(file_path)
    
    # Display results
    if result.success:
        logger.info(f"Successfully processed file into {len(result.documents)} chunks")
        
        # Show sample chunks
        for i, doc in enumerate(result.documents[:3]):
            logger.info(f"Chunk {i+1}:")
            logger.info(f"  Content: {doc.page_content[:100]}...")
            logger.info(f"  Metadata: {doc.metadata}")
        
        if len(result.documents) > 3:
            logger.info(f"... and {len(result.documents) - 3} more chunks")
    else:
        logger.error(f"Processing failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
    
    return result

async def main():
    """Main function to run the example."""
    # Load settings
    config_path = os.environ.get("CONFIG_PATH", "examples/rag_config_example.json")
    settings = Settings(config_path)
    
    # Check for document files
    sample_files = []
    
    # Add files from command line arguments or use a prompt
    import sys
    if len(sys.argv) > 1:
        sample_files = sys.argv[1:]
    else:
        # Prompt for file paths
        file_path = input("Enter path to a document file (PDF, DOCX, etc.): ")
        if file_path and os.path.exists(file_path):
            sample_files.append(file_path)
    
    # If no valid files provided, use a demo mode with text indexing
    if not sample_files:
        logger.info("No files provided, switching to demo mode...")
        
        # Process a text document directly
        await process_text_example(settings)
        return
    
    # Verify files exist
    valid_files = []
    for file_path in sample_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not valid_files:
        logger.error("No valid files to process")
        return
    
    # Option 1: Process file without indexing (just to see chunks)
    await process_file_without_indexing(settings, valid_files[0])
    
    # Option 2: Index files to vector database
    collection_name = input("Enter collection name for indexing (default: documents): ") or "documents"
    
    # Confirm indexing
    confirm = input(f"Index {len(valid_files)} files to collection '{collection_name}'? (y/n): ")
    if confirm.lower() == 'y':
        result = await index_documents(settings, valid_files, collection_name)
        
        # Display collection info
        logger.info(f"Documents indexed to collection: {result.collection_name}")
    else:
        logger.info("Indexing cancelled")

async def process_text_example(settings):
    """
    Example of processing text directly instead of from files.
    """
    logger.info("Demo: Processing text directly")
    
    # Sample text content
    text_content = """
    # Arshai Framework
    
    Arshai is a powerful AI multi-agent framework for building complex workflows with intelligent conversational agents, memory management, and tool integration.
    
    ## Core Features
    
    - **Agent Framework**: Build conversational AI agents with structured memory
    - **LLM Integration**: Seamless integration with various LLM providers
    - **Memory Management**: Sophisticated conversation memory handling with different storage options
    - **Tool Integration**: Easy integration of custom tools and capabilities
    - **Workflow Orchestration**: Build complex multi-agent workflows with state management
    
    ## Architecture
    
    Arshai follows a clean architecture with well-defined interfaces.
    The framework is organized into several key components:
    
    1. **Workflow System**: The orchestration layer that manages agent interactions
    2. **Agent System**: The intelligent components that perform specific tasks
    3. **Memory System**: Stores and retrieves conversation context and knowledge
    4. **Tool System**: Extends agent capabilities with specific functionalities
    5. **LLM Integration**: Connects to large language models for natural language processing
    6. **Indexing System**: Provides document indexing capabilities for RAG applications
    """
    
    # Create document orchestrator
    orchestrator = DocumentOrchestrator(settings)
    
    # Create a temporary file to simulate file loading
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as temp_file:
        temp_file.write(text_content)
        temp_path = temp_file.name
    
    try:
        # Process the file
        logger.info(f"Processing temporary markdown file: {temp_path}")
        result = orchestrator.process_file(temp_path)
        
        if result.success:
            logger.info(f"Successfully processed text into {len(result.documents)} chunks")
            
            # Show all chunks
            for i, doc in enumerate(result.documents):
                logger.info(f"Chunk {i+1}:")
                logger.info(f"  Content: {doc.page_content}")
                logger.info(f"  Metadata: {doc.metadata}")
            
            # Index the text
            index_confirm = input("Index this text to vector database? (y/n): ")
            if index_confirm.lower() == 'y':
                collection_name = input("Enter collection name (default: text_examples): ") or "text_examples"
                
                # Create indexing request
                request = IndexingRequest(
                    file_paths=[temp_path],
                    collection_name=collection_name,
                    metadata={"source": "demo_text", "format": "markdown"},
                    chunking_strategy="semantic",  # Use semantic chunking for markdown
                    add_wider_context=True,
                    add_context=True
                )
                
                # Execute indexing
                index_result = await orchestrator.aindex_documents(request)
                
                if index_result.success:
                    logger.info(f"Successfully indexed text to collection '{index_result.collection_name}'")
                else:
                    logger.error(f"Indexing failed: {index_result.message}")
        else:
            logger.error(f"Processing failed: {result.message}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    asyncio.run(main()) 