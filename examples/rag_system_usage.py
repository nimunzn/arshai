"""
Example usage of the RAG system in Arshai.

This script demonstrates how to use the RAG system for:
1. Indexing documents
2. Retrieving information based on queries
"""

import asyncio
import os
from pathlib import Path
from arshai.config.settings import Settings

async def index_documents(settings):
    """Index documents using the RAG system."""
    print("=== Indexing Documents ===")
    
    # Create the indexer
    indexer = settings.create_indexer()
    if not indexer:
        print("Indexer not configured. Check your settings.")
        return
    
    # Example: Index a specific document
    # Replace with an actual document path
    document_path = input("Enter the path to a document to index (PDF, DOCX, etc.): ")
    if os.path.exists(document_path):
        result = await indexer.aindex_document(document_path)
        if result.get("success"):
            print(f"Successfully indexed document: {document_path}")
            print(f"Processed {result.get('chunks_processed')} chunks")
        else:
            print(f"Failed to index document: {result.get('message', 'Unknown error')}")
    else:
        print(f"Document not found: {document_path}")
        
        # Alternative: Index text directly
        print("\nIndexing sample text instead...")
        sample_text = """
        Arshai is a lightweight framework for building modular, extensible AI agent workflows. 
        It provides a simplified architecture for creating, configuring, and orchestrating 
        AI agents within structured workflows.
        
        The package follows key principles:
        - Simplicity: Direct instantiation of components with minimal boilerplate
        - Extensibility: Easy extension points for custom business applications
        - Modularity: Independently usable and testable components
        - Consistency: Standard patterns for all component interactions
        """
        
        metadata = {
            "title": "Arshai Overview",
            "author": "Arshai Team",
            "category": "Documentation"
        }
        
        result = await indexer.aindex_text(sample_text, metadata)
        if result.get("success"):
            print("Successfully indexed sample text")
            print(f"Processed {result.get('chunks_processed')} chunks")
        else:
            print(f"Failed to index sample text: {result.get('message', 'Unknown error')}")

async def retrieve_information(settings):
    """Retrieve information using the RAG system."""
    print("\n=== Retrieving Information ===")
    
    # Create the retriever
    retriever = settings.create_retriever()
    if not retriever:
        print("Retriever not configured. Check your settings.")
        return
    
    # Example: Retrieve information based on a query
    query = input("Enter a query to search for information: ")
    if not query.strip():
        query = "What is Arshai?"
        print(f"Using default query: {query}")
    
    try:
        result = await retriever.ainvoke(query)
        print("\nRetrieved Information:")
        print(result)
    except Exception as e:
        print(f"Error retrieving information: {str(e)}")

async def main():
    """Main function to run the example."""
    # Load settings
    config_path = "examples/rag_config_example.json"
    settings = Settings(config_path)
    
    # Index documents
    await index_documents(settings)
    
    # Retrieve information
    await retrieve_information(settings)

if __name__ == "__main__":
    asyncio.run(main()) 