"""
Document loading and processing utilities.

This module provides utilities for loading and processing documents, following a 3-step approach:
1. Extraction/Parsing: Raw extraction without processing, part of file loaders
2. Chunking/Splitting: Breaking down raw data into chunks, also part of loaders
3. Processing: Optional post-processing steps that are case-specific

Each step is modular and can be used independently or combined.
"""

# File loaders (steps 1 & 2)
from arshai.document_loaders.file_loaders.base_loader import BaseFileLoader
from arshai.document_loaders.file_loaders.unstructured_loader import UnstructuredLoader
from arshai.document_loaders.file_loaders.pdf_loader import PDFLoader
from arshai.document_loaders.file_loaders.audio_loader import AudioLoader
from arshai.document_loaders.file_loaders.word_loader import UnstructuredWordDocumentLoader
from arshai.document_loaders.file_loaders.powerpoint_loader import UnstructuredPowerPointLoader
from arshai.document_loaders.file_loaders.html_loader import UnstructuredHTMLLoader
from arshai.document_loaders.file_loaders.excel_loader import UnstructuredExcelLoader

# Text splitters (step 2)
from arshai.core.interfaces import ITextSplitter, ITextSplitterConfig
from arshai.document_loaders.text_splitters.recursive_splitter import RecursiveTextSplitter

# Processors (step 3)
from arshai.document_loaders.processors.text_cleaner import TextCleaner
from arshai.document_loaders.processors.context_enricher import ContextEnricher
from arshai.document_loaders.processors.utils import save_docs_to_json

# Configs
from arshai.document_loaders.config import (
    UnstructuredLoaderConfig,
    AudioLoaderConfig,
    UnstructuredProcessorConfig,
    TextCleanerConfig,
    ContextEnricherConfig
)

# Import interfaces from seedwork for convenience
from arshai.core.interfaces import Document
from arshai.core.interfaces import IFileLoader, IFileLoaderConfig
from arshai.core.interfaces import ITextProcessor, ITextProcessorConfig

__all__ = [
    # File loaders
    "BaseFileLoader",
    "UnstructuredLoader",
    "PDFLoader",
    "AudioLoader",
    "UnstructuredWordDocumentLoader",
    "UnstructuredPowerPointLoader",
    "UnstructuredHTMLLoader",
    "UnstructuredExcelLoader",
    
    # Text splitters
    "ITextSplitter",
    "ITextSplitterConfig",
    "RecursiveTextSplitter",
    
    # Processors
    "TextCleaner", 
    "TextCleanerConfig",
    "ContextEnricher",
    "ContextEnricherConfig",
    "ITextProcessor",
    "ITextProcessorConfig",
    
    # Utility functions
    "save_docs_to_json",
    
    # Configs
    "UnstructuredLoaderConfig",
    "AudioLoaderConfig",
    "UnstructuredProcessorConfig",
    
    # Seedwork interfaces
    "Document",
    "IFileLoader",
    "IFileLoaderConfig"
]
