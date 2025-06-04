from abc import ABC, abstractmethod
from typing import List, Optional, Union, Protocol
from pathlib import Path
from pydantic import Field

from .idocument import Document
from .idto import IDTO

class IFileLoaderConfig(IDTO):
    """Base configuration for file loaders.
    
    This configuration is used by file loaders to determine how to process files.
    """
    languages: List[str] = Field(default_factory=lambda: ["eng"], description="Languages to process")
    max_characters: int = Field(default=8192, description="Maximum characters per chunk")
    chunk_size: int = Field(default=1024, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")

class IFileLoader(Protocol):
    """Base interface for all file loaders.
    
    File loaders are responsible for loading documents from specific file types.
    Each file loader should be specialized for a particular file format.
    """
    
    @abstractmethod
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single file and return a list of documents.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        ...
    
    @abstractmethod
    def load_files(self, file_paths: List[Union[str, Path]], separator: Optional[str] = None) -> List[Document]:
        """Load multiple files and return a list of documents.
        
        Args:
            file_paths: List of paths to files to load
            separator: Optional separator for splitting text into documents
            
        Returns:
            List of Document objects
        """
        ... 