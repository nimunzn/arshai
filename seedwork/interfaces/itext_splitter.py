from typing import Any, List, Optional, Protocol
from pydantic import Field

from .idto import IDTO

class ITextSplitterConfig(IDTO):
    """Configuration for text splitters."""
    chunk_size: int = Field(default=4000, description="Maximum size of chunks to return")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    length_function: Any = Field(default=len, description="Function that measures the length of given chunks")
    keep_separator: bool = Field(default=True, description="Whether to keep the separator in chunks")
    add_start_index: bool = Field(default=False, description="Whether to add start index to chunks metadata")
    strip_whitespace: bool = Field(default=True, description="Whether to strip whitespace from chunks")

class ITextSplitter(Protocol):
    """Interface for text splitters.
    
    Text splitters are responsible for breaking down text into smaller chunks
    that can be processed more effectively.
    """

    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        ...

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[dict]:
        """Create documents from a list of texts.
        
        Args:
            texts: List of texts to convert to documents
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document dictionaries
        """
        ...

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents into multiple smaller documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of split document dictionaries
        """
        ... 