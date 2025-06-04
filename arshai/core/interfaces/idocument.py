from typing import Protocol, Any, List, Dict
from pydantic import Field
from .idto import IDTO

class Document(IDTO):
    """Base document class for storing text content and metadata."""
    
    page_content: str = Field(description="The string content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary of metadata about the document")

    def __repr__(self) -> str:
        """Return string representation of the document."""
        return f"Document(page_content={self.page_content[:50]}..., metadata={self.metadata})" 