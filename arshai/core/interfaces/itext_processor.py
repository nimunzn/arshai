from typing import List, Optional, Callable, Protocol, Dict, Any
from pydantic import Field

from .idto import IDTO
from .idocument import Document

class ITextProcessorConfig(IDTO):
    """Configuration for text processors.
    
    Base configuration with minimal common parameters.
    Specific processor implementations should extend this with their own configuration.
    """
    # Common configuration options for all processor types
    api_key: Optional[str] = Field(default=None, description="API key for external services (if needed)")

class ITextProcessor(Protocol):
    """Interface for text processors.
    
    Text processors are responsible for transforming document content in various ways:
    - Cleaning and normalizing text
    - Enriching with context from neighboring documents
    - Adding metadata and additional information
    """
    
    def process(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed Document objects
        """
        ...
    
    def process_text(self, text: str) -> Optional[str]:
        """Process a single text string.
        
        Note: Not all processors need to implement this - context processors 
        may return the original text.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text or None if processing failed or is not applicable
        """
        ...
    
    def register_processor(self, name: str, processor_func: Callable[[List[Document]], List[Document]]) -> None:
        """Register a custom processor function.
        
        Args:
            name: Name of the processor
            processor_func: Function that takes and returns a list of Documents
        """
        ...
    
    def run_custom_processor(self, name: str, documents: List[Document]) -> List[Document]:
        """Run a registered custom processor.
        
        Args:
            name: Name of the processor to run
            documents: List of Document objects
            
        Returns:
            Processed Document objects
        """
        ... 