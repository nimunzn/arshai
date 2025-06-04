from typing import Dict, Protocol, Optional
from PIL import Image
from pydantic import Field


class IIndexing(Protocol):
    """Interface for indexing multimodal documents (PDFs with images).
    
    Responsible for converting PDFs to images, processing them, 
    generating embeddings, and storing them in a vector database.
    """

    
    def create_collection(self) -> None:
        """
        Create the collection for multimodal embeddings.
        """
        ...
    
    def index_document(self, pdf_path: str, save_pages: bool = False, pages_dir: str = "pages") -> None:
        """
        Index a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            save_pages: Whether to save the page images to disk
            pages_dir: Directory to save page images
        """
        ...
    
    def index_directory(self, save_pages: bool = False, pages_dir: str = "pages") -> None:
        """
        Index all PDF documents in the configured directory.
        
        Args:
            save_pages: Whether to save the page images to disk
            pages_dir: Directory to save page images
        """
        ...
    
