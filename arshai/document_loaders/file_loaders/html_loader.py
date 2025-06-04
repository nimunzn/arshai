from pathlib import Path
from typing import Optional, Dict, Any, List

from .unstructured_loader import UnstructuredLoader
from arshai.document_loaders.config import UnstructuredLoaderConfig
from arshai.core.interfaces.itext_splitter import ITextSplitter

class UnstructuredHTMLLoader(UnstructuredLoader):
    """HTML document loader based on the Unstructured package.
    
    This is a specialized version of the UnstructuredLoader for HTML files.
    """
    
    def __init__(self, config: UnstructuredLoaderConfig, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the HTML loader.
        
        Args:
            config: Configuration for the loader
            text_splitter: Optional text splitter for chunking documents
        """
        super().__init__(config, text_splitter)
        self.logger.info("Initializing UnstructuredHTMLLoader")
    
    def _get_partition_params(self, file_path: Path) -> Dict[str, Any]:
        """Get parameters for Unstructured's partition function.
        
        This overrides the base implementation to add HTML-specific parameters.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of parameters
        """
        params = super()._get_partition_params(file_path)
        
        # For HTML, we want to extract links when possible
        params["extract_links"] = True
        
        # HTML-specific settings
        params["html_assemble_articles"] = True
        params["html_include_links"] = True
        
        return params
    
    def _extract_metadata(self, element: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from an Unstructured element.
        
        This overrides the base implementation to add HTML-specific metadata.
        
        Args:
            element: Unstructured Element object
            file_path: Path to the source file
            
        Returns:
            Dictionary of metadata
        """
        metadata = super()._extract_metadata(element, file_path)
        
        # Add HTML-specific metadata
        metadata["document_type"] = "web_document"
        
        # Extract and add link information if available
        if hasattr(element, "metadata") and hasattr(element.metadata, "links"):
            metadata["links"] = element.metadata.links
            
        # Extract HTML element type if available
        if hasattr(element, "metadata") and hasattr(element.metadata, "html_element"):
            metadata["html_element"] = element.metadata.html_element
        
        return metadata
    
    def load_file(self, file_path: Path) -> List[Any]:
        """Load a single file and return a list of documents.
        
        This method adds validation to ensure only HTML files are processed.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        suffix = file_path.suffix.lower()
        
        # Verify file type
        if suffix not in ['.html', '.htm', '.xhtml']:
            self.logger.warning(f"File {file_path} is not an HTML document. Skipping.")
            return []
            
        return super().load_file(file_path) 