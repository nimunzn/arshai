from pathlib import Path
from typing import Optional, Dict, Any, List

from .unstructured_loader import UnstructuredLoader
from arshai.document_loaders.config import UnstructuredLoaderConfig
from arshai.core.interfaces.itext_splitter import ITextSplitter

class UnstructuredWordDocumentLoader(UnstructuredLoader):
    """Word document loader based on the Unstructured package.
    
    This is a specialized version of the UnstructuredLoader for Word documents (DOC, DOCX).
    """
    
    def __init__(self, config: UnstructuredLoaderConfig, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the Word document loader.
        
        Args:
            config: Configuration for the loader
            text_splitter: Optional text splitter for chunking documents
        """
        super().__init__(config, text_splitter)
        self.logger.info("Initializing UnstructuredWordDocumentLoader")
    
    def _get_partition_params(self, file_path: Path) -> Dict[str, Any]:
        """Get parameters for Unstructured's partition function.
        
        This overrides the base implementation to add Word-specific parameters.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of parameters
        """
        params = super()._get_partition_params(file_path)
        
        # For Word documents, we want to preserve formatting
        params["preserve_formatting"] = True
        
        return params
    
    def load_file(self, file_path: Path) -> List[Any]:
        """Load a single file and return a list of documents.
        
        This method adds validation to ensure only Word files are processed.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        suffix = file_path.suffix.lower()
        
        # Verify file type
        if suffix not in ['.doc', '.docx']:
            self.logger.warning(f"File {file_path} is not a Word document. Skipping.")
            return []
            
        return super().load_file(file_path) 