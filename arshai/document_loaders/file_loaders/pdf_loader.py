from pathlib import Path
from typing import Optional, Dict, Any

from .unstructured_loader import UnstructuredLoader
from arshai.document_loaders.config import UnstructuredLoaderConfig
from arshai.core.interfaces.itext_splitter import ITextSplitter

class PDFLoader(UnstructuredLoader):
    """PDF file loader based on the Unstructured package.
    
    This is a specialized version of the UnstructuredLoader for PDF files.
    """
    
    def __init__(self, config: UnstructuredLoaderConfig, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the PDF loader.
        
        Args:
            config: Configuration for the loader
            text_splitter: Optional text splitter for chunking documents
        """
        super().__init__(config, text_splitter)
        self.logger.info("Initializing PDFLoader")
    
    def _get_partition_params(self, file_path: Path) -> Dict[str, Any]:
        """Get parameters for Unstructured's partition function.
        
        This overrides the base implementation to add PDF-specific parameters.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of parameters
        """
        params = super()._get_partition_params(file_path)
        
        # Add PDF-specific parameters
        params["pdf_infer_table_structure"] = self.config.pdf_infer_table_structure
        
        # For PDFs, we always want to include page breaks
        params["include_page_breaks"] = True
        
        # For PDFs, we may want to use OCR if text extraction fails
        if self.config.strategy == "hi_res" and not self.config.ocr_languages:
            params["ocr_languages"] = self.config.languages
            
        return params 