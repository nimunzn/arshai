from pathlib import Path
from typing import Optional, Dict, Any, List

from .unstructured_loader import UnstructuredLoader
from arshai.document_loaders.config import UnstructuredLoaderConfig
from arshai.core.interfaces.itext_splitter import ITextSplitter

class UnstructuredPowerPointLoader(UnstructuredLoader):
    """PowerPoint presentation loader based on the Unstructured package.
    
    This is a specialized version of the UnstructuredLoader for PowerPoint presentations (PPT, PPTX).
    """
    
    def __init__(self, config: UnstructuredLoaderConfig, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the PowerPoint loader.
        
        Args:
            config: Configuration for the loader
            text_splitter: Optional text splitter for chunking documents
        """
        super().__init__(config, text_splitter)
        self.logger.info("Initializing UnstructuredPowerPointLoader")
    
    def _get_partition_params(self, file_path: Path) -> Dict[str, Any]:
        """Get parameters for Unstructured's partition function.
        
        This overrides the base implementation to add PowerPoint-specific parameters.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of parameters
        """
        params = super()._get_partition_params(file_path)
        
        # For PowerPoint, we always want to include page breaks to separate slides
        params["include_page_breaks"] = True
        
        # Handle image extraction and OCR if strategy is hi_res
        if self.config.strategy == "hi_res":
            params["extract_images_in_pdf"] = True
            if not self.config.ocr_languages:
                params["ocr_languages"] = self.config.languages
        
        return params
    
    def _extract_metadata(self, element: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from an Unstructured element.
        
        This overrides the base implementation to add PowerPoint-specific metadata.
        
        Args:
            element: Unstructured Element object
            file_path: Path to the source file
            
        Returns:
            Dictionary of metadata
        """
        metadata = super()._extract_metadata(element, file_path)
        
        # Add PowerPoint-specific metadata
        # For PowerPoint, page_number corresponds to slide number
        if "page_number" in metadata:
            metadata["slide_number"] = metadata["page_number"]
        
        # Mark this as a presentation document
        metadata["document_type"] = "presentation"
        
        return metadata
    
    def load_file(self, file_path: Path) -> List[Any]:
        """Load a single file and return a list of documents.
        
        This method adds validation to ensure only PowerPoint files are processed.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        suffix = file_path.suffix.lower()
        
        # Verify file type
        if suffix not in ['.ppt', '.pptx']:
            self.logger.warning(f"File {file_path} is not a PowerPoint presentation. Skipping.")
            return []
            
        return super().load_file(file_path) 