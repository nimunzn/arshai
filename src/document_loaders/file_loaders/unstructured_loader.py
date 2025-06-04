from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dict
from unstructured.documents.elements import Element, Text, Title, NarrativeText, ListItem, Table

from .base_loader import BaseFileLoader
from arshai.document_loaders.config import UnstructuredLoaderConfig
from arshai.core.interfaces import ITextSplitter

class UnstructuredLoader(BaseFileLoader):
    """File loader using the Unstructured package.
    
    This loader extracts content from various file types using the 
    Unstructured library's partition function.
    """
    
    def __init__(self, config: UnstructuredLoaderConfig, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the unstructured loader.
        
        Args:
            config: Configuration for the loader
            text_splitter: Optional text splitter for chunking documents
        """
        super().__init__(config, text_splitter)
        self.config = config  # Override with the correct type
    
    def _extract_content(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract content from a file using Unstructured.
        
        This method implements step 1 of our approach:
        - Extract raw content without any processing
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of dictionaries with 'content' and 'metadata' keys
        """
        try:
            self.logger.info(f"Extracting content from: {file_path}")
            
            # Get partition parameters
            partition_params = self._get_partition_params(file_path)
            
            # Extract elements using Unstructured's partition function
            elements = partition(**partition_params)
            
            # Convert elements to content dictionaries
            contents = []
            for element in elements:
                # Extract text from element
                text = self._get_element_text(element)
                if not text:
                    continue
                
                # Extract metadata
                metadata = self._extract_metadata(element, file_path)
                
                # Add to contents
                contents.append({
                    'content': text,
                    'metadata': metadata
                })
            
            self.logger.info(f"Extracted {len(contents)} elements from {file_path}")
            return contents
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {str(e)}")
            raise
    
    def _get_partition_params(self, file_path: Path) -> Dict[str, Any]:
        """Get parameters for Unstructured's partition function.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of parameters
        """
        params = {
            "filename": str(file_path),
            "strategy": self.config.strategy,
            "languages": self.config.languages,
            "include_page_breaks": self.config.include_page_breaks,
            "max_characters": self.config.max_characters,
            "preserve_formatting": self.config.preserve_formatting
        }
        
        # Add OCR languages if specified
        if self.config.ocr_languages:
            params["ocr_languages"] = self.config.ocr_languages
        
        # Add PDF-specific parameters
        if file_path.suffix.lower() == '.pdf':
            params["pdf_infer_table_structure"] = self.config.pdf_infer_table_structure
        
        return params
    
    def _get_element_text(self, element: Element) -> str:
        """Get text from an Unstructured element.
        
        Args:
            element: Unstructured Element object
            
        Returns:
            Text string
        """
        if isinstance(element, (Text, Title, NarrativeText)):
            return element.text.strip()
        elif isinstance(element, ListItem):
            return f"- {element.text.strip()}"
        elif isinstance(element, Table):
            return json.dumps(element.metadata.get("text_as_html", ""))
        return ""
    
    def _extract_metadata(self, element: Element, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from an Unstructured element.
        
        Args:
            element: Unstructured Element object
            file_path: Path to the source file
            
        Returns:
            Dictionary of metadata
        """
        # Convert element to dict format
        element_dict = convert_to_dict(element)
        
        # Base metadata
        metadata = {
            "source": str(file_path),
            "element_type": element_dict.get("type"),
            "page_number": element_dict.get("page_number"),
            "coordinates": element_dict.get("coordinates"),
            "file_type": self.get_file_type(file_path)
        }
        
        # Add table-specific metadata
        if isinstance(element, Table):
            metadata["is_table"] = True
            if hasattr(element.metadata, "text_as_html"):
                metadata["html"] = element.metadata.text_as_html
        
        return metadata 