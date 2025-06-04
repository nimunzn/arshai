from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from .unstructured_loader import UnstructuredLoader
from arshai.document_loaders.config import UnstructuredLoaderConfig
from arshai.core.interfaces import ITextSplitter

class UnstructuredExcelLoader(UnstructuredLoader):
    """Excel document loader based on the Unstructured package.
    
    This is a specialized version of the UnstructuredLoader for Excel spreadsheets (XLS, XLSX).
    """
    
    def __init__(self, config: UnstructuredLoaderConfig, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the Excel loader.
        
        Args:
            config: Configuration for the loader
            text_splitter: Optional text splitter for chunking documents
        """
        super().__init__(config, text_splitter)
        self.logger.info("Initializing UnstructuredExcelLoader")
    
    def _get_partition_params(self, file_path: Path) -> Dict[str, Any]:
        """Get parameters for Unstructured's partition function.
        
        This overrides the base implementation to add Excel-specific parameters.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of parameters
        """
        params = super()._get_partition_params(file_path)
        
        # Excel-specific settings
        params["xlsx_extract_all_sheets"] = True
        
        # For Excel, we want to infer table structure
        params["infer_table_structure"] = True
        
        return params
    
    def _extract_metadata(self, element: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from an Unstructured element.
        
        This overrides the base implementation to add Excel-specific metadata.
        
        Args:
            element: Unstructured Element object
            file_path: Path to the source file
            
        Returns:
            Dictionary of metadata
        """
        metadata = super()._extract_metadata(element, file_path)
        
        # Add Excel-specific metadata
        metadata["document_type"] = "spreadsheet"
        
        # Extract sheet name if available
        if hasattr(element, "metadata") and hasattr(element.metadata, "sheet_name"):
            metadata["sheet_name"] = element.metadata.sheet_name
            
        # For tables, add Excel-specific information
        if metadata.get("element_type") == "Table":
            metadata["is_table"] = True
            metadata["table_format"] = "excel"
            
            # Add the table data as structured JSON if available
            if hasattr(element, "metadata") and hasattr(element.metadata, "text_as_json"):
                metadata["table_data"] = element.metadata.text_as_json
        
        return metadata
    
    def _get_element_text(self, element: Any) -> str:
        """Get text from an Unstructured element.
        
        This overrides the base implementation to handle Excel-specific elements.
        
        Args:
            element: Unstructured Element object
            
        Returns:
            Text string
        """
        # For tables, convert to a structured format
        if hasattr(element, "metadata") and element.__class__.__name__ == "Table":
            # Try to get the table as HTML or JSON for better structure
            if hasattr(element.metadata, "text_as_html"):
                return element.metadata.text_as_html
            elif hasattr(element.metadata, "text_as_json"):
                return json.dumps(element.metadata.text_as_json)
        
        # For other elements, use the base implementation
        return super()._get_element_text(element)
    
    def load_file(self, file_path: Path) -> List[Any]:
        """Load a single file and return a list of documents.
        
        This method adds validation to ensure only Excel files are processed.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        suffix = file_path.suffix.lower()
        
        # Verify file type
        if suffix not in ['.xls', '.xlsx', '.xlsm', '.xlsb']:
            self.logger.warning(f"File {file_path} is not an Excel spreadsheet. Skipping.")
            return []
            
        return super().load_file(file_path) 