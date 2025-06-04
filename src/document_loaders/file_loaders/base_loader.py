from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import logging

from arshai.core.interfaces import IFileLoader, IFileLoaderConfig
from arshai.core.interfaces import Document
from arshai.core.interfaces import ITextSplitter

logger = logging.getLogger(__name__)

class BaseFileLoader(IFileLoader, ABC):
    """Base class for all file loaders.
    
    This class implements the core functionality of file loaders according to the 
    2-step approach:
    1. Extract raw content from files
    2. Apply chunking to extracted content
    
    Each specific file loader only needs to implement the _extract_content method.
    """
    
    def __init__(self, config: IFileLoaderConfig, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the base file loader.
        
        Args:
            config: Configuration for the loader
            text_splitter: Optional text splitter for chunking documents
        """
        self.config = config
        self.text_splitter = text_splitter
        self.logger = logger
    
    @abstractmethod
    def _extract_content(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract content from a file without any processing.
        
        This is the first step in the document loading process.
        Each file loader implementation must override this method.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of dictionaries with 'content' and 'metadata' keys
        """
        pass
    
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single file and return a list of documents.
        
        This method handles the full 2-step process:
        1. Extract content using the specialized _extract_content method
        2. Apply chunking if a text splitter is provided
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        try:
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            self.logger.info(f"Loading file: {file_path}")
            
            # Step 1: Extract raw content
            raw_contents = self._extract_content(file_path)
            
            # Convert to Document objects
            documents = [
                Document(
                    page_content=item['content'],
                    metadata=item['metadata']
                )
                for item in raw_contents
                if item['content']  # Skip empty content
            ]
            
            # Step 2: Apply chunking if text splitter is provided
            if self.text_splitter and documents:
                # Convert to format expected by text splitter
                docs_for_splitting = [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ]
                
                # Split the documents
                chunked_docs = self.text_splitter.split_documents(docs_for_splitting)
                
                # Convert back to Document objects
                documents = [
                    Document(page_content=doc["page_content"], metadata=doc["metadata"])
                    for doc in chunked_docs
                ]
            
            self.logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return []
    
    def load_files(self, file_paths: List[Union[str, Path]], separator: Optional[str] = None) -> List[Document]:
        """Load multiple files and return a list of documents.
        
        Args:
            file_paths: List of paths to files to load
            separator: Optional separator for custom chunking
            
        Returns:
            List of Document objects
        """
        all_documents = []
        
        for file_path in file_paths:
            # Handle normal chunking by default
            documents = self.load_file(file_path)
            
            # If separator is provided, apply custom chunking
            if separator and documents:
                chunked_docs = []
                
                for doc in documents:
                    # Split content by separator
                    split_content = doc.page_content.split(separator)
                    
                    # Create a new document for each chunk
                    for i, chunk in enumerate(split_content):
                        if not chunk.strip():
                            continue
                        
                        new_metadata = doc.metadata.copy()
                        new_metadata["chunk"] = i
                        
                        chunked_docs.append(Document(
                            page_content=chunk,
                            metadata=new_metadata
                        ))
                
                documents = chunked_docs
            
            all_documents.extend(documents)
        
        return all_documents
    
    def get_file_type(self, file_path: Path) -> str:
        """Get the file type from a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String representing the file type
        """
        extension = file_path.suffix.lower().lstrip('.')
        
        # Map extensions to file types
        extension_map = {
            # PDF
            'pdf': 'pdf_document',
            
            # Word documents
            'doc': 'word_document',
            'docx': 'word_document',
            
            # Excel documents
            'xls': 'spreadsheet',
            'xlsx': 'spreadsheet',
            
            # PowerPoint documents
            'ppt': 'presentation',
            'pptx': 'presentation',
            
            # HTML/XML documents
            'html': 'web_document',
            'htm': 'web_document',
            'xml': 'web_document',
            
            # Audio
            'mp3': 'audio',
            'wav': 'audio',
            'ogg': 'audio',
            
            # Video
            'mp4': 'video',
            'avi': 'video',
            'mov': 'video',
            'mkv': 'video',
        }
        
        return extension_map.get(extension, 'unknown') 