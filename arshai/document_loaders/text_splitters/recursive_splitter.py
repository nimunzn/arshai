import re
from typing import List, Dict, Any, Optional, Callable

from arshai.core.interfaces.itext_splitter import ITextSplitter, ITextSplitterConfig

class RecursiveTextSplitter(ITextSplitter):
    """Recursive text splitter that splits text by a list of separators.
    
    This splitter recursively tries a list of separators to split text into chunks.
    """
    
    def __init__(self, config: ITextSplitterConfig, separators: Optional[List[str]] = None):
        """Initialize the recursive text splitter.
        
        Args:
            config: Configuration for the text splitter
            separators: Optional list of separators to use, in order of priority
        """
        self.config = config
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.length_function = config.length_function
        self.keep_separator = config.keep_separator
        self.strip_whitespace = config.strip_whitespace
        self.add_start_index = config.add_start_index

    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components using recursive splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Final chunks
        chunks = []
        
        # Clean text if needed
        if self.strip_whitespace:
            text = text.strip()
        
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        # Find appropriate separator for this text
        separator = self._find_separator(text)
        
        # Split by separator
        splits = self._split_by_separator(text, separator)
        
        # Process chunks
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # If we're at the chunk size limit, add the chunk to the list of chunks
            if current_length + self.length_function(split) > self.chunk_size and current_chunk:
                chunk_text = self._merge_splits(current_chunk)
                chunks.append(chunk_text)
                
                # Keep overlap for next chunk
                overlap_splits = []
                overlap_length = 0
                
                for s in reversed(current_chunk):
                    if overlap_length + self.length_function(s) > self.chunk_overlap:
                        break
                    overlap_splits.insert(0, s)
                    overlap_length += self.length_function(s)
                
                # Reset for next chunk
                current_chunk = overlap_splits
                current_length = overlap_length
            
            # Add the current split to the current chunk
            current_chunk.append(split)
            current_length += self.length_function(split)
        
        # Add the last chunk if there is one
        if current_chunk:
            chunk_text = self._merge_splits(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _find_separator(self, text: str) -> str:
        """Find the appropriate separator for this text.
        
        Args:
            text: Text to split
            
        Returns:
            Appropriate separator
        """
        for separator in self.separators:
            if separator in text:
                return separator
        
        # If no separator is found, use the last one (usually character-level)
        return self.separators[-1]
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator.
        
        Args:
            text: Text to split
            separator: Separator to use
            
        Returns:
            List of text segments
        """
        if not separator:
            return list(text)
            
        # If we want to keep the separator, append it to each segment
        if self.keep_separator:
            splits = []
            for segment in text.split(separator):
                if segment:
                    if self.strip_whitespace:
                        segment = segment.strip()
                    if segment:
                        splits.append(segment + (separator if separator else ""))
            return splits
        else:
            return [s for s in text.split(separator) if s]
    
    def _merge_splits(self, splits: List[str]) -> str:
        """Merge a list of splits into a single text.
        
        Args:
            splits: List of text segments
            
        Returns:
            Merged text
        """
        return "".join(splits)
    
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[dict]:
        """Create documents from a list of texts.
        
        Args:
            texts: List of texts to convert to documents
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document dictionaries
        """
        metadatas = metadatas or [{}] * len(texts)
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i].copy()
            for chunk in self.split_text(text):
                doc = {
                    "page_content": chunk,
                    "metadata": metadata.copy()
                }
                
                # Add start index if requested
                if self.add_start_index:
                    doc["metadata"]["start_index"] = text.find(chunk)
                    
                documents.append(doc)
        
        return documents
    
    def split_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents into multiple smaller documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of split document dictionaries
        """
        texts, metadatas = [], []
        
        for doc in documents:
            texts.append(doc["page_content"])
            metadatas.append(doc["metadata"])
            
        return self.create_documents(texts, metadatas) 