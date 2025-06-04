import re
import unicodedata
import logging
from typing import List, Optional, Callable, Dict, Any

from arshai.core.interfaces import ITextProcessor
from arshai.core.interfaces import Document
from arshai.document_loaders.config import TextCleanerConfig

logger = logging.getLogger(__name__)

class TextCleaner(ITextProcessor):
    """Processor for cleaning and normalizing text.
    
    This processor can be used as part of the document loading pipeline
    to clean and normalize text after extraction and chunking.
    """
    
    def __init__(self, config: Optional[TextCleanerConfig] = None):
        """Initialize the text cleaner.
        
        Args:
            config: Configuration for the cleaner. If None, uses TextCleanerConfig defaults.
        """
        self.config = config or TextCleanerConfig()
        self._custom_processors = {}
        
        # Compile regex patterns
        self._url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self._email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self._phone_pattern = re.compile(r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
        self._hashtag_pattern = re.compile(r'#\w+')
        self._user_handle_pattern = re.compile(r'@\w+')
        self._whitespace_pattern = re.compile(r'\s+')
        self._newline_pattern = re.compile(r'\n{3,}')
        self._citation_pattern = re.compile(r'\[\d+\]|\(\d+\)|\d+\.')
    
    def register_processor(self, name: str, processor_func: Callable[[List[Document]], List[Document]]) -> None:
        """Register a custom processor function.
        
        Args:
            name: Name of the processor
            processor_func: Function that takes and returns a list of Documents
        """
        if name in self._custom_processors:
            logger.warning(f"Processor with name {name} already exists and will be overwritten")
            
        self._custom_processors[name] = processor_func
        logger.info(f"Registered custom processor: {name}")
    
    def process(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents to clean their text.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed Document objects
        """
        if not documents:
            logger.warning("No documents provided for processing")
            return []
            
        processed_docs = []
        
        try:
            for doc in documents:
                # Process the text
                processed_text = self.process_text(doc.page_content)
                
                # Skip empty results
                if not processed_text:
                    continue
                    
                # Create a new document with processed text
                processed_docs.append(Document(
                    page_content=processed_text,
                    metadata=doc.metadata.copy()
                ))
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error cleaning document text: {str(e)}")
            # Return original documents in case of error
            return documents
    
    def process_text(self, text: str) -> Optional[str]:
        """Clean a single text string according to configuration.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text or None if text is empty after cleaning
        """
        if text is None or text == "":
            return None
            
        try:
            # Apply cleaning operations based on config
            if self.config.strip_whitespace:
                text = text.strip()
            
            if self.config.lowercase:
                text = text.lower()
            
            if self.config.remove_urls:
                text = self._url_pattern.sub('', text)
            
            if self.config.remove_emails:
                text = self._email_pattern.sub('', text)
            
            if self.config.remove_phone_numbers:
                text = self._phone_pattern.sub('', text)
            
            if self.config.remove_hashtags:
                text = self._hashtag_pattern.sub('', text)
            
            if self.config.remove_user_handles:
                text = self._user_handle_pattern.sub('', text)
            
            if self.config.replace_unicode_quotes:
                text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
            
            if self.config.replace_unicode_bullets:
                text = text.replace('•', '*').replace('·', '*').replace('○', '*').replace('◦', '*')
            
            if self.config.remove_citations:
                text = self._citation_pattern.sub('', text)
            
            # Apply custom character replacements
            for char, replacement in self.config.replace_special_chars.items():
                text = text.replace(char, replacement)
            
            if self.config.remove_digits:
                text = re.sub(r'\d', '', text)
            
            if self.config.remove_punctuation:
                text = re.sub(r'[^\w\s]', '', text)
            
            if self.config.remove_extra_whitespace:
                text = self._whitespace_pattern.sub(' ', text)
            
            if self.config.remove_extra_newlines and self.config.max_consecutive_newlines > 0:
                # Replace 3+ consecutive newlines with config.max_consecutive_newlines newlines
                replacement = '\n' * self.config.max_consecutive_newlines
                text = self._newline_pattern.sub(replacement, text)
            
            if hasattr(self.config, 'normalize_unicode') and self.config.normalize_unicode:
                text = unicodedata.normalize('NFKC', text)
            
            # Apply final whitespace stripping if enabled
            if self.config.strip_whitespace:
                text = text.strip()
            
            # Check minimum length
            if hasattr(self.config, 'min_text_length') and len(text) < self.config.min_text_length:
                return None
            
            return text or None
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return text  # Return original in case of error
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text while preserving single spaces and newlines.
        
        Args:
            text: Text to process
            
        Returns:
            Text with extra whitespace removed
        """
        if not text:
            return text
            
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces at the beginning and end of lines
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        
        # Additional cleaning: Replace multiple newlines with double newline
        text = re.sub(r'\n+', '\n\n', text)
        
        # Clean up line endings
        text = '\n'.join(line.strip() for line in text.splitlines())
        
        return text.strip()
    
    def run_custom_processor(self, name: str, documents: List[Document]) -> List[Document]:
        """Run a registered custom processor.
        
        Args:
            name: Name of the processor to run
            documents: List of Document objects
            
        Returns:
            Processed Document objects
        """
        if name not in self._custom_processors:
            error_msg = f"No processor registered with name: {name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            return self._custom_processors[name](documents)
        except Exception as e:
            logger.error(f"Error running custom processor {name}: {str(e)}")
            # Return original documents in case of error
            return documents 