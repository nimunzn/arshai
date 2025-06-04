from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

from arshai.core.interfaces.ifile_loader import IFileLoaderConfig
from arshai.core.interfaces.itext_processor import ITextProcessorConfig

class UnstructuredLoaderConfig(IFileLoaderConfig):
    """Configuration for the Unstructured file loaders."""
    
    strategy: str = Field(
        default="hi_res",
        description="Strategy for processing documents (fast, hi_res, ocr_only)"
    )
    
    include_page_breaks: bool = Field(
        default=True,
        description="Whether to include page breaks in the output"
    )
    
    max_characters: int = Field(
        default=8192,
        description="Maximum number of characters per element"
    )
    
    ocr_languages: Optional[List[str]] = Field(
        default=None,
        description="Languages to use for OCR (if needed)"
    )
    
    pdf_infer_table_structure: bool = Field(
        default=True,
        description="Whether to attempt to infer table structure in PDFs"
    )
    
    preserve_formatting: bool = Field(
        default=True,
        description="Whether to preserve text formatting when possible"
    )

class AudioLoaderConfig(IFileLoaderConfig):
    """Configuration for the audio file loader."""
    
    model: str = Field(
        default="whisper-1",
        description="Model to use for audio transcription"
    )
    
    language: str = Field(
        default="eng",
        description="Language of the audio content"
    )
    
    correction_model: str = Field(
        default="gpt-4",
        description="Model to use for transcript correction"
    )
    
    api_key: str = Field(
        description="API key for external services"
    )

class TextCleanerConfig(ITextProcessorConfig):
    """Configuration for text cleaner.
    
    Provides specific options for text cleaning processors.
    """
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip whitespace from text"
    )
    
    lowercase: bool = Field(
        default=False,
        description="Whether to convert text to lowercase"
    )
    
    remove_urls: bool = Field(
        default=False,
        description="Whether to remove URLs from text"
    )
    
    remove_emails: bool = Field(
        default=False,
        description="Whether to remove email addresses from text"
    )
    
    remove_phone_numbers: bool = Field(
        default=False,
        description="Whether to remove phone numbers from text"
    )
    
    remove_hashtags: bool = Field(
        default=False,
        description="Whether to remove hashtags from text"
    )
    
    remove_user_handles: bool = Field(
        default=False,
        description="Whether to remove user handles from text"
    )
    
    remove_extra_whitespace: bool = Field(
        default=True,
        description="Whether to remove extra whitespace from text"
    )
    
    remove_extra_newlines: bool = Field(
        default=True,
        description="Whether to remove extra newlines from text"
    )
    
    max_consecutive_newlines: int = Field(
        default=2,
        description="Maximum number of consecutive newlines to allow"
    )
    
    replace_unicode_quotes: bool = Field(
        default=True,
        description="Whether to replace Unicode quotes with ASCII quotes"
    )
    
    replace_unicode_bullets: bool = Field(
        default=True,
        description="Whether to replace Unicode bullets with ASCII bullets"
    )
    
    remove_citations: bool = Field(
        default=False,
        description="Whether to remove citations from text"
    )
    
    normalize_unicode: bool = Field(
        default=False,
        description="Whether to normalize Unicode characters"
    )
    
    replace_special_chars: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of special characters to replace"
    )
    
    remove_digits: bool = Field(
        default=False,
        description="Whether to remove digits from text"
    )
    
    remove_punctuation: bool = Field(
        default=False,
        description="Whether to remove punctuation from text"
    )
    
    min_text_length: int = Field(
        default=0,
        description="Minimum length of text to keep after cleaning"
    )

class ContextEnricherConfig(ITextProcessorConfig):
    """Configuration for context enricher.
    
    Provides specific options for context enrichment processors.
    """
    window_size: int = Field(
        default=3,
        description="Number of documents to include in context window"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in context"
    )
    
    add_source_documents: bool = Field(
        default=False,
        description="Whether to add source documents to metadata"
    )
    
    gpt_model: str = Field(
        default="gpt-4",
        description="Model to use for GPT context enrichment"
    )

class UnstructuredProcessorConfig(ITextProcessorConfig):
    """Configuration for the UnstructuredFileProcessor."""
    
    languages: List[str] = Field(
        default=["eng"],
        description="List of languages to consider during processing"
    )
    
    strategy: str = Field(
        default="hi_res",
        description="Strategy for processing documents (fast, hi_res, ocr_only)"
    )
    
    include_page_breaks: bool = Field(
        default=True,
        description="Whether to include page breaks in the output"
    )
    
    max_characters: int = Field(
        default=8192,
        description="Maximum number of characters per element"
    )
    
    ocr_languages: Optional[List[str]] = Field(
        default=None,
        description="Languages to use for OCR (if needed)"
    )
    
    pdf_infer_table_structure: bool = Field(
        default=True,
        description="Whether to attempt to infer table structure in PDFs"
    )
    
    preserve_formatting: bool = Field(
        default=True,
        description="Whether to preserve text formatting when possible"
    ) 