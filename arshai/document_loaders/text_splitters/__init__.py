"""
Text splitters for chunking text into smaller pieces.

These splitters are used as part of step 2 of the document loading process:
2. Chunking/Splitting: Breaking down raw data into chunks
"""

from arshai.core.interfaces.itext_splitter import ITextSplitter, ITextSplitterConfig
from .recursive_splitter import RecursiveTextSplitter

__all__ = [
    "ITextSplitter",
    "ITextSplitterConfig",
    "RecursiveTextSplitter"
] 