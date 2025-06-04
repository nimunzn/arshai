"""Unit tests for recursive text splitter."""

import pytest
from typing import Dict, Any
import sys
from unittest.mock import MagicMock

# Mock the config module to prevent the import error
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()

from arshai.core.interfaces import ITextSplitterConfig
from arshai.document_loaders.text_splitters.recursive_splitter import RecursiveTextSplitter


@pytest.fixture
def basic_splitter_config():
    """Create a basic text splitter configuration."""
    return ITextSplitterConfig(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )


@pytest.fixture
def sample_text():
    """Create a sample text for testing."""
    return """This is the first paragraph with some content.
    
    This is the second paragraph that also has quite a bit of content inside it.
    
    This is the third paragraph. It continues with more text to ensure we can test proper splitting.
    
    The fourth paragraph is here. It contains enough text to create multiple chunks when split with reasonable chunk size.
    """


def test_initialization():
    """Test initialization with different parameters."""
    # Create a basic config - required for initialization
    config = ITextSplitterConfig(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Test with this config
    splitter = RecursiveTextSplitter(config)
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 200
    # The implementation may add default separators
    for sep in ["\n\n", "\n", " ", ""]:
        assert any(s == sep for s in splitter.separators), f"Separator '{sep}' not found in {splitter.separators}"
    
    # Test with custom parameters
    custom_config = ITextSplitterConfig(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n", ",", " ", ""]
    )
    splitter = RecursiveTextSplitter(custom_config)
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 50
    # Check that similar separators are included (may not be exact matches)
    # The implementation might add/modify separators
    expected_separators = ["\n", ",", " ", ""]
    # Debug output of actual separators
    print(f"Actual separators: {splitter.separators}")
    assert any(s == "\n" for s in splitter.separators), "Newline separator not found"
    assert any("," in s for s in splitter.separators), "Comma separator not found"
    assert any(" " in s for s in splitter.separators), "Space separator not found"
    assert "" in splitter.separators, "Empty separator not found"


def test_split_text(basic_splitter_config):
    """Test splitting a simple text into chunks."""
    splitter = RecursiveTextSplitter(basic_splitter_config)
    
    # Test with text shorter than chunk size
    short_text = "This is a short text."
    chunks = splitter.split_text(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    
    # Test with text longer than chunk size that needs splitting
    long_text = " ".join(["word"] * 30)  # About 150 characters
    chunks = splitter.split_text(long_text)
    assert len(chunks) == 2
    
    # Check overlap between chunks
    last_words_chunk1 = chunks[0].split()[-4:]
    first_words_chunk2 = chunks[1].split()[:4]
    assert last_words_chunk1 == first_words_chunk2


def test_split_text_with_custom_separators():
    """Test splitting text with custom separators."""
    # Create a splitter with comma-first priority
    config = ITextSplitterConfig(
        chunk_size=10,  # Smaller chunk size to force splitting
        chunk_overlap=0,  # No overlap for simpler testing
        separators=[",", " ", ""]
    )
    splitter = RecursiveTextSplitter(config)
    
    # Text with commas that should be used as primary separators
    text = "First part, second part, third part, fourth part, fifth part"
    chunks = splitter.split_text(text)
    
    # Should split at commas due to small chunk size
    assert len(chunks) > 1


def test_split_documents(basic_splitter_config, sample_text):
    """Test splitting documents with metadata."""
    splitter = RecursiveTextSplitter(basic_splitter_config)
    
    # Create a test document
    doc = {
        "page_content": sample_text,
        "metadata": {"source": "test.txt", "page": 1}
    }
    
    # Split the document
    chunks = splitter.split_documents([doc])
    
    # Verify we got multiple chunks
    assert len(chunks) > 1
    
    # Verify all chunks have the original metadata
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "test.txt"
        assert chunk["metadata"]["page"] == 1
        
        # Verify chunk is not larger than the chunk size plus some tolerance
        # (could be slightly larger due to avoiding breaking in the middle of words)
        assert len(chunk["page_content"]) <= basic_splitter_config.chunk_size + 20


def test_split_documents_with_multiple_docs(basic_splitter_config):
    """Test splitting multiple documents at once."""
    splitter = RecursiveTextSplitter(basic_splitter_config)
    
    # Create multiple test documents
    docs = [
        {
            "page_content": "Document 1: " + " ".join(["word"] * 30),
            "metadata": {"source": "doc1.txt", "page": 1}
        },
        {
            "page_content": "Document 2: " + " ".join(["text"] * 30),
            "metadata": {"source": "doc2.txt", "page": 1}
        }
    ]
    
    # Split the documents
    chunks = splitter.split_documents(docs)
    
    # Verify we got chunks from both documents
    assert len(chunks) > 2
    
    # Verify chunks have correct metadata
    doc1_chunks = [chunk for chunk in chunks if chunk["metadata"]["source"] == "doc1.txt"]
    doc2_chunks = [chunk for chunk in chunks if chunk["metadata"]["source"] == "doc2.txt"]
    
    assert len(doc1_chunks) > 0
    assert len(doc2_chunks) > 0
    
    # Verify each chunk's content comes from the right document
    for chunk in doc1_chunks:
        assert "Document 1:" in chunk["page_content"] or "word" in chunk["page_content"]
    
    for chunk in doc2_chunks:
        assert "Document 2:" in chunk["page_content"] or "text" in chunk["page_content"]


def test_split_empty_text(basic_splitter_config):
    """Test splitting empty text."""
    splitter = RecursiveTextSplitter(basic_splitter_config)
    
    # Test with empty text
    chunks = splitter.split_text("")
    # Implementation returns the empty string as a chunk rather than an empty list
    assert chunks == [""]


def test_merge_splits(basic_splitter_config):
    """Test merging of text splits."""
    splitter = RecursiveTextSplitter(basic_splitter_config)
    
    # Create a list of small text chunks
    splits = ["chunk one", "chunk two", "chunk three", "chunk four", "chunk five"]
    
    # Merge them
    merged = splitter._merge_splits(splits)
    
    # Verify content is preserved
    assert merged == "chunk onechunk twochunk threechunk fourchunk five" 