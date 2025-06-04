"""Unit tests for the base file loader."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import List, Dict, Any, Union

from arshai.core.interfaces import IFileLoaderConfig
from arshai.core.interfaces import Document
from arshai.core.interfaces import ITextSplitter, ITextSplitterConfig
from arshai.document_loaders.file_loaders.base_loader import BaseFileLoader
from arshai.document_loaders.text_splitters.recursive_splitter import RecursiveCharacterTextSplitter


# Create a concrete implementation of BaseFileLoader for testing
class TestFileLoader(BaseFileLoader):
    """Test implementation of BaseFileLoader."""
    
    def _extract_content(self, file_path: Path) -> List[Dict[str, Any]]:
        """Mock implementation of the abstract method."""
        # The content returned depends on the file extension for testing
        if file_path.suffix == '.txt':
            return [
                {
                    'content': f"Content from {file_path.name}",
                    'metadata': {'source': str(file_path), 'type': 'text'}
                }
            ]
        elif file_path.suffix == '.pdf':
            # Return multiple content chunks for PDFs
            return [
                {
                    'content': f"Page 1 from {file_path.name}",
                    'metadata': {'source': str(file_path), 'page': 1, 'type': 'pdf'}
                },
                {
                    'content': f"Page 2 from {file_path.name}",
                    'metadata': {'source': str(file_path), 'page': 2, 'type': 'pdf'}
                }
            ]
        elif file_path.suffix == '.empty':
            # Return empty content for testing empty content handling
            return [
                {
                    'content': '',
                    'metadata': {'source': str(file_path), 'type': 'empty'}
                }
            ]
        else:
            # Return an error message for unknown types
            return [
                {
                    'content': f"Unknown file type: {file_path.suffix}",
                    'metadata': {'source': str(file_path), 'type': 'unknown'}
                }
            ]


@pytest.fixture
def loader_config():
    """Create a basic file loader configuration."""
    return IFileLoaderConfig()


@pytest.fixture
def mock_text_splitter():
    """Create a mock text splitter."""
    splitter = MagicMock(spec=ITextSplitter)
    
    # Mock the split_documents method to return chunked documents
    def split_documents(documents):
        # For each document, split content into two chunks
        result = []
        for doc in documents:
            content = doc["page_content"]
            metadata = doc["metadata"]
            
            # Skip empty content
            if not content:
                continue
                
            # Split into two chunks for testing
            middle = len(content) // 2
            chunk1 = content[:middle]
            chunk2 = content[middle:]
            
            # Create metadata for each chunk
            meta1 = metadata.copy()
            meta1["chunk"] = 1
            meta2 = metadata.copy()
            meta2["chunk"] = 2
            
            result.append({"page_content": chunk1, "metadata": meta1})
            result.append({"page_content": chunk2, "metadata": meta2})
            
        return result
    
    splitter.split_documents.side_effect = split_documents
    return splitter


@pytest.fixture
def file_loader(loader_config, mock_text_splitter):
    """Create a test file loader instance."""
    return TestFileLoader(loader_config, mock_text_splitter)


def test_initialization():
    """Test initialization with different parameters."""
    # Test with minimal configuration
    config = IFileLoaderConfig()
    loader = TestFileLoader(config)
    assert loader.config == config
    assert loader.text_splitter is None
    
    # Test with text splitter
    splitter_config = ITextSplitterConfig(chunk_size=100, chunk_overlap=10)
    splitter = RecursiveCharacterTextSplitter(splitter_config)
    loader = TestFileLoader(config, splitter)
    assert loader.text_splitter == splitter


def test_load_file_txt(file_loader):
    """Test loading a text file."""
    with patch('pathlib.Path.is_file', return_value=True):
        documents = file_loader.load_file("test.txt")
        
        # Since we have a text splitter, the single document should be split into two
        assert len(documents) == 2
        
        # Check the content of the chunks
        assert documents[0].page_content == "Content"
        assert documents[1].page_content == " from test.txt"
        
        # Check metadata
        assert documents[0].metadata["source"] == "test.txt"
        assert documents[0].metadata["type"] == "text"
        assert documents[0].metadata["chunk"] == 1
        assert documents[1].metadata["chunk"] == 2


def test_load_file_pdf(file_loader):
    """Test loading a PDF file with multiple pages."""
    with patch('pathlib.Path.is_file', return_value=True):
        documents = file_loader.load_file("sample.pdf")
        
        # Each page becomes a document, and each document is split into two chunks
        assert len(documents) == 4
        
        # Check the first page chunks
        assert documents[0].page_content == "Page 1 fr"
        assert documents[1].page_content == "om sample.pdf"
        assert documents[0].metadata["page"] == 1
        assert documents[1].metadata["page"] == 1
        
        # Check the second page chunks
        assert documents[2].page_content == "Page 2 fr"
        assert documents[3].page_content == "om sample.pdf"
        assert documents[2].metadata["page"] == 2
        assert documents[3].metadata["page"] == 2


def test_load_file_empty(file_loader):
    """Test loading a file with empty content."""
    with patch('pathlib.Path.is_file', return_value=True):
        documents = file_loader.load_file("empty.empty")
        
        # Empty content should be skipped
        assert len(documents) == 0


def test_load_file_without_splitter(loader_config):
    """Test loading a file without a text splitter."""
    # Create loader without text splitter
    loader = TestFileLoader(loader_config)
    
    with patch('pathlib.Path.is_file', return_value=True):
        documents = loader.load_file("test.txt")
        
        # Without a splitter, we should get original documents
        assert len(documents) == 1
        assert documents[0].page_content == "Content from test.txt"
        assert documents[0].metadata["source"] == "test.txt"
        assert documents[0].metadata["type"] == "text"


def test_load_file_error_handling(file_loader):
    """Test error handling when loading a file."""
    # Mock _extract_content to raise an exception
    with patch.object(TestFileLoader, '_extract_content', side_effect=Exception("Test error")):
        with patch('pathlib.Path.is_file', return_value=True):
            documents = file_loader.load_file("error.txt")
            
            # On error, should return empty list
            assert documents == []


def test_load_files(file_loader):
    """Test loading multiple files."""
    with patch('pathlib.Path.is_file', return_value=True):
        documents = file_loader.load_files(["test1.txt", "test2.txt"])
        
        # Each file produces 2 chunks with the splitter
        assert len(documents) == 4
        
        # First file chunks
        assert documents[0].metadata["source"] == "test1.txt"
        assert documents[1].metadata["source"] == "test1.txt"
        
        # Second file chunks
        assert documents[2].metadata["source"] == "test2.txt"
        assert documents[3].metadata["source"] == "test2.txt"


def test_load_files_with_separator(file_loader):
    """Test loading files with custom separator."""
    # Mock load_file to return a document with content containing a separator
    def mock_load_file(file_path):
        return [Document(
            page_content="Part 1///Part 2///Part 3",
            metadata={"source": str(file_path)}
        )]
    
    with patch.object(file_loader, 'load_file', side_effect=mock_load_file):
        documents = file_loader.load_files(["test.txt"], separator="///")
        
        # Should split into 3 parts by the separator
        assert len(documents) == 3
        assert documents[0].page_content == "Part 1"
        assert documents[1].page_content == "Part 2"
        assert documents[2].page_content == "Part 3"
        
        # Check metadata includes chunk information
        assert all(doc.metadata["source"] == "test.txt" for doc in documents)
        assert documents[0].metadata["chunk"] == 0
        assert documents[1].metadata["chunk"] == 1
        assert documents[2].metadata["chunk"] == 2


def test_get_file_type(file_loader):
    """Test file type detection."""
    # Test common file types
    assert file_loader.get_file_type(Path("document.pdf")) == "pdf_document"
    assert file_loader.get_file_type(Path("document.docx")) == "word_document"
    assert file_loader.get_file_type(Path("spreadsheet.xlsx")) == "spreadsheet"
    assert file_loader.get_file_type(Path("presentation.pptx")) == "presentation"
    assert file_loader.get_file_type(Path("webpage.html")) == "web_document"
    assert file_loader.get_file_type(Path("audio.mp3")) == "audio"
    assert file_loader.get_file_type(Path("video.mp4")) == "video"
    
    # Test uppercase extension
    assert file_loader.get_file_type(Path("document.PDF")) == "pdf_document"
    
    # Test unknown extension
    assert file_loader.get_file_type(Path("unknown.xyz")) == "unknown" 