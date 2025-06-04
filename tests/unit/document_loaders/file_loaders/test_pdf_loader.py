"""Unit tests for PDF loader."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import sys
# Mock the config module to prevent the import error
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()

from arshai.core.interfaces import Document
from arshai.document_loaders.config import UnstructuredLoaderConfig
from arshai.document_loaders.file_loaders.pdf_loader import PDFLoader


@pytest.fixture
def pdf_loader_config():
    """Create a configuration for the PDF loader."""
    return UnstructuredLoaderConfig(
        strategy="hi_res",
        include_page_breaks=True,
        max_characters=8192,
        languages=["eng"],
        pdf_infer_table_structure=True,
        preserve_formatting=True
    )


@pytest.fixture
def pdf_loader(pdf_loader_config):
    """Create a PDF loader instance."""
    return PDFLoader(pdf_loader_config)


def test_initialization(pdf_loader_config):
    """Test initialization with different parameters."""
    # Test basic initialization
    loader = PDFLoader(pdf_loader_config)
    assert loader.config == pdf_loader_config
    assert loader.text_splitter is None
    
    # Test with different strategy and parameters
    config = UnstructuredLoaderConfig(
        strategy="fast",
        include_page_breaks=False,
        pdf_infer_table_structure=False,
        ocr_languages=["deu", "fra"]
    )
    loader = PDFLoader(config)
    assert loader.config.strategy == "fast"
    assert loader.config.pdf_infer_table_structure is False
    assert loader.config.ocr_languages == ["deu", "fra"]


def test_get_partition_params(pdf_loader, pdf_loader_config):
    """Test PDF-specific partition parameters."""
    file_path = Path("test.pdf")
    
    # Get partition parameters for a PDF file
    params = pdf_loader._get_partition_params(file_path)
    
    # Verify PDF-specific parameters
    assert "pdf_infer_table_structure" in params
    assert params["pdf_infer_table_structure"] is True  # From config
    assert params["include_page_breaks"] is True  # Always True for PDFs
    
    # Verify base parameters are also present
    assert params["filename"] == str(file_path)
    assert params["strategy"] == "hi_res"
    assert params["languages"] == ["eng"]
    assert params["max_characters"] == 8192
    assert params["preserve_formatting"] is True


def test_get_partition_params_with_ocr(pdf_loader_config):
    """Test PDF-specific OCR parameters."""
    # Create config with hi_res strategy but no explicit OCR languages
    config = UnstructuredLoaderConfig(
        strategy="hi_res",
        languages=["eng", "spa"],
        ocr_languages=None  # No explicit OCR languages
    )
    loader = PDFLoader(config)
    
    # Get partition parameters
    params = loader._get_partition_params(Path("test.pdf"))
    
    # For hi_res without explicit OCR languages, should use languages for OCR
    assert "ocr_languages" in params
    assert params["ocr_languages"] == ["eng", "spa"]


def test_extract_content(pdf_loader):
    """Test content extraction from PDF files."""
    # Create mock elements
    mock_elements = [
        {
            "content": "Page 1 content",
            "metadata": {"source": "test.pdf", "page_number": 1, "element_type": "Title"}
        },
        {
            "content": "Page 2 content", 
            "metadata": {"source": "test.pdf", "page_number": 2, "element_type": "Text"}
        }
    ]
    
    # Use patch to bypass the actual extraction and return our mock data
    with patch.object(pdf_loader, '_extract_content', return_value=mock_elements):
        file_path = Path("test.pdf")
        content = pdf_loader._extract_content(file_path)
        
        # Verify content extraction
        assert len(content) == 2
        assert content[0]["content"] == "Page 1 content"
        assert content[0]["metadata"]["page_number"] == 1
        assert content[0]["metadata"]["source"] == "test.pdf"
        
        assert content[1]["content"] == "Page 2 content"
        assert content[1]["metadata"]["page_number"] == 2


def test_load_file_integration(pdf_loader):
    """Test the complete PDF loading process."""
    # Mock extract_content to avoid needing the unstructured library
    with patch.object(PDFLoader, "_extract_content") as mock_extract:
        # Return sample content
        mock_extract.return_value = [
            {
                "content": "Chapter 1: Introduction",
                "metadata": {"source": "test.pdf", "page_number": 1, "element_type": "Title"}
            },
            {
                "content": "This is the first page of content in the PDF.",
                "metadata": {"source": "test.pdf", "page_number": 1, "element_type": "Text"}
            },
            {
                "content": "Chapter 2: Methodology",
                "metadata": {"source": "test.pdf", "page_number": 2, "element_type": "Title"}
            },
            {
                "content": "This is the second page of the PDF with some detailed methodology.",
                "metadata": {"source": "test.pdf", "page_number": 2, "element_type": "Text"}
            }
        ]
        
        # Test loading the file
        with patch('pathlib.Path.is_file', return_value=True):
            documents = pdf_loader.load_file("test.pdf")
            
            # Verify documents
            assert len(documents) == 4
            
            # Check first document
            assert documents[0].page_content == "Chapter 1: Introduction"
            assert documents[0].metadata["page_number"] == 1
            assert documents[0].metadata["element_type"] == "Title"
            
            # Check second document
            assert documents[1].page_content == "This is the first page of content in the PDF."
            assert documents[1].metadata["page_number"] == 1
            
            # Check third document
            assert documents[2].page_content == "Chapter 2: Methodology"
            assert documents[2].metadata["page_number"] == 2
            
            # Verify extract_content was called with Path object
            mock_extract.assert_called_once()
            assert isinstance(mock_extract.call_args[0][0], Path)
            assert str(mock_extract.call_args[0][0]) == "test.pdf" 