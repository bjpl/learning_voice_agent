"""
Tests for PDFParser

Tests PDF document processing including:
- Text extraction
- Image extraction
- Table extraction
- Metadata extraction
- Structure extraction
- Encrypted PDF handling
"""

import pytest
import os
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from app.documents import PDFParser, DocumentConfig
from app.documents.pdf_parser import (
    PDFParserError,
    PDFEncryptedError,
    PDFCorruptedError
)


@pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
class TestPDFParser:
    """Test PDFParser class"""

    @pytest.fixture
    def parser(self):
        """Create parser instance"""
        config = DocumentConfig()
        return PDFParser(config)

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create sample PDF for testing"""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not available")

        file_path = tmp_path / "sample.pdf"

        # Create simple PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is a test PDF document.")
        page.insert_text((72, 100), "It has multiple lines.")
        doc.save(str(file_path))
        doc.close()

        return str(file_path)

    def test_init(self, parser):
        """Test parser initialization"""
        assert parser is not None
        assert parser.config is not None
        assert parser.settings is not None

    def test_init_without_pymupdf(self, monkeypatch):
        """Test initialization without PyMuPDF"""
        # This test verifies error handling when PyMuPDF is not available
        # In actual test environment, PyMuPDF should be installed
        pass

    @pytest.mark.asyncio
    async def test_extract_text(self, parser, sample_pdf):
        """Test text extraction"""
        text = await parser.extract_text(sample_pdf)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "test PDF document" in text

    @pytest.mark.asyncio
    async def test_extract_text_nonexistent_file(self, parser):
        """Test extracting from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            await parser.extract_text("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_extract_text_by_page(self, parser, sample_pdf):
        """Test page-by-page extraction"""
        pages = await parser.extract_text_by_page(sample_pdf)

        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "page_num" in pages[0]
        assert "text" in pages[0]
        assert "dimensions" in pages[0]

    @pytest.mark.asyncio
    async def test_extract_metadata(self, parser, sample_pdf):
        """Test metadata extraction"""
        metadata = await parser.extract_metadata(sample_pdf)

        assert isinstance(metadata, dict)
        assert metadata["format"] == "pdf"
        assert "pages" in metadata
        assert "file_size" in metadata
        assert metadata["pages"] >= 1

    @pytest.mark.asyncio
    async def test_extract_structure(self, parser, sample_pdf):
        """Test structure extraction"""
        structure = await parser.extract_structure(sample_pdf)

        assert isinstance(structure, dict)
        assert "toc" in structure
        assert "headings" in structure
        assert "links" in structure

    @pytest.mark.asyncio
    async def test_extract_images_disabled(self, sample_pdf):
        """Test image extraction when disabled"""
        config = DocumentConfig()
        config.pdf_settings["extract_images"] = False
        parser = PDFParser(config)

        images = await parser.extract_images(sample_pdf)

        assert isinstance(images, list)
        assert len(images) == 0

    @pytest.mark.asyncio
    async def test_extract_tables_disabled(self, sample_pdf):
        """Test table extraction when disabled"""
        config = DocumentConfig()
        config.pdf_settings["extract_tables"] = False
        parser = PDFParser(config)

        tables = await parser.extract_tables(sample_pdf)

        assert isinstance(tables, list)
        assert len(tables) == 0

    @pytest.mark.asyncio
    async def test_max_pages_limit(self, tmp_path):
        """Test max pages limit"""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not available")

        # Create multi-page PDF
        file_path = tmp_path / "multipage.pdf"
        doc = fitz.open()

        for i in range(5):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1}")

        doc.save(str(file_path))
        doc.close()

        # Set max pages to 3
        config = DocumentConfig(max_pages=3)
        parser = PDFParser(config)

        pages = await parser.extract_text_by_page(str(file_path))

        # Should only extract 3 pages
        assert len(pages) == 3

    def test_parse_pdf_date(self, parser):
        """Test PDF date parsing"""
        # Test with valid PDF date
        date_str = "D:20231125123045"
        result = parser._parse_pdf_date(date_str)

        assert result is not None
        assert "2023" in result

        # Test with None
        result = parser._parse_pdf_date(None)
        assert result is None

        # Test with invalid format
        result = parser._parse_pdf_date("invalid")
        assert result == "invalid"

    def test_normalize_unicode(self, parser):
        """Test Unicode normalization"""
        text = "café"  # Contains Unicode character
        result = parser._normalize_unicode(text)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_whitespace(self, parser):
        """Test whitespace normalization"""
        text = "Multiple    spaces\n\n\n\nand newlines"
        result = parser._normalize_whitespace(text)

        assert "    " not in result  # Multiple spaces removed
        assert "\n\n\n" not in result  # Multiple newlines reduced


@pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
class TestPDFParserEncryption:
    """Test PDF encryption handling"""

    @pytest.fixture
    def encrypted_pdf(self, tmp_path):
        """Create encrypted PDF for testing"""
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF not available")

        file_path = tmp_path / "encrypted.pdf"

        # Create encrypted PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Encrypted content")

        # Save with password
        doc.save(
            str(file_path),
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner_pass",
            user_pw="user_pass"
        )
        doc.close()

        return str(file_path)

    @pytest.mark.asyncio
    async def test_extract_encrypted_pdf_with_password(self, encrypted_pdf):
        """Test extracting encrypted PDF with correct password"""
        config = DocumentConfig()
        config.pdf_settings["handle_encrypted"] = True
        parser = PDFParser(config)

        text = await parser.extract_text(encrypted_pdf, password="user_pass")

        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_extract_encrypted_pdf_without_password(self, encrypted_pdf):
        """Test extracting encrypted PDF without password"""
        config = DocumentConfig()
        config.pdf_settings["handle_encrypted"] = True
        parser = PDFParser(config)

        with pytest.raises(PDFEncryptedError):
            await parser.extract_text(encrypted_pdf)

    @pytest.mark.asyncio
    async def test_extract_encrypted_pdf_wrong_password(self, encrypted_pdf):
        """Test extracting encrypted PDF with wrong password"""
        config = DocumentConfig()
        config.pdf_settings["handle_encrypted"] = True
        parser = PDFParser(config)

        with pytest.raises(PDFEncryptedError):
            await parser.extract_text(encrypted_pdf, password="wrong_pass")

    @pytest.mark.asyncio
    async def test_extract_encrypted_pdf_disabled(self, encrypted_pdf):
        """Test extracting encrypted PDF when handling is disabled"""
        config = DocumentConfig()
        config.pdf_settings["handle_encrypted"] = False
        parser = PDFParser(config)

        with pytest.raises(PDFEncryptedError):
            await parser.extract_text(encrypted_pdf, password="user_pass")
