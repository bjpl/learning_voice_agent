"""
Tests for DocumentConfig

Tests configuration settings including:
- Default values
- Validation
- Preset configurations
- Format support checking
"""

import pytest
import os
import tempfile
from pathlib import Path

from app.documents.config import (
    DocumentConfig,
    PresetConfigs,
    default_config
)


class TestDocumentConfig:
    """Test DocumentConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = DocumentConfig()

        assert config.max_file_size == 10 * 1024 * 1024  # 10MB
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_pages == 1000
        assert config.processing_timeout == 30
        assert config.parallel_workers == 3

    def test_custom_config(self):
        """Test custom configuration"""
        config = DocumentConfig(
            max_file_size=5 * 1024 * 1024,
            chunk_size=500,
            chunk_overlap=100,
            max_pages=500,
        )

        assert config.max_file_size == 5 * 1024 * 1024
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.max_pages == 500

    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation"""
        # Valid: overlap < chunk_size
        config = DocumentConfig(chunk_size=1000, chunk_overlap=200)
        assert config.chunk_overlap == 200

        # Invalid: overlap >= chunk_size
        with pytest.raises(ValueError):
            DocumentConfig(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            DocumentConfig(chunk_size=100, chunk_overlap=150)

    def test_chunk_size_bounds(self):
        """Test chunk size bounds"""
        # Valid sizes
        config = DocumentConfig(chunk_size=100)
        assert config.chunk_size == 100

        config = DocumentConfig(chunk_size=4000)
        assert config.chunk_size == 4000

        # Invalid sizes
        with pytest.raises(ValueError):
            DocumentConfig(chunk_size=50)  # Too small

        with pytest.raises(ValueError):
            DocumentConfig(chunk_size=5000)  # Too large

    def test_is_format_supported(self):
        """Test format support checking"""
        config = DocumentConfig()

        assert config.is_format_supported("pdf") is True
        assert config.is_format_supported("docx") is True
        assert config.is_format_supported("txt") is True
        assert config.is_format_supported("md") is True

        # With leading dot
        assert config.is_format_supported(".pdf") is True

        # Case insensitive
        assert config.is_format_supported("PDF") is True
        assert config.is_format_supported("DOCX") is True

        # Unsupported
        assert config.is_format_supported("xyz") is False

    def test_get_format_settings(self):
        """Test getting format-specific settings"""
        config = DocumentConfig()

        pdf_settings = config.get_format_settings("pdf")
        assert isinstance(pdf_settings, dict)
        assert "extract_images" in pdf_settings

        docx_settings = config.get_format_settings("docx")
        assert isinstance(docx_settings, dict)
        assert "extract_tables" in docx_settings

        text_settings = config.get_format_settings("txt")
        assert isinstance(text_settings, dict)
        assert "parse_markdown" in text_settings

        md_settings = config.get_format_settings("md")
        assert isinstance(md_settings, dict)

        # Unknown format
        unknown_settings = config.get_format_settings("xyz")
        assert unknown_settings == {}

    def test_validate_file_size(self, tmp_path):
        """Test file size validation"""
        config = DocumentConfig(max_file_size=1000)  # 1000 bytes

        # Create small file (valid)
        small_file = tmp_path / "small.txt"
        small_file.write_text("x" * 500)
        assert config.validate_file_size(str(small_file)) is True

        # Create large file (invalid)
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 2000)
        assert config.validate_file_size(str(large_file)) is False

        # Nonexistent file
        with pytest.raises(FileNotFoundError):
            config.validate_file_size("/nonexistent/file.txt")

    def test_supported_formats(self):
        """Test supported formats set"""
        config = DocumentConfig()

        assert "pdf" in config.supported_formats
        assert "docx" in config.supported_formats
        assert "txt" in config.supported_formats
        assert "md" in config.supported_formats

    def test_pdf_settings(self):
        """Test PDF-specific settings"""
        config = DocumentConfig()

        assert config.pdf_settings["extract_images"] is True
        assert config.pdf_settings["extract_tables"] is True
        assert config.pdf_settings["preserve_layout"] is True
        assert config.pdf_settings["handle_encrypted"] is True

        # Custom settings
        config = DocumentConfig(
            pdf_settings={
                "extract_images": False,
                "extract_tables": False,
                "preserve_layout": False,
                "handle_encrypted": False,
            }
        )

        assert config.pdf_settings["extract_images"] is False

    def test_docx_settings(self):
        """Test DOCX-specific settings"""
        config = DocumentConfig()

        assert config.docx_settings["extract_images"] is True
        assert config.docx_settings["extract_tables"] is True
        assert config.docx_settings["preserve_formatting"] is True
        assert config.docx_settings["extract_comments"] is False

    def test_text_settings(self):
        """Test text/Markdown settings"""
        config = DocumentConfig()

        assert config.text_settings["parse_markdown"] is True
        assert config.text_settings["extract_code_blocks"] is True
        assert config.text_settings["extract_links"] is True
        assert config.text_settings["detect_structure"] is True


class TestPresetConfigs:
    """Test preset configurations"""

    def test_fast_processing_preset(self):
        """Test fast processing preset"""
        config = PresetConfigs.fast_processing()

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.parallel_workers == 5
        assert config.pdf_settings["extract_images"] is False
        assert config.pdf_settings["extract_tables"] is False
        assert config.extract_full_metadata is False

    def test_comprehensive_extraction_preset(self):
        """Test comprehensive extraction preset"""
        config = PresetConfigs.comprehensive_extraction()

        assert config.chunk_size == 1500
        assert config.chunk_overlap == 300
        assert config.max_file_size == 50 * 1024 * 1024
        assert config.pdf_settings["extract_images"] is True
        assert config.pdf_settings["extract_tables"] is True
        assert config.docx_settings["extract_comments"] is True
        assert config.extract_full_metadata is True

    def test_rag_optimized_preset(self):
        """Test RAG-optimized preset"""
        config = PresetConfigs.rag_optimized()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.parallel_workers == 4
        assert config.pdf_settings["extract_images"] is False
        assert config.pdf_settings["extract_tables"] is True
        assert config.normalize_unicode is True
        assert config.remove_page_numbers is True

    def test_preset_configs_are_valid(self):
        """Test that all preset configs are valid"""
        # Should not raise exceptions
        fast = PresetConfigs.fast_processing()
        comprehensive = PresetConfigs.comprehensive_extraction()
        rag = PresetConfigs.rag_optimized()

        # All should have valid chunk_overlap < chunk_size
        assert fast.chunk_overlap < fast.chunk_size
        assert comprehensive.chunk_overlap < comprehensive.chunk_size
        assert rag.chunk_overlap < rag.chunk_size


class TestDefaultConfig:
    """Test default config instance"""

    def test_default_config_instance(self):
        """Test that default_config is available"""
        assert default_config is not None
        assert isinstance(default_config, DocumentConfig)
        assert default_config.chunk_size == 1000


class TestConfigModification:
    """Test configuration modification"""

    def test_modify_config(self):
        """Test modifying configuration"""
        config = DocumentConfig()

        # Modify values
        config.chunk_size = 800
        config.chunk_overlap = 150

        assert config.chunk_size == 800
        assert config.chunk_overlap == 150

    def test_invalid_modification(self):
        """Test invalid configuration modification"""
        config = DocumentConfig()

        # Try to set invalid chunk_size
        with pytest.raises(ValueError):
            config.chunk_size = 50  # Too small
