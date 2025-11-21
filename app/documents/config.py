"""
Document Processing Configuration

Configuration settings for document processing, including:
- File size limits
- Chunk sizes for RAG
- Supported formats
- Processing options
- Security settings
"""

from typing import Dict, List, Set
from pydantic import BaseModel, Field, field_validator
import os


class DocumentConfig(BaseModel):
    """Configuration for document processing"""

    # File size limits (in bytes)
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file size in bytes"
    )

    # Chunking configuration for RAG
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Target chunk size in tokens"
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks in tokens"
    )

    # Processing limits
    max_pages: int = Field(
        default=1000,
        description="Maximum number of pages to process"
    )

    processing_timeout: int = Field(
        default=30,
        description="Processing timeout in seconds per document"
    )

    parallel_workers: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of parallel workers for page processing"
    )

    # Supported formats
    supported_formats: Set[str] = Field(
        default={"pdf", "docx", "txt", "md"},
        description="Supported document formats"
    )

    # Format-specific settings
    pdf_settings: Dict[str, bool] = Field(
        default={
            "extract_images": True,
            "extract_tables": True,
            "preserve_layout": True,
            "handle_encrypted": True,
        },
        description="PDF-specific extraction settings"
    )

    docx_settings: Dict[str, bool] = Field(
        default={
            "extract_images": True,
            "extract_tables": True,
            "preserve_formatting": True,
            "extract_comments": False,
        },
        description="DOCX-specific extraction settings"
    )

    text_settings: Dict[str, bool] = Field(
        default={
            "parse_markdown": True,
            "extract_code_blocks": True,
            "extract_links": True,
            "detect_structure": True,
        },
        description="Text/Markdown extraction settings"
    )

    # Security settings
    allow_password_pdfs: bool = Field(
        default=False,
        description="Allow processing of password-protected PDFs"
    )

    max_password_attempts: int = Field(
        default=3,
        description="Maximum password attempts for encrypted files"
    )

    # Output settings
    preserve_whitespace: bool = Field(
        default=False,
        description="Preserve exact whitespace in extracted text"
    )

    normalize_unicode: bool = Field(
        default=True,
        description="Normalize Unicode characters"
    )

    remove_page_numbers: bool = Field(
        default=False,
        description="Attempt to remove page numbers from text"
    )

    # Metadata settings
    extract_full_metadata: bool = Field(
        default=True,
        description="Extract all available metadata"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size"""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    def is_format_supported(self, format: str) -> bool:
        """Check if a file format is supported"""
        return format.lower().lstrip('.') in self.supported_formats

    def get_format_settings(self, format: str) -> Dict[str, bool]:
        """Get settings for a specific format"""
        format = format.lower().lstrip('.')
        if format == "pdf":
            return self.pdf_settings
        elif format == "docx":
            return self.docx_settings
        elif format in ("txt", "md"):
            return self.text_settings
        return {}

    def validate_file_size(self, file_path: str) -> bool:
        """Validate file size is within limits"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        return file_size <= self.max_file_size

    class Config:
        """Pydantic config"""
        validate_assignment = True
        frozen = False


# Default configuration instance
default_config = DocumentConfig()


# Preset configurations
class PresetConfigs:
    """Preset configurations for common use cases"""

    @staticmethod
    def fast_processing() -> DocumentConfig:
        """Fast processing with minimal extraction"""
        return DocumentConfig(
            chunk_size=500,
            chunk_overlap=50,
            parallel_workers=5,
            pdf_settings={
                "extract_images": False,
                "extract_tables": False,
                "preserve_layout": False,
                "handle_encrypted": False,
            },
            docx_settings={
                "extract_images": False,
                "extract_tables": False,
                "preserve_formatting": False,
                "extract_comments": False,
            },
            extract_full_metadata=False,
        )

    @staticmethod
    def comprehensive_extraction() -> DocumentConfig:
        """Comprehensive extraction with all features"""
        return DocumentConfig(
            chunk_size=1500,
            chunk_overlap=300,
            parallel_workers=3,
            max_file_size=50 * 1024 * 1024,  # 50MB
            pdf_settings={
                "extract_images": True,
                "extract_tables": True,
                "preserve_layout": True,
                "handle_encrypted": True,
            },
            docx_settings={
                "extract_images": True,
                "extract_tables": True,
                "preserve_formatting": True,
                "extract_comments": True,
            },
            text_settings={
                "parse_markdown": True,
                "extract_code_blocks": True,
                "extract_links": True,
                "detect_structure": True,
            },
            extract_full_metadata=True,
        )

    @staticmethod
    def rag_optimized() -> DocumentConfig:
        """Optimized for RAG systems"""
        return DocumentConfig(
            chunk_size=1000,
            chunk_overlap=200,
            parallel_workers=4,
            pdf_settings={
                "extract_images": False,
                "extract_tables": True,
                "preserve_layout": False,
                "handle_encrypted": False,
            },
            docx_settings={
                "extract_images": False,
                "extract_tables": True,
                "preserve_formatting": False,
                "extract_comments": False,
            },
            preserve_whitespace=False,
            normalize_unicode=True,
            remove_page_numbers=True,
        )
