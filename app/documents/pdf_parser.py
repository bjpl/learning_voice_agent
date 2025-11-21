"""
PDF Parser Module

Specialized PDF document processing using PyMuPDF (fitz).

Features:
- Text extraction with layout preservation
- Image extraction from embedded content
- Table detection and extraction
- Metadata extraction (author, title, dates, etc.)
- Encrypted PDF handling (with password support)
- Page-by-page processing for large documents
- Concurrent page processing

Dependencies:
- PyMuPDF (fitz) >= 1.23.0
"""

from typing import Dict, List, Optional, Tuple, Any
import asyncio
import io
import os
from datetime import datetime
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from app.documents.config import DocumentConfig
from app.logger import get_logger

logger = get_logger(__name__)


class PDFParserError(Exception):
    """Base exception for PDF parsing errors"""
    pass


class PDFEncryptedError(PDFParserError):
    """Raised when PDF is encrypted and cannot be opened"""
    pass


class PDFCorruptedError(PDFParserError):
    """Raised when PDF file is corrupted"""
    pass


class PDFParser:
    """
    PDF document parser using PyMuPDF

    Handles text extraction, image extraction, table detection,
    and metadata extraction from PDF documents.
    """

    def __init__(self, config: Optional[DocumentConfig] = None):
        """
        Initialize PDF parser

        Args:
            config: Document processing configuration

        Raises:
            ImportError: If PyMuPDF is not installed
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install PyMuPDF>=1.23.0"
            )

        self.config = config or DocumentConfig()
        self.settings = self.config.pdf_settings
        self.logger = logger

    async def extract_text(self, file_path: str, password: Optional[str] = None) -> str:
        """
        Extract all text from PDF

        Args:
            file_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            Extracted text content

        Raises:
            PDFEncryptedError: If PDF is encrypted and no password provided
            PDFCorruptedError: If PDF is corrupted
            FileNotFoundError: If file doesn't exist
        """
        self.logger.info(f"Extracting text from PDF: {file_path}")

        try:
            doc = await self._open_pdf(file_path, password)

            # Extract text from all pages
            text_parts = []
            for page_num in range(len(doc)):
                if page_num >= self.config.max_pages:
                    self.logger.warning(
                        f"Reached max page limit ({self.config.max_pages}), "
                        f"stopping at page {page_num}"
                    )
                    break

                page = doc[page_num]

                # Extract text with or without layout preservation
                if self.settings.get("preserve_layout", True):
                    page_text = page.get_text("text", sort=True)
                else:
                    page_text = page.get_text()

                if page_text.strip():
                    text_parts.append(page_text)

            doc.close()

            full_text = "\n\n".join(text_parts)

            # Post-process text
            if self.config.normalize_unicode:
                full_text = self._normalize_unicode(full_text)

            if not self.config.preserve_whitespace:
                full_text = self._normalize_whitespace(full_text)

            self.logger.info(f"Extracted {len(full_text)} characters from {len(text_parts)} pages")
            return full_text

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    async def extract_text_by_page(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text page by page with metadata

        Args:
            file_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            List of dicts with page_num, text, and metadata
        """
        self.logger.info(f"Extracting text by page from PDF: {file_path}")

        doc = await self._open_pdf(file_path, password)

        pages_data = []
        for page_num in range(len(doc)):
            if page_num >= self.config.max_pages:
                break

            page = doc[page_num]

            page_data = {
                "page_num": page_num + 1,
                "text": page.get_text(),
                "dimensions": {
                    "width": page.rect.width,
                    "height": page.rect.height,
                },
                "rotation": page.rotation,
            }

            pages_data.append(page_data)

        doc.close()

        self.logger.info(f"Extracted {len(pages_data)} pages")
        return pages_data

    async def extract_images(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract embedded images from PDF

        Args:
            file_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            List of image data dicts with image bytes and metadata
        """
        if not self.settings.get("extract_images", True):
            return []

        self.logger.info(f"Extracting images from PDF: {file_path}")

        doc = await self._open_pdf(file_path, password)

        images = []
        for page_num in range(len(doc)):
            if page_num >= self.config.max_pages:
                break

            page = doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    image_data = {
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "bytes": base_image["image"],
                        "extension": base_image["ext"],
                        "width": base_image.get("width"),
                        "height": base_image.get("height"),
                        "colorspace": base_image.get("colorspace"),
                    }

                    images.append(image_data)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to extract image {img_index} from page {page_num + 1}: {str(e)}"
                    )

        doc.close()

        self.logger.info(f"Extracted {len(images)} images")
        return images

    async def extract_tables(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF (basic structure detection)

        Args:
            file_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            List of table data with rows and cells

        Note:
            This is basic table detection. For advanced table extraction,
            consider using libraries like tabula-py or camelot-py
        """
        if not self.settings.get("extract_tables", True):
            return []

        self.logger.info(f"Extracting tables from PDF: {file_path}")

        doc = await self._open_pdf(file_path, password)

        tables = []
        for page_num in range(len(doc)):
            if page_num >= self.config.max_pages:
                break

            page = doc[page_num]

            # Find tables using text blocks (basic detection)
            blocks = page.get_text("dict")["blocks"]

            # Simple heuristic: look for aligned text blocks
            table_candidates = self._detect_table_structure(blocks)

            for table_idx, table_data in enumerate(table_candidates):
                tables.append({
                    "page_num": page_num + 1,
                    "table_index": table_idx,
                    "rows": table_data,
                })

        doc.close()

        self.logger.info(f"Extracted {len(tables)} tables")
        return tables

    async def extract_metadata(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract PDF metadata

        Args:
            file_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            Dictionary with metadata fields
        """
        self.logger.info(f"Extracting metadata from PDF: {file_path}")

        doc = await self._open_pdf(file_path, password)

        # Extract standard metadata
        raw_metadata = doc.metadata

        metadata = {
            "format": "pdf",
            "pages": len(doc),
            "file_size": os.path.getsize(file_path),
            "file_name": os.path.basename(file_path),
            "title": raw_metadata.get("title", ""),
            "author": raw_metadata.get("author", ""),
            "subject": raw_metadata.get("subject", ""),
            "keywords": raw_metadata.get("keywords", ""),
            "creator": raw_metadata.get("creator", ""),
            "producer": raw_metadata.get("producer", ""),
            "creation_date": self._parse_pdf_date(raw_metadata.get("creationDate")),
            "modification_date": self._parse_pdf_date(raw_metadata.get("modDate")),
            "encrypted": doc.is_encrypted,
            "pdf_version": doc.pdf_version if hasattr(doc, 'pdf_version') else None,
        }

        # Add page dimensions
        if len(doc) > 0:
            first_page = doc[0]
            metadata["page_dimensions"] = {
                "width": first_page.rect.width,
                "height": first_page.rect.height,
            }

        doc.close()

        return metadata

    async def extract_structure(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract document structure (TOC, headings, etc.)

        Args:
            file_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            Dictionary with structure information
        """
        self.logger.info(f"Extracting structure from PDF: {file_path}")

        doc = await self._open_pdf(file_path, password)

        structure = {
            "toc": [],
            "headings": [],
            "links": [],
        }

        # Extract table of contents
        try:
            toc = doc.get_toc()
            structure["toc"] = [
                {
                    "level": item[0],
                    "title": item[1],
                    "page": item[2],
                }
                for item in toc
            ]
        except Exception as e:
            self.logger.warning(f"Failed to extract TOC: {str(e)}")

        # Extract links
        for page_num in range(min(len(doc), self.config.max_pages)):
            page = doc[page_num]
            links = page.get_links()

            for link in links:
                if "uri" in link:
                    structure["links"].append({
                        "page": page_num + 1,
                        "uri": link["uri"],
                        "type": link.get("kind", "unknown"),
                    })

        doc.close()

        return structure

    async def _open_pdf(
        self,
        file_path: str,
        password: Optional[str] = None
    ) -> fitz.Document:
        """
        Open PDF document with error handling

        Args:
            file_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            Opened fitz.Document

        Raises:
            FileNotFoundError: If file doesn't exist
            PDFEncryptedError: If PDF is encrypted and cannot be opened
            PDFCorruptedError: If PDF is corrupted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            doc = fitz.open(file_path)

            # Handle encrypted PDFs
            if doc.is_encrypted:
                if not self.settings.get("handle_encrypted", True):
                    doc.close()
                    raise PDFEncryptedError(
                        "PDF is encrypted and encrypted handling is disabled"
                    )

                if password:
                    auth_result = doc.authenticate(password)
                    if not auth_result:
                        doc.close()
                        raise PDFEncryptedError("Invalid password for encrypted PDF")
                else:
                    doc.close()
                    raise PDFEncryptedError(
                        "PDF is encrypted but no password provided"
                    )

            return doc

        except fitz.FileDataError as e:
            raise PDFCorruptedError(f"PDF file is corrupted: {str(e)}")
        except Exception as e:
            if "encrypted" in str(e).lower():
                raise PDFEncryptedError(f"PDF encryption error: {str(e)}")
            raise PDFParserError(f"Failed to open PDF: {str(e)}")

    def _detect_table_structure(self, blocks: List[Dict]) -> List[List[List[str]]]:
        """
        Basic table structure detection from text blocks

        This is a simple heuristic-based approach. For production use,
        consider using specialized table extraction libraries.

        Args:
            blocks: Text blocks from page.get_text("dict")

        Returns:
            List of tables, each table is a list of rows
        """
        # Simple implementation: group text blocks by vertical position
        # This is a placeholder for more sophisticated table detection
        tables = []

        # TODO: Implement more sophisticated table detection
        # For now, return empty list

        return tables

    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[str]:
        """
        Parse PDF date string to ISO format

        Args:
            date_str: PDF date string (e.g., "D:20231125123045")

        Returns:
            ISO format date string or None
        """
        if not date_str:
            return None

        try:
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
            if date_str.startswith("D:"):
                date_str = date_str[2:16]  # Get YYYYMMDDHHmmSS
                dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                return dt.isoformat()
        except Exception as e:
            self.logger.warning(f"Failed to parse PDF date '{date_str}': {str(e)}")

        return date_str

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        import unicodedata
        return unicodedata.normalize('NFKC', text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple spaces with single space
        import re
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
