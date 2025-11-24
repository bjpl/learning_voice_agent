"""
Text Parser Module

Parser for plain text and Markdown documents.

Features:
- Plain text extraction
- Markdown structure parsing
- Code block detection and extraction
- Link extraction
- Heading hierarchy extraction
- List detection
- Basic text chunking

Dependencies:
- markdown >= 3.5.0 (for Markdown parsing)
"""

from typing import Dict, List, Optional, Any
import os
import re
from pathlib import Path

try:
    import markdown
    from markdown.extensions import Extension
    from markdown.treeprocessors import Treeprocessor
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

from app.documents.config import DocumentConfig
from app.logger import get_logger

logger = get_logger(__name__)


class TextParserError(Exception):
    """Base exception for text parsing errors"""
    pass


class TextParser:
    """
    Plain text and Markdown parser

    Handles text extraction, structure detection, and content analysis
    for both plain text and Markdown documents.
    """

    def __init__(self, config: Optional[DocumentConfig] = None):
        """
        Initialize text parser

        Args:
            config: Document processing configuration
        """
        self.config = config or DocumentConfig()
        self.settings = self.config.text_settings
        self.logger = logger

    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from file

        Args:
            file_path: Path to text file

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is invalid
        """
        self.logger.info(f"Extracting text from file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                self.logger.warning(f"File {file_path} read with latin-1 encoding")
            except Exception as e:
                raise TextParserError(f"Failed to read file: {str(e)}")

        # Post-process
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)

        if not self.config.preserve_whitespace:
            text = self._normalize_whitespace(text)

        self.logger.info(f"Extracted {len(text)} characters")
        return text

    async def extract_markdown_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Extract structure from Markdown file

        Args:
            file_path: Path to Markdown file

        Returns:
            Dictionary with headings, code blocks, links, etc.
        """
        if not self.settings.get("parse_markdown", True):
            return {}

        self.logger.info(f"Extracting Markdown structure from: {file_path}")

        text = await self.extract_text(file_path)

        structure = {
            "headings": self._extract_headings(text),
            "code_blocks": [],
            "links": [],
            "lists": [],
        }

        if self.settings.get("extract_code_blocks", True):
            structure["code_blocks"] = self._extract_code_blocks(text)

        if self.settings.get("extract_links", True):
            structure["links"] = self._extract_links(text)

        if self.settings.get("detect_structure", True):
            structure["lists"] = self._extract_lists(text)

        return structure

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from text file

        Args:
            file_path: Path to text file

        Returns:
            Dictionary with metadata
        """
        self.logger.info(f"Extracting metadata from: {file_path}")

        extension = os.path.splitext(file_path)[1].lower().lstrip('.')

        metadata = {
            "format": extension or "txt",
            "file_size": os.path.getsize(file_path),
            "file_name": os.path.basename(file_path),
            "encoding": "utf-8",  # Assumed
        }

        # Count lines and words
        text = await self.extract_text(file_path)
        lines = text.split('\n')
        words = text.split()

        metadata["num_lines"] = len(lines)
        metadata["num_words"] = len(words)
        metadata["num_characters"] = len(text)

        # Detect if it's Markdown
        if extension == "md" or self._is_markdown(text):
            metadata["is_markdown"] = True
            structure = await self.extract_markdown_structure(file_path)
            metadata["num_headings"] = len(structure.get("headings", []))
            metadata["num_code_blocks"] = len(structure.get("code_blocks", []))
            metadata["num_links"] = len(structure.get("links", []))
        else:
            metadata["is_markdown"] = False

        return metadata

    async def extract_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Extract document structure

        Args:
            file_path: Path to file

        Returns:
            Structure dictionary
        """
        extension = os.path.splitext(file_path)[1].lower().lstrip('.')

        if extension == "md":
            return await self.extract_markdown_structure(file_path)
        else:
            # Basic structure for plain text
            text = await self.extract_text(file_path)
            return {
                "paragraphs": [p for p in text.split('\n\n') if p.strip()],
                "lines": text.split('\n'),
            }

    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Markdown headings

        Args:
            text: Markdown text

        Returns:
            List of heading dicts with level and text
        """
        headings = []

        # Match # Heading syntax
        pattern = r'^(#{1,6})\s+(.+)$'

        for match in re.finditer(pattern, text, re.MULTILINE):
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            headings.append({
                "level": level,
                "text": heading_text,
            })

        return headings

    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Markdown code blocks

        Args:
            text: Markdown text

        Returns:
            List of code block dicts with language and code
        """
        code_blocks = []

        # Match ```language\ncode\n``` syntax
        pattern = r'```(\w*)\n(.*?)```'

        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()

            code_blocks.append({
                "language": language,
                "code": code,
                "length": len(code),
            })

        return code_blocks

    def _extract_links(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Markdown links

        Args:
            text: Markdown text

        Returns:
            List of link dicts with text and URL
        """
        links = []

        # Match [text](url) syntax
        pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

        for match in re.finditer(pattern, text):
            link_text = match.group(1)
            url = match.group(2)

            links.append({
                "text": link_text,
                "url": url,
            })

        return links

    def _extract_lists(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Markdown lists

        Args:
            text: Markdown text

        Returns:
            List of list dicts (ordered/unordered)
        """
        lists = []

        # Simple list detection
        # Ordered: 1. Item
        # Unordered: - Item or * Item

        lines = text.split('\n')
        current_list = None

        for line in lines:
            # Check for ordered list
            ordered_match = re.match(r'^\s*(\d+)\.\s+(.+)$', line)
            if ordered_match:
                if current_list and current_list["type"] == "ordered":
                    current_list["items"].append(ordered_match.group(2))
                else:
                    if current_list:
                        lists.append(current_list)
                    current_list = {
                        "type": "ordered",
                        "items": [ordered_match.group(2)],
                    }
                continue

            # Check for unordered list
            unordered_match = re.match(r'^\s*[-*+]\s+(.+)$', line)
            if unordered_match:
                if current_list and current_list["type"] == "unordered":
                    current_list["items"].append(unordered_match.group(1))
                else:
                    if current_list:
                        lists.append(current_list)
                    current_list = {
                        "type": "unordered",
                        "items": [unordered_match.group(1)],
                    }
                continue

            # Empty line ends current list
            if not line.strip() and current_list:
                lists.append(current_list)
                current_list = None

        # Add last list if exists
        if current_list:
            lists.append(current_list)

        return lists

    def _is_markdown(self, text: str) -> bool:
        """
        Detect if text is Markdown

        Args:
            text: Text to check

        Returns:
            True if text appears to be Markdown
        """
        # Simple heuristic: check for Markdown syntax
        markdown_indicators = [
            r'^#{1,6}\s',  # Headings
            r'^\s*[-*+]\s',  # Unordered lists
            r'^\s*\d+\.\s',  # Ordered lists
            r'\[.+\]\(.+\)',  # Links
            r'```',  # Code blocks
            r'^\s*>',  # Blockquotes
        ]

        for pattern in markdown_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True

        return False

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        import unicodedata
        return unicodedata.normalize('NFKC', text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        import re
        # Don't be too aggressive with text files
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n\n+', '\n\n', text)
        return text.strip()
