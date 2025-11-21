"""
Tests for TextParser

Tests text and Markdown document processing including:
- Plain text extraction
- Markdown structure parsing
- Heading extraction
- Code block detection
- Link extraction
- List detection
"""

import pytest
import os
from pathlib import Path

from app.documents import TextParser, DocumentConfig
from app.documents.text_parser import TextParserError


class TestTextParser:
    """Test TextParser class"""

    @pytest.fixture
    def parser(self):
        """Create parser instance"""
        config = DocumentConfig()
        return TextParser(config)

    @pytest.fixture
    def sample_text_file(self, tmp_path):
        """Create sample text file"""
        file_path = tmp_path / "sample.txt"
        content = """This is a test document.

It has multiple paragraphs.

And some more content here.
With multiple lines."""
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)

    @pytest.fixture
    def sample_markdown_file(self, tmp_path):
        """Create sample Markdown file"""
        file_path = tmp_path / "sample.md"
        content = """# Main Title

## Section 1

This is some content in section 1.

### Subsection 1.1

More content here.

## Section 2

- Item 1
- Item 2
- Item 3

1. First
2. Second
3. Third

```python
def hello():
    print("world")
```

```javascript
console.log("Hello");
```

[Link text](https://example.com)
[Another link](https://test.com)

> This is a blockquote

Regular paragraph text.
"""
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)

    def test_init(self, parser):
        """Test parser initialization"""
        assert parser is not None
        assert parser.config is not None
        assert parser.settings is not None

    @pytest.mark.asyncio
    async def test_extract_text(self, parser, sample_text_file):
        """Test text extraction"""
        text = await parser.extract_text(sample_text_file)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "test document" in text

    @pytest.mark.asyncio
    async def test_extract_text_nonexistent_file(self, parser):
        """Test extracting from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            await parser.extract_text("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_extract_markdown_structure(self, parser, sample_markdown_file):
        """Test Markdown structure extraction"""
        structure = await parser.extract_markdown_structure(sample_markdown_file)

        assert isinstance(structure, dict)
        assert "headings" in structure
        assert "code_blocks" in structure
        assert "links" in structure
        assert "lists" in structure

        # Verify headings
        assert len(structure["headings"]) >= 3
        assert structure["headings"][0]["level"] == 1
        assert structure["headings"][0]["text"] == "Main Title"

        # Verify code blocks
        assert len(structure["code_blocks"]) >= 2
        assert any(cb["language"] == "python" for cb in structure["code_blocks"])
        assert any(cb["language"] == "javascript" for cb in structure["code_blocks"])

        # Verify links
        assert len(structure["links"]) >= 2

        # Verify lists
        assert len(structure["lists"]) >= 2

    @pytest.mark.asyncio
    async def test_extract_metadata(self, parser, sample_text_file):
        """Test metadata extraction"""
        metadata = await parser.extract_metadata(sample_text_file)

        assert isinstance(metadata, dict)
        assert metadata["format"] == "txt"
        assert "file_size" in metadata
        assert "num_lines" in metadata
        assert "num_words" in metadata
        assert "num_characters" in metadata
        assert metadata["is_markdown"] is False

    @pytest.mark.asyncio
    async def test_extract_metadata_markdown(self, parser, sample_markdown_file):
        """Test metadata extraction for Markdown"""
        metadata = await parser.extract_metadata(sample_markdown_file)

        assert metadata["format"] == "md"
        assert metadata["is_markdown"] is True
        assert "num_headings" in metadata
        assert "num_code_blocks" in metadata
        assert "num_links" in metadata
        assert metadata["num_headings"] >= 3

    @pytest.mark.asyncio
    async def test_extract_structure_text(self, parser, sample_text_file):
        """Test structure extraction for plain text"""
        structure = await parser.extract_structure(sample_text_file)

        assert isinstance(structure, dict)
        assert "paragraphs" in structure or "lines" in structure

    @pytest.mark.asyncio
    async def test_extract_structure_markdown(self, parser, sample_markdown_file):
        """Test structure extraction for Markdown"""
        structure = await parser.extract_structure(sample_markdown_file)

        assert isinstance(structure, dict)
        assert "headings" in structure
        assert len(structure["headings"]) >= 3

    def test_extract_headings(self, parser):
        """Test heading extraction"""
        text = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6

Not a heading
"""
        headings = parser._extract_headings(text)

        assert len(headings) == 6
        assert headings[0]["level"] == 1
        assert headings[0]["text"] == "Level 1"
        assert headings[5]["level"] == 6

    def test_extract_code_blocks(self, parser):
        """Test code block extraction"""
        text = """Some text

```python
def test():
    pass
```

More text

```javascript
console.log("test");
```

```
Plain code block
```
"""
        code_blocks = parser._extract_code_blocks(text)

        assert len(code_blocks) == 3
        assert code_blocks[0]["language"] == "python"
        assert "def test():" in code_blocks[0]["code"]
        assert code_blocks[1]["language"] == "javascript"
        assert code_blocks[2]["language"] == "text"

    def test_extract_links(self, parser):
        """Test link extraction"""
        text = """Some text with [link 1](https://example.com) and
[link 2](https://test.com) and [another](http://foo.bar)."""

        links = parser._extract_links(text)

        assert len(links) == 3
        assert links[0]["text"] == "link 1"
        assert links[0]["url"] == "https://example.com"
        assert links[1]["text"] == "link 2"

    def test_extract_lists(self, parser):
        """Test list extraction"""
        text = """
- Item 1
- Item 2
- Item 3

Some text

1. First
2. Second
3. Third

More text

* Bullet 1
* Bullet 2
"""
        lists = parser._extract_lists(text)

        assert len(lists) >= 2
        assert any(lst["type"] == "unordered" for lst in lists)
        assert any(lst["type"] == "ordered" for lst in lists)

    def test_is_markdown(self, parser):
        """Test Markdown detection"""
        # Should detect as Markdown
        assert parser._is_markdown("# Heading\n\nContent") is True
        assert parser._is_markdown("- List item") is True
        assert parser._is_markdown("1. Ordered item") is True
        assert parser._is_markdown("[Link](url)") is True
        assert parser._is_markdown("```code```") is True

        # Should not detect as Markdown
        assert parser._is_markdown("Plain text without any markdown") is False

    def test_normalize_unicode(self, parser):
        """Test Unicode normalization"""
        text = "café résumé naïve"
        result = parser._normalize_unicode(text)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_whitespace(self, parser):
        """Test whitespace normalization"""
        text = "Multiple    spaces\n\n\n\nand newlines"
        result = parser._normalize_whitespace(text)

        assert "    " not in result
        assert "\n\n\n\n" not in result


class TestTextParserAdvanced:
    """Advanced text parser tests"""

    @pytest.mark.asyncio
    async def test_complex_markdown_document(self, tmp_path):
        """Test parsing complex Markdown document"""
        file_path = tmp_path / "complex.md"
        content = """# Documentation

## Introduction

This is a comprehensive guide.

### Features

- Feature 1
- Feature 2
- Feature 3

### Installation

1. Step 1
2. Step 2
3. Step 3

## Code Examples

```python
# Python example
import sys

def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
```

```bash
# Bash example
echo "Hello"
```

## References

- [Python Docs](https://docs.python.org)
- [GitHub](https://github.com)
- [Stack Overflow](https://stackoverflow.com)

## Conclusion

That's all folks!
"""
        file_path.write_text(content, encoding='utf-8')

        parser = TextParser()
        structure = await parser.extract_markdown_structure(str(file_path))

        # Verify comprehensive extraction
        assert len(structure["headings"]) >= 5
        assert len(structure["code_blocks"]) >= 2
        assert len(structure["links"]) >= 3
        assert len(structure["lists"]) >= 2

    @pytest.mark.asyncio
    async def test_unicode_file_encoding(self, tmp_path):
        """Test handling files with Unicode content"""
        file_path = tmp_path / "unicode.txt"
        content = "Hello 世界 مرحبا Привет"
        file_path.write_text(content, encoding='utf-8')

        parser = TextParser()
        text = await parser.extract_text(str(file_path))

        assert "世界" in text
        assert "مرحبا" in text
        assert "Привет" in text

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path):
        """Test handling empty file"""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding='utf-8')

        parser = TextParser()
        text = await parser.extract_text(str(file_path))

        assert text == ""

    @pytest.mark.asyncio
    async def test_markdown_with_nested_lists(self, tmp_path):
        """Test Markdown with nested lists"""
        file_path = tmp_path / "nested.md"
        content = """
- Top level 1
  - Nested 1.1
  - Nested 1.2
- Top level 2
  - Nested 2.1
"""
        file_path.write_text(content, encoding='utf-8')

        parser = TextParser()
        structure = await parser.extract_markdown_structure(str(file_path))

        # Basic list detection (doesn't fully handle nesting)
        assert "lists" in structure
