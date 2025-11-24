# Sample Markdown Document

This is a comprehensive Markdown document for testing the document processing pipeline.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Code Examples](#code-examples)
- [References](#references)

## Introduction

Markdown is a lightweight markup language that's easy to read and write. This document demonstrates various Markdown features that the parser should be able to extract.

### Why Markdown?

Markdown is popular because it:

- Is human-readable
- Converts easily to HTML
- Supports code blocks
- Has wide tool support

## Features

### Text Formatting

You can use **bold**, *italic*, and ***bold italic*** text. You can also use ~~strikethrough~~ text.

### Lists

Unordered lists:
- Item 1
- Item 2
  - Nested item 2.1
  - Nested item 2.2
- Item 3

Ordered lists:
1. First item
2. Second item
3. Third item
   1. Nested 3.1
   2. Nested 3.2

### Links

Here are some useful links:
- [Python Documentation](https://docs.python.org)
- [Markdown Guide](https://www.markdownguide.org)
- [GitHub](https://github.com)

## Code Examples

### Python Example

```python
def process_document(file_path):
    """Process a document and extract content"""
    with open(file_path, 'r') as f:
        content = f.read()
    return analyze(content)

# Example usage
result = process_document('sample.md')
print(f"Processed {result['word_count']} words")
```

### JavaScript Example

```javascript
async function fetchData(url) {
  try {
    const response = await fetch(url);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Bash Script

```bash
#!/bin/bash

# Process all markdown files
for file in *.md; do
  echo "Processing $file"
  python process.py "$file"
done
```

## Tables

| Feature | PDF | DOCX | TXT | MD |
|---------|-----|------|-----|----|
| Text Extraction | ✓ | ✓ | ✓ | ✓ |
| Images | ✓ | ✓ | ✗ | ✗ |
| Tables | ✓ | ✓ | ✗ | ✓ |
| Metadata | ✓ | ✓ | ✓ | ✓ |

## Blockquotes

> This is a blockquote. It can span multiple lines
> and is often used for citations or emphasis.

> Nested quotes are also possible:
>> This is nested inside the first quote

## Horizontal Rules

---

## Images

Images can be embedded (though not displayed in this text):

![Sample Image](https://example.com/image.png)

## Emphasis and Strong

You can use *emphasis* or _emphasis_ and **strong** or __strong__.

## References

For more information, see:

1. [Markdown Specification](https://spec.commonmark.org/)
2. [GitHub Flavored Markdown](https://github.github.com/gfm/)
3. [Markdown Cheatsheet](https://www.markdownguide.org/cheat-sheet/)

## Conclusion

This document demonstrates various Markdown features including:
- Multiple heading levels
- Lists (ordered and unordered)
- Code blocks with syntax highlighting
- Links and references
- Tables
- Blockquotes
- Text formatting

The document processor should be able to extract all of these elements and their structure.
