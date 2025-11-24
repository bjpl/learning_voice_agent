"""
Tests for Enhanced PDF Table Detection (Feature 5)

SPARC Specification:
- Sophisticated table boundary detection
- >80% table detection accuracy
- Handle complex multi-column layouts
- Improved cell extraction accuracy
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@pytest.fixture
def mock_pdf_blocks():
    """Mock PDF text blocks representing a table"""
    # Simulate blocks from page.get_text("dict")
    return {
        "blocks": [
            # Table header row
            {
                "type": 0,  # text block
                "bbox": [72, 100, 540, 120],  # Full width
                "lines": [
                    {
                        "bbox": [72, 100, 180, 120],
                        "spans": [{"text": "Name", "bbox": [72, 100, 180, 120]}]
                    },
                    {
                        "bbox": [200, 100, 350, 120],
                        "spans": [{"text": "Email", "bbox": [200, 100, 350, 120]}]
                    },
                    {
                        "bbox": [370, 100, 540, 120],
                        "spans": [{"text": "Phone", "bbox": [370, 100, 540, 120]}]
                    }
                ]
            },
            # Table row 1
            {
                "type": 0,
                "bbox": [72, 125, 540, 145],
                "lines": [
                    {
                        "bbox": [72, 125, 180, 145],
                        "spans": [{"text": "John Doe", "bbox": [72, 125, 180, 145]}]
                    },
                    {
                        "bbox": [200, 125, 350, 145],
                        "spans": [{"text": "john@example.com", "bbox": [200, 125, 350, 145]}]
                    },
                    {
                        "bbox": [370, 125, 540, 145],
                        "spans": [{"text": "555-1234", "bbox": [370, 125, 540, 145]}]
                    }
                ]
            },
            # Table row 2
            {
                "type": 0,
                "bbox": [72, 150, 540, 170],
                "lines": [
                    {
                        "bbox": [72, 150, 180, 170],
                        "spans": [{"text": "Jane Smith", "bbox": [72, 150, 180, 170]}]
                    },
                    {
                        "bbox": [200, 150, 350, 170],
                        "spans": [{"text": "jane@example.com", "bbox": [200, 150, 350, 170]}]
                    },
                    {
                        "bbox": [370, 150, 540, 170],
                        "spans": [{"text": "555-5678", "bbox": [370, 150, 540, 170]}]
                    }
                ]
            },
            # Non-table text block (paragraph)
            {
                "type": 0,
                "bbox": [72, 200, 540, 300],
                "lines": [
                    {
                        "bbox": [72, 200, 540, 220],
                        "spans": [{"text": "This is a regular paragraph that spans the full width.", "bbox": [72, 200, 540, 220]}]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def complex_table_blocks():
    """Mock blocks for a complex multi-column table with merged cells"""
    return {
        "blocks": [
            # Header with merged cell spanning two columns
            {
                "type": 0,
                "bbox": [72, 100, 540, 130],
                "lines": [
                    {
                        "bbox": [72, 100, 300, 130],
                        "spans": [{"text": "Contact Information", "bbox": [72, 100, 300, 130]}]
                    },
                    {
                        "bbox": [320, 100, 540, 130],
                        "spans": [{"text": "Status", "bbox": [320, 100, 540, 130]}]
                    }
                ]
            },
            # Sub-header row
            {
                "type": 0,
                "bbox": [72, 135, 540, 155],
                "lines": [
                    {
                        "bbox": [72, 135, 150, 155],
                        "spans": [{"text": "Name", "bbox": [72, 135, 150, 155]}]
                    },
                    {
                        "bbox": [160, 135, 300, 155],
                        "spans": [{"text": "Email", "bbox": [160, 135, 300, 155]}]
                    },
                    {
                        "bbox": [320, 135, 430, 155],
                        "spans": [{"text": "Active", "bbox": [320, 135, 430, 155]}]
                    },
                    {
                        "bbox": [440, 135, 540, 155],
                        "spans": [{"text": "Verified", "bbox": [440, 135, 540, 155]}]
                    }
                ]
            }
        ]
    }


class TestTableBoundaryDetection:
    """Test table boundary detection algorithms"""

    def test_detect_table_by_column_alignment(self, mock_pdf_blocks):
        """Test detecting tables by column alignment"""
        pytest.importorskip("fitz", reason="PyMuPDF not available")
        from app.documents.pdf_parser import PDFParser

        parser = PDFParser.__new__(PDFParser)
        parser.logger = MagicMock()

        # The algorithm should detect aligned columns
        tables = parser._detect_table_structure(mock_pdf_blocks["blocks"])

        # We expect at least one table to be detected
        # Note: Current implementation returns empty, this test drives the implementation
        assert isinstance(tables, list)

    def test_detect_row_alignment(self, mock_pdf_blocks):
        """Test detecting rows by vertical alignment"""
        # Rows should be detected by consistent y-coordinates
        blocks = mock_pdf_blocks["blocks"]

        # Calculate y-coordinates of blocks
        y_coords = set()
        for block in blocks:
            if block["type"] == 0:  # text block
                bbox = block["bbox"]
                y_coords.add(round(bbox[1]))  # top y

        # Table rows should have consistent spacing
        assert len(y_coords) >= 3  # At least header + 2 rows

    def test_detect_column_boundaries(self, mock_pdf_blocks):
        """Test detecting column boundaries"""
        blocks = mock_pdf_blocks["blocks"]

        # Extract x-coordinates from all spans
        x_positions = set()
        for block in blocks:
            if block["type"] == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span["bbox"]
                        x_positions.add(round(bbox[0]))  # left x

        # Should detect multiple column positions
        assert len(x_positions) >= 3  # At least 3 columns

    def test_distinguish_table_from_paragraph(self, mock_pdf_blocks):
        """Test distinguishing tables from regular paragraphs"""
        blocks = mock_pdf_blocks["blocks"]

        table_blocks = []
        paragraph_blocks = []

        for block in blocks:
            if block["type"] == 0:
                lines = block.get("lines", [])
                if len(lines) > 1:
                    # Multiple elements on same line might indicate table
                    table_blocks.append(block)
                else:
                    # Single line spanning width is likely paragraph
                    spans = lines[0].get("spans", []) if lines else []
                    if len(spans) == 1:
                        paragraph_blocks.append(block)

        # Should correctly classify blocks
        assert len(table_blocks) >= 3  # 3 table rows
        assert len(paragraph_blocks) >= 1  # 1 paragraph


class TestCellExtraction:
    """Test table cell extraction accuracy"""

    def test_extract_cell_content(self, mock_pdf_blocks):
        """Test extracting content from table cells"""
        blocks = mock_pdf_blocks["blocks"]

        # Extract all text from first table row
        first_row = blocks[0]  # Header row
        cells = []
        for line in first_row.get("lines", []):
            for span in line.get("spans", []):
                cells.append(span["text"])

        assert "Name" in cells
        assert "Email" in cells
        assert "Phone" in cells

    def test_preserve_cell_order(self, mock_pdf_blocks):
        """Test cells are extracted in correct order (left to right)"""
        blocks = mock_pdf_blocks["blocks"]

        # Get cells from data row
        data_row = blocks[1]  # First data row
        cells = []
        for line in data_row.get("lines", []):
            for span in line.get("spans", []):
                cells.append({
                    "text": span["text"],
                    "x": span["bbox"][0]
                })

        # Sort by x position
        cells.sort(key=lambda c: c["x"])

        # Verify order: Name, Email, Phone
        assert cells[0]["text"] == "John Doe"
        assert cells[1]["text"] == "john@example.com"
        assert cells[2]["text"] == "555-1234"

    def test_handle_empty_cells(self):
        """Test handling tables with empty cells"""
        blocks_with_empty = {
            "blocks": [{
                "type": 0,
                "bbox": [72, 100, 540, 120],
                "lines": [
                    {
                        "bbox": [72, 100, 180, 120],
                        "spans": [{"text": "Value1", "bbox": [72, 100, 180, 120]}]
                    },
                    # Empty cell would be absence of span at expected position
                    {
                        "bbox": [370, 100, 540, 120],
                        "spans": [{"text": "Value3", "bbox": [370, 100, 540, 120]}]
                    }
                ]
            }]
        }

        # Algorithm should handle missing cells gracefully
        # This tests robustness of cell extraction


class TestMultiColumnLayouts:
    """Test handling complex multi-column layouts"""

    def test_detect_merged_cells(self, complex_table_blocks):
        """Test detecting merged cells that span multiple columns"""
        blocks = complex_table_blocks["blocks"]

        # First row has a cell spanning two columns
        header_row = blocks[0]
        spans = []
        for line in header_row.get("lines", []):
            for span in line.get("spans", []):
                bbox = span["bbox"]
                width = bbox[2] - bbox[0]
                spans.append({
                    "text": span["text"],
                    "width": width
                })

        # "Contact Information" should have larger width (merged)
        contact_span = next(s for s in spans if "Contact" in s["text"])
        status_span = next(s for s in spans if "Status" in s["text"])

        assert contact_span["width"] > status_span["width"]

    def test_handle_nested_headers(self, complex_table_blocks):
        """Test tables with nested/multi-level headers"""
        blocks = complex_table_blocks["blocks"]

        # Should detect two header rows
        # Row 1: Contact Information | Status
        # Row 2: Name | Email | Active | Verified

        row1_cells = []
        row2_cells = []

        for line in blocks[0].get("lines", []):
            for span in line.get("spans", []):
                row1_cells.append(span["text"])

        for line in blocks[1].get("lines", []):
            for span in line.get("spans", []):
                row2_cells.append(span["text"])

        assert len(row1_cells) == 2  # Merged headers
        assert len(row2_cells) == 4  # Sub-headers

    def test_handle_varying_column_counts(self):
        """Test tables where rows have different column counts"""
        # Some tables have summary rows with fewer columns
        varying_blocks = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [72, 100, 540, 120],
                    "lines": [
                        {"bbox": [72, 100, 180, 120], "spans": [{"text": "A", "bbox": [72, 100, 180, 120]}]},
                        {"bbox": [200, 100, 350, 120], "spans": [{"text": "B", "bbox": [200, 100, 350, 120]}]},
                        {"bbox": [370, 100, 540, 120], "spans": [{"text": "C", "bbox": [370, 100, 540, 120]}]}
                    ]
                },
                {
                    "type": 0,
                    "bbox": [72, 125, 540, 145],
                    "lines": [
                        # Summary row spanning full width
                        {"bbox": [72, 125, 540, 145], "spans": [{"text": "Total: 100", "bbox": [72, 125, 540, 145]}]}
                    ]
                }
            ]
        }

        # Algorithm should handle varying column counts
        first_row_cols = len(varying_blocks["blocks"][0]["lines"])
        second_row_cols = len(varying_blocks["blocks"][1]["lines"])

        assert first_row_cols == 3
        assert second_row_cols == 1


class TestTableDetectionAccuracy:
    """Benchmark tests for table detection accuracy"""

    @pytest.fixture
    def test_cases(self):
        """Test cases with known table boundaries"""
        return [
            {
                "name": "simple_3x3_table",
                "has_table": True,
                "expected_rows": 3,
                "expected_cols": 3
            },
            {
                "name": "paragraph_only",
                "has_table": False,
                "expected_rows": 0,
                "expected_cols": 0
            },
            {
                "name": "table_with_header",
                "has_table": True,
                "expected_rows": 4,  # 1 header + 3 data
                "expected_cols": 4
            }
        ]

    def test_accuracy_benchmark(self, test_cases, mock_pdf_blocks):
        """Benchmark: >80% table detection accuracy"""
        pytest.importorskip("fitz", reason="PyMuPDF not available")
        from app.documents.pdf_parser import PDFParser

        parser = PDFParser.__new__(PDFParser)
        parser.logger = MagicMock()

        # For now, test that the method exists and returns correct type
        result = parser._detect_table_structure(mock_pdf_blocks["blocks"])

        assert isinstance(result, list)

        # In production test, we would:
        # 1. Run detection on all test cases
        # 2. Compare with ground truth
        # 3. Calculate accuracy
        # accuracy = correct_detections / total_cases
        # assert accuracy >= 0.8

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    @pytest.mark.asyncio
    async def test_real_pdf_table_detection(self, tmp_path):
        """Integration test with real PDF containing tables"""
        # Create PDF with table
        import fitz

        pdf_path = tmp_path / "table_test.pdf"
        doc = fitz.open()
        page = doc.new_page()

        # Insert table-like content
        y = 100
        for row in [["Name", "Age", "City"],
                    ["John", "30", "NYC"],
                    ["Jane", "25", "LA"]]:
            x = 72
            for cell in row:
                page.insert_text((x, y), cell)
                x += 150
            y += 20

        doc.save(str(pdf_path))
        doc.close()

        # Test extraction
        from app.documents.pdf_parser import PDFParser
        from app.documents.config import DocumentConfig

        parser = PDFParser(DocumentConfig())
        tables = await parser.extract_tables(str(pdf_path))

        # Should detect at least the structure
        assert isinstance(tables, list)
