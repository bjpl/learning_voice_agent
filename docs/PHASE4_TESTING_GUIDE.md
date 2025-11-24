# Phase 4 Testing Guide: Multi-Modal Capabilities

**Version:** 1.0.0
**Date:** 2025-01-21
**Status:** Test Specification

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Test Environment Setup](#test-environment-setup)
3. [Vision Analysis Testing](#vision-analysis-testing)
4. [Document Processing Testing](#document-processing-testing)
5. [File Upload Testing](#file-upload-testing)
6. [Integration Testing](#integration-testing)
7. [Performance Testing](#performance-testing)
8. [Coverage Goals](#coverage-goals)

---

## Testing Strategy

### Test Pyramid

```
                    ┌─────────────┐
                    │   E2E (5%)  │
                    │   25 tests  │
                    └─────────────┘
                ┌──────────────────────┐
                │ Integration (20%)    │
                │    30 tests          │
                └──────────────────────┘
            ┌─────────────────────────────┐
            │    Unit Tests (75%)         │
            │      120 tests              │
            └─────────────────────────────┘
```

### Coverage Targets

- **Vision Components**: 85%+
- **Document Processing**: 85%+
- **Storage Components**: 80%+
- **API Endpoints**: 85%+
- **Integration Tests**: 75%+
- **Overall Phase 4**: 80%+

### Test Categories

1. **Unit Tests** (120 tests)
   - VisionAnalyzer (25 tests)
   - ImageProcessor (20 tests)
   - DocumentProcessor (30 tests)
   - FileManager (25 tests)
   - MultiModalIndexer (20 tests)

2. **Integration Tests** (30 tests)
   - Upload Endpoints (20 tests)
   - Multi-Modal RAG (10 tests)

3. **End-to-End Tests** (25 tests)
   - Full upload → analysis → RAG flow
   - Multi-modal conversation scenarios
   - Performance benchmarks

---

## Test Environment Setup

### Dependencies

```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install responses aioresponses
pip install pillow PyPDF2 python-docx
```

### Fixture Files

Create test fixtures directory:

```bash
mkdir -p tests/fixtures/images
mkdir -p tests/fixtures/documents
mkdir -p tests/fixtures/expected
```

**Test Images:**
- `sample_diagram.png` (1024x768, simple diagram)
- `large_image.jpg` (4000x3000, for resize testing)
- `screenshot.png` (with text for OCR)
- `corrupted.png` (invalid image)

**Test Documents:**
- `sample.pdf` (5 pages, simple text)
- `sample.docx` (Word document with formatting)
- `sample.txt` (plain text)
- `sample.md` (Markdown with headers)
- `encrypted.pdf` (password-protected, for error testing)

### Mock Setup

#### Mock Claude Vision API

```python
# tests/vision/conftest.py

import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for vision analysis"""
    mock = AsyncMock()

    # Mock successful response
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="A diagram showing system architecture with API Gateway, services, and database.")]
    mock_message.usage = MagicMock(input_tokens=1000, output_tokens=150)

    mock.messages.create = AsyncMock(return_value=mock_message)

    return mock

@pytest.fixture
def vision_analyzer(mock_anthropic_client):
    """VisionAnalyzer with mocked API"""
    from app.vision.vision_analyzer import VisionAnalyzer

    analyzer = VisionAnalyzer()
    analyzer.client = mock_anthropic_client
    return analyzer
```

#### Mock File Uploads

```python
# tests/api/conftest.py

import pytest
from io import BytesIO
from PIL import Image

@pytest.fixture
def sample_image_upload():
    """Generate sample image upload"""
    img = Image.new('RGB', (800, 600), color='white')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

@pytest.fixture
def sample_pdf_upload():
    """Generate sample PDF upload"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Sample PDF Document")
    c.drawString(100, 730, "This is test content.")
    c.save()
    buffer.seek(0)
    return buffer
```

---

## Vision Analysis Testing

### Unit Tests: VisionAnalyzer

**File:** `tests/vision/test_vision_analyzer.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import tempfile
import os

@pytest.mark.asyncio
async def test_analyze_image_success(vision_analyzer, tmp_path):
    """Test successful image analysis"""
    # Arrange
    img_path = tmp_path / "test.png"
    img = Image.new('RGB', (800, 600), color='blue')
    img.save(img_path)

    # Act
    result = await vision_analyzer.analyze_image(
        str(img_path),
        prompt="What is in this image?"
    )

    # Assert
    assert "analysis" in result
    assert result["analysis"] == "A diagram showing system architecture with API Gateway, services, and database."
    assert result["dimensions"] == (800, 600)
    assert result["format"] == "PNG"
    assert result["tokens_used"] == 1150

@pytest.mark.asyncio
async def test_analyze_image_with_ocr(vision_analyzer, tmp_path):
    """Test image analysis with OCR extraction"""
    # Arrange
    img_path = tmp_path / "screenshot.png"
    img = Image.new('RGB', (800, 600), color='white')
    img.save(img_path)

    # Mock OCR response
    vision_analyzer.client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(text="Screenshot showing: Text: Login, Username, Password, Submit")],
            usage=MagicMock(input_tokens=1100, output_tokens=200)
        )
    )

    # Act
    result = await vision_analyzer.analyze_image(
        str(img_path),
        include_ocr=True
    )

    # Assert
    assert result["ocr_text"] is not None
    assert "Login" in result["ocr_text"] or "text:" in result["analysis"].lower()

@pytest.mark.asyncio
async def test_analyze_image_file_not_found(vision_analyzer):
    """Test error when image file doesn't exist"""
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        await vision_analyzer.analyze_image("/nonexistent/image.png")

@pytest.mark.asyncio
async def test_analyze_image_invalid_format(vision_analyzer, tmp_path):
    """Test error with invalid image format"""
    # Arrange
    invalid_file = tmp_path / "not_an_image.txt"
    invalid_file.write_text("This is not an image")

    # Act & Assert
    with pytest.raises(Exception):  # PIL will raise exception
        await vision_analyzer.analyze_image(str(invalid_file))

@pytest.mark.asyncio
async def test_analyze_diagram(vision_analyzer, tmp_path):
    """Test specialized diagram analysis"""
    # Arrange
    img_path = tmp_path / "flowchart.png"
    img = Image.new('RGB', (1024, 768), color='white')
    img.save(img_path)

    vision_analyzer.client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(text="Flowchart showing: 1. Type: Process Flow 2. Components: Start, Process, Decision, End")],
            usage=MagicMock(input_tokens=1200, output_tokens=250)
        )
    )

    # Act
    result = await vision_analyzer.analyze_diagram(str(img_path))

    # Assert
    assert "analysis" in result
    assert "flowchart" in result["analysis"].lower() or "diagram" in result["analysis"].lower()

@pytest.mark.asyncio
async def test_compare_images(vision_analyzer, tmp_path):
    """Test image comparison"""
    # Arrange
    img1_path = tmp_path / "before.png"
    img2_path = tmp_path / "after.png"

    img1 = Image.new('RGB', (800, 600), color='red')
    img1.save(img1_path)

    img2 = Image.new('RGB', (800, 600), color='blue')
    img2.save(img2_path)

    # Act
    result = await vision_analyzer.compare_images(
        str(img1_path),
        str(img2_path)
    )

    # Assert
    assert "image1_analysis" in result
    assert "image2_analysis" in result
    assert "comparison" in result

@pytest.mark.asyncio
async def test_vision_api_timeout(vision_analyzer, tmp_path):
    """Test handling of API timeout"""
    # Arrange
    img_path = tmp_path / "test.png"
    img = Image.new('RGB', (800, 600), color='white')
    img.save(img_path)

    vision_analyzer.client.messages.create = AsyncMock(
        side_effect=TimeoutError("API timeout")
    )

    # Act & Assert
    with pytest.raises(TimeoutError):
        await vision_analyzer.analyze_image(str(img_path))

@pytest.mark.asyncio
async def test_vision_api_rate_limit(vision_analyzer, tmp_path):
    """Test handling of rate limit errors"""
    # Arrange
    img_path = tmp_path / "test.png"
    img = Image.new('RGB', (800, 600), color='white')
    img.save(img_path)

    vision_analyzer.client.messages.create = AsyncMock(
        side_effect=Exception("Rate limit exceeded")
    )

    # Act & Assert
    with pytest.raises(Exception, match="Rate limit"):
        await vision_analyzer.analyze_image(str(img_path))
```

### Unit Tests: ImageProcessor

**File:** `tests/vision/test_image_processor.py`

```python
import pytest
from PIL import Image
from app.vision.image_processor import ImageProcessor

def test_validate_image_success(tmp_path):
    """Test successful image validation"""
    # Arrange
    img_path = tmp_path / "valid.png"
    img = Image.new('RGB', (800, 600), color='white')
    img.save(img_path)

    # Act
    is_valid, message = ImageProcessor.validate_image(str(img_path))

    # Assert
    assert is_valid is True
    assert message == "Valid"

def test_validate_image_file_not_found():
    """Test validation fails for missing file"""
    # Act
    is_valid, message = ImageProcessor.validate_image("/nonexistent.png")

    # Assert
    assert is_valid is False
    assert "not found" in message.lower()

def test_validate_image_too_large(tmp_path):
    """Test validation fails for oversized file"""
    # This would require creating a >10MB file
    # Mocking the file size check instead
    pass  # Implementation specific

def test_validate_image_invalid_format(tmp_path):
    """Test validation fails for non-image file"""
    # Arrange
    file_path = tmp_path / "not_image.txt"
    file_path.write_text("This is text")

    # Act
    is_valid, message = ImageProcessor.validate_image(str(file_path))

    # Assert
    assert is_valid is False
    assert "not an image" in message.lower()

def test_resize_if_needed_no_resize(tmp_path):
    """Test resize is skipped for small images"""
    # Arrange
    img_path = tmp_path / "small.png"
    img = Image.new('RGB', (800, 600), color='white')
    img.save(img_path)

    # Act
    result_path = ImageProcessor.resize_if_needed(str(img_path), max_dimension=1024)

    # Assert
    assert result_path == str(img_path)
    result_img = Image.open(result_path)
    assert result_img.size == (800, 600)

def test_resize_if_needed_resize_width(tmp_path):
    """Test resize for image exceeding max width"""
    # Arrange
    img_path = tmp_path / "wide.png"
    img = Image.new('RGB', (2048, 1024), color='white')
    img.save(img_path)

    # Act
    result_path = ImageProcessor.resize_if_needed(str(img_path), max_dimension=1024)

    # Assert
    result_img = Image.open(result_path)
    assert result_img.width <= 1024
    assert result_img.height <= 512  # Maintains aspect ratio

def test_resize_if_needed_resize_height(tmp_path):
    """Test resize for image exceeding max height"""
    # Arrange
    img_path = tmp_path / "tall.png"
    img = Image.new('RGB', (1024, 2048), color='white')
    img.save(img_path)

    # Act
    result_path = ImageProcessor.resize_if_needed(str(img_path), max_dimension=1024)

    # Assert
    result_img = Image.open(result_path)
    assert result_img.height <= 1024
    assert result_img.width <= 512

def test_generate_thumbnail(tmp_path):
    """Test thumbnail generation"""
    # Arrange
    img_path = tmp_path / "image.png"
    img = Image.new('RGB', (800, 600), color='white')
    img.save(img_path)

    # Act
    thumb_path = ImageProcessor.generate_thumbnail(str(img_path), size=(128, 128))

    # Assert
    assert os.path.exists(thumb_path)
    assert "_thumb" in thumb_path
    thumb_img = Image.open(thumb_path)
    assert thumb_img.width <= 128
    assert thumb_img.height <= 128

def test_compute_hash_same_images(tmp_path):
    """Test hash computation for identical images"""
    # Arrange
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"

    img = Image.new('RGB', (100, 100), color='red')
    img.save(img1_path)
    img.save(img2_path)

    # Act
    hash1 = ImageProcessor.compute_hash(str(img1_path))
    hash2 = ImageProcessor.compute_hash(str(img2_path))

    # Assert
    assert hash1 == hash2

def test_compute_hash_different_images(tmp_path):
    """Test hash computation for different images"""
    # Arrange
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"

    img1 = Image.new('RGB', (100, 100), color='red')
    img1.save(img1_path)

    img2 = Image.new('RGB', (100, 100), color='blue')
    img2.save(img2_path)

    # Act
    hash1 = ImageProcessor.compute_hash(str(img1_path))
    hash2 = ImageProcessor.compute_hash(str(img2_path))

    # Assert
    assert hash1 != hash2

def test_extract_exif(tmp_path):
    """Test EXIF extraction"""
    # Arrange
    img_path = tmp_path / "photo.jpg"
    img = Image.new('RGB', (800, 600), color='white')
    # Add EXIF data (simplified)
    img.save(img_path, "JPEG", quality=95)

    # Act
    exif = ImageProcessor.extract_exif(str(img_path))

    # Assert
    assert isinstance(exif, dict)
    # Note: Simple test images may not have EXIF data
```

---

## Document Processing Testing

### Unit Tests: DocumentProcessor

**File:** `tests/documents/test_document_processor.py`

```python
import pytest
from app.documents.document_processor import DocumentProcessor, DocumentChunk

@pytest.mark.asyncio
async def test_process_pdf_success(document_processor, sample_pdf):
    """Test successful PDF processing"""
    # Act
    result = await document_processor.process_document(sample_pdf)

    # Assert
    assert "text" in result
    assert len(result["text"]) > 0
    assert "chunks" in result
    assert len(result["chunks"]) > 0
    assert result["format"] == "pdf"
    assert result["page_count"] > 0

@pytest.mark.asyncio
async def test_process_pdf_multipage(document_processor, multipage_pdf):
    """Test processing multi-page PDF"""
    # Act
    result = await document_processor.process_document(multipage_pdf)

    # Assert
    assert result["page_count"] == 5
    assert len(result["chunks"]) >= 5  # At least one chunk per page

@pytest.mark.asyncio
async def test_process_pdf_metadata(document_processor, pdf_with_metadata):
    """Test PDF metadata extraction"""
    # Act
    result = await document_processor.process_document(pdf_with_metadata)

    # Assert
    assert "metadata" in result
    assert result["metadata"].get("author") == "Test Author"
    assert result["metadata"].get("title") == "Test Document"

@pytest.mark.asyncio
async def test_process_docx_success(document_processor, sample_docx):
    """Test successful DOCX processing"""
    # Act
    result = await document_processor.process_document(sample_docx)

    # Assert
    assert result["format"] == "docx"
    assert len(result["text"]) > 0
    assert len(result["chunks"]) > 0

@pytest.mark.asyncio
async def test_process_txt_success(document_processor, sample_txt):
    """Test successful TXT processing"""
    # Act
    result = await document_processor.process_document(sample_txt)

    # Assert
    assert result["format"] == "txt"
    assert len(result["text"]) > 0

@pytest.mark.asyncio
async def test_process_markdown_success(document_processor, sample_md):
    """Test successful Markdown processing"""
    # Act
    result = await document_processor.process_document(sample_md)

    # Assert
    assert result["format"] == "md"
    assert "# " in result["text"]  # Has markdown headers

@pytest.mark.asyncio
async def test_chunk_text_proper_size(document_processor):
    """Test text chunking respects size limits"""
    # Arrange
    text = "word " * 500  # 2500 characters

    # Act
    chunks = document_processor._chunk_text(text)

    # Assert
    for chunk in chunks:
        assert len(chunk.text) <= document_processor.CHUNK_SIZE + 100  # Allow small overflow

@pytest.mark.asyncio
async def test_chunk_text_has_overlap(document_processor):
    """Test chunks have proper overlap"""
    # Arrange
    text = "word " * 500

    # Act
    chunks = document_processor._chunk_text(text)

    # Assert
    if len(chunks) > 1:
        # Check overlap exists
        chunk1_end = chunks[0].text[-50:]
        chunk2_start = chunks[1].text[:50]
        # Some overlap expected
        assert len(set(chunk1_end.split()) & set(chunk2_start.split())) > 0

@pytest.mark.asyncio
async def test_process_encrypted_pdf_fails(document_processor, encrypted_pdf):
    """Test encrypted PDF raises error"""
    # Act & Assert
    with pytest.raises(Exception):
        await document_processor.process_document(encrypted_pdf)

@pytest.mark.asyncio
async def test_process_corrupted_document_fails(document_processor, corrupted_doc):
    """Test corrupted document raises error"""
    # Act & Assert
    with pytest.raises(Exception):
        await document_processor.process_document(corrupted_doc)

@pytest.mark.asyncio
async def test_process_unsupported_format_fails(document_processor, tmp_path):
    """Test unsupported format raises ValueError"""
    # Arrange
    unsupported = tmp_path / "file.xyz"
    unsupported.write_bytes(b"random data")

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported document format"):
        await document_processor.process_document(str(unsupported))
```

---

## File Upload Testing

### Integration Tests: Upload Endpoints

**File:** `tests/api/test_upload_endpoints.py`

```python
import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

@pytest.mark.asyncio
async def test_upload_image_success(client, sample_image_upload):
    """Test successful image upload"""
    # Act
    response = client.post(
        "/api/upload/image",
        files={"file": ("test.png", sample_image_upload, "image/png")},
        data={
            "session_id": "test_session_123",
            "description": "Test image",
            "analyze": "true"
        }
    )

    # Assert
    assert response.status_code == 200
    result = response.json()
    assert "file_id" in result
    assert "url" in result
    assert "thumbnail_url" in result
    assert "analysis" in result

@pytest.mark.asyncio
async def test_upload_image_without_analysis(client, sample_image_upload):
    """Test image upload without vision analysis"""
    # Act
    response = client.post(
        "/api/upload/image",
        files={"file": ("test.png", sample_image_upload, "image/png")},
        data={
            "session_id": "test_session_123",
            "analyze": "false"
        }
    )

    # Assert
    assert response.status_code == 200
    result = response.json()
    assert result["analysis"] is None

@pytest.mark.asyncio
async def test_upload_image_invalid_format(client):
    """Test image upload with invalid format"""
    # Arrange
    invalid_file = BytesIO(b"not an image")

    # Act
    response = client.post(
        "/api/upload/image",
        files={"file": ("test.txt", invalid_file, "text/plain")},
        data={"session_id": "test_session_123"}
    )

    # Assert
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_upload_document_pdf_success(client, sample_pdf_upload):
    """Test successful PDF upload"""
    # Act
    response = client.post(
        "/api/upload/document",
        files={"file": ("test.pdf", sample_pdf_upload, "application/pdf")},
        data={
            "session_id": "test_session_123",
            "description": "Test PDF"
        }
    )

    # Assert
    assert response.status_code == 200
    result = response.json()
    assert "file_id" in result
    assert "url" in result
    assert "text" in result
    assert "chunks" in result

@pytest.mark.asyncio
async def test_get_uploaded_file(client, sample_image_upload):
    """Test retrieving uploaded file"""
    # Arrange - Upload first
    upload_response = client.post(
        "/api/upload/image",
        files={"file": ("test.png", sample_image_upload, "image/png")},
        data={"session_id": "test_session_123"}
    )
    file_id = upload_response.json()["file_id"]

    # Act
    response = client.get(f"/api/upload/files/images/{file_id}")

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

@pytest.mark.asyncio
async def test_get_nonexistent_file(client):
    """Test retrieving non-existent file returns 404"""
    # Act
    response = client.get("/api/upload/files/images/nonexistent-id")

    # Assert
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_multimodal_conversation(client, sample_image_upload):
    """Test multi-modal conversation"""
    # Arrange - Upload image first
    upload_response = client.post(
        "/api/upload/image",
        files={"file": ("diagram.png", sample_image_upload, "image/png")},
        data={"session_id": "test_session_123"}
    )
    file_id = upload_response.json()["file_id"]

    # Act
    response = client.post(
        "/api/upload/conversation/multimodal",
        data={
            "text": "Explain this architecture",
            "session_id": "test_session_123",
            "file_ids": [file_id]
        }
    )

    # Assert
    assert response.status_code == 200
    result = response.json()
    assert "response" in result
    assert "sources" in result
```

---

## Integration Testing

### End-to-End Tests

**File:** `tests/integration/test_phase4_integration.py`

```python
@pytest.mark.asyncio
async def test_complete_image_workflow(client):
    """Test complete image upload → analysis → RAG workflow"""
    # Step 1: Upload image
    # Step 2: Verify analysis
    # Step 3: Query with multi-modal context
    # Step 4: Verify sources include image
    pass

@pytest.mark.asyncio
async def test_complete_document_workflow(client):
    """Test complete document upload → extraction → indexing → RAG"""
    # Step 1: Upload PDF
    # Step 2: Verify extraction
    # Step 3: Verify indexing
    # Step 4: Search document content
    pass

@pytest.mark.asyncio
async def test_multimodal_rag_integration():
    """Test RAG with mixed text, image, and document context"""
    pass
```

---

## Performance Testing

### Benchmarks

```python
@pytest.mark.benchmark
def test_image_upload_performance(benchmark, client, sample_image_upload):
    """Benchmark image upload speed"""
    def upload():
        return client.post(
            "/api/upload/image",
            files={"file": ("test.png", sample_image_upload, "image/png")},
            data={"session_id": "test_session_123"}
        )

    result = benchmark(upload)
    assert result.status_code == 200
    # Target: < 2 seconds

@pytest.mark.benchmark
def test_vision_analysis_performance(benchmark, vision_analyzer, sample_image):
    """Benchmark vision analysis speed"""
    async def analyze():
        return await vision_analyzer.analyze_image(sample_image)

    result = benchmark(analyze)
    # Target: < 3 seconds
```

---

## Coverage Goals

Run tests with coverage:

```bash
pytest tests/vision --cov=app.vision --cov-report=html
pytest tests/documents --cov=app.documents --cov-report=html
pytest tests/storage --cov=app.storage --cov-report=html
pytest tests/api --cov=app.api --cov-report=html
pytest tests/integration --cov=app --cov-report=html

# Generate combined report
pytest --cov=app --cov-report=html --cov-report=term
```

**Target Coverage:**
- Lines: 80%+
- Branches: 75%+
- Functions: 85%+

---

For implementation details, see [PHASE4_IMPLEMENTATION_GUIDE.md](PHASE4_IMPLEMENTATION_GUIDE.md).

For API reference, see [PHASE4_API_REFERENCE.md](PHASE4_API_REFERENCE.md).
