# Phase 4: Upload API and Multi-Modal Integration

## Overview

Phase 4 implements comprehensive multi-modal capabilities, enabling the learning voice agent to process and understand images and documents alongside text conversations.

## Implementation Summary

### ✅ Completed Components

1. **Multimodal Services Module** (`app/multimodal/`)
   - File Manager: Upload validation and storage
   - Vision Analyzer: Claude Vision API integration
   - Document Processor: PDF/DOCX/TXT/MD extraction
   - Metadata Store: SQLite persistence layer
   - Multimodal Indexer: Vector embeddings for images/documents

2. **API Endpoints** (4 new endpoints in `app/main.py`)
   - `POST /api/upload/image` - Image upload with analysis
   - `POST /api/upload/document` - Document upload with extraction
   - `GET /api/files/{file_id}` - File retrieval
   - `POST /api/conversation/multimodal` - Multi-modal conversation

3. **Data Models** (`app/models.py`)
   - ImageUploadResponse
   - DocumentUploadResponse
   - MultiModalConversationRequest
   - MultiModalConversationResponse
   - FileMetadata
   - VisionAnalysisResult

4. **Testing** (`tests/test_multimodal_endpoints.py`)
   - 15+ comprehensive test cases
   - Integration tests
   - Error handling tests
   - Rate limiting tests

### 📊 Code Statistics

- **Total Lines Added**: ~2,333 lines
- **New Files Created**: 7 files
- **Test Coverage**: 487 lines of test code

### 🗂️ File Structure

```
app/
├── multimodal/
│   ├── __init__.py (39 lines)
│   ├── file_manager.py (328 lines)
│   ├── vision_analyzer.py (308 lines)
│   ├── document_processor.py (371 lines)
│   ├── metadata_store.py (375 lines)
│   └── multimodal_indexer.py (425 lines)
├── models.py (+124 lines)
└── main.py (+426 lines)

uploads/
├── images/
└── documents/

tests/
└── test_multimodal_endpoints.py (487 lines)
```

## Features

### 1. Image Upload & Analysis

**Endpoint**: `POST /api/upload/image`

**Features**:
- File type validation (PNG, JPEG, GIF, WebP)
- Size limit enforcement (5MB max)
- Magic byte verification
- Optional Claude Vision analysis
- Vector indexing for semantic search
- Rate limiting: 10 requests/minute

**Example**:
```bash
curl -X POST "http://localhost:8000/api/upload/image" \
  -F "file=@diagram.png" \
  -F "session_id=my-session" \
  -F "analyze=true"
```

**Response**:
```json
{
  "file_id": "uuid-1234",
  "url": "/api/files/uuid-1234",
  "filename": "diagram.png",
  "size": 245678,
  "mime_type": "image/png",
  "analysis": {
    "success": true,
    "analysis": "This image shows a system architecture diagram...",
    "processing_time_ms": 1250.5
  }
}
```

### 2. Document Upload & Processing

**Endpoint**: `POST /api/upload/document`

**Features**:
- Supports PDF, DOCX, TXT, MD formats
- Size limit: 10MB
- Text extraction and chunking
- Metadata extraction (author, title, pages)
- Vector indexing for RAG
- Rate limiting: 5 requests/minute

**Example**:
```bash
curl -X POST "http://localhost:8000/api/upload/document" \
  -F "file=@research.pdf" \
  -F "session_id=my-session" \
  -F "extract_text=true"
```

**Response**:
```json
{
  "file_id": "uuid-5678",
  "url": "/api/files/uuid-5678",
  "filename": "research.pdf",
  "size": 1245678,
  "mime_type": "application/pdf",
  "text_preview": "Introduction to Machine Learning...",
  "chunk_count": 15,
  "metadata": {
    "title": "ML Research Paper",
    "author": "John Doe",
    "page_count": 10
  }
}
```

### 3. File Retrieval

**Endpoint**: `GET /api/files/{file_id}`

**Features**:
- Content negotiation
- Access tracking
- Proper MIME types
- Security validation

**Example**:
```bash
curl "http://localhost:8000/api/files/uuid-1234?file_type=image"
```

### 4. Multi-Modal Conversation

**Endpoint**: `POST /api/conversation/multimodal`

**Features**:
- Combine text, images, and documents
- Context enrichment with image analyses
- Document content integration
- Session management
- Rate limiting: 20 requests/minute

**Example**:
```bash
curl -X POST "http://localhost:8000/api/conversation/multimodal" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain the relationship between the image and document",
    "image_ids": ["uuid-1234"],
    "document_ids": ["uuid-5678"],
    "session_id": "my-session"
  }'
```

**Response**:
```json
{
  "session_id": "my-session",
  "user_text": "Explain the relationship...",
  "agent_text": "Based on the image and document, the system architecture shows...",
  "intent": "multimodal_query",
  "image_count": 1,
  "document_count": 1,
  "processing_time_ms": 2340.5
}
```

## Architecture

### Security Measures

1. **File Type Validation**
   - Magic byte verification
   - MIME type checking
   - Extension validation

2. **Size Limits**
   - Images: 5MB maximum
   - Documents: 10MB maximum

3. **Rate Limiting**
   - Image upload: 10/minute
   - Document upload: 5/minute
   - Multimodal conversation: 20/minute

4. **Session-Based Access**
   - Files tied to sessions
   - Access tracking
   - Metadata persistence

### Performance Optimizations

1. **Async Operations**
   - Non-blocking file I/O
   - Parallel processing
   - Background indexing

2. **Chunking Strategy**
   - Default chunk size: 500 characters
   - Overlap: 50 characters
   - Sentence boundary detection

3. **Caching & Indexing**
   - Vector embeddings cached
   - Metadata indexed in SQLite
   - Fast retrieval paths

### Error Handling

1. **Validation Errors** (400)
   - Invalid file types
   - Oversized files
   - Malformed requests

2. **Not Found Errors** (404)
   - Missing file IDs
   - Deleted files

3. **Rate Limit Errors** (429)
   - Exceeded request limits
   - Retry-After headers

4. **Server Errors** (500)
   - Processing failures
   - API errors
   - Storage issues

## Testing

### Running Tests

```bash
# Run all multimodal tests
pytest tests/test_multimodal_endpoints.py -v

# Run with coverage
pytest tests/test_multimodal_endpoints.py --cov=app/multimodal --cov-report=html

# Run specific test
pytest tests/test_multimodal_endpoints.py::test_image_upload_with_analysis -v
```

### Test Categories

1. **Image Upload Tests** (6 tests)
   - With/without analysis
   - Invalid file types
   - Size limits
   - Session handling

2. **Document Upload Tests** (3 tests)
   - Text extraction
   - Without extraction
   - Invalid types

3. **File Retrieval Tests** (2 tests)
   - Nonexistent files
   - After upload

4. **Multimodal Conversation Tests** (3 tests)
   - Text only
   - With images
   - With documents

5. **Integration Tests** (1 test)
   - Full workflow

6. **Error Handling Tests** (2 tests)
   - Malformed requests
   - Empty files

## Dependencies

### New Dependencies Added

```
# Vision and Image Processing
Pillow>=10.1.0
python-magic>=0.4.27

# Document Processing
PyPDF2==3.0.1
python-docx>=1.1.0
markdown>=3.5.0
chardet==5.2.0

# Async Operations
aiofiles==23.2.1
```

### Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your_key_here  # For Claude Vision
OPENAI_API_KEY=your_key_here     # For embeddings

# Optional
MAX_IMAGE_SIZE=5242880           # 5MB in bytes
MAX_DOCUMENT_SIZE=10485760       # 10MB in bytes
UPLOAD_DIR=uploads               # Upload directory
```

### Directory Setup

```bash
# Create upload directories
mkdir -p uploads/images
mkdir -p uploads/documents
```

## Usage Examples

### Python Client

```python
import requests
from pathlib import Path

# Upload image
with open('diagram.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload/image',
        files={'file': f},
        data={'session_id': 'my-session', 'analyze': 'true'}
    )
    image_data = response.json()
    image_id = image_data['file_id']

# Upload document
with open('research.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload/document',
        files={'file': f},
        data={'session_id': 'my-session', 'extract_text': 'true'}
    )
    doc_data = response.json()
    doc_id = doc_data['file_id']

# Multi-modal conversation
response = requests.post(
    'http://localhost:8000/api/conversation/multimodal',
    json={
        'text': 'Explain the relationship between the image and document',
        'image_ids': [image_id],
        'document_ids': [doc_id],
        'session_id': 'my-session'
    }
)
conversation = response.json()
print(conversation['agent_text'])
```

### JavaScript Client

```javascript
// Upload image
const formData = new FormData();
formData.append('file', imageFile);
formData.append('session_id', 'my-session');
formData.append('analyze', 'true');

const imageResponse = await fetch('http://localhost:8000/api/upload/image', {
  method: 'POST',
  body: formData
});
const imageData = await imageResponse.json();

// Multi-modal conversation
const convResponse = await fetch('http://localhost:8000/api/conversation/multimodal', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'What do you see in this image?',
    image_ids: [imageData.file_id],
    document_ids: [],
    session_id: 'my-session'
  })
});
const conversation = await convResponse.json();
console.log(conversation.agent_text);
```

## Future Enhancements

1. **Additional Formats**
   - Audio file support (MP3, WAV)
   - Video analysis
   - Spreadsheet parsing (XLSX, CSV)

2. **Advanced Vision**
   - Object detection
   - OCR improvements
   - Image classification

3. **Document Intelligence**
   - Table extraction
   - Layout analysis
   - Form parsing

4. **Security**
   - Virus scanning (ClamAV)
   - Content moderation
   - Encryption at rest

5. **Performance**
   - CDN integration
   - Image optimization
   - Compression

## Troubleshooting

### Common Issues

**Issue**: "Unsupported file type" error
**Solution**: Ensure file has correct magic bytes. The system validates actual file content, not just extension.

**Issue**: Upload fails with size error
**Solution**: Check file size is under limits (5MB for images, 10MB for documents).

**Issue**: Vision analysis fails
**Solution**: Verify ANTHROPIC_API_KEY is set and valid.

**Issue**: Document indexing fails
**Solution**: Verify OPENAI_API_KEY is set for embeddings generation.

**Issue**: File not found after upload
**Solution**: Check that upload directories exist and have write permissions.

## Monitoring

### Key Metrics

1. **Upload Performance**
   - Processing time (p50, p95, p99)
   - Success rate
   - Error types

2. **Analysis Quality**
   - Vision API latency
   - Document extraction success rate
   - Chunk count distribution

3. **Storage**
   - Total files uploaded
   - Storage usage by type
   - Access patterns

### Logging

All operations are logged with structured logging:

```python
api_logger.info(
    "image_upload_complete",
    session_id=session_id,
    file_id=file_metadata["file_id"],
    analyzed=analyze,
    processing_time_ms=round(processing_time, 2)
)
```

## API Documentation

OpenAPI/Swagger documentation is auto-generated and available at:

```
http://localhost:8000/docs
```

## Support

For issues or questions:
- GitHub Issues: [project-repo]/issues
- Documentation: `/docs` directory
- API Reference: `/docs` endpoint

---

**Phase 4 Status**: ✅ Complete
**Total Implementation Time**: Single session
**Code Quality**: Production-ready with comprehensive tests
**Security**: Validated and rate-limited
**Performance**: Optimized with async operations
