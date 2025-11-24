# Phase 4 Multi-Modal API Reference

**Version:** 1.0.0
**Date:** 2025-01-21

Complete API documentation for Phase 4 multi-modal components.

## Table of Contents

1. [VisionAnalyzer API](#visionanalyzer-api)
2. [ImageProcessor API](#imageprocessor-api)
3. [DocumentProcessor API](#documentprocessor-api)
4. [FileManager API](#filemanager-api)
5. [MultiModalIndexer API](#multimodalindexer-api)
6. [Upload Endpoints](#upload-endpoints)
7. [Configuration API](#configuration-api)
8. [Error Handling](#error-handling)

---

## VisionAnalyzer API

**Module:** `app.vision.vision_analyzer`

### Class: `VisionAnalyzer`

Analyze images using Claude 3.5 Sonnet Vision API.

#### Constructor

```python
VisionAnalyzer(api_key: Optional[str] = None)
```

**Parameters:**
- `api_key` (str, optional): Anthropic API key. Uses `ANTHROPIC_API_KEY` env var if None.

**Example:**
```python
from app.vision.vision_analyzer import vision_analyzer

# Use singleton
analyzer = vision_analyzer

# Or create custom instance
custom_analyzer = VisionAnalyzer(api_key="sk-ant-...")
```

---

#### `analyze_image()`

Analyze image and extract information.

```python
async def analyze_image(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    include_ocr: bool = False
) -> Dict[str, Any]
```

**Parameters:**
- `image_path` (str): Path to image file
- `prompt` (str): Analysis prompt. Default: "Describe this image in detail."
- `include_ocr` (bool): Extract text from image. Default: False

**Returns:** Dict with keys:
- `analysis` (str): Text description of image
- `ocr_text` (str|None): Extracted text if `include_ocr=True`
- `dimensions` (tuple): Image dimensions (width, height)
- `format` (str): Image format (PNG, JPEG, etc.)
- `tokens_used` (int): Total API tokens consumed
- `model` (str): Model used for analysis

**Raises:**
- `FileNotFoundError`: Image file not found
- `ValueError`: Invalid image format
- `RuntimeError`: API call failed

**Example:**
```python
result = await vision_analyzer.analyze_image(
    image_path="/path/to/diagram.png",
    prompt="What does this system architecture diagram show?",
    include_ocr=True
)

print(result["analysis"])
# Output: "This diagram shows a microservices architecture with..."

print(result["ocr_text"])
# Output: "API Gateway, Auth Service, User Service..."

print(result["tokens_used"])
# Output: 1245
```

---

#### `analyze_diagram()`

Specialized analysis for diagrams and flowcharts.

```python
async def analyze_diagram(image_path: str) -> Dict[str, Any]
```

**Parameters:**
- `image_path` (str): Path to diagram image

**Returns:** Analysis result with structured description

**Example:**
```python
result = await vision_analyzer.analyze_diagram("/path/to/flowchart.png")

print(result["analysis"])
# Output: "This is a flowchart showing an authentication process.
#          Components: Login Form -> Validate Credentials -> Check 2FA..."
```

---

#### `compare_images()`

Compare two images and describe differences.

```python
async def compare_images(
    image_path1: str,
    image_path2: str
) -> Dict[str, Any]
```

**Parameters:**
- `image_path1` (str): First image path
- `image_path2` (str): Second image path

**Returns:** Dict with keys:
- `image1_analysis` (str): First image description
- `image2_analysis` (str): Second image description
- `comparison` (str): Key similarities and differences
- `tokens_used` (int): Total tokens used

**Example:**
```python
result = await vision_analyzer.compare_images(
    "/path/to/before.png",
    "/path/to/after.png"
)

print(result["comparison"])
# Output: "The main difference is the navigation menu has been redesigned..."
```

---

## ImageProcessor API

**Module:** `app.vision.image_processor`

### Class: `ImageProcessor`

Handle image validation, resizing, and format conversion.

#### `validate_image()` (static)

Validate image file format, size, and dimensions.

```python
@staticmethod
def validate_image(file_path: str) -> Tuple[bool, str]
```

**Parameters:**
- `file_path` (str): Path to image file

**Returns:** Tuple `(is_valid, error_message)`
- `is_valid` (bool): Whether image is valid
- `error_message` (str): Error description or "Valid"

**Example:**
```python
from app.vision.image_processor import image_processor

is_valid, message = image_processor.validate_image("/path/to/image.png")

if is_valid:
    print("Image is valid!")
else:
    print(f"Validation failed: {message}")
```

**Validation Rules:**
- File exists
- Size ≤ 10 MB
- MIME type is image/*
- Format in: PNG, JPEG, JPG, GIF, WEBP
- Dimensions ≤ 4096 x 4096 pixels

---

#### `resize_if_needed()` (static)

Resize image if it exceeds maximum dimension.

```python
@staticmethod
def resize_if_needed(
    image_path: str,
    max_dimension: int = 2048,
    output_path: Optional[str] = None
) -> str
```

**Parameters:**
- `image_path` (str): Input image path
- `max_dimension` (int): Maximum width/height. Default: 2048
- `output_path` (str, optional): Output path. Overwrites input if None.

**Returns:** Path to resized image (str)

**Example:**
```python
# Resize in-place
resized_path = image_processor.resize_if_needed(
    "/path/to/large_image.png",
    max_dimension=1024
)

# Save to new file
resized_path = image_processor.resize_if_needed(
    "/path/to/input.png",
    max_dimension=1024,
    output_path="/path/to/output.png"
)
```

**Behavior:**
- Maintains aspect ratio
- Uses LANCZOS resampling (high quality)
- No-op if image already within limits
- Preserves original format

---

#### `generate_thumbnail()` (static)

Generate thumbnail image.

```python
@staticmethod
def generate_thumbnail(
    image_path: str,
    size: Tuple[int, int] = (256, 256)
) -> str
```

**Parameters:**
- `image_path` (str): Source image
- `size` (tuple): Thumbnail size (width, height). Default: (256, 256)

**Returns:** Path to thumbnail (str)

**Example:**
```python
thumb_path = image_processor.generate_thumbnail(
    "/path/to/image.png",
    size=(128, 128)
)

print(thumb_path)
# Output: "/path/to/image_thumb.png"
```

---

#### `compute_hash()` (static)

Compute SHA256 hash for deduplication.

```python
@staticmethod
def compute_hash(image_path: str) -> str
```

**Parameters:**
- `image_path` (str): Image to hash

**Returns:** SHA256 hash (str)

**Example:**
```python
hash1 = image_processor.compute_hash("/path/to/image1.png")
hash2 = image_processor.compute_hash("/path/to/image2.png")

if hash1 == hash2:
    print("Images are identical!")
```

---

#### `extract_exif()` (static)

Extract EXIF metadata from image.

```python
@staticmethod
def extract_exif(image_path: str) -> dict
```

**Parameters:**
- `image_path` (str): Image file

**Returns:** EXIF data dictionary

**Example:**
```python
exif = image_processor.extract_exif("/path/to/photo.jpg")

print(exif.get("DateTimeOriginal"))
# Output: "2025:01:21 10:30:00"

print(exif.get("Model"))
# Output: "iPhone 15 Pro"
```

---

## DocumentProcessor API

**Module:** `app.documents.document_processor`

### Class: `DocumentProcessor`

Process and extract text from various document formats.

#### `process_document()`

Process document and extract text.

```python
async def process_document(file_path: str) -> Dict[str, Any]
```

**Parameters:**
- `file_path` (str): Path to document

**Returns:** Dict with keys:
- `text` (str): Full extracted text
- `chunks` (List[DocumentChunk]): Text chunks for indexing
- `metadata` (dict): Document metadata (author, title, etc.)
- `page_count` (int|None): Number of pages (if applicable)
- `format` (str): Document format (pdf, docx, txt, md)

**Raises:**
- `ValueError`: Unsupported document format
- `RuntimeError`: Document processing failed

**Supported Formats:**
- PDF (application/pdf)
- DOCX (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
- TXT (text/plain)
- MD (text/markdown)

**Example:**
```python
from app.documents.document_processor import document_processor

result = await document_processor.process_document("/path/to/paper.pdf")

print(f"Extracted {len(result['text'])} characters")
print(f"Split into {len(result['chunks'])} chunks")
print(f"Pages: {result['page_count']}")
print(f"Author: {result['metadata'].get('author')}")

# Access chunks
for chunk in result['chunks']:
    print(f"Chunk {chunk.chunk_index}: {chunk.text[:100]}...")
```

---

### Class: `DocumentChunk`

**Dataclass:** Represents a text chunk from document.

```python
@dataclass
class DocumentChunk:
    text: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    metadata: Optional[Dict[str, Any]] = None
```

**Attributes:**
- `text` (str): Chunk text content
- `page_number` (int, optional): Source page number
- `chunk_index` (int): Chunk sequence number
- `metadata` (dict, optional): Additional metadata

**Example:**
```python
chunk = DocumentChunk(
    text="This is the introduction section...",
    page_number=1,
    chunk_index=0,
    metadata={"section": "introduction"}
)
```

---

## FileManager API

**Module:** `app.storage.file_manager`

### Class: `FileManager`

Manage uploaded files with organization and deduplication.

#### `save_file()`

Save file with organization.

```python
async def save_file(
    source_path: str,
    file_type: str,
    session_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters:**
- `source_path` (str): Source file path
- `file_type` (str): "image" or "document"
- `session_id` (str): Session identifier
- `metadata` (dict, optional): Additional metadata

**Returns:** File ID (str, UUID format)

**Storage Structure:**
```
data/uploads/
├── images/
│   └── {session_id[:8]}/
│       └── {file_id}.{ext}
├── documents/
│   └── {session_id[:8]}/
│       └── {file_id}.{ext}
└── metadata/
    └── {file_id}.json
```

**Example:**
```python
from app.storage.file_manager import file_manager

file_id = await file_manager.save_file(
    source_path="/tmp/upload.png",
    file_type="image",
    session_id="sess_abc123",
    metadata={
        "original_name": "diagram.png",
        "description": "System architecture"
    }
)

print(file_id)
# Output: "a1b2c3d4-5678-90ab-cdef-1234567890ab"
```

---

#### `get_file_path()`

Get file path by ID.

```python
async def get_file_path(file_id: str) -> Optional[str]
```

**Parameters:**
- `file_id` (str): File identifier

**Returns:** File path (str) or None if not found

**Example:**
```python
path = await file_manager.get_file_path("a1b2c3d4-...")

if path:
    print(f"File location: {path}")
else:
    print("File not found")
```

---

#### `get_file_info()`

Get file metadata.

```python
async def get_file_info(file_id: str) -> Optional[Dict[str, Any]]
```

**Parameters:**
- `file_id` (str): File identifier

**Returns:** Metadata dict or None

**Example:**
```python
info = await file_manager.get_file_info("a1b2c3d4-...")

if info:
    print(f"Type: {info['file_type']}")
    print(f"Size: {info['size']} bytes")
    print(f"Created: {info['created']}")
    print(f"Session: {info['session_id']}")
```

---

#### `delete_file()`

Delete file and metadata.

```python
async def delete_file(file_id: str) -> bool
```

**Parameters:**
- `file_id` (str): File identifier

**Returns:** True if deleted, False if not found

**Example:**
```python
deleted = await file_manager.delete_file("a1b2c3d4-...")

if deleted:
    print("File deleted successfully")
else:
    print("File not found")
```

---

## MultiModalIndexer API

**Module:** `app.storage.multimodal_indexer`

### Class: `MultiModalIndexer`

Index images and documents in vector database.

#### `initialize()`

Initialize vector collections.

```python
async def initialize() -> None
```

**Collections Created:**
- `multimodal_images`: Image analysis embeddings
- `multimodal_documents`: Document chunk embeddings

**Example:**
```python
from app.storage.multimodal_indexer import multimodal_indexer

await multimodal_indexer.initialize()
```

---

#### `index_image()`

Index image analysis in vector store.

```python
async def index_image(
    file_id: str,
    image_path: str,
    analysis: Dict[str, Any],
    metadata: Optional[Dict] = None
) -> None
```

**Parameters:**
- `file_id` (str): File identifier
- `image_path` (str): Path to image
- `analysis` (dict): Vision analysis result
- `metadata` (dict, optional): Additional metadata

**Example:**
```python
await multimodal_indexer.index_image(
    file_id="a1b2c3d4-...",
    image_path="/path/to/image.png",
    analysis={
        "analysis": "A diagram showing microservices architecture",
        "dimensions": (1024, 768),
        "format": "PNG"
    },
    metadata={
        "session_id": "sess_123",
        "description": "Architecture diagram"
    }
)
```

---

#### `index_document()`

Index document chunks in vector store.

```python
async def index_document(
    file_id: str,
    chunks: List[DocumentChunk],
    metadata: Optional[Dict] = None
) -> None
```

**Parameters:**
- `file_id` (str): File identifier
- `chunks` (List[DocumentChunk]): Document chunks
- `metadata` (dict, optional): Additional metadata

**Example:**
```python
result = await document_processor.process_document("/path/to/paper.pdf")

await multimodal_indexer.index_document(
    file_id="b2c3d4e5-...",
    chunks=result["chunks"],
    metadata={
        "session_id": "sess_123",
        "title": result["metadata"].get("title"),
        "author": result["metadata"].get("author")
    }
)
```

---

#### `retrieve_context()`

Retrieve multi-modal context for query.

```python
async def retrieve_context(
    query: str,
    session_id: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    k: int = 5
) -> Dict[str, Any]
```

**Parameters:**
- `query` (str): Search query
- `session_id` (str, optional): Filter by session
- `file_ids` (List[str], optional): Filter by specific files
- `k` (int): Number of results per type. Default: 5

**Returns:** Dict with key `sources` (list of result dicts)

**Example:**
```python
context = await multimodal_indexer.retrieve_context(
    query="microservices architecture patterns",
    session_id="sess_123",
    k=3
)

for source in context["sources"]:
    print(f"Type: {source['type']}")
    print(f"Content: {source['content'][:100]}...")
    print(f"Metadata: {source['metadata']}")
    print("---")
```

**Source Format:**
```python
{
    "type": "image" | "document",
    "content": "extracted text or analysis",
    "metadata": {
        "file_id": "...",
        "dimensions": [...],  # for images
        "chunk_index": 0,      # for documents
        # ... other metadata
    }
}
```

---

## Upload Endpoints

**Router:** `app.api.upload_routes`

All endpoints are prefixed with `/api/upload`.

### POST /api/upload/image

Upload and analyze an image.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file` (File): Image file (PNG, JPEG, GIF, WEBP)
  - `session_id` (str): Session identifier
  - `description` (str, optional): Image description
  - `analyze` (bool): Run vision analysis. Default: true

**Response:** 200 OK
```json
{
    "file_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
    "url": "/files/images/a1b2c3d4-...",
    "thumbnail_url": "/files/thumbnails/a1b2c3d4-..._thumb",
    "analysis": {
        "analysis": "A diagram showing...",
        "dimensions": [1024, 768],
        "format": "PNG",
        "tokens_used": 1245
    },
    "metadata": {
        "format": "PNG",
        "dimensions": [1024, 768],
        "size_bytes": 123456
    }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/upload/image \
  -F "file=@diagram.png" \
  -F "session_id=sess_123" \
  -F "description=System architecture" \
  -F "analyze=true"
```

**Python Example:**
```python
import requests

with open("diagram.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload/image",
        files={"file": f},
        data={
            "session_id": "sess_123",
            "description": "System architecture",
            "analyze": "true"
        }
    )

result = response.json()
print(f"Uploaded: {result['file_id']}")
print(f"Analysis: {result['analysis']['analysis']}")
```

---

### POST /api/upload/document

Upload and process a document.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file` (File): Document file (PDF, DOCX, TXT, MD)
  - `session_id` (str): Session identifier
  - `description` (str, optional): Document description

**Response:** 200 OK
```json
{
    "file_id": "b2c3d4e5-6789-01ab-cdef-234567890abc",
    "url": "/files/documents/b2c3d4e5-...",
    "text": "Introduction\n\nThis paper discusses...",
    "chunks": 25,
    "metadata": {
        "author": "John Doe",
        "title": "Microservices Architecture",
        "page_count": 10
    },
    "page_count": 10
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/upload/document \
  -F "file=@paper.pdf" \
  -F "session_id=sess_123" \
  -F "description=Research paper on microservices"
```

**Python Example:**
```python
with open("paper.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload/document",
        files={"file": f},
        data={
            "session_id": "sess_123",
            "description": "Research paper"
        }
    )

result = response.json()
print(f"Uploaded: {result['file_id']}")
print(f"Extracted {result['chunks']} chunks")
print(f"Author: {result['metadata']['author']}")
```

---

### GET /api/upload/files/{file_type}/{file_id}

Retrieve uploaded file.

**Parameters:**
- `file_type` (path): "images" or "documents"
- `file_id` (path): File identifier

**Response:** File content with appropriate Content-Type

**Example:**
```bash
curl http://localhost:8000/api/upload/files/images/a1b2c3d4-... -o downloaded.png
```

---

### POST /api/upload/conversation/multimodal

Enhanced conversation with multi-modal context.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `text` (str): User question/input
  - `session_id` (str): Session identifier
  - `file_ids` (List[str], optional): File IDs for context

**Response:** 200 OK
```json
{
    "response": "Based on the architecture diagram you shared, the system uses...",
    "sources": [
        {
            "type": "text",
            "content": "Previous conversation about microservices..."
        },
        {
            "type": "image",
            "file_id": "a1b2c3d4-...",
            "analysis": "A diagram showing microservices architecture..."
        },
        {
            "type": "document",
            "file_id": "b2c3d4e5-...",
            "excerpt": "Microservices pattern involves..."
        }
    ]
}
```

**Python Example:**
```python
response = requests.post(
    "http://localhost:8000/api/upload/conversation/multimodal",
    data={
        "text": "Explain the authentication flow in this architecture",
        "session_id": "sess_123",
        "file_ids": ["a1b2c3d4-...", "b2c3d4e5-..."]
    }
)

result = response.json()
print(f"Response: {result['response']}")
print(f"Sources: {len(result['sources'])}")
```

---

## Configuration API

**Module:** `app.multimodal.config`

### Class: `MultiModalConfig`

Configuration for multi-modal features.

```python
from app.multimodal.config import multimodal_config

# Access configuration
print(multimodal_config.max_file_size)  # 10485760
print(multimodal_config.max_image_dimension)  # 4096
print(multimodal_config.doc_chunk_size)  # 1000
print(multimodal_config.upload_dir)  # Path("data/uploads")
```

**Attributes:**
- `claude_vision_model` (str): Vision model ID
- `vision_max_tokens` (int): Max tokens for vision API
- `max_file_size` (int): Maximum file size in bytes
- `max_image_dimension` (int): Maximum image width/height
- `thumbnail_size` (tuple): Thumbnail dimensions
- `doc_chunk_size` (int): Document chunk size in characters
- `doc_chunk_overlap` (int): Overlap between chunks
- `upload_dir` (Path): Upload directory path
- `file_retention_days` (int): Auto-cleanup after N days
- `supported_image_formats` (set): Supported image formats
- `supported_doc_formats` (set): Supported document formats

---

## Error Handling

### Common Errors

#### Validation Errors (400)

```json
{
    "detail": "File too large (max 10MB)"
}
```

**Causes:**
- File size exceeds limit
- Unsupported format
- Invalid dimensions
- Corrupted file

#### Not Found (404)

```json
{
    "detail": "File not found"
}
```

**Causes:**
- Invalid file_id
- File deleted
- Path not accessible

#### API Errors (500)

```json
{
    "detail": "Vision analysis failed: API timeout"
}
```

**Causes:**
- Claude API timeout
- Rate limit exceeded
- Network error

#### Processing Errors (500)

```json
{
    "detail": "Document extraction failed: Encrypted PDF"
}
```

**Causes:**
- Encrypted documents
- Corrupted files
- Unsupported format variant

### Error Handling Example

```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(
        "http://localhost:8000/api/upload/image",
        files={"file": open("image.png", "rb")},
        data={"session_id": "sess_123"}
    )
    response.raise_for_status()
    result = response.json()
    print(f"Success: {result['file_id']}")

except RequestException as e:
    if e.response:
        print(f"Error {e.response.status_code}: {e.response.json()['detail']}")
    else:
        print(f"Network error: {e}")
```

---

## Rate Limits

**Default Limits:**
- Image uploads: 20 per minute
- Document uploads: 10 per minute
- Vision analysis: 30 per minute (Claude API limit)
- File retrievals: 100 per minute

**Headers:**
```
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1642780800
```

---

## Performance Metrics

**Typical Response Times:**
- Image upload: < 2 seconds
- Vision analysis: < 3 seconds
- Document upload: < 1 second per MB
- Document processing: < 5 seconds per page
- File retrieval: < 100ms
- Multi-modal search: < 300ms

---

For implementation details, see [PHASE4_IMPLEMENTATION_GUIDE.md](PHASE4_IMPLEMENTATION_GUIDE.md).

For usage examples, see [PHASE4_USAGE_EXAMPLES.md](PHASE4_USAGE_EXAMPLES.md).

---

**Phase 4 Status:** Ready for implementation.
