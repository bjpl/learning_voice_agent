# Phase 4: Multi-Modal Storage and Indexing

## Overview

Phase 4 implements a comprehensive storage and indexing system for multi-modal content including images, documents (PDF, DOCX, TXT), and audio files. The system provides:

- **Organized File Storage**: Hierarchical directory organization by date and session
- **Metadata Management**: SQLite-based metadata with full-text search
- **Deduplication**: SHA256 hash-based to save storage space
- **Vector Indexing**: Semantic search using ChromaDB embeddings
- **Knowledge Graph Integration**: Link files to concepts in Neo4j
- **Automatic Cleanup**: Configurable retention policies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    File Upload (bytes)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │    FileManager        │
           │  - Validation         │
           │  - Deduplication      │
           │  - Organization       │
           │  - Storage            │
           └───────┬───────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌──────────┐
   │  Disk  │ │Metadata│ │ Indexer  │
   │Storage │ │ Store  │ │          │
   └────────┘ └───┬────┘ └────┬─────┘
                  │            │
                  │      ┌─────┴──────┬──────────┐
                  │      │            │          │
                  ▼      ▼            ▼          ▼
              ┌───────┐ ┌──────┐ ┌────────┐ ┌──────┐
              │SQLite │ │Vector│ │FullText│ │ Neo4j│
              │  DB   │ │Store │ │ Search │ │  KG  │
              └───────┘ └──────┘ └────────┘ └──────┘
```

## Components

### 1. StorageConfig (`app/storage/config.py`)

Configuration management for storage system:

```python
from app.storage import storage_config

# Get file type configuration
image_config = storage_config.get_file_type_config("image")
print(f"Max size: {image_config.max_size_mb}MB")
print(f"Extensions: {image_config.allowed_extensions}")
print(f"Retention: {image_config.retention_days} days")

# Validate file
is_valid, error = storage_config.validate_file(
    file_type="image",
    filename="photo.jpg",
    size_bytes=5_000_000
)

# Get storage path
path = storage_config.get_storage_path("image", "session_123")
print(f"Storage path: {path}")
```

**Default Limits:**
- Images: 10MB, 30 days retention
- PDFs: 25MB, 90 days retention
- Documents: 20MB, 90 days retention
- Text: 5MB, 60 days retention
- Audio: 50MB, 30 days retention
- Video: 100MB, 14 days retention

### 2. FileManager (`app/storage/file_manager.py`)

High-level file operations:

```python
from app.storage import file_manager

# Initialize
await file_manager.initialize()

# Save file
result = await file_manager.save_file(
    file_data=file_bytes,
    original_filename="document.pdf",
    file_type="pdf",
    session_id="session_123",
    metadata={"source": "upload", "user_id": "user_456"}
)

# Returns:
# {
#     "file_id": "session_123_abc123...",
#     "stored_path": "/data/uploads/documents/2025/01/session_123/...",
#     "file_hash": "sha256_hash",
#     "file_size": 1234567,
#     "deduplicated": False
# }

# Retrieve file
file_data = await file_manager.get_file(result["file_id"])

# Get metadata only
metadata = await file_manager.get_file_metadata(result["file_id"])

# List files
files = await file_manager.list_files(
    session_id="session_123",
    file_type="pdf",
    limit=50
)

# Delete file
await file_manager.delete_file(result["file_id"])

# Cleanup old files
stats = await file_manager.cleanup_old_files(dry_run=True)
print(f"Would delete: {stats['deleted_count']} files")
print(f"Would free: {stats['freed_mb']:.2f} MB")
```

**Deduplication:**

If the same file is uploaded twice (same content hash), only one copy is stored on disk, but separate metadata entries are created:

```python
# First upload
result1 = await file_manager.save_file(...)
# deduplicated: False (new file)

# Second upload (same content)
result2 = await file_manager.save_file(...)
# deduplicated: True (reuses existing file)
# original_file_id: points to first upload
```

### 3. MetadataStore (`app/storage/metadata_store.py`)

SQLite-based metadata management with full-text search:

```python
from app.storage import metadata_store

# Initialize
await metadata_store.initialize()

# Save file metadata (usually done by FileManager)
await metadata_store.save_file_metadata(
    file_id="file_123",
    session_id="session_456",
    file_type="image",
    original_filename="photo.jpg",
    stored_path="/path/to/file",
    file_size=1234567,
    file_hash="abc123...",
    metadata={"camera": "iPhone"}
)

# Save analysis results
await metadata_store.save_analysis(
    file_id="file_123",
    analysis_type="vision",
    result={
        "objects": ["cat", "tree"],
        "description": "A cat under a tree"
    }
)

# Get file by ID
file = await metadata_store.get_file_by_id("file_123")

# Get file by hash (deduplication check)
existing = await metadata_store.get_file_by_hash("abc123...")

# Get analysis results
analyses = await metadata_store.get_file_analysis("file_123")

# Full-text search on analysis
results = await metadata_store.search_analysis("cat in garden")

# Link to knowledge graph
await metadata_store.link_file_to_concept(
    file_id="file_123",
    concept_name="animal",
    link_type="detected",
    confidence=0.95
)

# Get linked concepts
concepts = await metadata_store.get_file_concepts("file_123")
```

### 4. MultiModalIndexer (`app/storage/indexer.py`)

Multi-backend indexing for semantic search and knowledge graph:

```python
from app.storage import multimodal_indexer

# Initialize (starts vector store and knowledge graph)
await multimodal_indexer.initialize()

# Index image with vision analysis
await multimodal_indexer.index_image(
    file_id="img_123",
    vision_analysis={
        "description": "A tabby cat sitting on a fence",
        "objects": ["cat", "fence", "outdoor"],
        "labels": [
            {"name": "cat", "confidence": 0.98},
            {"name": "fence", "confidence": 0.92}
        ]
    },
    session_id="session_123",
    metadata={"source": "google_vision"}
)

# Index document with extracted text
await multimodal_indexer.index_document(
    file_id="doc_123",
    extracted_text="Machine learning is...",
    session_id="session_123",
    document_metadata={
        "pages": 5,
        "author": "John Doe",
        "title": "ML Overview"
    }
)

# Index audio with transcription
await multimodal_indexer.index_audio(
    file_id="audio_123",
    transcription="Hello, this is a test recording...",
    session_id="session_123",
    audio_metadata={
        "duration_seconds": 120,
        "speaker": "John"
    }
)

# Search similar files
results = await multimodal_indexer.search_similar_files(
    query="cats outdoors",
    file_type="image",
    n_results=10,
    min_score=0.7
)

# Re-index after analysis update
await multimodal_indexer.update_index("file_123")
```

## Database Schema

### Tables

**multimodal_files**: File metadata
- `file_id` (TEXT, UNIQUE): File identifier
- `session_id` (TEXT): Session identifier
- `file_type` (TEXT): image, pdf, docx, txt, audio, video
- `original_filename` (TEXT): Original filename
- `stored_path` (TEXT): Path on disk
- `file_size` (INTEGER): Size in bytes
- `mime_type` (TEXT): MIME type
- `file_hash` (TEXT): SHA256 hash for deduplication
- `uploaded_at` (DATETIME): Upload timestamp
- `metadata` (TEXT): JSON metadata

**file_analysis**: Analysis results
- `file_id` (TEXT): References multimodal_files
- `analysis_type` (TEXT): vision, ocr, extraction, transcription
- `analysis_result` (TEXT): JSON result
- `analyzed_at` (DATETIME): Analysis timestamp

**file_concept_links**: Knowledge graph links
- `file_id` (TEXT): References multimodal_files
- `concept_name` (TEXT): Concept in knowledge graph
- `link_type` (TEXT): extracted, mentioned, related
- `confidence` (REAL): Link confidence 0.0-1.0

**file_analysis_fts**: Full-text search (FTS5 virtual table)
- Indexes analysis_result for fast text search

### Indexes

- `idx_file_session`: Files by session and date
- `idx_file_type`: Files by type and date
- `idx_file_hash`: Files by hash (deduplication)
- `idx_analysis_file`: Analysis by file
- `idx_concept_links`: Concept links by file and concept

## Migration

Run the Phase 4 migration to create the database schema:

```bash
# Run migration script
python scripts/run_phase4_migration.py

# Or manually apply SQL
sqlite3 data/storage_metadata.db < scripts/phase4_storage_migration.sql
```

Verify migration:

```python
from app.storage import metadata_store

await metadata_store.initialize()
stats = await metadata_store.get_storage_stats()
print(stats)
```

## Integration with Phase 3

### Vector Store Integration

Files are automatically indexed in ChromaDB for semantic search:

```python
# When indexing, embeddings are generated and stored
await multimodal_indexer.index_image(
    file_id="img_123",
    vision_analysis={"description": "A cat..."},
    ...
)

# Search uses vector similarity
results = await multimodal_indexer.search_similar_files(
    query="cute animals",
    n_results=10
)
```

### Knowledge Graph Integration

Files are linked to concepts in Neo4j:

```python
# Concepts extracted from analysis are added to graph
await multimodal_indexer.index_document(
    file_id="doc_123",
    extracted_text="Machine learning uses neural networks...",
    ...
)

# Creates nodes: machine_learning, neural_networks
# Creates relationships: MENTIONED_IN
# Links via metadata: file_concept_links table

# Query from knowledge graph side
from app.knowledge_graph import kg_store

# Get files mentioning a concept
files = await metadata_store.get_files_for_concept("machine_learning")
```

### Hybrid Search Integration

Combine full-text search, vector search, and knowledge graph:

```python
from app.search import hybrid_search_engine

# Search across all modalities
results = await hybrid_search_engine.search(
    query="neural networks in computer vision",
    include_files=True,
    include_conversations=True
)

# Results include:
# - Text matches (FTS5)
# - Semantic matches (vector)
# - Related concepts (knowledge graph)
# - Linked files (multi-modal)
```

## Usage Examples

### Example 1: Image Upload and Analysis

```python
# Upload image
with open("photo.jpg", "rb") as f:
    image_data = f.read()

result = await file_manager.save_file(
    file_data=image_data,
    original_filename="photo.jpg",
    file_type="image",
    session_id="session_001"
)

# Analyze with vision API (Google Vision, AWS Rekognition, etc.)
# This is your external vision API call
vision_result = analyze_image_with_api(image_data)

# Index the analysis
await multimodal_indexer.index_image(
    file_id=result["file_id"],
    vision_analysis=vision_result,
    session_id="session_001"
)

# Now searchable
results = await multimodal_indexer.search_similar_files(
    query="outdoor scenes with animals",
    file_type="image"
)
```

### Example 2: PDF Document Processing

```python
import PyPDF2

# Upload PDF
with open("paper.pdf", "rb") as f:
    pdf_data = f.read()

result = await file_manager.save_file(
    file_data=pdf_data,
    original_filename="paper.pdf",
    file_type="pdf",
    session_id="session_002"
)

# Extract text
pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Index document
await multimodal_indexer.index_document(
    file_id=result["file_id"],
    extracted_text=text,
    session_id="session_002",
    document_metadata={
        "pages": len(pdf_reader.pages),
        "title": pdf_reader.metadata.get('/Title', '')
    }
)
```

### Example 3: Audio Transcription

```python
# Upload audio
with open("recording.mp3", "rb") as f:
    audio_data = f.read()

result = await file_manager.save_file(
    file_data=audio_data,
    original_filename="recording.mp3",
    file_type="audio",
    session_id="session_003"
)

# Transcribe (using Whisper, Google Speech, etc.)
transcription = transcribe_audio(audio_data)

# Index audio
await multimodal_indexer.index_audio(
    file_id=result["file_id"],
    transcription=transcription,
    session_id="session_003",
    audio_metadata={"duration_seconds": 120}
)
```

## Configuration

Environment variables:

```bash
# Storage configuration
export STORAGE_BASE_DIR="./data/uploads"
export STORAGE_METADATA_DB="./data/storage_metadata.db"
export STORAGE_RETENTION_DAYS=30
export STORAGE_MAX_USER_GB=1.0

# Feature flags
export STORAGE_ENABLE_VECTOR_INDEXING=true
export STORAGE_ENABLE_KNOWLEDGE_GRAPH=true
export STORAGE_ENABLE_DEDUPLICATION=true
```

Python configuration:

```python
from app.storage import storage_config

# Update configuration
storage_config.max_storage_per_user_gb = 5.0
storage_config.retention_days = 60
storage_config.deduplication_enabled = True

# Add custom file type
from app.storage.config import FileTypeConfig

storage_config.file_types["csv"] = FileTypeConfig(
    max_size_mb=50,
    allowed_extensions=[".csv", ".tsv"],
    storage_path="data",
    retention_days=90,
    enable_deduplication=True
)
```

## Monitoring and Maintenance

### Storage Statistics

```python
stats = await file_manager.get_storage_stats()

print(f"Total files: {stats['total_files']}")
print(f"Total size: {stats['total_bytes'] / (1024**3):.2f} GB")
print(f"Unique sessions: {stats['unique_sessions']}")

# By type
for type_stat in stats['by_type']:
    print(f"{type_stat['file_type']}: {type_stat['count']} files")
```

### Cleanup

```python
# Dry run to see what would be deleted
stats = await file_manager.cleanup_old_files(dry_run=True)
print(f"Would delete {stats['deleted_count']} files")
print(f"Would free {stats['freed_mb']:.2f} MB")

# Actually delete
stats = await file_manager.cleanup_old_files(dry_run=False)
```

### Scheduled Cleanup

```python
import asyncio

async def scheduled_cleanup():
    while True:
        # Run cleanup every 24 hours
        await asyncio.sleep(storage_config.cleanup_interval_hours * 3600)

        try:
            stats = await file_manager.cleanup_old_files(dry_run=False)
            logger.info("cleanup_complete", **stats)
        except Exception as e:
            logger.error("cleanup_failed", error=str(e))

# Start in background
asyncio.create_task(scheduled_cleanup())
```

## Testing

Run Phase 4 tests:

```bash
# Run all storage tests
pytest tests/storage/ -v

# Run specific test files
pytest tests/storage/test_file_manager.py -v
pytest tests/storage/test_metadata_store.py -v
pytest tests/storage/test_indexer.py -v

# Run with coverage
pytest tests/storage/ --cov=app/storage --cov-report=html
```

## Performance Considerations

1. **Batch Operations**: Use batch methods when processing multiple files
2. **Async I/O**: All file operations are async for non-blocking I/O
3. **Chunking**: Large documents are chunked for better vector search
4. **Caching**: Embedding cache reduces redundant computations
5. **Indexes**: Database indexes optimize common queries

## Security

1. **Validation**: All files validated against type and size limits
2. **Sandboxing**: Files stored in isolated directory structure
3. **Hash Verification**: SHA256 hashes prevent tampering
4. **Access Control**: Session-based file isolation (implement as needed)
5. **Cleanup**: Automatic cleanup prevents unbounded growth

## Troubleshooting

### Issue: Files not being indexed

Check initialization:

```python
# Ensure all components initialized
await file_manager.initialize()
await metadata_store.initialize()
await multimodal_indexer.initialize()

# Check status
info = embedding_generator.get_model_info()
print(f"Embeddings: {info['status']}")
```

### Issue: Deduplication not working

Check configuration:

```python
print(f"Deduplication enabled: {storage_config.deduplication_enabled}")
```

Verify hash calculation:

```python
import hashlib
file_hash = hashlib.sha256(file_data).hexdigest()
existing = await metadata_store.get_file_by_hash(file_hash)
```

### Issue: Cleanup not removing files

Check retention policy:

```python
config = storage_config.get_file_type_config("image")
print(f"Retention days: {config.retention_days}")

# Check old files
old_files = await metadata_store.get_old_files(days=30)
print(f"Old files: {len(old_files)}")
```

## Next Steps

- **Phase 5**: Real-time processing and streaming
- **Phase 6**: Advanced analytics and reporting
- **API Integration**: REST API for file uploads
- **Web UI**: File management interface
- **Advanced Analysis**: OCR, NER, object detection

## References

- ChromaDB: https://docs.trychroma.com/
- Neo4j: https://neo4j.com/docs/
- SQLite FTS5: https://www.sqlite.org/fts5.html
- Async I/O: https://docs.python.org/3/library/asyncio.html
