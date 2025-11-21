# Phase 4 Implementation Guide: Multi-Modal Capabilities

**Version:** 1.0.0
**Date:** 2025-01-21
**Status:** Specification (Ready for Implementation)

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Vision Analysis Setup](#vision-analysis-setup)
4. [Document Processing](#document-processing)
5. [File Upload System](#file-upload-system)
6. [Storage and Indexing](#storage-and-indexing)
7. [Integration with Existing Agents](#integration-with-existing-agents)
8. [Configuration](#configuration)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Phase 4 extends the Learning Voice Agent with multi-modal capabilities, enabling processing of images, documents, and rich media alongside voice conversations.

### Key Features

- **Vision Analysis** (Claude 3.5 Sonnet Vision) for image understanding
- **Document Processing** (PDF, DOCX, TXT, MD) with text extraction
- **File Upload API** with validation and storage
- **Multi-Modal Indexing** in vector database
- **Semantic Search** across text, images, and documents
- **RAG Enhancement** with multi-modal context

### Benefits

- **Enhanced Learning**: Capture visual diagrams, screenshots, and documents
- **Richer Context**: Combine voice, text, and visual information
- **Document Analysis**: Extract insights from PDFs, research papers, notes
- **Image Understanding**: Describe diagrams, charts, and visual concepts
- **Knowledge Integration**: Connect visual and textual information semantically

### Performance Targets

- **Image Upload**: < 2 seconds (including validation)
- **Vision Analysis**: < 3 seconds per image
- **Document Upload**: < 1 second per MB
- **Document Processing**: < 5 seconds per page
- **Multi-Modal Search**: < 300ms
- **File Retrieval**: < 100ms

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 4: MULTI-MODAL SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Input Layer                           │   │
│  ├──────────────┬──────────────┬──────────────────────────┤   │
│  │    Voice     │    Image     │      Document            │   │
│  │   (Audio)    │  (PNG/JPG)   │   (PDF/DOCX/TXT)         │   │
│  └──────┬───────┴──────┬───────┴────────┬─────────────────┘   │
│         │              │                │                       │
│         ▼              ▼                ▼                       │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────┐   │
│  │  Audio   │   │    Image     │   │    Document         │   │
│  │ Pipeline │   │  Processor   │   │   Processor         │   │
│  │(Whisper) │   │  • Validate  │   │   • PDF Extract     │   │
│  └────┬─────┘   │  • Resize    │   │   • DOCX Parse      │   │
│       │         │  • Format    │   │   • Text Chunk      │   │
│       │         └──────┬───────┘   └─────────┬───────────┘   │
│       │                │                     │                 │
│       └────────────────┴─────────────────────┘                 │
│                        │                                        │
│                        ▼                                        │
│         ┌─────────────────────────────────┐                   │
│         │      Vision Analyzer            │                   │
│         │  (Claude 3.5 Sonnet Vision)     │                   │
│         │  • Image Understanding          │                   │
│         │  • OCR Extraction               │                   │
│         │  • Diagram Analysis             │                   │
│         │  • Chart Interpretation         │                   │
│         └──────────────┬──────────────────┘                   │
│                        │                                        │
│                        ▼                                        │
│         ┌─────────────────────────────────┐                   │
│         │      File Manager               │                   │
│         │  • Organize by type/date        │                   │
│         │  • Hash-based deduplication     │                   │
│         │  • Metadata storage             │                   │
│         │  • Cleanup policies             │                   │
│         └──────────────┬──────────────────┘                   │
│                        │                                        │
│                        ▼                                        │
│         ┌─────────────────────────────────┐                   │
│         │   Multi-Modal Indexer           │                   │
│         │  • Generate embeddings          │                   │
│         │  • Store in ChromaDB            │                   │
│         │  • Link to knowledge graph      │                   │
│         │  • Enable hybrid search         │                   │
│         └──────────────┬──────────────────┘                   │
│                        │                                        │
│         ┌──────────────┴──────────────────────────┐           │
│         │                                          │           │
│         ▼                                          ▼           │
│  ┌──────────────┐                         ┌──────────────┐   │
│  │   Vector DB  │                         │  File Store  │   │
│  │  (ChromaDB)  │                         │ (Local/S3)   │   │
│  │              │                         │              │   │
│  │  Embeddings  │                         │   Images     │   │
│  │  + Metadata  │                         │   Documents  │   │
│  └──────────────┘                         └──────────────┘   │
│                                                                 │
│         ┌──────────────────────────────────────────┐          │
│         │      RAG System Integration              │          │
│         │  • Multi-modal context retrieval         │          │
│         │  • Text + Image + Document context       │          │
│         │  • Enhanced response generation          │          │
│         └──────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Image Upload Flow**
   ```
   Client → Upload API → Validate → Process → Store File
                                    ↓
                              Vision Analysis
                                    ↓
                              Extract Text/Data
                                    ↓
                              Generate Embedding
                                    ↓
                              Index in ChromaDB
                                    ↓
                           Link to Knowledge Graph
   ```

2. **Document Upload Flow**
   ```
   Client → Upload API → Validate → Extract Text
                                    ↓
                              Chunk Content
                                    ↓
                              Store File + Chunks
                                    ↓
                              Generate Embeddings
                                    ↓
                              Index All Chunks
                                    ↓
                           Update Search Index
   ```

3. **Multi-Modal Conversation**
   ```
   User Query → Analyze Intent
              ↓
         Retrieve Context (Text + Images + Docs)
              ↓
         Build Multi-Modal Prompt
              ↓
         Generate Response (Claude)
              ↓
         Include Citations (with thumbnails)
   ```

---

## Vision Analysis Setup

### Component Overview

The **VisionAnalyzer** uses Claude 3.5 Sonnet's vision capabilities to understand images.

### Installation

```bash
# Required dependencies
pip install anthropic Pillow python-magic-bin
```

### Module: `app.vision.vision_analyzer`

```python
"""
Vision analysis using Claude 3.5 Sonnet Vision API
Location: app/vision/vision_analyzer.py
"""
from anthropic import AsyncAnthropic
from PIL import Image
import base64
from io import BytesIO
from typing import Dict, Any, Optional

class VisionAnalyzer:
    """Analyze images using Claude Vision API"""

    MODEL = "claude-3-5-sonnet-20241022"
    MAX_TOKENS = 1024

    def __init__(self, api_key: Optional[str] = None):
        """Initialize vision analyzer

        Args:
            api_key: Anthropic API key (uses env var if None)
        """
        self.client = AsyncAnthropic(api_key=api_key)

    async def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        include_ocr: bool = False
    ) -> Dict[str, Any]:
        """Analyze image and extract information

        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            include_ocr: Whether to extract text from image

        Returns:
            Dict with keys:
                - analysis: Text description
                - ocr_text: Extracted text (if include_ocr=True)
                - dimensions: Image dimensions (width, height)
                - format: Image format (PNG, JPEG, etc.)
                - tokens_used: API token count
                - model: Model used
        """
        # Load and encode image
        with Image.open(image_path) as img:
            dimensions = img.size
            img_format = img.format

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format=img_format)
            image_data = base64.b64encode(buffer.getvalue()).decode()

        # Build prompt
        full_prompt = prompt
        if include_ocr:
            full_prompt += "\n\nAlso extract any text visible in the image."

        # Call Claude Vision API
        message = await self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{img_format.lower()}",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": full_prompt
                    }
                ]
            }]
        )

        analysis_text = message.content[0].text

        # Extract OCR text if requested
        ocr_text = None
        if include_ocr:
            ocr_text = self._extract_ocr_from_analysis(analysis_text)

        return {
            "analysis": analysis_text,
            "ocr_text": ocr_text,
            "dimensions": dimensions,
            "format": img_format,
            "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
            "model": self.MODEL
        }

    async def analyze_diagram(self, image_path: str) -> Dict[str, Any]:
        """Specialized analysis for diagrams and flowcharts

        Args:
            image_path: Path to diagram image

        Returns:
            Analysis result with structure description
        """
        prompt = """Analyze this diagram or flowchart:

        1. What type of diagram is this?
        2. What are the main components/nodes?
        3. How are they connected/related?
        4. What process or concept does it illustrate?
        5. Extract any labels or text.

        Provide a structured description."""

        return await self.analyze_image(
            image_path,
            prompt=prompt,
            include_ocr=True
        )

    async def compare_images(
        self,
        image_path1: str,
        image_path2: str
    ) -> Dict[str, Any]:
        """Compare two images and describe differences

        Args:
            image_path1: First image path
            image_path2: Second image path

        Returns:
            Comparison analysis
        """
        # Analyze both images
        result1 = await self.analyze_image(image_path1)
        result2 = await self.analyze_image(image_path2)

        # Generate comparison prompt
        prompt = f"""Compare these two descriptions:

        Image 1: {result1['analysis']}
        Image 2: {result2['analysis']}

        What are the key similarities and differences?
        """

        # Note: In full implementation, would send both images
        # to Claude in single request for better comparison

        return {
            "image1_analysis": result1["analysis"],
            "image2_analysis": result2["analysis"],
            "comparison": "Comparison result here",
            "tokens_used": result1["tokens_used"] + result2["tokens_used"]
        }

    def _extract_ocr_from_analysis(self, analysis: str) -> Optional[str]:
        """Extract OCR text from analysis response

        Args:
            analysis: Full analysis text

        Returns:
            Extracted text or None
        """
        # Parse analysis to find extracted text
        # This is a simple heuristic - adjust as needed
        if "text:" in analysis.lower():
            parts = analysis.lower().split("text:")
            if len(parts) > 1:
                return parts[1].strip()
        return None

# Singleton instance
vision_analyzer = VisionAnalyzer()
```

### Image Processor

**Module:** `app.vision.image_processor`

```python
"""
Image processing utilities
Location: app/vision/image_processor.py
"""
from PIL import Image
import magic
from pathlib import Path
from typing import Tuple, Optional
import hashlib

class ImageProcessor:
    """Handle image validation, resizing, and format conversion"""

    SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "GIF", "WEBP"}
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_DIMENSION = 4096  # Max width/height
    THUMBNAIL_SIZE = (256, 256)

    @staticmethod
    def validate_image(file_path: str) -> Tuple[bool, str]:
        """Validate image file

        Args:
            file_path: Path to image

        Returns:
            (is_valid, error_message)
        """
        path = Path(file_path)

        # Check existence
        if not path.exists():
            return False, "File not found"

        # Check size
        if path.stat().st_size > ImageProcessor.MAX_SIZE:
            return False, f"File too large (max {ImageProcessor.MAX_SIZE / 1024 / 1024}MB)"

        # Check MIME type
        mime = magic.from_file(str(path), mime=True)
        if not mime.startswith("image/"):
            return False, f"Not an image file: {mime}"

        # Try opening with PIL
        try:
            with Image.open(path) as img:
                # Check format
                if img.format not in ImageProcessor.SUPPORTED_FORMATS:
                    return False, f"Unsupported format: {img.format}"

                # Check dimensions
                if img.width > ImageProcessor.MAX_DIMENSION or img.height > ImageProcessor.MAX_DIMENSION:
                    return False, f"Image too large (max {ImageProcessor.MAX_DIMENSION}x{ImageProcessor.MAX_DIMENSION})"

        except Exception as e:
            return False, f"Invalid image: {str(e)}"

        return True, "Valid"

    @staticmethod
    def resize_if_needed(
        image_path: str,
        max_dimension: int = 2048,
        output_path: Optional[str] = None
    ) -> str:
        """Resize image if it exceeds max dimension

        Args:
            image_path: Input image path
            max_dimension: Maximum width/height
            output_path: Output path (overwrites input if None)

        Returns:
            Path to resized image
        """
        if output_path is None:
            output_path = image_path

        with Image.open(image_path) as img:
            # Check if resize needed
            if img.width <= max_dimension and img.height <= max_dimension:
                if output_path != image_path:
                    img.save(output_path)
                return output_path

            # Calculate new dimensions (maintain aspect ratio)
            ratio = min(max_dimension / img.width, max_dimension / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))

            # Resize
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            resized.save(output_path)

        return output_path

    @staticmethod
    def generate_thumbnail(
        image_path: str,
        size: Tuple[int, int] = THUMBNAIL_SIZE
    ) -> str:
        """Generate thumbnail

        Args:
            image_path: Source image
            size: Thumbnail size (width, height)

        Returns:
            Path to thumbnail
        """
        path = Path(image_path)
        thumbnail_path = path.parent / f"{path.stem}_thumb{path.suffix}"

        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path)

        return str(thumbnail_path)

    @staticmethod
    def compute_hash(image_path: str) -> str:
        """Compute image hash for deduplication

        Args:
            image_path: Image to hash

        Returns:
            SHA256 hash
        """
        hasher = hashlib.sha256()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def extract_exif(image_path: str) -> dict:
        """Extract EXIF metadata

        Args:
            image_path: Image file

        Returns:
            EXIF data dict
        """
        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif:
                return {
                    key: str(value)
                    for key, value in exif.items()
                }
        return {}

# Singleton
image_processor = ImageProcessor()
```

---

## Document Processing

### PDF Processing

**Module:** `app.documents.document_processor`

```python
"""
Document processing for PDFs, DOCX, and text files
Location: app/documents/document_processor.py
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import docx
import magic
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Document text chunk for indexing"""
    text: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    metadata: Optional[Dict[str, Any]] = None

class DocumentProcessor:
    """Process and extract text from various document formats"""

    CHUNK_SIZE = 1000  # Characters per chunk
    CHUNK_OVERLAP = 200  # Overlap between chunks

    SUPPORTED_FORMATS = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "text/plain": "txt",
        "text/markdown": "md"
    }

    async def process_document(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """Process document and extract text

        Args:
            file_path: Path to document

        Returns:
            Dict with:
                - text: Full extracted text
                - chunks: List of DocumentChunk objects
                - metadata: Document metadata
                - page_count: Number of pages (if applicable)
                - format: Document format
        """
        # Detect format
        mime_type = magic.from_file(file_path, mime=True)
        format_type = self.SUPPORTED_FORMATS.get(mime_type)

        if not format_type:
            raise ValueError(f"Unsupported document format: {mime_type}")

        # Process based on format
        if format_type == "pdf":
            return await self._process_pdf(file_path)
        elif format_type == "docx":
            return await self._process_docx(file_path)
        elif format_type in ("txt", "md"):
            return await self._process_text(file_path)
        else:
            raise ValueError(f"Unknown format: {format_type}")

    async def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF document"""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)

            # Extract text from all pages
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                pages_text.append({
                    "page": i + 1,
                    "text": text
                })

            # Combine all text
            full_text = "\n\n".join(p["text"] for p in pages_text)

            # Create chunks
            chunks = self._chunk_text(full_text, pages_text)

            # Extract metadata
            metadata = {
                "author": reader.metadata.get("/Author", ""),
                "title": reader.metadata.get("/Title", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            } if reader.metadata else {}

            return {
                "text": full_text,
                "chunks": chunks,
                "metadata": metadata,
                "page_count": page_count,
                "format": "pdf"
            }

    async def _process_docx(self, file_path: str) -> Dict[str, Any]:
        """Process DOCX document"""
        doc = docx.Document(file_path)

        # Extract paragraphs
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)

        # Create chunks
        chunks = self._chunk_text(full_text)

        # Extract properties
        props = doc.core_properties
        metadata = {
            "author": props.author or "",
            "title": props.title or "",
            "subject": props.subject or "",
            "keywords": props.keywords or "",
            "created": str(props.created) if props.created else "",
            "modified": str(props.modified) if props.modified else ""
        }

        return {
            "text": full_text,
            "chunks": chunks,
            "metadata": metadata,
            "page_count": None,
            "format": "docx"
        }

    async def _process_text(self, file_path: str) -> Dict[str, Any]:
        """Process plain text or markdown"""
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        chunks = self._chunk_text(full_text)

        return {
            "text": full_text,
            "chunks": chunks,
            "metadata": {},
            "page_count": None,
            "format": Path(file_path).suffix[1:]
        }

    def _chunk_text(
        self,
        text: str,
        pages: Optional[List[Dict]] = None
    ) -> List[DocumentChunk]:
        """Split text into chunks for indexing

        Args:
            text: Full text
            pages: Optional page information

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > self.CHUNK_SIZE * 0.8:
                    end = start + last_period + 1
                    chunk_text = text[start:end]

            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                metadata={"start": start, "end": end}
            ))

            start = end - self.CHUNK_OVERLAP
            chunk_index += 1

        return chunks

# Singleton
document_processor = DocumentProcessor()
```

---

## File Upload System

### Upload Endpoints

**Module:** `app.api.upload_routes`

```python
"""
File upload API endpoints
Location: app/api/upload_routes.py (to be added to main.py)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path
import shutil

from app.vision.vision_analyzer import vision_analyzer
from app.vision.image_processor import image_processor
from app.documents.document_processor import document_processor
from app.storage.file_manager import file_manager
from app.storage.multimodal_indexer import multimodal_indexer

router = APIRouter(prefix="/api/upload", tags=["upload"])

@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    description: Optional[str] = Form(None),
    analyze: bool = Form(True)
):
    """Upload and analyze an image

    Request:
        - file: Image file (PNG, JPEG, GIF, WEBP)
        - session_id: Session identifier
        - description: Optional description
        - analyze: Whether to run vision analysis

    Response:
        {
            "file_id": "uuid",
            "url": "/files/images/uuid.png",
            "thumbnail_url": "/files/thumbnails/uuid_thumb.png",
            "analysis": {...} or null,
            "metadata": {
                "format": "PNG",
                "dimensions": [800, 600],
                "size_bytes": 123456
            }
        }
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Save temporary file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Validate image
        is_valid, error = image_processor.validate_image(temp_path)
        if not is_valid:
            raise HTTPException(400, error)

        # Resize if needed
        image_processor.resize_if_needed(temp_path, max_dimension=2048)

        # Generate thumbnail
        thumb_path = image_processor.generate_thumbnail(temp_path)

        # Store file
        file_id = await file_manager.save_file(
            temp_path,
            file_type="image",
            session_id=session_id,
            metadata={
                "original_name": file.filename,
                "description": description
            }
        )

        # Vision analysis
        analysis = None
        if analyze:
            analysis = await vision_analyzer.analyze_image(
                temp_path,
                prompt=description or "Describe this image in detail."
            )

            # Index in vector database
            await multimodal_indexer.index_image(
                file_id=file_id,
                image_path=temp_path,
                analysis=analysis,
                metadata={
                    "session_id": session_id,
                    "description": description
                }
            )

        # Get file info
        file_info = await file_manager.get_file_info(file_id)

        return JSONResponse({
            "file_id": file_id,
            "url": f"/files/images/{file_id}",
            "thumbnail_url": f"/files/thumbnails/{file_id}_thumb",
            "analysis": analysis,
            "metadata": file_info.get("metadata", {})
        })

    finally:
        # Cleanup temp files
        Path(temp_path).unlink(missing_ok=True)

@router.post("/document")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    description: Optional[str] = Form(None)
):
    """Upload and process a document

    Request:
        - file: Document file (PDF, DOCX, TXT, MD)
        - session_id: Session identifier
        - description: Optional description

    Response:
        {
            "file_id": "uuid",
            "url": "/files/documents/uuid.pdf",
            "text": "Extracted text...",
            "chunks": [{...}],
            "metadata": {...},
            "page_count": 10
        }
    """
    # Save temporary file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process document
        result = await document_processor.process_document(temp_path)

        # Store file
        file_id = await file_manager.save_file(
            temp_path,
            file_type="document",
            session_id=session_id,
            metadata={
                "original_name": file.filename,
                "description": description,
                "page_count": result.get("page_count"),
                "format": result.get("format")
            }
        )

        # Index all chunks
        await multimodal_indexer.index_document(
            file_id=file_id,
            chunks=result["chunks"],
            metadata={
                "session_id": session_id,
                "description": description,
                **result.get("metadata", {})
            }
        )

        return JSONResponse({
            "file_id": file_id,
            "url": f"/files/documents/{file_id}",
            "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
            "chunks": len(result["chunks"]),
            "metadata": result.get("metadata", {}),
            "page_count": result.get("page_count")
        })

    finally:
        Path(temp_path).unlink(missing_ok=True)

@router.get("/files/{file_type}/{file_id}")
async def get_file(file_type: str, file_id: str):
    """Retrieve uploaded file"""
    file_path = await file_manager.get_file_path(file_id)
    if not file_path:
        raise HTTPException(404, "File not found")

    from fastapi.responses import FileResponse
    return FileResponse(file_path)

@router.post("/conversation/multimodal")
async def multimodal_conversation(
    text: str = Form(...),
    session_id: str = Form(...),
    file_ids: Optional[List[str]] = Form(None)
):
    """Enhanced conversation with multi-modal context

    Request:
        - text: User question/input
        - session_id: Session ID
        - file_ids: Optional list of file IDs for context

    Response:
        {
            "response": "AI response...",
            "sources": [
                {"type": "text", "content": "..."},
                {"type": "image", "file_id": "...", "analysis": "..."},
                {"type": "document", "file_id": "...", "excerpt": "..."}
            ]
        }
    """
    # Retrieve multi-modal context
    context = await multimodal_indexer.retrieve_context(
        query=text,
        session_id=session_id,
        file_ids=file_ids,
        k=5
    )

    # Generate response with context
    # (Integration with ConversationAgent)
    response = "Multi-modal response generation here"

    return JSONResponse({
        "response": response,
        "sources": context.get("sources", [])
    })
```

---

## Storage and Indexing

### File Manager

**Module:** `app.storage.file_manager`

```python
"""
File storage and management
Location: app/storage/file_manager.py
"""
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
import shutil
import json
from datetime import datetime

class FileManager:
    """Manage uploaded files with organization and deduplication"""

    BASE_DIR = Path("data/uploads")

    def __init__(self):
        """Initialize file manager"""
        self.base_dir = self.BASE_DIR
        self._ensure_directories()

    def _ensure_directories(self):
        """Create storage directories"""
        (self.base_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "documents").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "metadata").mkdir(parents=True, exist_ok=True)

    async def save_file(
        self,
        source_path: str,
        file_type: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save file with organization

        Args:
            source_path: Source file path
            file_type: "image" or "document"
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            File ID
        """
        # Generate ID
        file_id = str(uuid.uuid4())

        # Determine destination
        suffix = Path(source_path).suffix
        dest_dir = self.base_dir / file_type / session_id[:8]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{file_id}{suffix}"

        # Copy file
        shutil.copy2(source_path, dest_path)

        # Save metadata
        meta = {
            "file_id": file_id,
            "file_type": file_type,
            "session_id": session_id,
            "path": str(dest_path),
            "size": dest_path.stat().st_size,
            "created": datetime.utcnow().isoformat(),
            **(metadata or {})
        }

        meta_path = self.base_dir / "metadata" / f"{file_id}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return file_id

    async def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file path by ID"""
        meta_path = self.base_dir / "metadata" / f"{file_id}.json"
        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        return meta.get("path")

    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata"""
        meta_path = self.base_dir / "metadata" / f"{file_id}.json"
        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            return json.load(f)

    async def delete_file(self, file_id: str) -> bool:
        """Delete file and metadata"""
        # Get metadata
        info = await self.get_file_info(file_id)
        if not info:
            return False

        # Delete file
        file_path = Path(info["path"])
        file_path.unlink(missing_ok=True)

        # Delete metadata
        meta_path = self.base_dir / "metadata" / f"{file_id}.json"
        meta_path.unlink(missing_ok=True)

        return True

# Singleton
file_manager = FileManager()
```

### Multi-Modal Indexer

**Module:** `app.storage.multimodal_indexer`

```python
"""
Multi-modal content indexing in vector database
Location: app/storage/multimodal_indexer.py
"""
from typing import List, Dict, Any, Optional
from app.vector.vector_store import vector_store
from app.vector.embeddings import embedding_generator

class MultiModalIndexer:
    """Index images and documents in vector database"""

    COLLECTION_IMAGES = "multimodal_images"
    COLLECTION_DOCUMENTS = "multimodal_documents"

    async def initialize(self):
        """Initialize collections"""
        await vector_store.initialize()
        # Collections created automatically by ChromaDB

    async def index_image(
        self,
        file_id: str,
        image_path: str,
        analysis: Dict[str, Any],
        metadata: Optional[Dict] = None
    ):
        """Index image analysis in vector store

        Args:
            file_id: File identifier
            image_path: Path to image
            analysis: Vision analysis result
            metadata: Additional metadata
        """
        # Use analysis text for embedding
        text = f"{analysis.get('analysis', '')} {metadata.get('description', '')}"

        await vector_store.add_embedding(
            collection_name=self.COLLECTION_IMAGES,
            text=text,
            document_id=file_id,
            metadata={
                "file_id": file_id,
                "type": "image",
                "path": image_path,
                "dimensions": analysis.get("dimensions"),
                "format": analysis.get("format"),
                **( metadata or {})
            }
        )

    async def index_document(
        self,
        file_id: str,
        chunks: List[Any],
        metadata: Optional[Dict] = None
    ):
        """Index document chunks in vector store

        Args:
            file_id: File identifier
            chunks: Document chunks
            metadata: Additional metadata
        """
        # Index each chunk
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_id": file_id,
                "type": "document",
                "chunk_index": i,
                **(metadata or {})
            }
            for i in range(len(chunks))
        ]

        await vector_store.add_batch(
            collection_name=self.COLLECTION_DOCUMENTS,
            texts=texts,
            document_ids=chunk_ids,
            metadatas=metadatas
        )

    async def retrieve_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """Retrieve multi-modal context for query

        Args:
            query: Search query
            session_id: Filter by session
            file_ids: Filter by specific files
            k: Number of results

        Returns:
            Context with sources
        """
        sources = []

        # Search images
        image_results = await vector_store.search(
            collection_name=self.COLLECTION_IMAGES,
            query_text=query,
            n_results=k
        )

        for result in image_results.get("documents", []):
            sources.append({
                "type": "image",
                "content": result.get("text"),
                "metadata": result.get("metadata")
            })

        # Search documents
        doc_results = await vector_store.search(
            collection_name=self.COLLECTION_DOCUMENTS,
            query_text=query,
            n_results=k
        )

        for result in doc_results.get("documents", []):
            sources.append({
                "type": "document",
                "content": result.get("text"),
                "metadata": result.get("metadata")
            })

        return {"sources": sources}

# Singleton
multimodal_indexer = MultiModalIndexer()
```

---

## Integration with Existing Agents

### ConversationAgent Enhancement

To integrate multi-modal capabilities with the existing `ConversationAgent`:

```python
# In app/agents/conversation_agent.py

from app.storage.multimodal_indexer import multimodal_indexer

class ConversationAgent:
    # ... existing code ...

    async def process_with_multimodal(
        self,
        user_input: str,
        session_id: str,
        file_ids: Optional[List[str]] = None
    ):
        """Process input with multi-modal context

        Args:
            user_input: User text input
            session_id: Session ID
            file_ids: Optional file IDs for context
        """
        # Retrieve multi-modal context
        context = await multimodal_indexer.retrieve_context(
            query=user_input,
            session_id=session_id,
            file_ids=file_ids,
            k=3
        )

        # Build enhanced prompt
        context_text = self._build_multimodal_prompt(context)

        # Generate response
        response = await self.process(
            user_input=user_input,
            context=context_text
        )

        return {
            "response": response,
            "sources": context.get("sources", [])
        }

    def _build_multimodal_prompt(self, context: Dict) -> str:
        """Build prompt with multi-modal context"""
        parts = []

        for source in context.get("sources", []):
            if source["type"] == "image":
                parts.append(f"[Image context: {source['content']}]")
            elif source["type"] == "document":
                parts.append(f"[Document excerpt: {source['content']}]")

        return "\n\n".join(parts)
```

---

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Phase 4: Multi-Modal Configuration

# Vision Analysis
CLAUDE_VISION_MODEL=claude-3-5-sonnet-20241022
VISION_MAX_TOKENS=1024

# File Upload
MAX_FILE_SIZE=10485760  # 10 MB
MAX_IMAGE_DIMENSION=4096
THUMBNAIL_SIZE=256

# Document Processing
DOC_CHUNK_SIZE=1000
DOC_CHUNK_OVERLAP=200

# Storage
UPLOAD_DIR=data/uploads
FILE_RETENTION_DAYS=90  # Auto-cleanup after 90 days
```

### Configuration Module

**File:** `app/multimodal/config.py`

```python
"""
Multi-modal configuration
Location: app/multimodal/config.py
"""
from pydantic import BaseModel
from pathlib import Path
import os

class MultiModalConfig(BaseModel):
    """Configuration for multi-modal features"""

    # Vision
    claude_vision_model: str = "claude-3-5-sonnet-20241022"
    vision_max_tokens: int = 1024

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    max_image_dimension: int = 4096
    thumbnail_size: tuple = (256, 256)

    # Document Processing
    doc_chunk_size: int = 1000
    doc_chunk_overlap: int = 200

    # Storage
    upload_dir: Path = Path("data/uploads")
    file_retention_days: int = 90

    # Supported formats
    supported_image_formats: set = {"PNG", "JPEG", "JPG", "GIF", "WEBP"}
    supported_doc_formats: set = {"pdf", "docx", "txt", "md"}

    class Config:
        env_prefix = ""

# Singleton
multimodal_config = MultiModalConfig()
```

---

## Performance Tuning

### Image Processing Optimization

1. **Resize large images before upload** (client-side)
2. **Use lazy loading** for thumbnails
3. **Implement caching** for vision analysis results
4. **Batch process** multiple images together

### Document Processing Optimization

1. **Stream large documents** instead of loading entirely
2. **Parallel chunk processing** using asyncio
3. **Cache extracted text** to avoid re-processing
4. **Index incrementally** as pages are processed

### Vector Database Optimization

1. **Batch insert** document chunks
2. **Use appropriate embedding dimension** (384 for all-MiniLM-L6-v2)
3. **Configure ChromaDB** for SSD storage
4. **Implement result caching** for common queries

---

## Troubleshooting

### Common Issues

1. **"Image validation failed"**
   - Check supported formats
   - Verify file size < 10 MB
   - Ensure image dimensions < 4096px

2. **"Vision analysis timeout"**
   - Reduce image size
   - Check API rate limits
   - Verify API key is valid

3. **"Document extraction failed"**
   - Ensure PDF is not encrypted
   - Check document format is supported
   - Verify file is not corrupted

4. **"File not found"**
   - Check upload directory permissions
   - Verify file_id is correct
   - Ensure storage path is accessible

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing

See [PHASE4_TESTING_GUIDE.md](PHASE4_TESTING_GUIDE.md) for comprehensive testing strategies.

---

## Next Steps

1. **Implement all modules** as specified
2. **Create test suite** (150+ tests)
3. **Deploy to staging** and validate
4. **Performance benchmark** all operations
5. **Production deployment** with monitoring

For API reference, see [PHASE4_API_REFERENCE.md](PHASE4_API_REFERENCE.md).

For usage examples, see [PHASE4_USAGE_EXAMPLES.md](PHASE4_USAGE_EXAMPLES.md).

---

**Phase 4 Status:** Ready for implementation following SPARC methodology.
