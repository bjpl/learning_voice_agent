# Phase 4: Multi-Modal Architecture (Vision, Documents, Audio)

**Date:** 2025-11-21
**Status:** Architecture Design
**Methodology:** SPARC
**Phase:** Architecture
**Target:** Production-ready multi-modal learning system

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Specification](#specification)
3. [Architecture Overview](#architecture-overview)
4. [Vision Analysis System](#vision-analysis-system)
5. [Document Processing Pipeline](#document-processing-pipeline)
6. [Multi-Modal Storage](#multi-modal-storage)
7. [API Design](#api-design)
8. [Integration Strategy](#integration-strategy)
9. [Performance Targets](#performance-targets)
10. [Security Considerations](#security-considerations)
11. [Migration Strategy](#migration-strategy)
12. [Implementation Plan](#implementation-plan)

---

## Executive Summary

Phase 4 extends the learning_voice_agent v2.0 system with **multi-modal capabilities** for processing images, documents, and maintaining unified semantic memory across all modalities.

### Current State (Phase 1-3 Complete)
- ✅ **Phase 1:** Logging, resilience, metrics, testing (87% coverage)
- ✅ **Phase 2:** Multi-agent system (5 agents, LangGraph orchestration)
- ✅ **Phase 3:** Vector memory, RAG, hybrid search, knowledge graph

### Phase 4 Objectives
1. **Vision Analysis:** Process images/screenshots with Claude 3.5 Sonnet vision
2. **Document Processing:** Extract and index PDF, DOCX, TXT, MD files
3. **Multi-Modal Storage:** Unified storage for files, metadata, and embeddings
4. **API Extensions:** Upload and process multi-modal content
5. **RAG Integration:** Retrieve relevant context from images and documents

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Vision API** | Claude 3.5 Sonnet (native) | Already in use, unified model, no extra cost |
| **PDF Parser** | PyMuPDF (fitz) | Fastest, most reliable, Apache license |
| **DOCX Parser** | python-docx | Standard library, simple API |
| **Image Storage** | Local filesystem (./data/uploads/) | Simple, Railway-compatible, migrate to S3 later |
| **Embeddings** | Text-only (Sentence Transformers) | Reuse existing pipeline, CLIP optional in Phase 5 |
| **Integration** | Extend ConversationAgent | Minimal changes, unified interface |

---

## Specification

### Requirements

#### Functional Requirements
1. **FR1:** Accept image uploads (PNG, JPEG, GIF, WebP) up to 5MB
2. **FR2:** Accept document uploads (PDF, DOCX, TXT, MD) up to 10MB
3. **FR3:** Analyze images with Claude 3.5 Sonnet vision API
4. **FR4:** Extract text from documents with proper formatting
5. **FR5:** Generate embeddings for extracted text
6. **FR6:** Store files, metadata, and embeddings persistently
7. **FR7:** Retrieve relevant multi-modal context in conversations
8. **FR8:** Support searching across text, image descriptions, and documents

#### Non-Functional Requirements
1. **NFR1:** Image processing < 2 seconds per image
2. **NFR2:** Document processing < 5 seconds per page
3. **NFR3:** 99.9% uptime for upload endpoints
4. **NFR4:** File validation and virus scanning (if production)
5. **NFR5:** MIME type validation and content verification
6. **NFR6:** Graceful degradation if storage is full

### Out of Scope (Phase 5+)
- Video processing (deferred to Phase 5)
- Real-time image generation (deferred to Phase 5)
- CLIP/ImageBind multi-modal embeddings (optional enhancement)
- OCR for handwritten text (use Claude vision instead)
- Audio file uploads (already handled via Whisper pipeline)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  MULTI-MODAL INPUT LAYER                     │
├──────────────┬──────────────┬──────────────┬────────────────┤
│    Images    │     PDFs     │    DOCX      │   Text/MD      │
│ (PNG, JPEG,  │ (PyMuPDF)    │ (python-docx)│  (Built-in)    │
│  GIF, WebP)  │              │              │                │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       v              v              v                v
┌─────────────────────────────────────────────────────────────┐
│                  PROCESSING LAYER                            │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Vision       │ PDF          │ DOCX         │ Text           │
│ Analysis     │ Extraction   │ Extraction   │ Processing     │
│              │              │              │                │
│ Claude 3.5   │ PyMuPDF      │ python-docx  │ Direct         │
│ Sonnet       │ (fitz)       │              │                │
│              │              │              │                │
│ • Image      │ • Text       │ • Paragraphs │ • Markdown     │
│   description│   extraction │ • Tables     │ • Plain text   │
│ • OCR        │ • Metadata   │ • Headers    │                │
│ • Analysis   │ • Page info  │ • Lists      │                │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       v              v              v                v
┌─────────────────────────────────────────────────────────────┐
│               EMBEDDING GENERATION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  Sentence Transformers (all-MiniLM-L6-v2, 384-dim)          │
│  • Convert vision descriptions to embeddings                 │
│  • Convert extracted document text to embeddings             │
│  • Maintain unified vector space for all modalities          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│              STORAGE & INDEXING LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐│
│  │ File Storage  │  │ Metadata DB   │  │  Vector Store    ││
│  │               │  │               │  │                  ││
│  │ ./data/       │  │ SQLite        │  │  ChromaDB        ││
│  │ uploads/      │  │ multimodal_   │  │  multimodal      ││
│  │ ├── images/   │  │ files table   │  │  collection      ││
│  │ ├── documents/│  │               │  │                  ││
│  │ └── temp/     │  │ • file_id     │  │  • embeddings    ││
│  │               │  │ • file_path   │  │  • metadata      ││
│  │               │  │ • mime_type   │  │  • ref to files  ││
│  │               │  │ • size        │  │                  ││
│  │               │  │ • extracted   │  │                  ││
│  │               │  │   _text       │  │                  ││
│  │               │  │ • vision_     │  │                  ││
│  │               │  │   analysis    │  │                  ││
│  └───────────────┘  └───────────────┘  └──────────────────┘│
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                  RAG & RETRIEVAL LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  • Multi-modal hybrid search (text + image descriptions)     │
│  • Context augmentation with relevant files                  │
│  • ConversationAgent integration                             │
│  • Knowledge graph updates with document concepts            │
└─────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           NEW: Multi-Modal Endpoints                      │ │
│  │  • POST /api/upload/image                                 │ │
│  │  • POST /api/upload/document                              │ │
│  │  • GET  /api/files/{file_id}                              │ │
│  │  • POST /api/analyze/image                                │ │
│  │  • POST /api/conversation (enhanced with image support)   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         NEW: Multi-Modal Processing Modules               │ │
│  │                                                            │ │
│  │  ┌───────────────────┐  ┌────────────────────────┐       │ │
│  │  │ VisionProcessor   │  │ DocumentProcessor      │       │ │
│  │  │                   │  │                        │       │ │
│  │  │ • analyze_image() │  │ • extract_from_pdf()   │       │ │
│  │  │ • validate_image()│  │ • extract_from_docx()  │       │ │
│  │  │ • resize_image()  │  │ • extract_from_text()  │       │ │
│  │  └───────────────────┘  └────────────────────────┘       │ │
│  │                                                            │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │ FileStorageManager                                  │  │ │
│  │  │                                                      │  │ │
│  │  │ • save_file()                                        │  │ │
│  │  │ • get_file()                                         │  │ │
│  │  │ • delete_file()                                      │  │ │
│  │  │ • generate_thumbnail()  # For images                │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │      ENHANCED: Existing Components                        │ │
│  │                                                            │ │
│  │  • ConversationAgent (+ vision support)                   │ │
│  │  • VectorStore (multimodal collection)                    │ │
│  │  • HybridSearch (multi-modal queries)                     │ │
│  │  • Database (new multimodal_files table)                  │ │
│  │  • RAGRetriever (file context retrieval)                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow: Image Upload and Analysis

```
User uploads image
       │
       v
┌────────────────────────────────────────────────────────────────┐
│ 1. API Endpoint: POST /api/upload/image                        │
│    • Validate file size (< 5MB)                                │
│    • Validate MIME type (PNG, JPEG, GIF, WebP)                 │
│    • Generate unique file_id                                   │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────────────┐
│ 2. FileStorageManager.save_file()                              │
│    • Save to ./data/uploads/images/{file_id}.ext               │
│    • Generate thumbnail (200x200)                              │
│    • Calculate file hash                                       │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────────────┐
│ 3. VisionProcessor.analyze_image()                             │
│    • Read image file                                           │
│    • Encode to base64                                          │
│    • Call Claude 3.5 Sonnet with vision prompt:               │
│      "Describe this image in detail. Include:                  │
│       - Main subjects and objects                              │
│       - Text content (OCR)                                     │
│       - Context and setting                                    │
│       - Notable details"                                       │
│    • Parse structured response                                 │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────────────┐
│ 4. Generate Embeddings                                         │
│    • Use EmbeddingGenerator.generate_embedding()               │
│    • Input: vision analysis text                               │
│    • Model: all-MiniLM-L6-v2 (384-dim)                        │
│    • Store embedding vector                                    │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────────────┐
│ 5. Persist to Storage                                          │
│    ┌────────────────────────────────────────────────────────┐ │
│    │ SQLite: multimodal_files table                         │ │
│    │ INSERT INTO multimodal_files VALUES (                  │ │
│    │   file_id, file_path, mime_type, size,                 │ │
│    │   extracted_text, vision_analysis, metadata            │ │
│    │ )                                                       │ │
│    └────────────────────────────────────────────────────────┘ │
│    ┌────────────────────────────────────────────────────────┐ │
│    │ ChromaDB: multimodal collection                        │ │
│    │ add(                                                    │ │
│    │   ids=[file_id],                                        │ │
│    │   embeddings=[embedding],                               │ │
│    │   documents=[vision_analysis],                          │ │
│    │   metadatas=[{file_type: 'image', ...}]                │ │
│    │ )                                                       │ │
│    └────────────────────────────────────────────────────────┘ │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────────────┐
│ 6. Return Response                                             │
│    {                                                           │
│      "file_id": "uuid",                                        │
│      "status": "processed",                                    │
│      "vision_analysis": "Detailed description...",             │
│      "extracted_text": "OCR text...",                          │
│      "embedding_id": "chroma_id",                              │
│      "thumbnail_url": "/api/files/{file_id}/thumbnail"         │
│    }                                                           │
└────────────────────────────────────────────────────────────────┘
```

---

## Vision Analysis System

### Claude 3.5 Sonnet Vision Integration

#### Capabilities
- **Image Understanding:** Analyze images, screenshots, diagrams, charts
- **OCR:** Extract text from images (built-in, no external OCR needed)
- **Contextual Analysis:** Understand relationships between visual elements
- **Technical Diagrams:** Analyze architecture diagrams, flowcharts, UML
- **Document Images:** Process screenshots of documents, slides, whiteboards

#### Image Format Support
- **Supported:** PNG, JPEG, GIF, WebP
- **Max Size:** 5MB per image (Claude API limit)
- **Recommended:** JPEG for photos, PNG for screenshots/diagrams
- **Processing:** Automatic resize if > 5MB (maintain aspect ratio)

### VisionProcessor Module

**Location:** `app/multimodal/vision_processor.py`

```python
"""
Vision Processing with Claude 3.5 Sonnet
PATTERN: Async image analysis with structured output
WHY: Unified vision processing for all image types
"""

from typing import Dict, Any, Optional
import base64
from pathlib import Path
from PIL import Image
import anthropic

from app.config import settings
from app.logger import get_logger

logger = get_logger("vision_processor")


class VisionProcessor:
    """
    SPECIFICATION: Image analysis with Claude 3.5 Sonnet vision
    PATTERN: Single responsibility - vision processing only
    WHY: Centralized vision logic, easy to test and extend
    """

    def __init__(self, max_image_size: int = 5 * 1024 * 1024):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.max_image_size = max_image_size
        self.max_image_dimension = 1568  # Claude's max dimension

    async def analyze_image(
        self,
        image_path: Path,
        prompt: Optional[str] = None,
        detail_level: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Analyze image with Claude 3.5 Sonnet vision

        Args:
            image_path: Path to image file
            prompt: Optional custom prompt (uses default if None)
            detail_level: "brief", "detailed", or "comprehensive"

        Returns:
            {
                "description": str,        # Main visual description
                "extracted_text": str,     # OCR text content
                "objects": List[str],      # Identified objects
                "context": str,            # Contextual understanding
                "raw_response": str        # Full Claude response
            }
        """
        try:
            # Validate and prepare image
            image_data, mime_type = await self._prepare_image(image_path)

            # Build prompt based on detail level
            analysis_prompt = self._build_prompt(detail_level, prompt)

            # Call Claude vision API
            logger.info("vision_analysis_started",
                       image_path=str(image_path),
                       detail_level=detail_level)

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": analysis_prompt
                            }
                        ],
                    }
                ],
            )

            # Parse structured response
            result = self._parse_response(response.content[0].text)

            logger.info("vision_analysis_complete",
                       image_path=str(image_path),
                       tokens_used=response.usage.input_tokens + response.usage.output_tokens)

            return result

        except Exception as e:
            logger.error("vision_analysis_failed",
                        image_path=str(image_path),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True)
            raise

    async def _prepare_image(self, image_path: Path) -> tuple[str, str]:
        """
        Prepare image for Claude API
        - Resize if too large
        - Convert to base64
        - Determine MIME type
        """
        # Check file size
        file_size = image_path.stat().st_size

        if file_size > self.max_image_size:
            logger.warning("image_too_large",
                          image_path=str(image_path),
                          size_mb=file_size / (1024 * 1024))
            # Resize image
            image_path = await self._resize_image(image_path)

        # Read and encode
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        # Determine MIME type
        mime_type = self._get_mime_type(image_path)

        return image_data, mime_type

    async def _resize_image(self, image_path: Path) -> Path:
        """Resize image to fit Claude's limits"""
        img = Image.open(image_path)

        # Calculate new dimensions
        ratio = min(
            self.max_image_dimension / img.width,
            self.max_image_dimension / img.height
        )

        if ratio < 1:
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save resized image
            resized_path = image_path.parent / f"resized_{image_path.name}"
            img.save(resized_path, optimize=True, quality=85)

            logger.info("image_resized",
                       original=str(image_path),
                       resized=str(resized_path),
                       new_dimensions=new_size)

            return resized_path

        return image_path

    def _build_prompt(self, detail_level: str, custom_prompt: Optional[str]) -> str:
        """Build analysis prompt based on detail level"""
        if custom_prompt:
            return custom_prompt

        prompts = {
            "brief": """Provide a brief description of this image in 2-3 sentences.
Include any visible text.""",

            "detailed": """Analyze this image in detail. Provide:

1. **Main Description:** What is shown in the image?
2. **Text Content:** Extract all visible text (OCR)
3. **Key Objects:** List main objects and elements
4. **Context:** What is the purpose or context?

Format your response as:
DESCRIPTION: ...
TEXT: ...
OBJECTS: ...
CONTEXT: ...""",

            "comprehensive": """Provide a comprehensive analysis of this image:

1. **Visual Description:** Detailed description of what's shown
2. **Text Extraction (OCR):** All visible text, preserving layout
3. **Objects and Elements:** Complete inventory of visible items
4. **Spatial Layout:** Arrangement and positioning
5. **Context and Purpose:** Inferred purpose and usage
6. **Technical Details:** Colors, composition, style
7. **Notable Features:** Anything particularly interesting

Format your response with clear sections."""
        }

        return prompts.get(detail_level, prompts["detailed"])

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's response into structured format"""
        result = {
            "description": "",
            "extracted_text": "",
            "objects": [],
            "context": "",
            "raw_response": response_text
        }

        # Simple parsing - extract sections
        lines = response_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("DESCRIPTION:"):
                current_section = "description"
                result["description"] = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("TEXT:"):
                current_section = "extracted_text"
                result["extracted_text"] = line.replace("TEXT:", "").strip()
            elif line.startswith("OBJECTS:"):
                current_section = "objects"
                objects_str = line.replace("OBJECTS:", "").strip()
                result["objects"] = [obj.strip() for obj in objects_str.split(',')]
            elif line.startswith("CONTEXT:"):
                current_section = "context"
                result["context"] = line.replace("CONTEXT:", "").strip()
            elif current_section and line:
                # Continue previous section
                if current_section == "objects":
                    result["objects"].extend([obj.strip() for obj in line.split(',')])
                else:
                    result[current_section] += " " + line

        # If no structured format, use full response as description
        if not result["description"]:
            result["description"] = response_text

        return result

    def _get_mime_type(self, image_path: Path) -> str:
        """Determine MIME type from file extension"""
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(image_path.suffix.lower(), 'image/jpeg')


# Global instance
vision_processor = VisionProcessor()
```

### Vision Analysis Prompts

#### Default Prompt (Balanced)
```
Analyze this image in detail. Provide:

1. **Main Description:** What is shown in the image?
2. **Text Content:** Extract all visible text (OCR)
3. **Key Objects:** List main objects and elements
4. **Context:** What is the purpose or context?

Format your response as:
DESCRIPTION: ...
TEXT: ...
OBJECTS: ...
CONTEXT: ...
```

#### Code/Technical Diagram Prompt
```
Analyze this technical diagram/code screenshot:

1. **Type:** What kind of diagram/code is this?
2. **Text/Code:** Extract all text and code
3. **Components:** List key components or functions
4. **Relationships:** How do components relate?
5. **Purpose:** What does this diagram/code accomplish?
```

#### Document/Whiteboard Prompt
```
Extract information from this document/whiteboard:

1. **Content Type:** What kind of document is this?
2. **Main Points:** Key ideas and concepts
3. **Text Content:** All visible text
4. **Structure:** Headings, lists, organization
5. **Actionable Items:** Any tasks or next steps
```

---

## Document Processing Pipeline

### Supported Formats

| Format | Library | Extraction Capabilities | Max Size |
|--------|---------|-------------------------|----------|
| **PDF** | PyMuPDF (fitz) | Text, metadata, page info | 10MB |
| **DOCX** | python-docx | Paragraphs, tables, headers | 10MB |
| **TXT** | Built-in | Plain text | 10MB |
| **MD** | Built-in | Markdown with structure | 10MB |

### DocumentProcessor Module

**Location:** `app/multimodal/document_processor.py`

```python
"""
Document Processing Pipeline
PATTERN: Strategy pattern for different document types
WHY: Extensible document processing with unified interface
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import re

from app.logger import get_logger

logger = get_logger("document_processor")


class DocumentExtractor(ABC):
    """Base class for document extractors"""

    @abstractmethod
    async def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract content from document

        Returns:
            {
                "text": str,              # Full text content
                "metadata": dict,         # Document metadata
                "structure": dict,        # Document structure info
                "page_count": int,        # Number of pages (if applicable)
                "word_count": int         # Total words
            }
        """
        pass


class PDFExtractor(DocumentExtractor):
    """Extract content from PDF files using PyMuPDF"""

    async def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF

        PATTERN: Page-by-page extraction with structure preservation
        WHY: Maintain document hierarchy and metadata
        """
        try:
            logger.info("pdf_extraction_started", file_path=str(file_path))

            doc = fitz.open(file_path)

            # Extract text from all pages
            pages = []
            full_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "word_count": len(text.split())
                })
                full_text.append(text)

            # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            }

            combined_text = "\n\n".join(full_text)

            result = {
                "text": combined_text,
                "metadata": metadata,
                "structure": {
                    "pages": pages
                },
                "page_count": len(doc),
                "word_count": len(combined_text.split())
            }

            doc.close()

            logger.info("pdf_extraction_complete",
                       file_path=str(file_path),
                       page_count=len(doc),
                       word_count=result["word_count"])

            return result

        except Exception as e:
            logger.error("pdf_extraction_failed",
                        file_path=str(file_path),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True)
            raise


class DOCXExtractor(DocumentExtractor):
    """Extract content from DOCX files"""

    async def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text and structure from DOCX

        PATTERN: Paragraph and table extraction with formatting
        WHY: Preserve document structure for better context
        """
        try:
            logger.info("docx_extraction_started", file_path=str(file_path))

            doc = DocxDocument(file_path)

            # Extract paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append({
                        "text": para.text,
                        "style": para.style.name
                    })

            # Extract tables
            tables = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
                tables.append(table_data)

            # Combine text
            full_text = "\n\n".join([p["text"] for p in paragraphs])

            # Add table text
            if tables:
                table_text = "\n\n".join([
                    "\n".join([" | ".join(row) for row in table])
                    for table in tables
                ])
                full_text += "\n\n" + table_text

            # Extract core properties (metadata)
            metadata = {
                "title": doc.core_properties.title or "",
                "author": doc.core_properties.author or "",
                "subject": doc.core_properties.subject or "",
                "keywords": doc.core_properties.keywords or "",
                "created": str(doc.core_properties.created) if doc.core_properties.created else "",
                "modified": str(doc.core_properties.modified) if doc.core_properties.modified else ""
            }

            result = {
                "text": full_text,
                "metadata": metadata,
                "structure": {
                    "paragraphs": paragraphs,
                    "tables": tables,
                    "paragraph_count": len(paragraphs),
                    "table_count": len(tables)
                },
                "page_count": None,  # DOCX doesn't have fixed pages
                "word_count": len(full_text.split())
            }

            logger.info("docx_extraction_complete",
                       file_path=str(file_path),
                       paragraph_count=len(paragraphs),
                       table_count=len(tables),
                       word_count=result["word_count"])

            return result

        except Exception as e:
            logger.error("docx_extraction_failed",
                        file_path=str(file_path),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True)
            raise


class TextExtractor(DocumentExtractor):
    """Extract content from plain text and markdown files"""

    async def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from TXT/MD files

        PATTERN: Simple text extraction with markdown structure detection
        WHY: Handle plain text efficiently
        """
        try:
            logger.info("text_extraction_started", file_path=str(file_path))

            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Detect if markdown
            is_markdown = file_path.suffix.lower() == '.md'

            # Extract headings if markdown
            structure = {}
            if is_markdown:
                headings = re.findall(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
                structure["headings"] = [
                    {"level": len(h[0]), "text": h[1]}
                    for h in headings
                ]

            result = {
                "text": text,
                "metadata": {
                    "file_type": "markdown" if is_markdown else "plain_text",
                    "encoding": "utf-8"
                },
                "structure": structure,
                "page_count": None,
                "word_count": len(text.split())
            }

            logger.info("text_extraction_complete",
                       file_path=str(file_path),
                       is_markdown=is_markdown,
                       word_count=result["word_count"])

            return result

        except Exception as e:
            logger.error("text_extraction_failed",
                        file_path=str(file_path),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True)
            raise


class DocumentProcessor:
    """
    SPECIFICATION: Unified document processing interface
    PATTERN: Factory pattern for document type selection
    WHY: Single entry point for all document processing
    """

    def __init__(self):
        self.extractors = {
            '.pdf': PDFExtractor(),
            '.docx': DOCXExtractor(),
            '.txt': TextExtractor(),
            '.md': TextExtractor()
        }

    async def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process document and extract content

        Args:
            file_path: Path to document file

        Returns:
            Extracted content and metadata

        Raises:
            ValueError: If file type not supported
        """
        suffix = file_path.suffix.lower()

        if suffix not in self.extractors:
            raise ValueError(f"Unsupported file type: {suffix}")

        extractor = self.extractors[suffix]
        return await extractor.extract(file_path)

    def supports_file_type(self, file_path: Path) -> bool:
        """Check if file type is supported"""
        return file_path.suffix.lower() in self.extractors


# Global instance
document_processor = DocumentProcessor()
```

### Document Processing Flow

```
1. Upload Document
   └─> Validate file type and size

2. Save to ./data/uploads/documents/
   └─> Generate unique file_id

3. Extract Content
   ├─> PDF: PyMuPDF extraction
   ├─> DOCX: python-docx extraction
   └─> TXT/MD: Direct text extraction

4. Generate Embeddings
   └─> Chunk long documents (max 512 tokens per chunk)
   └─> Generate embeddings for each chunk

5. Store in Database
   ├─> SQLite: metadata + extracted_text
   └─> ChromaDB: embeddings with document chunks

6. Update Knowledge Graph
   └─> Extract concepts from document
   └─> Create document-concept relationships
```

---

## Multi-Modal Storage

### Database Schema Extensions

#### SQLite: multimodal_files Table

```sql
-- New table for multi-modal file storage
CREATE TABLE multimodal_files (
    file_id TEXT PRIMARY KEY,
    file_type TEXT NOT NULL,  -- 'image', 'pdf', 'docx', 'txt', 'md'
    mime_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,

    -- Extracted content
    extracted_text TEXT,
    vision_analysis TEXT,  -- For images only

    -- Metadata
    original_filename TEXT,
    session_id TEXT,  -- Optional: link to conversation session
    capture_id INTEGER,  -- Optional: link to conversation capture

    -- Processing info
    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_duration_ms REAL,

    -- Additional metadata (JSON)
    metadata TEXT,  -- JSON: page_count, word_count, author, etc.

    -- Timestamps
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL,
    FOREIGN KEY (capture_id) REFERENCES captures(id) ON DELETE SET NULL
);

-- Index for fast lookups
CREATE INDEX idx_multimodal_session ON multimodal_files(session_id);
CREATE INDEX idx_multimodal_type ON multimodal_files(file_type);
CREATE INDEX idx_multimodal_uploaded ON multimodal_files(uploaded_at);

-- Full-text search for extracted content
CREATE VIRTUAL TABLE multimodal_files_fts USING fts5(
    file_id UNINDEXED,
    extracted_text,
    vision_analysis,
    content=multimodal_files
);

-- Triggers to keep FTS in sync
CREATE TRIGGER multimodal_files_ai AFTER INSERT ON multimodal_files BEGIN
    INSERT INTO multimodal_files_fts(rowid, file_id, extracted_text, vision_analysis)
    VALUES (new.rowid, new.file_id, new.extracted_text, new.vision_analysis);
END;

CREATE TRIGGER multimodal_files_ad AFTER DELETE ON multimodal_files BEGIN
    DELETE FROM multimodal_files_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER multimodal_files_au AFTER UPDATE ON multimodal_files BEGIN
    UPDATE multimodal_files_fts
    SET extracted_text = new.extracted_text,
        vision_analysis = new.vision_analysis
    WHERE rowid = new.rowid;
END;
```

### ChromaDB: multimodal Collection

```python
# Collection configuration for multi-modal embeddings
multimodal_collection = {
    "name": "multimodal",
    "metadata_schema": {
        "file_id": "string",
        "file_type": "string",  # image, pdf, docx, txt, md
        "session_id": "string",
        "chunk_index": "int",  # For chunked documents
        "total_chunks": "int",
        "source_type": "string"  # vision_analysis, document_text
    },
    "distance_metric": "cosine"
}
```

### File Storage Structure

```
./data/
├── uploads/
│   ├── images/
│   │   ├── {file_id}.jpg
│   │   ├── {file_id}.png
│   │   └── thumbnails/
│   │       └── {file_id}_thumb.jpg
│   ├── documents/
│   │   ├── {file_id}.pdf
│   │   ├── {file_id}.docx
│   │   ├── {file_id}.txt
│   │   └── {file_id}.md
│   └── temp/  # Temporary processing files
│       └── (auto-cleaned after 24h)
└── chroma/
    └── multimodal/  # ChromaDB collection
```

### Storage Management

```python
"""
File Storage Manager
PATTERN: Repository pattern with lifecycle management
WHY: Centralized file operations with cleanup
"""

from pathlib import Path
from typing import Optional
import shutil
from datetime import datetime, timedelta

class FileStorageManager:
    """Manage multi-modal file storage"""

    def __init__(self, base_path: Path = Path("./data/uploads")):
        self.base_path = base_path
        self.image_path = base_path / "images"
        self.document_path = base_path / "documents"
        self.temp_path = base_path / "temp"
        self.thumbnail_path = self.image_path / "thumbnails"

        # Create directories
        for path in [self.image_path, self.document_path,
                     self.temp_path, self.thumbnail_path]:
            path.mkdir(parents=True, exist_ok=True)

    async def save_file(
        self,
        file_id: str,
        file_content: bytes,
        file_type: str,
        extension: str
    ) -> Path:
        """Save file to appropriate directory"""
        if file_type == "image":
            target_dir = self.image_path
        else:
            target_dir = self.document_path

        file_path = target_dir / f"{file_id}{extension}"

        with open(file_path, 'wb') as f:
            f.write(file_content)

        logger.info("file_saved",
                   file_id=file_id,
                   file_type=file_type,
                   file_path=str(file_path),
                   size_bytes=len(file_content))

        return file_path

    async def get_file(self, file_id: str) -> Optional[Path]:
        """Get file path by ID"""
        # Search in images
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            path = self.image_path / f"{file_id}{ext}"
            if path.exists():
                return path

        # Search in documents
        for ext in ['.pdf', '.docx', '.txt', '.md']:
            path = self.document_path / f"{file_id}{ext}"
            if path.exists():
                return path

        return None

    async def delete_file(self, file_id: str) -> bool:
        """Delete file and associated resources"""
        file_path = await self.get_file(file_id)
        if file_path:
            file_path.unlink()

            # Delete thumbnail if image
            if file_path.parent == self.image_path:
                thumb_path = self.thumbnail_path / f"{file_id}_thumb.jpg"
                if thumb_path.exists():
                    thumb_path.unlink()

            logger.info("file_deleted", file_id=file_id)
            return True

        return False

    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than max_age"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        for file_path in self.temp_path.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff:
                    file_path.unlink()
                    logger.debug("temp_file_deleted", file_path=str(file_path))


# Global instance
file_storage = FileStorageManager()
```

---

## API Design

### New Endpoints

#### 1. Upload Image

```http
POST /api/upload/image
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - session_id: string (optional)
  - analyze: boolean (default: true)
  - detail_level: "brief" | "detailed" | "comprehensive" (default: "detailed")

Response: 200 OK
{
  "file_id": "uuid",
  "status": "processed",
  "file_type": "image",
  "mime_type": "image/jpeg",
  "file_size": 2048000,
  "vision_analysis": {
    "description": "A technical diagram showing...",
    "extracted_text": "API → Database → Cache",
    "objects": ["server", "database", "arrows"],
    "context": "System architecture diagram"
  },
  "embedding_id": "chroma_uuid",
  "thumbnail_url": "/api/files/uuid/thumbnail",
  "processing_time_ms": 1834,
  "uploaded_at": "2025-11-21T10:30:00Z"
}
```

#### 2. Upload Document

```http
POST /api/upload/document
Content-Type: multipart/form-data

Parameters:
  - file: document file (required)
  - session_id: string (optional)
  - chunk_size: int (default: 512 tokens)

Response: 200 OK
{
  "file_id": "uuid",
  "status": "processed",
  "file_type": "pdf",
  "mime_type": "application/pdf",
  "file_size": 5242880,
  "extracted_content": {
    "text": "Full document text...",
    "metadata": {
      "title": "Architecture Document",
      "author": "John Doe",
      "page_count": 15
    },
    "word_count": 3500
  },
  "embeddings": {
    "chunk_count": 7,
    "embedding_ids": ["id1", "id2", ...]
  },
  "processing_time_ms": 3421,
  "uploaded_at": "2025-11-21T10:30:00Z"
}
```

#### 3. Analyze Image (without upload)

```http
POST /api/analyze/image
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - prompt: string (optional custom prompt)
  - detail_level: "brief" | "detailed" | "comprehensive"

Response: 200 OK
{
  "vision_analysis": {
    "description": "...",
    "extracted_text": "...",
    "objects": [...],
    "context": "..."
  },
  "processing_time_ms": 1650
}
```

#### 4. Get File

```http
GET /api/files/{file_id}

Response: 200 OK
Content-Type: image/jpeg (or appropriate MIME type)
[File binary data]
```

#### 5. Get File Metadata

```http
GET /api/files/{file_id}/metadata

Response: 200 OK
{
  "file_id": "uuid",
  "file_type": "image",
  "mime_type": "image/jpeg",
  "file_size": 2048000,
  "original_filename": "diagram.jpg",
  "extracted_text": "...",
  "vision_analysis": {...},
  "metadata": {...},
  "session_id": "session_uuid",
  "uploaded_at": "2025-11-21T10:30:00Z"
}
```

#### 6. Enhanced Conversation Endpoint

```http
POST /api/conversation
Content-Type: application/json

{
  "text": "Explain this diagram",
  "image_ids": ["image_uuid1", "image_uuid2"],  # NEW: Reference uploaded images
  "document_ids": ["doc_uuid1"],  # NEW: Reference uploaded documents
  "session_id": "session_uuid"
}

Response: 200 OK
{
  "session_id": "session_uuid",
  "user_text": "Explain this diagram",
  "agent_text": "Based on the diagram you shared, this shows a microservices architecture...",
  "context_used": {
    "images": [
      {
        "file_id": "image_uuid1",
        "vision_analysis": "..."
      }
    ],
    "documents": [
      {
        "file_id": "doc_uuid1",
        "relevant_chunks": ["chunk1", "chunk2"]
      }
    ]
  },
  "intent": "question"
}
```

### API Implementation

**Location:** `app/main.py` (additions)

```python
from fastapi import UploadFile, File
from app.multimodal.vision_processor import vision_processor
from app.multimodal.document_processor import document_processor
from app.multimodal.file_storage import file_storage
from app.multimodal.models import (
    ImageUploadResponse,
    DocumentUploadResponse,
    FileMetadataResponse
)

@app.post("/api/upload/image")
@limiter.limit("10/minute")
async def upload_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    analyze: bool = True,
    detail_level: str = "detailed",
    http_request: Request = None
) -> ImageUploadResponse:
    """
    Upload and analyze image

    PATTERN: Upload → Store → Analyze → Embed → Persist
    WHY: Complete multi-modal processing pipeline
    """
    start_time = time.perf_counter()
    file_id = str(uuid.uuid4())

    try:
        # Validate file
        if file.content_type not in ['image/jpeg', 'image/png', 'image/gif', 'image/webp']:
            raise HTTPException(400, f"Unsupported image type: {file.content_type}")

        # Read file content
        content = await file.read()
        file_size = len(content)

        if file_size > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(413, "File too large (max 5MB)")

        logger.info("image_upload_started",
                   file_id=file_id,
                   filename=file.filename,
                   size_bytes=file_size,
                   mime_type=file.content_type)

        # Save file
        extension = Path(file.filename).suffix or '.jpg'
        file_path = await file_storage.save_file(
            file_id, content, "image", extension
        )

        # Analyze image (if requested)
        vision_analysis = None
        embedding_id = None

        if analyze:
            vision_result = await vision_processor.analyze_image(
                file_path,
                detail_level=detail_level
            )
            vision_analysis = vision_result

            # Generate embedding from vision analysis
            from app.vector.embeddings import embedding_generator
            await embedding_generator.initialize()

            # Combine description and extracted text for embedding
            embedding_text = f"{vision_result['description']}\n{vision_result['extracted_text']}"
            embedding = await embedding_generator.generate_embedding(embedding_text)

            # Store in ChromaDB
            from app.vector.vector_store import vector_store
            await vector_store.initialize()

            embedding_id = await vector_store.add_embedding(
                collection_name="multimodal",
                text=embedding_text,
                embedding=embedding,
                metadata={
                    "file_id": file_id,
                    "file_type": "image",
                    "session_id": session_id or "",
                    "source_type": "vision_analysis"
                }
            )

        # Store in database
        await db.execute(
            """
            INSERT INTO multimodal_files
            (file_id, file_type, mime_type, file_path, file_size,
             extracted_text, vision_analysis, original_filename,
             session_id, processing_duration_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                "image",
                file.content_type,
                str(file_path),
                file_size,
                vision_analysis.get("extracted_text", "") if vision_analysis else None,
                json.dumps(vision_analysis) if vision_analysis else None,
                file.filename,
                session_id,
                (time.perf_counter() - start_time) * 1000,
                json.dumps({"detail_level": detail_level})
            )
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info("image_upload_complete",
                   file_id=file_id,
                   processing_time_ms=processing_time_ms,
                   analyzed=analyze)

        return ImageUploadResponse(
            file_id=file_id,
            status="processed" if analyze else "uploaded",
            file_type="image",
            mime_type=file.content_type,
            file_size=file_size,
            vision_analysis=vision_analysis,
            embedding_id=embedding_id,
            thumbnail_url=f"/api/files/{file_id}/thumbnail",
            processing_time_ms=processing_time_ms,
            uploaded_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("image_upload_failed",
                    file_id=file_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True)
        raise HTTPException(500, f"Image upload failed: {str(e)}")


@app.post("/api/upload/document")
@limiter.limit("5/minute")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    chunk_size: int = 512,
    http_request: Request = None
) -> DocumentUploadResponse:
    """
    Upload and process document

    PATTERN: Upload → Store → Extract → Chunk → Embed → Persist
    WHY: Handle long documents with chunking for better retrieval
    """
    start_time = time.perf_counter()
    file_id = str(uuid.uuid4())

    try:
        # Validate file type
        allowed_types = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'text/plain': '.txt',
            'text/markdown': '.md'
        }

        if file.content_type not in allowed_types:
            raise HTTPException(400, f"Unsupported document type: {file.content_type}")

        # Read file content
        content = await file.read()
        file_size = len(content)

        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(413, "File too large (max 10MB)")

        logger.info("document_upload_started",
                   file_id=file_id,
                   filename=file.filename,
                   size_bytes=file_size,
                   mime_type=file.content_type)

        # Save file
        extension = allowed_types[file.content_type]
        file_path = await file_storage.save_file(
            file_id, content, "document", extension
        )

        # Extract content
        extracted = await document_processor.process_document(file_path)

        # Chunk text for embedding (if > chunk_size tokens)
        from app.vector.embeddings import embedding_generator
        chunks = await embedding_generator.chunk_text(
            extracted["text"],
            chunk_size=chunk_size
        )

        # Generate embeddings for chunks
        await embedding_generator.initialize()
        embeddings = await embedding_generator.generate_batch(chunks)

        # Store embeddings in ChromaDB
        from app.vector.vector_store import vector_store
        await vector_store.initialize()

        embedding_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            emb_id = await vector_store.add_embedding(
                collection_name="multimodal",
                text=chunk,
                embedding=embedding,
                metadata={
                    "file_id": file_id,
                    "file_type": extension.lstrip('.'),
                    "session_id": session_id or "",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_type": "document_text"
                }
            )
            embedding_ids.append(emb_id)

        # Store in database
        await db.execute(
            """
            INSERT INTO multimodal_files
            (file_id, file_type, mime_type, file_path, file_size,
             extracted_text, original_filename, session_id,
             processing_duration_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                extension.lstrip('.'),
                file.content_type,
                str(file_path),
                file_size,
                extracted["text"],
                file.filename,
                session_id,
                (time.perf_counter() - start_time) * 1000,
                json.dumps({
                    "page_count": extracted.get("page_count"),
                    "word_count": extracted["word_count"],
                    "chunk_count": len(chunks),
                    **extracted.get("metadata", {})
                })
            )
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info("document_upload_complete",
                   file_id=file_id,
                   chunk_count=len(chunks),
                   processing_time_ms=processing_time_ms)

        return DocumentUploadResponse(
            file_id=file_id,
            status="processed",
            file_type=extension.lstrip('.'),
            mime_type=file.content_type,
            file_size=file_size,
            extracted_content={
                "text": extracted["text"][:1000] + "..." if len(extracted["text"]) > 1000 else extracted["text"],
                "metadata": extracted["metadata"],
                "word_count": extracted["word_count"],
                "page_count": extracted.get("page_count")
            },
            embeddings={
                "chunk_count": len(chunks),
                "embedding_ids": embedding_ids
            },
            processing_time_ms=processing_time_ms,
            uploaded_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("document_upload_failed",
                    file_id=file_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True)
        raise HTTPException(500, f"Document upload failed: {str(e)}")
```

---

## Integration Strategy

### ConversationAgent Enhancement

**Pattern:** Extend existing ConversationAgent to support vision

**Changes to `app/agents/conversation_agent.py`:**

```python
class ConversationAgent(BaseAgent):
    """Enhanced with multi-modal support"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process conversation with optional image context"""

        content = message.content
        user_text = content.get("user_text", "")
        image_ids = content.get("image_ids", [])
        document_ids = content.get("document_ids", [])

        # Retrieve image context if provided
        image_context = []
        if image_ids:
            image_context = await self._get_image_context(image_ids)

        # Retrieve document context if provided
        document_context = []
        if document_ids:
            document_context = await self._get_document_context(document_ids)

        # Build enriched prompt
        enriched_prompt = await self._build_multimodal_prompt(
            user_text,
            image_context,
            document_context,
            content.get("context", [])
        )

        # Generate response with Claude (vision-capable model)
        response = await self._generate_with_vision(enriched_prompt, image_ids)

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.CONVERSATION_COMPLETE,
            content={
                "agent_text": response,
                "context_used": {
                    "images": image_context,
                    "documents": document_context
                }
            },
            correlation_id=message.message_id
        )

    async def _get_image_context(self, image_ids: List[str]) -> List[Dict]:
        """Retrieve vision analysis for images"""
        from app.database import db

        results = []
        for image_id in image_ids:
            row = await db.fetch_one(
                """
                SELECT file_id, vision_analysis, extracted_text
                FROM multimodal_files
                WHERE file_id = ? AND file_type = 'image'
                """,
                (image_id,)
            )
            if row:
                results.append({
                    "file_id": row["file_id"],
                    "vision_analysis": json.loads(row["vision_analysis"]) if row["vision_analysis"] else None,
                    "extracted_text": row["extracted_text"]
                })

        return results

    async def _get_document_context(self, document_ids: List[str]) -> List[Dict]:
        """Retrieve relevant chunks from documents"""
        from app.vector.vector_store import vector_store

        results = []
        for doc_id in document_ids:
            # Get document metadata
            row = await db.fetch_one(
                """
                SELECT file_id, extracted_text, metadata
                FROM multimodal_files
                WHERE file_id = ?
                """,
                (doc_id,)
            )

            if row:
                # Get relevant chunks from vector store
                chunks = await vector_store.search_similar(
                    collection_name="multimodal",
                    query_text=row["extracted_text"][:500],  # Use excerpt as query
                    metadata_filter={"file_id": doc_id},
                    n_results=3
                )

                results.append({
                    "file_id": row["file_id"],
                    "relevant_chunks": [c["document"] for c in chunks],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                })

        return results

    async def _build_multimodal_prompt(
        self,
        user_text: str,
        image_context: List[Dict],
        document_context: List[Dict],
        conversation_context: List[Dict]
    ) -> List[Dict]:
        """Build prompt with multi-modal context"""

        messages = []

        # Add conversation context
        for ctx in conversation_context[-5:]:  # Last 5 exchanges
            messages.append({
                "role": "user",
                "content": ctx["user_text"]
            })
            messages.append({
                "role": "assistant",
                "content": ctx["agent_text"]
            })

        # Build current user message with context
        current_message_parts = []

        # Add image context
        if image_context:
            image_descriptions = []
            for img in image_context:
                if img.get("vision_analysis"):
                    desc = img["vision_analysis"].get("description", "")
                    text = img.get("extracted_text", "")
                    image_descriptions.append(f"Image: {desc}")
                    if text:
                        image_descriptions.append(f"Text in image: {text}")

            if image_descriptions:
                current_message_parts.append(
                    "Context from images:\n" + "\n\n".join(image_descriptions)
                )

        # Add document context
        if document_context:
            doc_excerpts = []
            for doc in document_context:
                chunks = doc.get("relevant_chunks", [])
                if chunks:
                    doc_excerpts.append("Relevant document excerpt:\n" + "\n".join(chunks[:2]))

            if doc_excerpts:
                current_message_parts.append(
                    "Context from documents:\n" + "\n\n".join(doc_excerpts)
                )

        # Add user query
        current_message_parts.append(f"User query: {user_text}")

        messages.append({
            "role": "user",
            "content": "\n\n---\n\n".join(current_message_parts)
        })

        return messages

    async def _generate_with_vision(
        self,
        messages: List[Dict],
        image_ids: Optional[List[str]] = None
    ) -> str:
        """Generate response using Claude with vision support"""

        # If image_ids provided and we want direct image analysis,
        # we can fetch the actual images and include them in the API call
        # For now, we're using the pre-analyzed vision descriptions

        response = await self.client.messages.create(
            model=self.model,  # claude-3-5-sonnet-20241022
            max_tokens=2048,
            messages=messages
        )

        return response.content[0].text
```

### RAG Integration

**Enhance RAGRetriever to include multi-modal context:**

```python
# In app/rag/retriever.py

async def retrieve_context(
    self,
    query: str,
    session_id: Optional[str] = None,
    include_multimodal: bool = True,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Retrieve relevant context including multi-modal files

    Returns:
        {
            "conversation_context": [...],
            "vector_results": [...],
            "keyword_results": [...],
            "multimodal_files": [...]  # NEW
        }
    """
    # Existing retrieval logic...
    context = await self._retrieve_base_context(query, session_id, n_results)

    # Add multi-modal context
    if include_multimodal:
        multimodal_results = await self._retrieve_multimodal_context(
            query, session_id, n_results
        )
        context["multimodal_files"] = multimodal_results

    return context

async def _retrieve_multimodal_context(
    self,
    query: str,
    session_id: Optional[str],
    n_results: int
) -> List[Dict]:
    """Retrieve relevant multi-modal files"""

    # Search vector store for relevant files
    results = await self.vector_store.search_similar(
        collection_name="multimodal",
        query_text=query,
        metadata_filter={"session_id": session_id} if session_id else None,
        n_results=n_results
    )

    # Enrich with file metadata
    enriched = []
    for result in results:
        file_id = result["metadata"]["file_id"]
        file_metadata = await self._get_file_metadata(file_id)
        enriched.append({
            **result,
            "file_metadata": file_metadata
        })

    return enriched
```

---

## Performance Targets

### Processing Time Targets

| Operation | Target | Acceptable | Notes |
|-----------|--------|------------|-------|
| **Image Upload** | < 1s | < 2s | File save + validation |
| **Image Analysis (Claude)** | < 1.5s | < 3s | Vision API call |
| **Image Embedding** | < 0.3s | < 0.5s | Sentence Transformers |
| **Image Total** | **< 2s** | **< 4s** | End-to-end processing |
| **PDF Extraction (per page)** | < 0.5s | < 1s | PyMuPDF processing |
| **DOCX Extraction** | < 1s | < 2s | python-docx processing |
| **Document Embedding** | < 0.5s/chunk | < 1s/chunk | Batch processing |
| **Document Total (10 pages)** | **< 5s** | **< 10s** | End-to-end processing |
| **File Retrieval** | < 100ms | < 200ms | Local filesystem read |
| **Multi-modal Search** | < 500ms | < 1s | Vector + FTS5 hybrid |

### Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Concurrent Image Uploads** | 10/s | Rate limited to 10/min per user |
| **Concurrent Document Uploads** | 5/s | Rate limited to 5/min per user |
| **Storage Capacity** | 10GB | Railway persistent volume |
| **Max File Count** | 10,000 files | Indexed and searchable |

### Resource Utilization

| Resource | Normal | Peak | Limit |
|----------|--------|------|-------|
| **Memory (Image)** | 50MB | 200MB | Per request |
| **Memory (Document)** | 100MB | 500MB | Per request |
| **CPU (Vision Analysis)** | Negligible | N/A | API-bound |
| **CPU (Embedding)** | 20% | 80% | Model inference |
| **Disk I/O** | < 10 MB/s | < 50 MB/s | File operations |

---

## Security Considerations

### File Validation

```python
"""
File Security Validator
PATTERN: Defense in depth with multiple validation layers
WHY: Prevent malicious uploads and resource exhaustion
"""

class FileValidator:
    """Validate uploaded files for security"""

    # Allowed MIME types
    ALLOWED_IMAGES = {
        'image/jpeg', 'image/png', 'image/gif', 'image/webp'
    }

    ALLOWED_DOCUMENTS = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/markdown'
    }

    # Size limits (bytes)
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB

    # Extension validation
    ALLOWED_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.webp',  # Images
        '.pdf', '.docx', '.txt', '.md'  # Documents
    }

    @staticmethod
    async def validate_upload(
        file: UploadFile,
        expected_type: str  # 'image' or 'document'
    ) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded file

        Returns:
            (is_valid, error_message)
        """
        # 1. Check file extension
        extension = Path(file.filename).suffix.lower()
        if extension not in FileValidator.ALLOWED_EXTENSIONS:
            return False, f"Invalid file extension: {extension}"

        # 2. Check MIME type
        if expected_type == 'image':
            allowed_types = FileValidator.ALLOWED_IMAGES
            max_size = FileValidator.MAX_IMAGE_SIZE
        else:
            allowed_types = FileValidator.ALLOWED_DOCUMENTS
            max_size = FileValidator.MAX_DOCUMENT_SIZE

        if file.content_type not in allowed_types:
            return False, f"Invalid MIME type: {file.content_type}"

        # 3. Check file size (read in chunks to avoid memory issues)
        total_size = 0
        chunk_size = 8192

        # Save current position
        current_pos = file.file.tell()
        file.file.seek(0)

        while chunk := await file.read(chunk_size):
            total_size += len(chunk)
            if total_size > max_size:
                # Reset file pointer
                file.file.seek(current_pos)
                return False, f"File too large: {total_size} bytes (max {max_size})"

        # Reset file pointer for subsequent reads
        file.file.seek(current_pos)

        # 4. Validate file header (magic bytes) for images
        if expected_type == 'image':
            file.file.seek(0)
            header = await file.read(12)
            file.file.seek(current_pos)

            is_valid_header = FileValidator._validate_image_header(
                header, file.content_type
            )
            if not is_valid_header:
                return False, "Invalid file header (possible disguised file)"

        return True, None

    @staticmethod
    def _validate_image_header(header: bytes, mime_type: str) -> bool:
        """Validate image file header (magic bytes)"""
        # PNG
        if mime_type == 'image/png':
            return header[:8] == b'\x89PNG\r\n\x1a\n'

        # JPEG
        elif mime_type == 'image/jpeg':
            return header[:2] == b'\xff\xd8'

        # GIF
        elif mime_type == 'image/gif':
            return header[:6] in (b'GIF87a', b'GIF89a')

        # WebP
        elif mime_type == 'image/webp':
            return header[:4] == b'RIFF' and header[8:12] == b'WEBP'

        return False
```

### Additional Security Measures

#### 1. Rate Limiting
```python
# Stricter rate limits for upload endpoints
@app.post("/api/upload/image")
@limiter.limit("10/minute")  # Max 10 images per minute per IP
async def upload_image(...):
    pass

@app.post("/api/upload/document")
@limiter.limit("5/minute")  # Max 5 documents per minute per IP
async def upload_document(...):
    pass
```

#### 2. Content Security Policy
```python
# Add CSP headers for file serving
@app.get("/api/files/{file_id}")
async def get_file(file_id: str, response: Response):
    # Set restrictive CSP
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"

    # ... serve file
```

#### 3. Virus Scanning (Optional, for production)
```python
# Integration with ClamAV or similar (optional)
async def scan_file(file_path: Path) -> bool:
    """
    Scan file for viruses (production only)
    Requires ClamAV daemon running
    """
    try:
        import pyclamd
        cd = pyclamd.ClamdUnixSocket()

        # Scan file
        result = cd.scan_file(str(file_path))

        if result is None:
            # File is clean
            return True
        else:
            # Virus detected
            logger.warning("virus_detected",
                          file_path=str(file_path),
                          result=result)
            return False

    except Exception as e:
        logger.error("virus_scan_failed",
                    file_path=str(file_path),
                    error=str(e))
        # Fail open (allow file) or fail closed (reject file)?
        # For single-user Railway deployment, fail open
        return True
```

#### 4. Storage Quotas
```python
class StorageQuotaManager:
    """Manage storage quotas per user/session"""

    def __init__(self, max_storage_bytes: int = 1 * 1024 * 1024 * 1024):  # 1GB
        self.max_storage = max_storage_bytes

    async def check_quota(self, session_id: str, new_file_size: int) -> bool:
        """Check if upload would exceed quota"""

        # Get current storage usage
        result = await db.fetch_one(
            """
            SELECT SUM(file_size) as total_size
            FROM multimodal_files
            WHERE session_id = ?
            """,
            (session_id,)
        )

        current_usage = result["total_size"] or 0

        if current_usage + new_file_size > self.max_storage:
            logger.warning("storage_quota_exceeded",
                          session_id=session_id,
                          current_usage=current_usage,
                          new_file_size=new_file_size,
                          max_storage=self.max_storage)
            return False

        return True

    async def get_usage_stats(self, session_id: str) -> Dict[str, Any]:
        """Get storage usage statistics"""
        result = await db.fetch_one(
            """
            SELECT
                COUNT(*) as file_count,
                SUM(file_size) as total_size,
                SUM(CASE WHEN file_type = 'image' THEN 1 ELSE 0 END) as image_count,
                SUM(CASE WHEN file_type != 'image' THEN 1 ELSE 0 END) as document_count
            FROM multimodal_files
            WHERE session_id = ?
            """,
            (session_id,)
        )

        return {
            "file_count": result["file_count"],
            "total_size_bytes": result["total_size"] or 0,
            "total_size_mb": (result["total_size"] or 0) / (1024 * 1024),
            "image_count": result["image_count"],
            "document_count": result["document_count"],
            "quota_bytes": self.max_storage,
            "quota_used_percent": ((result["total_size"] or 0) / self.max_storage) * 100
        }
```

---

## Migration Strategy

### Backward Compatibility

**CRITICAL:** Phase 4 must NOT break existing functionality.

#### 1. Database Migration

```sql
-- Migration script: migrations/004_multimodal_tables.sql

-- Create new tables (no changes to existing tables)
CREATE TABLE IF NOT EXISTS multimodal_files (
    file_id TEXT PRIMARY KEY,
    file_type TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    extracted_text TEXT,
    vision_analysis TEXT,
    original_filename TEXT,
    session_id TEXT,
    capture_id INTEGER,
    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_duration_ms REAL,
    metadata TEXT,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL,
    FOREIGN KEY (capture_id) REFERENCES captures(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_multimodal_session ON multimodal_files(session_id);
CREATE INDEX IF NOT EXISTS idx_multimodal_type ON multimodal_files(file_type);
CREATE INDEX IF NOT EXISTS idx_multimodal_uploaded ON multimodal_files(uploaded_at);

-- FTS5 table for multi-modal content
CREATE VIRTUAL TABLE IF NOT EXISTS multimodal_files_fts USING fts5(
    file_id UNINDEXED,
    extracted_text,
    vision_analysis,
    content=multimodal_files
);

-- Triggers
CREATE TRIGGER IF NOT EXISTS multimodal_files_ai AFTER INSERT ON multimodal_files BEGIN
    INSERT INTO multimodal_files_fts(rowid, file_id, extracted_text, vision_analysis)
    VALUES (new.rowid, new.file_id, new.extracted_text, new.vision_analysis);
END;

CREATE TRIGGER IF NOT EXISTS multimodal_files_ad AFTER DELETE ON multimodal_files BEGIN
    DELETE FROM multimodal_files_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS multimodal_files_au AFTER UPDATE ON multimodal_files BEGIN
    UPDATE multimodal_files_fts
    SET extracted_text = new.extracted_text,
        vision_analysis = new.vision_analysis
    WHERE rowid = new.rowid;
END;
```

#### 2. ChromaDB Collection Setup

```python
# In app/vector/config.py - Add new collection

from app.vector.config import VectorConfig, CollectionConfig

# Extend existing config
multimodal_collection_config = CollectionConfig(
    name="multimodal",
    metadata_schema={
        "file_id": "string",
        "file_type": "string",
        "session_id": "string",
        "chunk_index": "int",
        "total_chunks": "int",
        "source_type": "string"
    },
    distance_metric="cosine"
)

# Update vector_config
vector_config.collections["multimodal"] = multimodal_collection_config
```

#### 3. API Versioning (Optional)

```python
# If needed, version the API
@app.post("/api/v2/conversation")  # New endpoint with multi-modal support
async def conversation_v2(...):
    pass

# Keep existing endpoint working
@app.post("/api/conversation")  # Legacy endpoint (no breaking changes)
async def conversation(...):
    pass
```

### Rollout Plan

#### Phase 4.1: Core Infrastructure (Week 1)
- [ ] Create database tables and migrations
- [ ] Implement FileStorageManager
- [ ] Set up ChromaDB multimodal collection
- [ ] Add file validation utilities
- [ ] Write unit tests for storage layer

#### Phase 4.2: Vision Processing (Week 2)
- [ ] Implement VisionProcessor module
- [ ] Integrate Claude 3.5 Sonnet vision API
- [ ] Add image upload endpoint
- [ ] Implement thumbnail generation
- [ ] Test image analysis pipeline

#### Phase 4.3: Document Processing (Week 3)
- [ ] Implement DocumentProcessor module
- [ ] Add PDF extraction (PyMuPDF)
- [ ] Add DOCX extraction (python-docx)
- [ ] Add document upload endpoint
- [ ] Test document chunking and embedding

#### Phase 4.4: Agent Integration (Week 4)
- [ ] Enhance ConversationAgent with multi-modal support
- [ ] Update RAGRetriever for multi-modal context
- [ ] Enhance hybrid search for multi-modal queries
- [ ] Add multi-modal conversation endpoint
- [ ] Test end-to-end multi-modal conversations

#### Phase 4.5: Testing & Optimization (Week 5)
- [ ] Comprehensive integration tests
- [ ] Performance benchmarking and optimization
- [ ] Security audit and hardening
- [ ] Documentation and examples
- [ ] Deployment preparation

### Testing Strategy

```python
# tests/test_multimodal.py

import pytest
from pathlib import Path

@pytest.mark.asyncio
async def test_image_upload():
    """Test image upload and analysis"""
    # Upload test image
    with open("tests/fixtures/test_diagram.png", "rb") as f:
        response = await client.post(
            "/api/upload/image",
            files={"file": ("diagram.png", f, "image/png")},
            data={"detail_level": "detailed"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"
    assert "vision_analysis" in data
    assert data["vision_analysis"]["description"]


@pytest.mark.asyncio
async def test_document_upload():
    """Test document upload and extraction"""
    with open("tests/fixtures/test_document.pdf", "rb") as f:
        response = await client.post(
            "/api/upload/document",
            files={"file": ("document.pdf", f, "application/pdf")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"
    assert data["embeddings"]["chunk_count"] > 0


@pytest.mark.asyncio
async def test_multimodal_conversation():
    """Test conversation with image context"""
    # Upload image first
    image_response = await upload_test_image()
    file_id = image_response.json()["file_id"]

    # Ask question about image
    response = await client.post(
        "/api/conversation",
        json={
            "text": "What does this diagram show?",
            "image_ids": [file_id]
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "diagram" in data["agent_text"].lower()
    assert data["context_used"]["images"]
```

---

## Implementation Plan

### Week 1: Core Infrastructure
**Deliverables:**
- Database schema and migrations
- FileStorageManager implementation
- ChromaDB multimodal collection setup
- File validation utilities
- Unit tests (80%+ coverage)

**Files to Create:**
- `migrations/004_multimodal_tables.sql`
- `app/multimodal/__init__.py`
- `app/multimodal/file_storage.py`
- `app/multimodal/file_validator.py`
- `app/multimodal/models.py` (Pydantic models)
- `tests/test_file_storage.py`
- `tests/test_file_validator.py`

### Week 2: Vision Processing
**Deliverables:**
- VisionProcessor module
- Claude 3.5 Sonnet vision integration
- Image upload API endpoint
- Thumbnail generation
- Vision analysis tests

**Files to Create:**
- `app/multimodal/vision_processor.py`
- `app/multimodal/thumbnail_generator.py`
- `tests/test_vision_processor.py`
- `tests/fixtures/test_images/`

**API Endpoints:**
- `POST /api/upload/image`
- `POST /api/analyze/image`
- `GET /api/files/{file_id}/thumbnail`

### Week 3: Document Processing
**Deliverables:**
- DocumentProcessor module with extractors
- PDF, DOCX, TXT, MD support
- Document upload API endpoint
- Text chunking and embedding
- Document processing tests

**Files to Create:**
- `app/multimodal/document_processor.py`
- `app/multimodal/text_chunker.py`
- `tests/test_document_processor.py`
- `tests/fixtures/test_documents/`

**API Endpoints:**
- `POST /api/upload/document`
- `GET /api/files/{file_id}`
- `GET /api/files/{file_id}/metadata`

### Week 4: Agent Integration
**Deliverables:**
- Enhanced ConversationAgent
- Multi-modal RAGRetriever
- Enhanced hybrid search
- Multi-modal conversation endpoint
- Integration tests

**Files to Modify:**
- `app/agents/conversation_agent.py`
- `app/rag/retriever.py`
- `app/search/hybrid_search.py`
- `app/main.py` (add endpoints)

**Files to Create:**
- `tests/test_multimodal_integration.py`
- `tests/test_multimodal_conversation.py`

**API Endpoints:**
- Enhanced `POST /api/conversation` (with image/document support)
- `POST /api/search/multimodal`

### Week 5: Testing & Documentation
**Deliverables:**
- Comprehensive test suite
- Performance benchmarks
- Security audit
- API documentation
- Usage examples
- Deployment guide

**Documentation to Create:**
- `docs/PHASE4_API_REFERENCE.md`
- `docs/PHASE4_USAGE_EXAMPLES.md`
- `docs/PHASE4_TESTING_GUIDE.md`
- `docs/PHASE4_DEPLOYMENT.md`

---

## Success Metrics

### Technical Metrics
- **Test Coverage:** > 85% for all multi-modal modules
- **Image Processing:** < 2s average (P99 < 4s)
- **Document Processing:** < 5s per page average
- **API Error Rate:** < 0.1%
- **Storage Efficiency:** < 100MB overhead per 1GB files

### Functional Metrics
- **Image Analysis Accuracy:** > 95% (manual review of 100 samples)
- **OCR Extraction Quality:** > 90% accuracy for printed text
- **Document Extraction:** 100% of text extracted from standard PDFs
- **Multi-modal Search Relevance:** > 85% (manual evaluation)

### User Experience Metrics
- **Upload Success Rate:** > 99.5%
- **Processing Reliability:** > 99.9% (no crashes)
- **Context Integration:** Relevant images/docs in > 80% of conversations

---

## Appendix

### A. Dependencies

```txt
# Add to requirements.txt

# Vision and Document Processing
PyMuPDF==1.23.8  # PDF processing
python-docx==1.1.0  # DOCX processing
Pillow==10.2.0  # Image processing

# Already in requirements.txt (verify versions):
# anthropic>=0.18.0  # Claude 3.5 Sonnet with vision
# chromadb>=0.4.22  # Vector store
# sentence-transformers>=2.2.2  # Embeddings
```

### B. Configuration

```python
# Add to app/config.py

class Settings(BaseSettings):
    # ... existing settings ...

    # Multi-modal settings
    multimodal_storage_path: str = "./data/uploads"
    max_image_size_mb: int = 5
    max_document_size_mb: int = 10
    max_storage_per_user_gb: int = 1
    enable_virus_scanning: bool = False

    # Vision processing
    vision_model: str = "claude-3-5-sonnet-20241022"
    vision_max_tokens: int = 2048

    # Document chunking
    document_chunk_size: int = 512  # tokens
    document_chunk_overlap: int = 50  # tokens
```

### C. Monitoring

```python
# Add to app/metrics.py

# Multi-modal metrics
multimodal_uploads_total = Counter(
    'multimodal_uploads_total',
    'Total multi-modal file uploads',
    ['file_type', 'status']
)

multimodal_processing_duration = Histogram(
    'multimodal_processing_duration_seconds',
    'Multi-modal processing duration',
    ['file_type', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

multimodal_storage_bytes = Gauge(
    'multimodal_storage_bytes',
    'Total storage used by multi-modal files',
    ['file_type']
)
```

---

## Conclusion

Phase 4 establishes a **production-ready multi-modal architecture** that:

1. ✅ **Extends existing system** without breaking changes
2. ✅ **Leverages Claude 3.5 Sonnet** for unified vision and conversation
3. ✅ **Uses proven libraries** (PyMuPDF, python-docx) for document processing
4. ✅ **Maintains performance targets** (< 2s images, < 5s/page documents)
5. ✅ **Integrates seamlessly** with Phase 2 agents and Phase 3 RAG
6. ✅ **Prioritizes security** with comprehensive validation
7. ✅ **Scales for Railway deployment** with local storage

### Next Steps

1. **Review and approve** this architecture document
2. **Create implementation tickets** for 5-week rollout
3. **Set up test fixtures** and validation datasets
4. **Begin Week 1 implementation** (Core Infrastructure)

**Status:** Ready for implementation
**Estimated Duration:** 5 weeks
**Risk Level:** Low (building on stable Phase 1-3 foundation)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Author:** System Architect Agent
**Review Status:** Pending approval
