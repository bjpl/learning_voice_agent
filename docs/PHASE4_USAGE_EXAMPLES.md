# Phase 4 Usage Examples: Multi-Modal Capabilities

**Version:** 1.0.0
**Date:** 2025-01-21

Complete end-to-end examples for Phase 4 multi-modal features.

## Table of Contents

1. [Image Upload and Analysis](#image-upload-and-analysis)
2. [Document Upload and Processing](#document-upload-and-processing)
3. [Multi-Modal Conversations](#multi-modal-conversations)
4. [RAG with Images and Documents](#rag-with-images-and-documents)
5. [Python Client Examples](#python-client-examples)
6. [curl Examples](#curl-examples)
7. [Advanced Use Cases](#advanced-use-cases)

---

## Image Upload and Analysis

### Example 1: Upload System Diagram

**Scenario:** Upload an architecture diagram and get AI analysis.

```python
import requests

# Upload diagram
with open("architecture_diagram.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload/image",
        files={"file": ("architecture.png", f, "image/png")},
        data={
            "session_id": "sess_architect_123",
            "description": "Microservices architecture diagram",
            "analyze": "true"
        }
    )

result = response.json()

print(f"✅ Uploaded: {result['file_id']}")
print(f"📊 Analysis: {result['analysis']['analysis']}")
print(f"🔢 Tokens used: {result['analysis']['tokens_used']}")
print(f"🖼️  Thumbnail: {result['thumbnail_url']}")

# Example output:
# ✅ Uploaded: a1b2c3d4-5678-90ab-cdef-1234567890ab
# 📊 Analysis: This diagram shows a microservices architecture with an API Gateway
#              at the front, routing requests to Auth Service, User Service, and
#              Order Service. Each service connects to its own database. A message
#              queue (RabbitMQ) handles asynchronous communication between services.
# 🔢 Tokens used: 1245
# 🖼️  Thumbnail: /files/thumbnails/a1b2c3d4-..._thumb
```

### Example 2: Screenshot with OCR

**Scenario:** Upload a screenshot and extract text from it.

```python
import requests

with open("error_screenshot.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload/image",
        files={"file": f},
        data={
            "session_id": "sess_debug_456",
            "description": "Error message screenshot",
            "analyze": "true"
        }
    )

result = response.json()

# The analysis will include extracted text
print(f"Error message: {result['analysis']['analysis']}")
# Example: "The screenshot shows an error dialog with text:
#           'Connection timeout: Unable to reach authentication server
#            at auth.api.example.com. Please check your network connection.'"
```

### Example 3: Compare Before/After Images

```python
from app.vision.vision_analyzer import vision_analyzer

# Upload and compare UI changes
result = await vision_analyzer.compare_images(
    "ui_before.png",
    "ui_after.png"
)

print("Comparison:")
print(result["comparison"])

# Example output:
# "The main differences are:
#  1. Navigation menu has been redesigned with a cleaner layout
#  2. Color scheme changed from blue to green
#  3. Search bar moved from top-right to center-top
#  4. User profile icon replaced with avatar image
#  5. Footer simplified with fewer links"
```

---

## Document Upload and Processing

### Example 4: Upload Research Paper (PDF)

```python
import requests

# Upload PDF
with open("research_paper.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload/document",
        files={"file": ("paper.pdf", f, "application/pdf")},
        data={
            "session_id": "sess_research_789",
            "description": "Paper on microservices patterns"
        }
    )

result = response.json()

print(f"✅ Uploaded: {result['file_id']}")
print(f"📄 Pages: {result['page_count']}")
print(f"📦 Chunks: {result['chunks']}")
print(f"👤 Author: {result['metadata']['author']}")
print(f"📝 Title: {result['metadata']['title']}")
print(f"\n📖 Preview:")
print(result['text'][:300] + "...")

# Example output:
# ✅ Uploaded: b2c3d4e5-6789-01ab-cdef-234567890abc
# 📄 Pages: 12
# 📦 Chunks: 25
# 👤 Author: Jane Smith
# 📝 Title: Microservices Design Patterns: A Comprehensive Guide
#
# 📖 Preview:
# Abstract
#
# This paper examines common design patterns in microservices architecture,
# including API Gateway, Service Discovery, Circuit Breaker, and Saga patterns.
# We analyze real-world implementations and provide recommendations for...
```

### Example 5: Upload Meeting Notes (Markdown)

```python
import requests

with open("meeting_notes.md", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload/document",
        files={"file": f},
        data={
            "session_id": "sess_meeting_101",
            "description": "Weekly team sync notes"
        }
    )

result = response.json()

print(f"Indexed {result['chunks']} sections from meeting notes")
# Now searchable: "What did we discuss about the API redesign?"
```

### Example 6: Process DOCX Document

```python
from app.documents.document_processor import document_processor

# Process Word document
result = await document_processor.process_document("proposal.docx")

print(f"Document format: {result['format']}")
print(f"Total text length: {len(result['text'])} characters")
print(f"Number of chunks: {len(result['chunks'])}")

# Access metadata
meta = result['metadata']
print(f"Author: {meta['author']}")
print(f"Created: {meta['created']}")
print(f"Modified: {meta['modified']}")

# Iterate through chunks
for i, chunk in enumerate(result['chunks'][:3]):
    print(f"\nChunk {i}:")
    print(chunk.text[:200] + "...")
```

---

## Multi-Modal Conversations

### Example 7: Ask About Uploaded Image

```python
import requests

# First, upload an image
with open("database_schema.png", "rb") as f:
    upload_response = requests.post(
        "http://localhost:8000/api/upload/image",
        files={"file": f},
        data={
            "session_id": "sess_learning_202",
            "description": "Database schema diagram"
        }
    )

file_id = upload_response.json()["file_id"]

# Now ask questions about it
conversation_response = requests.post(
    "http://localhost:8000/api/upload/conversation/multimodal",
    data={
        "text": "Explain the relationships in this database schema",
        "session_id": "sess_learning_202",
        "file_ids": [file_id]
    }
)

result = conversation_response.json()

print(f"🤖 AI Response: {result['response']}")
print(f"\n📚 Sources used:")
for source in result['sources']:
    print(f"  - {source['type']}: {source['content'][:100]}...")

# Example output:
# 🤖 AI Response: Based on the database schema you shared, this appears to be
#                 an e-commerce system. The Users table has a one-to-many
#                 relationship with Orders through the user_id foreign key...
#
# 📚 Sources used:
#   - image: A database schema diagram showing five tables: Users, Orders,
#            OrderItems, Products, and Categories. The Users table...
```

### Example 8: Multi-Modal Learning Session

```python
import requests

session_id = "sess_ml_learning_303"

# Upload course materials
materials = [
    ("neural_network_diagram.png", "Neural network architecture"),
    ("backpropagation_explained.pdf", "Backpropagation algorithm paper"),
    ("code_example.txt", "PyTorch implementation")
]

file_ids = []

for filename, description in materials:
    file_type = "image" if filename.endswith(".png") else "document"
    endpoint = f"/api/upload/{file_type}"

    with open(filename, "rb") as f:
        response = requests.post(
            f"http://localhost:8000{endpoint}",
            files={"file": f},
            data={
                "session_id": session_id,
                "description": description
            }
        )
        file_ids.append(response.json()["file_id"])

print(f"✅ Uploaded {len(file_ids)} materials")

# Now have a conversation with context from all materials
questions = [
    "How does backpropagation work in neural networks?",
    "Can you explain the diagram's architecture?",
    "How is this implemented in the code example?"
]

for question in questions:
    response = requests.post(
        "http://localhost:8000/api/upload/conversation/multimodal",
        data={
            "text": question,
            "session_id": session_id,
            "file_ids": file_ids
        }
    )

    result = response.json()
    print(f"\n❓ Q: {question}")
    print(f"🤖 A: {result['response']}")
```

---

## RAG with Images and Documents

### Example 9: Semantic Search Across Multi-Modal Content

```python
from app.storage.multimodal_indexer import multimodal_indexer

# Search for relevant content
context = await multimodal_indexer.retrieve_context(
    query="authentication best practices",
    session_id="sess_security_404",
    k=5
)

print(f"Found {len(context['sources'])} relevant sources:\n")

for i, source in enumerate(context['sources'], 1):
    print(f"{i}. Type: {source['type']}")
    if source['type'] == 'image':
        print(f"   Image analysis: {source['content'][:100]}...")
        print(f"   Dimensions: {source['metadata']['dimensions']}")
    elif source['type'] == 'document':
        print(f"   Text excerpt: {source['content'][:100]}...")
        print(f"   Page/Chunk: {source['metadata']['chunk_index']}")
    print()

# Example output:
# Found 5 relevant sources:
#
# 1. Type: document
#    Text excerpt: Authentication Best Practices: Always use HTTPS for login pages.
#                  Implement multi-factor authentication (MFA) for sensitive...
#    Page/Chunk: 3
#
# 2. Type: image
#    Image analysis: A diagram showing OAuth 2.0 authentication flow with
#                    Authorization Server, Client, and Resource Server...
#    Dimensions: (1200, 800)
#
# 3. Type: document
#    Text excerpt: JWT tokens should have short expiration times. Store refresh
#                  tokens securely. Never store passwords in plain text...
#    Page/Chunk: 15
```

### Example 10: Build Enhanced Context for RAG

```python
from app.agents.conversation_agent import ConversationAgent
from app.storage.multimodal_indexer import multimodal_indexer

async def enhanced_conversation(user_input: str, session_id: str):
    """Conversation with multi-modal RAG context"""

    # Retrieve relevant multi-modal context
    context = await multimodal_indexer.retrieve_context(
        query=user_input,
        session_id=session_id,
        k=3
    )

    # Build enhanced prompt with context
    context_parts = []

    for source in context['sources']:
        if source['type'] == 'image':
            context_parts.append(
                f"[IMAGE CONTEXT: {source['content']}]"
            )
        elif source['type'] == 'document':
            context_parts.append(
                f"[DOCUMENT CONTEXT: {source['content']}]"
            )

    enhanced_prompt = f"""
Context from uploaded materials:
{chr(10).join(context_parts)}

User question: {user_input}

Please answer based on the context provided, citing specific sources.
"""

    # Generate response with enhanced context
    agent = ConversationAgent()
    response = await agent.process(enhanced_prompt)

    return {
        "response": response,
        "sources": context['sources']
    }

# Usage
result = await enhanced_conversation(
    "What are the key components in our microservices architecture?",
    "sess_architect_123"
)

print(result["response"])
```

---

## Python Client Examples

### Example 11: Python SDK for Multi-Modal Operations

```python
class LearningVoiceClient:
    """Client SDK for Learning Voice Agent"""

    def __init__(self, base_url="http://localhost:8000", session_id=None):
        self.base_url = base_url
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"

    async def upload_image(self, image_path: str, description: str = None):
        """Upload image and get analysis"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/upload/image",
                files={"file": f},
                data={
                    "session_id": self.session_id,
                    "description": description,
                    "analyze": "true"
                }
            )
        return response.json()

    async def upload_document(self, doc_path: str, description: str = None):
        """Upload document and extract text"""
        with open(doc_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/upload/document",
                files={"file": f},
                data={
                    "session_id": self.session_id,
                    "description": description
                }
            )
        return response.json()

    async def ask(self, question: str, file_ids: list = None):
        """Ask question with multi-modal context"""
        response = requests.post(
            f"{self.base_url}/api/upload/conversation/multimodal",
            data={
                "text": question,
                "session_id": self.session_id,
                "file_ids": file_ids or []
            }
        )
        return response.json()

# Usage
client = LearningVoiceClient()

# Upload materials
diagram = await client.upload_image(
    "architecture.png",
    "System architecture diagram"
)

paper = await client.upload_document(
    "design_doc.pdf",
    "Design documentation"
)

# Ask questions
result = await client.ask(
    "How does the authentication service work?",
    file_ids=[diagram['file_id'], paper['file_id']]
)

print(result['response'])
```

---

## curl Examples

### Example 12: Upload Image with curl

```bash
# Upload image
curl -X POST http://localhost:8000/api/upload/image \
  -F "file=@diagram.png" \
  -F "session_id=sess_curl_test" \
  -F "description=Architecture diagram" \
  -F "analyze=true" \
  | jq .

# Response:
# {
#   "file_id": "a1b2c3d4-...",
#   "url": "/files/images/a1b2c3d4-...",
#   "thumbnail_url": "/files/thumbnails/a1b2c3d4-..._thumb",
#   "analysis": {
#     "analysis": "A diagram showing...",
#     "tokens_used": 1245
#   }
# }
```

### Example 13: Upload Document with curl

```bash
# Upload PDF
curl -X POST http://localhost:8000/api/upload/document \
  -F "file=@research_paper.pdf" \
  -F "session_id=sess_curl_test" \
  -F "description=Research paper" \
  | jq '.file_id, .page_count, .chunks'

# Response:
# "b2c3d4e5-..."
# 12
# 25
```

### Example 14: Multi-Modal Conversation with curl

```bash
# Ask question with context
curl -X POST http://localhost:8000/api/upload/conversation/multimodal \
  -d "text=Explain the authentication flow" \
  -d "session_id=sess_curl_test" \
  -d "file_ids=a1b2c3d4-..." \
  | jq '.response'

# Response:
# "Based on the architecture diagram you shared, the authentication flow
#  begins when a user submits credentials to the API Gateway..."
```

### Example 15: Download Uploaded File

```bash
# Download image
curl http://localhost:8000/api/upload/files/images/a1b2c3d4-... \
  -o downloaded_image.png

# Download document
curl http://localhost:8000/api/upload/files/documents/b2c3d4e5-... \
  -o downloaded_document.pdf
```

---

## Advanced Use Cases

### Example 16: Batch Upload Multiple Files

```python
import asyncio
import requests
from pathlib import Path

async def batch_upload(directory: str, session_id: str):
    """Upload all files in a directory"""

    directory_path = Path(directory)
    results = []

    for file_path in directory_path.iterdir():
        if file_path.is_file():
            # Determine file type
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                endpoint = "/api/upload/image"
            elif file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md']:
                endpoint = "/api/upload/document"
            else:
                continue

            # Upload
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"http://localhost:8000{endpoint}",
                    files={"file": f},
                    data={
                        "session_id": session_id,
                        "description": f"Uploaded from {file_path.name}"
                    }
                )

            if response.status_code == 200:
                results.append({
                    "filename": file_path.name,
                    "file_id": response.json()["file_id"]
                })
                print(f"✅ Uploaded: {file_path.name}")
            else:
                print(f"❌ Failed: {file_path.name}")

    return results

# Usage
results = await batch_upload("./course_materials", "sess_batch_505")
print(f"\n📦 Uploaded {len(results)} files total")
```

### Example 17: Progressive Document Analysis

```python
from app.documents.document_processor import document_processor
from app.storage.multimodal_indexer import multimodal_indexer

async def analyze_document_progressively(pdf_path: str, session_id: str):
    """Process and analyze large document page by page"""

    # Process document
    result = await document_processor.process_document(pdf_path)

    print(f"Processing {result['page_count']} pages...")

    # Index chunks progressively
    for i, chunk in enumerate(result['chunks']):
        await multimodal_indexer.index_document(
            file_id=f"doc_{session_id}_{i}",
            chunks=[chunk],
            metadata={
                "session_id": session_id,
                "chunk_index": i,
                "total_chunks": len(result['chunks'])
            }
        )

        # Show progress
        progress = (i + 1) / len(result['chunks']) * 100
        print(f"Progress: {progress:.1f}%", end='\r')

    print("\n✅ Document fully indexed and searchable")

# Usage
await analyze_document_progressively("large_book.pdf", "sess_book_606")
```

### Example 18: Image Comparison Workflow

```python
from app.vision.vision_analyzer import vision_analyzer

async def compare_ui_versions(before: str, after: str, session_id: str):
    """Compare two UI versions and generate change report"""

    # Analyze both versions
    before_result = await vision_analyzer.analyze_image(
        before,
        prompt="Describe this user interface in detail"
    )

    after_result = await vision_analyzer.analyze_image(
        after,
        prompt="Describe this user interface in detail"
    )

    # Compare
    comparison = await vision_analyzer.compare_images(before, after)

    # Generate report
    report = f"""
UI Comparison Report
{'='*50}

BEFORE:
{before_result['analysis']}

AFTER:
{after_result['analysis']}

CHANGES:
{comparison['comparison']}

TOKENS USED: {comparison['tokens_used']}
"""

    print(report)
    return report

# Usage
await compare_ui_versions(
    "ui_v1.png",
    "ui_v2.png",
    "sess_ui_design_707"
)
```

### Example 19: Multi-Modal Knowledge Base

```python
class MultiModalKnowledgeBase:
    """Build and query a multi-modal knowledge base"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_ids = []

    async def add_image(self, path: str, description: str):
        """Add image to knowledge base"""
        # Upload and store file_id
        pass

    async def add_document(self, path: str, description: str):
        """Add document to knowledge base"""
        pass

    async def query(self, question: str) -> dict:
        """Query the knowledge base"""
        response = requests.post(
            "http://localhost:8000/api/upload/conversation/multimodal",
            data={
                "text": question,
                "session_id": self.session_id,
                "file_ids": self.file_ids
            }
        )
        return response.json()

# Usage
kb = MultiModalKnowledgeBase("sess_kb_808")

# Build knowledge base
await kb.add_image("architecture.png", "System architecture")
await kb.add_image("database_schema.png", "Database design")
await kb.add_document("api_docs.pdf", "API documentation")
await kb.add_document("deployment_guide.md", "Deployment instructions")

# Query
result = await kb.query("How do I deploy the authentication service?")
print(result['response'])
```

---

## Performance Optimization

### Example 20: Caching Vision Analysis

```python
from functools import lru_cache
import hashlib

class CachedVisionAnalyzer:
    """Vision analyzer with result caching"""

    def __init__(self):
        self.analyzer = vision_analyzer
        self.cache = {}

    def _compute_cache_key(self, image_path: str, prompt: str) -> str:
        """Compute cache key from image hash and prompt"""
        with open(image_path, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
        return f"{image_hash}:{prompt}"

    async def analyze_image(self, image_path: str, prompt: str):
        """Analyze with caching"""
        cache_key = self._compute_cache_key(image_path, prompt)

        if cache_key in self.cache:
            print("✨ Using cached result")
            return self.cache[cache_key]

        result = await self.analyzer.analyze_image(image_path, prompt)
        self.cache[cache_key] = result
        return result

# Usage
cached_analyzer = CachedVisionAnalyzer()

# First call - hits API
result1 = await cached_analyzer.analyze_image("diagram.png", "Describe this")

# Second call - uses cache
result2 = await cached_analyzer.analyze_image("diagram.png", "Describe this")
```

---

For more details, see:
- [PHASE4_IMPLEMENTATION_GUIDE.md](PHASE4_IMPLEMENTATION_GUIDE.md) - Implementation details
- [PHASE4_API_REFERENCE.md](PHASE4_API_REFERENCE.md) - Complete API documentation
- [PHASE4_TESTING_GUIDE.md](PHASE4_TESTING_GUIDE.md) - Testing strategies

**Phase 4 Status:** Ready for implementation.
