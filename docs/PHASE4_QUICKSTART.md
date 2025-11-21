# Phase 4: Multi-Modal API Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Create .env file or export variables
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"
```

### 3. Create Upload Directories

```bash
mkdir -p uploads/images uploads/documents
```

### 4. Start Server

```bash
uvicorn app.main:app --reload --port 8000
```

## Quick Examples

### Upload & Analyze Image

```bash
curl -X POST "http://localhost:8000/api/upload/image" \
  -F "file=@diagram.png" \
  -F "session_id=test-session" \
  -F "analyze=true"
```

### Upload & Process Document

```bash
curl -X POST "http://localhost:8000/api/upload/document" \
  -F "file=@research.pdf" \
  -F "session_id=test-session" \
  -F "extract_text=true"
```

### Multi-Modal Conversation

```bash
curl -X POST "http://localhost:8000/api/conversation/multimodal" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is shown in this image?",
    "image_ids": ["<file-id-from-upload>"],
    "document_ids": [],
    "session_id": "test-session"
  }'
```

### Retrieve File

```bash
curl "http://localhost:8000/api/files/<file-id>?file_type=image" \
  --output retrieved_file.png
```

## Python Example

```python
import requests

# 1. Upload image
with open('diagram.png', 'rb') as f:
    resp = requests.post(
        'http://localhost:8000/api/upload/image',
        files={'file': f},
        data={'session_id': 'my-session', 'analyze': 'true'}
    )
    image_id = resp.json()['file_id']
    print(f"Image uploaded: {image_id}")

# 2. Have conversation
conv_resp = requests.post(
    'http://localhost:8000/api/conversation/multimodal',
    json={
        'text': 'Explain what you see',
        'image_ids': [image_id],
        'document_ids': [],
        'session_id': 'my-session'
    }
)
print(conv_resp.json()['agent_text'])
```

## Test Everything

```bash
# Run tests
pytest tests/test_multimodal_endpoints.py -v

# Check API docs
open http://localhost:8000/docs
```

## Supported Formats

**Images**: PNG, JPEG, GIF, WebP (max 5MB)
**Documents**: PDF, DOCX, TXT, MD (max 10MB)

## Rate Limits

- Image upload: 10/minute
- Document upload: 5/minute
- Multimodal conversation: 20/minute

## Next Steps

1. Try the [interactive API docs](http://localhost:8000/docs)
2. Read [full documentation](PHASE4_MULTIMODAL.md)
3. Explore example workflows
4. Integrate with your application

## Troubleshooting

**Upload fails?**
- Check file size limits
- Verify file format
- Ensure upload directories exist

**Analysis not working?**
- Verify ANTHROPIC_API_KEY is set
- Check API key has Claude Vision access

**Indexing fails?**
- Verify OPENAI_API_KEY is set
- Check embedding API is accessible

Need help? Check `/docs/PHASE4_MULTIMODAL.md` for detailed documentation.
