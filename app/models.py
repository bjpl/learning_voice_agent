"""
Pydantic Models for Request/Response Validation
PATTERN: Contract-first development
WHY: Type safety and automatic documentation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class ConversationRequest(BaseModel):
    """Request model for conversation endpoint"""
    session_id: Optional[str] = Field(None, description="Session ID for context")
    text: Optional[str] = Field(None, description="Text input")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "abc-123",
                "text": "I'm learning about distributed systems"
            }
        }

class ConversationResponse(BaseModel):
    """Response model for conversation"""
    session_id: str
    user_text: str
    agent_text: str
    intent: str = "statement"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SearchRequest(BaseModel):
    """Request model for search"""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    
class SearchResponse(BaseModel):
    """Response model for search"""
    query: str
    results: List[Dict]
    count: int

class TwilioVoiceRequest(BaseModel):
    """Twilio webhook request"""
    CallSid: str
    From: str
    To: str
    CallStatus: str
    RecordingUrl: Optional[str] = None
    SpeechResult: Optional[str] = None

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., pattern="^(audio|text|end|ping)$")
    audio: Optional[str] = None
    text: Optional[str] = None
    session_id: Optional[str] = None

class HybridSearchRequest(BaseModel):
    """Request model for hybrid search"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    strategy: Optional[str] = Field(
        None,
        description="Search strategy: semantic, keyword, hybrid, or adaptive (default)",
        pattern="^(semantic|keyword|hybrid|adaptive)$"
    )
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")

    class Config:
        schema_extra = {
            "example": {
                "query": "explain machine learning concepts",
                "strategy": "hybrid",
                "limit": 10
            }
        }

class HybridSearchResponse(BaseModel):
    """Response model for hybrid search"""
    query: str
    strategy: str
    results: List[Dict]
    total_count: int
    query_analysis: Dict
    execution_time_ms: float
    vector_results_count: int = 0
    keyword_results_count: int = 0

    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning",
                "strategy": "hybrid",
                "results": [
                    {
                        "id": 1,
                        "score": 0.95,
                        "user_text": "Tell me about machine learning",
                        "agent_text": "Machine learning is...",
                        "timestamp": "2025-11-21T10:00:00"
                    }
                ],
                "total_count": 10,
                "query_analysis": {"intent": "conceptual", "keywords": ["machine", "learning"]},
                "execution_time_ms": 45.2,
                "vector_results_count": 8,
                "keyword_results_count": 6
            }
        }

# ============================================================================
# MULTIMODAL MODELS (Phase 4)
# ============================================================================

class ImageUploadResponse(BaseModel):
    """Response model for image upload"""
    file_id: str
    url: str
    filename: str
    size: int
    mime_type: str
    analysis: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "file_id": "uuid-1234",
                "url": "/api/files/uuid-1234",
                "filename": "diagram.png",
                "size": 245678,
                "mime_type": "image/png",
                "analysis": {
                    "success": True,
                    "analysis": "This image shows a system architecture diagram...",
                    "processing_time_ms": 1250.5
                },
                "timestamp": "2025-11-21T10:00:00"
            }
        }

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    file_id: str
    url: str
    filename: str
    size: int
    mime_type: str
    text_preview: Optional[str] = None
    chunk_count: int = 0
    metadata: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
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
                },
                "timestamp": "2025-11-21T10:00:00"
            }
        }

class FileMetadata(BaseModel):
    """File metadata model"""
    file_id: str
    file_type: str
    original_filename: str
    mime_type: str
    file_size: int
    upload_timestamp: str
    session_id: str
    analysis_status: Optional[str] = None
    indexed: bool = False

class VisionAnalysisResult(BaseModel):
    """Vision analysis result model"""
    success: bool
    analysis: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: float
    tokens_used: Optional[int] = None
    timestamp: str

class MultiModalConversationRequest(BaseModel):
    """Request model for multimodal conversation"""
    text: str = Field(..., min_length=1, description="User message text")
    image_ids: List[str] = Field(default=[], description="List of uploaded image IDs")
    document_ids: List[str] = Field(default=[], description="List of uploaded document IDs")
    session_id: Optional[str] = Field(None, description="Session ID for context")

    class Config:
        schema_extra = {
            "example": {
                "text": "Can you explain what's in this image and relate it to the document?",
                "image_ids": ["uuid-1234"],
                "document_ids": ["uuid-5678"],
                "session_id": "session-abc"
            }
        }

class MultiModalConversationResponse(BaseModel):
    """Response model for multimodal conversation"""
    session_id: str
    user_text: str
    agent_text: str
    intent: str = "multimodal_query"
    image_count: int = 0
    document_count: int = 0
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session-abc",
                "user_text": "Can you explain what's in this image?",
                "agent_text": "This image shows a system architecture with three main components...",
                "intent": "multimodal_query",
                "image_count": 1,
                "document_count": 0,
                "processing_time_ms": 2340.5,
                "timestamp": "2025-11-21T10:00:00"
            }
        }