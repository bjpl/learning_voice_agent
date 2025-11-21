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