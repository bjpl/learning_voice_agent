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