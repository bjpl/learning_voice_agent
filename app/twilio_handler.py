"""
Twilio Voice Integration - SPARC Implementation

SPECIFICATION:
- Handle Twilio voice webhooks
- Stream audio for real-time transcription
- Manage phone conversation state
- Support TwiML responses

PSEUDOCODE:
1. Receive Twilio webhook
2. Extract audio or speech result
3. Process through conversation pipeline
4. Generate TwiML response with TTS
5. Return to Twilio

ARCHITECTURE:
- Webhook endpoint for Twilio
- TwiML generation for responses
- Session management for phone calls

CODE:
"""
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import Response
from typing import Dict, Optional, Any
from datetime import datetime
import hashlib
import hmac

# Conditional imports for twilio
try:
    from twilio.twiml.voice_response import VoiceResponse, Gather, Say
    from twilio.request_validator import RequestValidator
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    VoiceResponse = None
    Gather = None
    Say = None
    RequestValidator = None

from app.config import settings
from app.conversation_handler import conversation_handler
from app.state_manager import state_manager
from app.database import db
from app.audio_pipeline import audio_pipeline
from app.logger import logger

router = APIRouter(prefix="/twilio", tags=["twilio"])

class TwilioHandler:
    def __init__(self):
        if TWILIO_AVAILABLE and RequestValidator and settings.twilio_auth_token:
            self.validator = RequestValidator(settings.twilio_auth_token)
        else:
            self.validator = None
        
    def validate_request(self, request: Request, body: str) -> bool:
        """
        PATTERN: Request validation for security
        WHY: Ensure requests are from Twilio

        SECURITY FIX: Fail closed when validator not configured
        Previous behavior (insecure): return True when no validator
        Current behavior (secure): return False when no validator
        """
        import os
        import logging

        logger = logging.getLogger(__name__)

        if not self.validator:
            # SECURITY: Fail closed - reject if not properly configured
            environment = os.getenv("ENVIRONMENT", "development").lower()

            if environment == "production":
                logger.error(
                    "SECURITY: Twilio validator not configured in production! "
                    "Set TWILIO_AUTH_TOKEN environment variable. Rejecting request."
                )
                return False

            # In development, warn but allow (with explicit opt-in)
            allow_unvalidated = os.getenv("TWILIO_ALLOW_UNVALIDATED", "false").lower() == "true"
            if allow_unvalidated:
                logger.warning(
                    "SECURITY WARNING: Twilio validation disabled in development. "
                    "This should NEVER happen in production."
                )
                return True
            else:
                logger.warning(
                    "Twilio validator not configured. Set TWILIO_AUTH_TOKEN or "
                    "TWILIO_ALLOW_UNVALIDATED=true for development."
                )
                return False
            
        signature = request.headers.get('X-Twilio-Signature', '')
        url = str(request.url)
        
        return self.validator.validate(url, {}, signature)
    
    def create_gather_response(
        self,
        prompt: str,
        session_id: str,
        timeout: int = 3
    ) -> str:
        """
        PATTERN: TwiML response generation
        WHY: Twilio needs specific XML format
        """
        response = VoiceResponse()
        
        # Create gather for speech input
        gather = Gather(
            input='speech',
            timeout=timeout,
            language='en-US',
            action=f'/twilio/process-speech?session_id={session_id}',
            method='POST',
            speechTimeout='auto',
            enhanced=True  # Better speech recognition
        )
        
        # Use neural voice for better quality
        gather.say(
            prompt,
            voice='Polly.Joanna-Neural',
            language='en-US'
        )
        
        response.append(gather)
        
        # Fallback if no input
        response.say("I didn't catch that. Let's try again.")
        response.redirect(f'/twilio/voice?session_id={session_id}')
        
        return str(response)
    
    def create_say_response(self, text: str, end_call: bool = False) -> str:
        """Generate simple TwiML say response"""
        response = VoiceResponse()
        
        response.say(
            text,
            voice='Polly.Joanna-Neural',
            language='en-US'
        )
        
        if end_call:
            response.hangup()
        
        return str(response)

# Create handler instance
twilio_handler = TwilioHandler()

@router.post("/voice")
async def handle_voice_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    PATTERN: Main Twilio voice webhook
    WHY: Entry point for phone calls
    """
    # Read body first for validation (stream can only be consumed once)
    body = await request.body()
    body_str = body.decode()

    # Validate request before processing
    if not twilio_handler.validate_request(request, body_str):
        raise HTTPException(403, "Invalid request signature")

    # Parse form data from body (since stream is consumed)
    from urllib.parse import parse_qs
    parsed = parse_qs(body_str)
    call_sid = parsed.get('CallSid', [''])[0]
    from_number = parsed.get('From', [''])[0]
    call_status = parsed.get('CallStatus', [''])[0]
    
    # Create or get session
    session_id = f"twilio_{call_sid}"
    
    # Handle different call statuses
    if call_status == 'ringing':
        # Initial greeting
        greeting = "Hello! I'm your learning companion. Tell me what you're working on or learning about today."
        response_xml = twilio_handler.create_gather_response(greeting, session_id)
        
    elif call_status == 'in-progress':
        # Continue conversation
        response_xml = twilio_handler.create_gather_response(
            "What else would you like to explore?",
            session_id
        )
        
    else:
        # End call
        response_xml = twilio_handler.create_say_response(
            "Thanks for sharing your thoughts. Goodbye!",
            end_call=True
        )
    
    # Update session metadata
    background_tasks.add_task(
        update_twilio_session,
        session_id,
        from_number,
        call_status
    )
    
    return Response(content=response_xml, media_type="application/xml")

@router.post("/process-speech")
async def process_speech(
    request: Request,
    session_id: str,
    background_tasks: BackgroundTasks
):
    """
    PATTERN: Process speech input from Twilio
    WHY: Handle transcribed speech and generate response
    """
    # Parse form data
    form = await request.form()
    speech_result = form.get('SpeechResult', '')
    confidence = float(form.get('Confidence', 0))
    
    if not speech_result:
        # No speech detected
        response_xml = twilio_handler.create_gather_response(
            "I didn't hear anything. Could you repeat that?",
            session_id
        )
        return Response(content=response_xml, media_type="application/xml")
    
    # Low confidence threshold
    if confidence < 0.5:
        response_xml = twilio_handler.create_gather_response(
            "I'm not sure I understood. Could you say that again?",
            session_id
        )
        return Response(content=response_xml, media_type="application/xml")
    
    try:
        # Get conversation context
        context = await state_manager.get_conversation_context(session_id)
        
        # Generate response using Claude
        agent_response = await conversation_handler.generate_response(
            speech_result,
            context
        )
        
        # Detect if conversation should end
        intent = conversation_handler.detect_intent(speech_result)
        
        if intent == "end_conversation":
            # Create summary and end call
            summary = conversation_handler.create_summary(context)
            response_xml = twilio_handler.create_say_response(
                f"{agent_response} {summary}",
                end_call=True
            )
        else:
            # Continue conversation
            response_xml = twilio_handler.create_gather_response(
                agent_response,
                session_id,
                timeout=5  # Longer timeout for thoughtful responses
            )
        
        # Save exchange in background
        background_tasks.add_task(
            save_twilio_exchange,
            session_id,
            speech_result,
            agent_response
        )
        
    except Exception as e:
        print(f"Error processing speech: {e}")
        response_xml = twilio_handler.create_gather_response(
            "I had trouble processing that. Let's try again.",
            session_id
        )
    
    return Response(content=response_xml, media_type="application/xml")

@router.post("/recording")
async def handle_recording(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    PATTERN: Handle recorded audio from Twilio
    WHY: Process longer audio recordings
    """
    form = await request.form()
    recording_url = form.get('RecordingUrl')
    call_sid = form.get('CallSid')
    
    if recording_url:
        # Process recording in background
        background_tasks.add_task(
            process_recording,
            recording_url,
            f"twilio_{call_sid}"
        )
    
    # Acknowledge receipt
    response = VoiceResponse()
    response.say("Got it. Processing your recording.")
    
    return Response(content=str(response), media_type="application/xml")

async def update_twilio_session(
    session_id: str,
    from_number: str,
    call_status: str
):
    """Update Twilio session metadata"""
    metadata = {
        "source": "twilio",
        "from_number": from_number,
        "call_status": call_status,
        "created_at": datetime.utcnow().isoformat()
    }
    
    await state_manager.update_session_metadata(session_id, metadata)

async def save_twilio_exchange(
    session_id: str,
    user_text: str,
    agent_text: str
):
    """Save Twilio conversation exchange"""
    # Update Redis context
    await state_manager.update_conversation_context(
        session_id,
        user_text,
        agent_text
    )
    
    # Save to database
    await db.save_exchange(
        session_id,
        user_text,
        agent_text,
        metadata={"source": "twilio"}
    )

async def process_recording(recording_url: str, session_id: str):
    """
    Process Twilio recording asynchronously
    NOTE: Simplified - full implementation would download and process
    """
    # In production, download recording from Twilio
    # Process through audio pipeline
    # Generate response and callback to user
    pass

# Add router to main app
def setup_twilio_routes(app):
    """Setup Twilio routes on main app"""
    if not TWILIO_AVAILABLE:
        logger.warning("Twilio routes not available - twilio package not installed")
        return
    app.include_router(router)