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
from twilio.twiml.voice_response import VoiceResponse, Gather, Say
from twilio.request_validator import RequestValidator
from typing import Dict, Optional
from datetime import datetime
import hashlib
import hmac

from app.config import settings
from app.conversation_handler import conversation_handler
from app.state_manager import state_manager
from app.database import db
from app.audio_pipeline import audio_pipeline
from app.logger import twilio_logger, set_request_context, clear_request_context

router = APIRouter(prefix="/twilio", tags=["twilio"])

class TwilioHandler:
    def __init__(self):
        self.validator = RequestValidator(settings.twilio_auth_token) if settings.twilio_auth_token else None
        
    def validate_request(self, request: Request, body: str) -> bool:
        """
        PATTERN: Request validation for security
        WHY: Ensure requests are from Twilio
        """
        if not self.validator:
            return True  # Skip validation in dev
            
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
    # Parse form data
    form = await request.form()
    call_sid = form.get('CallSid')
    from_number = form.get('From')
    call_status = form.get('CallStatus')
    
    # Validate request
    body = await request.body()
    if not twilio_handler.validate_request(request, body.decode()):
        raise HTTPException(403, "Invalid request signature")
    
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
    
    # Set request context for tracing
    set_request_context(session_id=session_id, endpoint="/twilio/process-speech")

    try:
        twilio_logger.info(
            "twilio_speech_received",
            session_id=session_id,
            confidence=confidence,
            speech_length=len(speech_result)
        )

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
            twilio_logger.info("twilio_call_ending", session_id=session_id)
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

        twilio_logger.info(
            "twilio_response_generated",
            session_id=session_id,
            intent=intent
        )

    except Exception as e:
        twilio_logger.error(
            "twilio_speech_processing_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        response_xml = twilio_handler.create_gather_response(
            "I had trouble processing that. Let's try again.",
            session_id
        )
    finally:
        clear_request_context()
    
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
    app.include_router(router)