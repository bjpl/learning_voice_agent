"""
FastAPI Main Application - Integration Layer
PATTERN: Dependency injection with lifecycle management
"""
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Optional, Dict
from datetime import datetime
import uuid
import json
import time
import traceback

from app.config import settings
from app.database import db
from app.state_manager import state_manager
from app.conversation_handler import conversation_handler
from app.audio_pipeline import audio_pipeline
from app.models import (
    ConversationRequest,
    ConversationResponse,
    SearchRequest,
    SearchResponse
)
from app.twilio_handler import setup_twilio_routes
from app.logger import api_logger, set_request_context, clear_request_context
from app.metrics import (
    metrics_collector,
    CONTENT_TYPE_LATEST,
    http_requests_in_progress,
    websocket_connections_active,
    track_in_progress
)
from app.resilience import RateLimiter, HealthCheck
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
rate_limiter = RateLimiter(max_calls=100, time_window=60)  # 100 calls per minute per IP

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    PATTERN: Application lifecycle management
    WHY: Proper initialization and cleanup
    """
    # Startup
    api_logger.info("application_startup_initiated")
    app.state.startup_time = time.time()
    await db.initialize()
    await state_manager.initialize()
    api_logger.info("application_ready", status="operational")

    yield

    # Shutdown
    api_logger.info("application_shutdown_initiated")
    await state_manager.close()
    api_logger.info("application_shutdown_complete")

# Create FastAPI app
app = FastAPI(
    title="Learning Voice Agent",
    description="AI-powered voice conversation for learning capture",
    version="1.0.0",
    lifespan=lifespan
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    PATTERN: Rate limiting for API protection
    WHY: Prevent abuse and manage API quotas
    """
    # Skip rate limiting for health checks and static files
    if request.url.path in ["/health", "/api/health", "/", "/docs", "/openapi.json", "/metrics", "/api/metrics"] or request.url.path.startswith("/static"):
        return await call_next(request)

    # Get client identifier
    client_id = get_remote_address(request)

    # Check rate limit
    if not await rate_limiter.acquire(client_id):
        retry_after = rate_limiter.get_retry_after(client_id)
        api_logger.warning(
            "rate_limit_exceeded",
            client_id=client_id,
            endpoint=request.url.path,
            retry_after=retry_after
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Please try again in {retry_after} seconds.",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )

    return await call_next(request)

# Metrics Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    PATTERN: Metrics collection middleware
    WHY: Automatic instrumentation of all HTTP requests
    """
    # Extract endpoint path template (not actual path with params)
    endpoint = request.url.path
    method = request.method

    # Track in-progress requests
    async with track_in_progress(http_requests_in_progress, {"endpoint": endpoint}):
        start_time = time.perf_counter()
        status_code = 500  # Default to error

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            api_logger.error(
                "unhandled_exception",
                endpoint=endpoint,
                error=str(e),
                exc_info=True
            )
            raise
        finally:
            # Track metrics
            duration = time.perf_counter() - start_time
            metrics_collector.track_http_request(
                endpoint=endpoint,
                method=method,
                status=status_code,
                duration=duration
            )

# API Routes

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "healthy",
        "service": "Learning Voice Agent",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/{session_id}",
            "twilio": "/twilio/voice",
            "search": "/api/search",
            "stats": "/api/stats",
            "metrics": "/metrics",
            "health": "/api/health",
            "metrics_json": "/api/metrics"
        }
    }

# ============================================================================
# OBSERVABILITY ENDPOINTS
# ============================================================================

@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint
    WHY: Standard endpoint for Prometheus scraping
    """
    # Update active sessions before returning metrics
    active_sessions = await state_manager.get_active_sessions()
    metrics_collector.update_active_sessions(len(active_sessions))

    metrics_data = metrics_collector.get_metrics()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

@app.get("/api/metrics")
async def metrics_json():
    """
    JSON metrics endpoint
    WHY: Human-readable metrics for dashboards and debugging
    """
    # Update gauges before returning
    active_sessions = await state_manager.get_active_sessions()
    metrics_collector.update_active_sessions(len(active_sessions))

    return metrics_collector.get_metrics_dict()

@app.get("/api/health")
async def health_check():
    """
    PATTERN: Comprehensive health check with dependency status
    WHY: Kubernetes/Docker health probes and monitoring
    RESILIENCE: Uses HealthCheck utility from resilience module
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {},
        "metrics": {}
    }

    all_healthy = True

    # Check database using HealthCheck utility
    db_health = await HealthCheck.check_database(db.db_path)
    health_status["dependencies"]["database"] = db_health
    if db_health["status"] == "unhealthy":
        all_healthy = False

    # Check Redis using HealthCheck utility
    if state_manager.redis_client and not state_manager._degraded_mode:
        redis_health = await HealthCheck.check_redis(state_manager.redis_client)
        health_status["dependencies"]["redis"] = redis_health
        if redis_health["status"] == "unhealthy":
            all_healthy = False
    else:
        health_status["dependencies"]["redis"] = {
            "status": "degraded",
            "message": "Operating without Redis cache"
        }
        # Don't mark as unhealthy - degraded mode is acceptable

    # External APIs (just report as configured)
    health_status["dependencies"]["claude_api"] = {
        "status": "configured" if settings.anthropic_api_key else "not_configured",
        "last_check": datetime.utcnow().isoformat()
    }

    health_status["dependencies"]["whisper_api"] = {
        "status": "configured" if settings.openai_api_key else "not_configured",
        "last_check": datetime.utcnow().isoformat()
    }

    # Add metrics summary
    active_sessions = await state_manager.get_active_sessions()
    uptime = time.time() - app.state.startup_time if hasattr(app.state, 'startup_time') else 0

    health_status["metrics"] = {
        "active_sessions": len(active_sessions),
        "uptime_seconds": round(uptime, 2)
    }

    # Set overall status
    health_status["status"] = "healthy" if all_healthy else "degraded"

    status_code = 200 if all_healthy else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.post("/api/conversation")
@limiter.limit("30/minute")
async def handle_conversation(
    request: ConversationRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
) -> ConversationResponse:
    """
    PATTERN: REST endpoint for conversation handling
    WHY: Simple integration for various clients
    """
    # Set request context for tracing
    session_id = request.session_id or str(uuid.uuid4())
    set_request_context(session_id=session_id, endpoint="/api/conversation")

    try:
        api_logger.info(
            "conversation_request_received",
            session_id=session_id,
            has_audio=bool(request.audio_base64),
            has_text=bool(request.text)
        )

        # Transcribe if audio provided
        if request.audio_base64:
            transcribe_start = time.perf_counter()
            user_text = await audio_pipeline.transcribe_base64(
                request.audio_base64,
                source="api"
            )
            # Metrics tracked internally by audio_pipeline
        else:
            user_text = request.text

        if not user_text:
            api_logger.warning("conversation_no_input", session_id=session_id)
            raise HTTPException(400, "No input provided")

        # Get conversation context
        context = await state_manager.get_conversation_context(session_id)

        # Generate response (metrics tracked internally)
        agent_response = await conversation_handler.generate_response(
            user_text,
            context
        )

        intent = conversation_handler.detect_intent(user_text)

        # Track conversation exchange
        metrics_collector.track_conversation_exchange(intent)

        # Update state in background
        background_tasks.add_task(
            update_conversation_state,
            session_id,
            user_text,
            agent_response
        )

        api_logger.info(
            "conversation_response_generated",
            session_id=session_id,
            intent=intent,
            user_text_length=len(user_text),
            agent_text_length=len(agent_response)
        )

        return ConversationResponse(
            session_id=session_id,
            user_text=user_text,
            agent_text=agent_response,
            intent=intent
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(
            "conversation_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(500, str(e))
    finally:
        clear_request_context()

async def update_conversation_state(
    session_id: str,
    user_text: str,
    agent_text: str
):
    """
    PATTERN: Background task for state updates
    WHY: Don't block the response for persistence
    """
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
        metadata={"source": "api"}
    )
    
    # Update session metadata
    metadata = await state_manager.get_session_metadata(session_id) or {
        "created_at": datetime.utcnow().isoformat(),
        "exchange_count": 0
    }
    metadata["exchange_count"] += 1
    await state_manager.update_session_metadata(session_id, metadata)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    PATTERN: WebSocket for real-time audio streaming
    WHY: Lower latency than REST for continuous conversation
    """
    set_request_context(session_id=session_id, endpoint="/ws")
    api_logger.info("websocket_connection_initiated", session_id=session_id)

    await websocket.accept()
    api_logger.info("websocket_connection_accepted", session_id=session_id)

    # Track WebSocket connection
    websocket_connections_active.inc()
    connection_start = time.perf_counter()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "audio":
                api_logger.debug("websocket_audio_received", session_id=session_id)

                # Handle audio data
                user_text = await audio_pipeline.transcribe_base64(
                    message["audio"],
                    source="websocket"
                )

                # Get context and generate response
                context = await state_manager.get_conversation_context(session_id)
                agent_response = await conversation_handler.generate_response(
                    user_text,
                    context
                )

                # Update state
                await update_conversation_state(
                    session_id,
                    user_text,
                    agent_response
                )

                intent = conversation_handler.detect_intent(user_text)

                # Send response
                await websocket.send_json({
                    "type": "response",
                    "user_text": user_text,
                    "agent_text": agent_response,
                    "intent": intent
                })

                api_logger.info(
                    "websocket_response_sent",
                    session_id=session_id,
                    intent=intent
                )

            elif message["type"] == "end":
                api_logger.info("websocket_end_requested", session_id=session_id)

                # End conversation
                summary = conversation_handler.create_summary(
                    await state_manager.get_conversation_context(session_id)
                )
                await websocket.send_json({
                    "type": "summary",
                    "text": summary
                })
                await state_manager.end_session(session_id)
                api_logger.info("websocket_session_ended", session_id=session_id)
                break

    except Exception as e:
        api_logger.error(
            "websocket_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
    finally:
        # Track connection duration
        connection_duration = time.perf_counter() - connection_start
        from app.metrics import websocket_connection_duration_seconds
        websocket_connection_duration_seconds.observe(connection_duration)

        websocket_connections_active.dec()
        await websocket.close()
        api_logger.info("websocket_connection_closed", session_id=session_id)
        clear_request_context()

@app.post("/api/search")
@limiter.limit("60/minute")
async def search_captures(request: SearchRequest, http_request: Request) -> SearchResponse:
    """
    PATTERN: FTS5 search endpoint
    WHY: Fast, relevant search across all captures
    """
    results = await db.search_captures(
        request.query,
        request.limit
    )
    
    return SearchResponse(
        query=request.query,
        results=results,
        count=len(results)
    )

@app.get("/api/stats")
async def get_stats() -> Dict:
    """System statistics and monitoring"""
    db_stats = await db.get_stats()
    active_sessions = await state_manager.get_active_sessions()
    
    return {
        "database": db_stats,
        "sessions": {
            "active": len(active_sessions),
            "ids": active_sessions
        }
    }

@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 20):
    """Get conversation history for a session"""
    history = await db.get_session_history(session_id, limit)
    return {
        "session_id": session_id,
        "history": history,
        "count": len(history)
    }

# Setup Twilio routes
setup_twilio_routes(app)

# Serve static files (PWA)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )