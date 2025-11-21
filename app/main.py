"""
FastAPI Main Application - Integration Layer
PATTERN: Dependency injection with lifecycle management
"""
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, Request, UploadFile, File, Form, Query
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
    SearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    ImageUploadResponse,
    DocumentUploadResponse,
    MultiModalConversationRequest,
    MultiModalConversationResponse
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
from app.search import create_hybrid_search_engine, SearchStrategy
from app.search.vector_store import vector_store as search_vector_store
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Multimodal imports (Phase 4)
from app.multimodal import (
    file_manager,
    vision_analyzer,
    document_processor,
    metadata_store,
    multimodal_indexer
)

# Learning/Feedback imports (Phase 5)
from app.learning.feedback_models import (
    ExplicitFeedbackRequest,
    ImplicitFeedbackRequest,
    CorrectionFeedbackRequest,
    SessionFeedbackResponse,
    FeedbackStatsResponse,
    ExplicitFeedback,
    ImplicitFeedback,
    CorrectionFeedback,
    CorrectionType,
)
from app.learning.feedback_store import FeedbackStore
from app.learning.config import feedback_config

# Global feedback store instance
feedback_store = FeedbackStore()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
rate_limiter = RateLimiter(max_calls=100, time_window=60)  # 100 calls per minute per IP

# Global hybrid search engine
hybrid_search_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    PATTERN: Application lifecycle management
    WHY: Proper initialization and cleanup
    """
    global hybrid_search_engine

    # Startup
    api_logger.info("application_startup_initiated")
    app.state.startup_time = time.time()
    await db.initialize()
    await state_manager.initialize()

    # Initialize multimodal services (Phase 4)
    await metadata_store.initialize()

    # Initialize feedback store (Phase 5)
    await feedback_store.initialize()
    api_logger.info("feedback_store_initialized")

    # Initialize hybrid search engine
    try:
        api_logger.info("hybrid_search_initialization_started")
        await search_vector_store.initialize()
        hybrid_search_engine = create_hybrid_search_engine(db)

        # Set OpenAI client for embeddings if available
        if settings.openai_api_key:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            hybrid_search_engine.set_embedding_client(openai_client)
            api_logger.info("hybrid_search_initialized", embedding_enabled=True)
        else:
            api_logger.warning("hybrid_search_initialized", embedding_enabled=False,
                             message="OpenAI API key not configured - semantic search unavailable")
    except Exception as e:
        api_logger.error("hybrid_search_initialization_failed", error=str(e), exc_info=True)
        # Continue without hybrid search
        hybrid_search_engine = None

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
            "hybrid_search": "/api/search/hybrid",
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
    capture_id = await db.save_exchange(
        session_id,
        user_text,
        agent_text,
        metadata={"source": "api"}
    )

    # Generate and store embedding for semantic search (if available)
    if hybrid_search_engine and hybrid_search_engine._embedding_client:
        try:
            # Combine user and agent text for embedding
            combined_text = f"User: {user_text}\nAgent: {agent_text}"

            # Generate embedding
            response = await hybrid_search_engine._embedding_client.embeddings.create(
                input=combined_text,
                model="text-embedding-ada-002"
            )
            import numpy as np
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Store in vector store
            await search_vector_store.add_embedding(
                capture_id=capture_id,
                embedding=embedding,
                model="text-embedding-ada-002"
            )

            api_logger.debug(
                "embedding_generated",
                capture_id=capture_id,
                session_id=session_id
            )
        except Exception as e:
            # Don't fail conversation if embedding generation fails
            api_logger.warning(
                "embedding_generation_failed",
                capture_id=capture_id,
                error=str(e)
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
    NOTE: For advanced semantic search, use /api/search/hybrid
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

@app.post("/api/search/hybrid")
@limiter.limit("30/minute")
async def hybrid_search(request: HybridSearchRequest, http_request: Request) -> HybridSearchResponse:
    """
    PATTERN: Hybrid search combining vector similarity and FTS5 keyword search
    WHY: Best of both worlds - semantic understanding + exact matching
    ALGORITHM: Reciprocal Rank Fusion (RRF) for result combination

    Search Strategies:
    - semantic: Pure vector search for conceptual queries
    - keyword: Pure FTS5 for exact phrase matching
    - hybrid: Combined search with RRF (balanced)
    - adaptive: Automatically choose based on query (default)

    Example:
        POST /api/search/hybrid
        {
            "query": "explain machine learning concepts",
            "strategy": "adaptive",
            "limit": 10
        }
    """
    if not hybrid_search_engine:
        api_logger.error("hybrid_search_unavailable", query=request.query)
        raise HTTPException(
            status_code=503,
            detail="Hybrid search is not available. Please use /api/search endpoint."
        )

    try:
        # Map strategy string to enum
        strategy = None
        if request.strategy:
            try:
                strategy = SearchStrategy(request.strategy)
            except ValueError:
                raise HTTPException(400, f"Invalid strategy: {request.strategy}")

        # Execute hybrid search
        response = await hybrid_search_engine.search(
            query=request.query,
            strategy=strategy,
            limit=request.limit
        )

        api_logger.info(
            "hybrid_search_complete",
            query=request.query,
            strategy=response.strategy,
            results_count=response.total_count,
            execution_time_ms=response.execution_time_ms
        )

        return HybridSearchResponse(**response.__dict__)

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(
            "hybrid_search_failed",
            query=request.query,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(500, f"Hybrid search failed: {str(e)}")

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

# ============================================================================
# MULTIMODAL ENDPOINTS (Phase 4)
# ============================================================================

@app.post("/api/upload/image")
@limiter.limit("10/minute")
async def upload_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    analyze: bool = Form(True),
    http_request: Request = None
) -> ImageUploadResponse:
    """
    Upload and optionally analyze an image

    PATTERN: Multimodal file upload with AI analysis
    WHY: Enable vision-enhanced conversations

    Parameters:
    - file: Image file (PNG, JPEG, GIF, WebP, max 5MB)
    - session_id: Session ID for context (optional)
    - analyze: Run vision analysis (default: True)

    Returns:
    - file_id, url, analysis (if requested)

    Rate Limit: 10 requests per minute
    """
    start_time = time.time()
    session_id = session_id or str(uuid.uuid4())

    set_request_context(session_id=session_id, endpoint="/api/upload/image")

    try:
        api_logger.info(
            "image_upload_started",
            session_id=session_id,
            filename=file.filename,
            content_type=file.content_type
        )

        # Read file data
        file_data = await file.read()

        # Validate and save file
        file_metadata = await file_manager.save_file(
            file_data=file_data,
            original_filename=file.filename,
            file_type="image",
            session_id=session_id
        )

        # Save metadata to database
        await metadata_store.save_file_metadata(file_metadata)

        # Analyze if requested
        analysis = None
        if analyze:
            analysis = await vision_analyzer.analyze_image(
                file_metadata["stored_path"]
            )

            # Save analysis
            await metadata_store.save_analysis(
                file_id=file_metadata["file_id"],
                analysis_type="vision",
                analysis_result=analysis
            )

            # Index in vector store if successful
            if analysis.get('success'):
                indexed = await multimodal_indexer.index_image(
                    file_id=file_metadata["file_id"],
                    analysis=analysis,
                    session_id=session_id
                )

                if indexed:
                    await metadata_store.mark_indexed(file_metadata["file_id"])

        processing_time = (time.time() - start_time) * 1000

        api_logger.info(
            "image_upload_complete",
            session_id=session_id,
            file_id=file_metadata["file_id"],
            analyzed=analyze,
            processing_time_ms=round(processing_time, 2)
        )

        return ImageUploadResponse(
            file_id=file_metadata["file_id"],
            url=f"/api/files/{file_metadata['file_id']}",
            filename=file.filename,
            size=file_metadata["file_size"],
            mime_type=file_metadata["mime_type"],
            analysis=analysis if analyze else None
        )

    except ValueError as e:
        api_logger.warning("image_upload_validation_error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        api_logger.error(
            "image_upload_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(500, f"Image upload failed: {str(e)}")
    finally:
        clear_request_context()

@app.post("/api/upload/document")
@limiter.limit("5/minute")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    extract_text: bool = Form(True),
    http_request: Request = None
) -> DocumentUploadResponse:
    """
    Upload and process a document

    PATTERN: Document upload with text extraction and indexing
    WHY: Enable document-enhanced conversations and RAG

    Parameters:
    - file: Document file (PDF, DOCX, TXT, MD, max 10MB)
    - session_id: Session ID for context
    - extract_text: Extract text and index (default: True)

    Returns:
    - file_id, metadata, text_preview, chunks

    Rate Limit: 5 requests per minute
    """
    start_time = time.time()
    session_id = session_id or str(uuid.uuid4())

    set_request_context(session_id=session_id, endpoint="/api/upload/document")

    try:
        api_logger.info(
            "document_upload_started",
            session_id=session_id,
            filename=file.filename,
            content_type=file.content_type
        )

        # Read file data
        file_data = await file.read()

        # Validate and save file
        file_metadata = await file_manager.save_file(
            file_data=file_data,
            original_filename=file.filename,
            file_type="document",
            session_id=session_id
        )

        # Save metadata to database
        await metadata_store.save_file_metadata(file_metadata)

        # Extract text if requested
        text_preview = None
        chunk_count = 0
        doc_metadata = None

        if extract_text:
            processing_result = await document_processor.process_document(
                file_metadata["stored_path"],
                extract_metadata=True,
                chunk_text=True
            )

            # Save processing result
            await metadata_store.save_analysis(
                file_id=file_metadata["file_id"],
                analysis_type="document",
                analysis_result=processing_result
            )

            if processing_result.get('success'):
                # Get preview
                text_preview = await document_processor.extract_preview(
                    file_metadata["stored_path"],
                    max_length=200
                )

                # Index chunks
                chunks = processing_result.get('chunks', [])
                chunk_count = len(chunks)

                if chunks:
                    index_result = await multimodal_indexer.index_document(
                        file_id=file_metadata["file_id"],
                        chunks=chunks,
                        session_id=session_id
                    )

                    if index_result.get('success'):
                        await metadata_store.mark_indexed(file_metadata["file_id"])

                # Extract metadata
                doc_metadata = processing_result.get('metadata')

        processing_time = (time.time() - start_time) * 1000

        api_logger.info(
            "document_upload_complete",
            session_id=session_id,
            file_id=file_metadata["file_id"],
            extracted=extract_text,
            chunk_count=chunk_count,
            processing_time_ms=round(processing_time, 2)
        )

        return DocumentUploadResponse(
            file_id=file_metadata["file_id"],
            url=f"/api/files/{file_metadata['file_id']}",
            filename=file.filename,
            size=file_metadata["file_size"],
            mime_type=file_metadata["mime_type"],
            text_preview=text_preview,
            chunk_count=chunk_count,
            metadata=doc_metadata
        )

    except ValueError as e:
        api_logger.warning("document_upload_validation_error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        api_logger.error(
            "document_upload_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(500, f"Document upload failed: {str(e)}")
    finally:
        clear_request_context()

@app.get("/api/files/{file_id}")
async def get_file(file_id: str, file_type: Optional[str] = "image"):
    """
    Retrieve uploaded file

    PATTERN: File retrieval with content negotiation
    WHY: Allow access to uploaded files

    Parameters:
    - file_id: Unique file ID
    - file_type: File type ('image' or 'document')

    Returns:
    - File content with appropriate MIME type
    """
    try:
        # Track access
        await metadata_store.track_access(file_id)

        # Retrieve file
        result = await file_manager.get_file(file_id, file_type)

        if result is None:
            raise HTTPException(404, "File not found")

        file_data, mime_type = result

        api_logger.info(
            "file_retrieved",
            file_id=file_id,
            file_type=file_type,
            mime_type=mime_type
        )

        return Response(content=file_data, media_type=mime_type)

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(
            "file_retrieval_error",
            file_id=file_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(500, f"File retrieval failed: {str(e)}")

@app.post("/api/conversation/multimodal")
@limiter.limit("20/minute")
async def multimodal_conversation(
    request: MultiModalConversationRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
) -> MultiModalConversationResponse:
    """
    Conversation with multi-modal context (text + images + documents)

    PATTERN: Multi-modal AI conversation
    WHY: Enable richer conversations with visual and document context

    Parameters:
    - text: User message
    - image_ids: List of uploaded image IDs
    - document_ids: List of uploaded document IDs
    - session_id: Session for context

    Returns:
    - Conversation response with multi-modal understanding

    Rate Limit: 20 requests per minute
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    set_request_context(session_id=session_id, endpoint="/api/conversation/multimodal")

    try:
        api_logger.info(
            "multimodal_conversation_started",
            session_id=session_id,
            image_count=len(request.image_ids),
            document_count=len(request.document_ids)
        )

        # Gather multimodal context
        image_analyses = []
        document_texts = []

        # Fetch image analyses
        for image_id in request.image_ids:
            metadata = await metadata_store.get_file_metadata(image_id)
            if metadata and metadata.get('analysis_result'):
                try:
                    analysis = json.loads(metadata['analysis_result'])
                    if analysis.get('success'):
                        image_analyses.append(analysis)
                except json.JSONDecodeError:
                    pass

        # Fetch document texts (from chunks)
        for doc_id in request.document_ids:
            metadata = await metadata_store.get_file_metadata(doc_id)
            if metadata and metadata.get('analysis_result'):
                try:
                    doc_result = json.loads(metadata['analysis_result'])
                    if doc_result.get('success'):
                        # Get preview or first few chunks
                        doc_text = doc_result.get('text', '')[:1000]  # Limit to 1000 chars
                        if doc_text:
                            document_texts.append(doc_text)
                except json.JSONDecodeError:
                    pass

        # Build enriched context
        enriched_text = request.text

        if image_analyses:
            enriched_text += "\n\n[Context from images]:\n"
            for i, analysis in enumerate(image_analyses, 1):
                enriched_text += f"Image {i}: {analysis.get('analysis', '')[:500]}\n"

        if document_texts:
            enriched_text += "\n\n[Context from documents]:\n"
            for i, doc_text in enumerate(document_texts, 1):
                enriched_text += f"Document {i}: {doc_text}\n"

        # Get conversation context
        context = await state_manager.get_conversation_context(session_id)

        # Generate response with enriched context
        agent_response = await conversation_handler.generate_response(
            enriched_text,
            context
        )

        intent = conversation_handler.detect_intent(request.text)

        # Track conversation exchange
        metrics_collector.track_conversation_exchange("multimodal_query")

        # Update state in background
        background_tasks.add_task(
            update_conversation_state,
            session_id,
            request.text,
            agent_response
        )

        processing_time = (time.time() - start_time) * 1000

        api_logger.info(
            "multimodal_conversation_complete",
            session_id=session_id,
            processing_time_ms=round(processing_time, 2)
        )

        return MultiModalConversationResponse(
            session_id=session_id,
            user_text=request.text,
            agent_text=agent_response,
            intent=intent,
            image_count=len(request.image_ids),
            document_count=len(request.document_ids),
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        api_logger.error(
            "multimodal_conversation_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(500, f"Multimodal conversation failed: {str(e)}")
    finally:
        clear_request_context()

# ============================================================================
# FEEDBACK COLLECTION ENDPOINTS (Phase 5)
# ============================================================================

@app.post("/api/feedback/explicit")
@limiter.limit("60/minute")
async def submit_explicit_feedback(
    request: ExplicitFeedbackRequest,
    http_request: Request
) -> Dict:
    """
    Submit explicit user feedback (ratings and comments).

    PATTERN: User-initiated feedback collection
    WHY: Direct signal of user satisfaction

    Parameters:
    - session_id: Session identifier
    - exchange_id: Exchange/turn identifier
    - rating: Rating from 1-5 stars
    - helpful: Whether the response was helpful
    - comment: Optional user comment

    Returns:
    - feedback_id: Generated feedback identifier
    - status: Success status

    Rate Limit: 60 requests per minute
    """
    try:
        api_logger.info(
            "explicit_feedback_received",
            session_id=request.session_id,
            exchange_id=request.exchange_id,
            rating=request.rating,
            helpful=request.helpful
        )

        feedback = ExplicitFeedback(
            session_id=request.session_id,
            exchange_id=request.exchange_id,
            rating=request.rating,
            helpful=request.helpful,
            comment=request.comment
        )

        feedback_id = await feedback_store.save_explicit(feedback)

        api_logger.info(
            "explicit_feedback_saved",
            feedback_id=feedback_id,
            session_id=request.session_id
        )

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully"
        }

    except Exception as e:
        api_logger.error(
            "explicit_feedback_error",
            session_id=request.session_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(500, f"Failed to submit feedback: {str(e)}")


@app.post("/api/feedback/implicit")
@limiter.limit("120/minute")
async def track_implicit_feedback(
    request: ImplicitFeedbackRequest,
    http_request: Request
) -> Dict:
    """
    Track implicit engagement metrics.

    PATTERN: Automatic engagement tracking
    WHY: Non-intrusive feedback collection

    Parameters:
    - session_id: Session identifier
    - response_time_ms: Agent response time in milliseconds
    - user_response_time_ms: Time until user's next message
    - follow_up_count: Number of follow-up questions
    - engagement_duration_seconds: Total engagement time
    - copy_action: Whether user copied text
    - share_action: Whether user shared the response
    - scroll_depth: How far user scrolled (0-1)

    Returns:
    - feedback_id: Generated feedback identifier
    - engagement_score: Computed engagement score

    Rate Limit: 120 requests per minute
    """
    try:
        feedback = ImplicitFeedback(
            session_id=request.session_id,
            response_time_ms=request.response_time_ms,
            user_response_time_ms=request.user_response_time_ms,
            engagement_duration_seconds=request.engagement_duration_seconds,
            follow_up_count=request.follow_up_count,
            scroll_depth=request.scroll_depth,
            copy_action=request.copy_action,
            share_action=request.share_action
        )

        feedback_id = await feedback_store.save_implicit(feedback)

        api_logger.debug(
            "implicit_feedback_tracked",
            feedback_id=feedback_id,
            session_id=request.session_id,
            engagement_score=feedback.engagement_score
        )

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "engagement_score": feedback.engagement_score
        }

    except Exception as e:
        api_logger.error(
            "implicit_feedback_error",
            session_id=request.session_id,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to track engagement: {str(e)}")


@app.post("/api/feedback/correction")
@limiter.limit("30/minute")
async def log_correction(
    request: CorrectionFeedbackRequest,
    http_request: Request
) -> Dict:
    """
    Log a user correction.

    PATTERN: Learning from user corrections
    WHY: Identify misunderstandings and improve responses

    Parameters:
    - session_id: Session identifier
    - original_text: Original text before correction
    - corrected_text: Corrected/rephrased text
    - correction_type: Type of correction (optional, auto-detected)
    - context: Optional context around the correction

    Returns:
    - correction_id: Generated correction identifier
    - correction_type: Detected correction type
    - edit_distance_ratio: Edit distance as ratio

    Rate Limit: 30 requests per minute
    """
    try:
        # Compute edit distance
        def compute_edit_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return compute_edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        edit_distance = compute_edit_distance(
            request.original_text,
            request.corrected_text
        )
        edit_distance_ratio = edit_distance / max(len(request.original_text), 1)

        # Auto-detect correction type if not provided
        correction_type = request.correction_type
        if correction_type is None:
            if edit_distance_ratio < 0.1:
                correction_type = CorrectionType.SPELLING
            elif edit_distance_ratio < 0.3:
                correction_type = CorrectionType.GRAMMAR
            elif len(request.corrected_text) > len(request.original_text) * 1.5:
                correction_type = CorrectionType.ELABORATION
            else:
                correction_type = CorrectionType.REPHRASE

        correction = CorrectionFeedback(
            session_id=request.session_id,
            original_text=request.original_text,
            corrected_text=request.corrected_text,
            correction_type=correction_type,
            edit_distance=edit_distance,
            edit_distance_ratio=edit_distance_ratio,
            context=request.context
        )

        correction_id = await feedback_store.save_correction(correction)

        api_logger.info(
            "correction_logged",
            correction_id=correction_id,
            session_id=request.session_id,
            correction_type=correction_type.value,
            edit_distance_ratio=round(edit_distance_ratio, 3)
        )

        return {
            "status": "success",
            "correction_id": correction_id,
            "correction_type": correction_type.value,
            "edit_distance_ratio": round(edit_distance_ratio, 3)
        }

    except ValueError as e:
        api_logger.warning(
            "correction_validation_error",
            session_id=request.session_id,
            error=str(e)
        )
        raise HTTPException(400, str(e))
    except Exception as e:
        api_logger.error(
            "correction_error",
            session_id=request.session_id,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to log correction: {str(e)}")


@app.get("/api/feedback/session/{session_id}")
@limiter.limit("30/minute")
async def get_session_feedback(
    session_id: str,
    http_request: Request
) -> SessionFeedbackResponse:
    """
    Get aggregated feedback for a session.

    PATTERN: Session-level feedback aggregation
    WHY: Understand overall conversation quality

    Parameters:
    - session_id: Session identifier

    Returns:
    - Session feedback summary with explicit items and corrections

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info(
            "session_feedback_requested",
            session_id=session_id
        )

        # Get aggregated feedback
        session_feedback = await feedback_store.get_session_feedback(session_id)

        # Get detailed items
        explicit_items = await feedback_store.get_explicit_by_session(session_id, limit=50)
        correction_items = await feedback_store.get_corrections_by_session(session_id, limit=50)

        return SessionFeedbackResponse(
            session_id=session_id,
            feedback=session_feedback,
            explicit_items=explicit_items,
            correction_items=correction_items
        )

    except Exception as e:
        api_logger.error(
            "session_feedback_error",
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get session feedback: {str(e)}")


@app.get("/api/feedback/stats")
@limiter.limit("10/minute")
async def get_feedback_stats(
    hours: int = 24,
    http_request: Request = None
) -> FeedbackStatsResponse:
    """
    Get aggregate feedback statistics.

    PATTERN: System-wide metrics aggregation
    WHY: Monitor overall system performance and user satisfaction

    Parameters:
    - hours: Time range in hours (default: 24)

    Returns:
    - Aggregate feedback statistics

    Rate Limit: 10 requests per minute
    """
    try:
        api_logger.info(
            "feedback_stats_requested",
            hours=hours
        )

        from datetime import timedelta
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        stats = await feedback_store.get_aggregate_stats(start_time, end_time)

        return FeedbackStatsResponse(
            stats=stats,
            cache_ttl_seconds=60
        )

    except Exception as e:
        api_logger.error(
            "feedback_stats_error",
            hours=hours,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get feedback stats: {str(e)}")


# ============================================================================
# ANALYTICS DASHBOARD ENDPOINTS (Phase 6)
# ============================================================================

# Import analytics components
from app.analytics.dashboard_service import dashboard_service, DashboardService
from app.analytics.dashboard_models import (
    OverviewResponse,
    ProgressChartResponse,
    TrendChartResponse,
    TopicBreakdownResponse,
    ActivityHeatmapResponse,
    GoalProgressResponse,
    InsightResponse,
    ExportResponse,
    ExportFormat,
    PeriodType,
)
from app.analytics.chart_data import ChartJSData


@app.get("/api/analytics/overview")
@limiter.limit("30/minute")
async def get_analytics_overview(
    http_request: Request
) -> OverviewResponse:
    """
    Get main dashboard overview data.

    PATTERN: Aggregated KPI dashboard
    WHY: Single call for landing page data

    Returns:
    - Overview cards with key metrics
    - Quick stats summary
    - Recent insights
    - Streak information

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("analytics_overview_requested")

        await dashboard_service.initialize()
        response = await dashboard_service.get_overview_data()

        api_logger.info(
            "analytics_overview_generated",
            cards_count=len(response.cards)
        )

        return response

    except Exception as e:
        api_logger.error(
            "analytics_overview_error",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(500, f"Failed to get analytics overview: {str(e)}")


@app.get("/api/analytics/progress")
@limiter.limit("30/minute")
async def get_analytics_progress(
    period: str = Query("week", regex="^(week|month|year)$"),
    http_request: Request = None
) -> ProgressChartResponse:
    """
    Get progress visualization data.

    PATTERN: Time-series progress data
    WHY: Track learning progress over time

    Parameters:
    - period: Time period (week, month, year)

    Returns:
    - Progress data points with sessions, exchanges, quality
    - Summary statistics
    - Chart configuration hints

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("analytics_progress_requested", period=period)

        await dashboard_service.initialize()
        response = await dashboard_service.get_progress_charts(period)

        api_logger.info(
            "analytics_progress_generated",
            period=period,
            data_points=len(response.data_points)
        )

        return response

    except Exception as e:
        api_logger.error(
            "analytics_progress_error",
            period=period,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get progress data: {str(e)}")


@app.get("/api/analytics/trends")
@limiter.limit("30/minute")
async def get_analytics_trends(
    metrics: str = Query("quality,engagement", description="Comma-separated metrics"),
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    http_request: Request = None
) -> TrendChartResponse:
    """
    Get trend data for specified metrics.

    PATTERN: Multi-metric trend comparison
    WHY: Compare trends across different metrics

    Parameters:
    - metrics: Comma-separated list (quality, engagement, sessions, duration, positive_rate)
    - days: Number of days to analyze (7-365)

    Returns:
    - Trend data for each requested metric
    - Direction and change percentage
    - Chart-ready data

    Rate Limit: 30 requests per minute
    """
    try:
        metric_list = [m.strip() for m in metrics.split(",")]
        api_logger.info(
            "analytics_trends_requested",
            metrics=metric_list,
            days=days
        )

        await dashboard_service.initialize()
        response = await dashboard_service.get_trend_charts(metric_list, days)

        api_logger.info(
            "analytics_trends_generated",
            metrics_count=len(response.metrics),
            days=days
        )

        return response

    except Exception as e:
        api_logger.error(
            "analytics_trends_error",
            metrics=metrics,
            days=days,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get trend data: {str(e)}")


@app.get("/api/analytics/topics")
@limiter.limit("30/minute")
async def get_analytics_topics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    http_request: Request = None
) -> TopicBreakdownResponse:
    """
    Get topic analytics and distribution.

    PATTERN: Categorical analytics with quality correlation
    WHY: Understand content distribution and performance by topic

    Parameters:
    - days: Number of days to analyze (1-365)

    Returns:
    - Topic statistics with session counts and quality
    - Top and emerging topics
    - Pie chart data

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("analytics_topics_requested", days=days)

        await dashboard_service.initialize()
        response = await dashboard_service.get_topic_breakdown(days)

        api_logger.info(
            "analytics_topics_generated",
            total_topics=response.total_topics
        )

        return response

    except Exception as e:
        api_logger.error(
            "analytics_topics_error",
            days=days,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get topic analytics: {str(e)}")


@app.get("/api/analytics/activity")
@limiter.limit("30/minute")
async def get_analytics_activity(
    year: int = Query(None, ge=2020, le=2030, description="Year for heatmap"),
    http_request: Request = None
) -> ActivityHeatmapResponse:
    """
    Get activity heatmap data (GitHub-style calendar).

    PATTERN: Calendar heatmap visualization
    WHY: Show activity patterns over the year

    Parameters:
    - year: Year to generate heatmap for (defaults to current year)

    Returns:
    - Weekly activity cells with intensity levels
    - Total active days and sessions
    - Month labels for display

    Rate Limit: 30 requests per minute
    """
    try:
        if year is None:
            year = date.today().year

        api_logger.info("analytics_activity_requested", year=year)

        await dashboard_service.initialize()
        response = await dashboard_service.get_activity_heatmap(year)

        api_logger.info(
            "analytics_activity_generated",
            year=year,
            active_days=response.total_active_days
        )

        return response

    except Exception as e:
        api_logger.error(
            "analytics_activity_error",
            year=year,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get activity heatmap: {str(e)}")


@app.get("/api/analytics/goals")
@limiter.limit("30/minute")
async def get_analytics_goals(
    http_request: Request
) -> GoalProgressResponse:
    """
    Get goal progress tracking data.

    PATTERN: Goal progress tracking
    WHY: Track learning objectives and milestones

    Returns:
    - Active and completed goals
    - Completion rate
    - Next milestone information
    - Suggested goals

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("analytics_goals_requested")

        await dashboard_service.initialize()
        response = await dashboard_service.get_goal_progress()

        api_logger.info(
            "analytics_goals_generated",
            active=len(response.active_goals),
            completed=len(response.completed_goals)
        )

        return response

    except Exception as e:
        api_logger.error(
            "analytics_goals_error",
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get goal progress: {str(e)}")


@app.get("/api/analytics/insights")
@limiter.limit("30/minute")
async def get_analytics_insights(
    limit: int = Query(10, ge=1, le=100, description="Maximum insights to return"),
    http_request: Request = None
) -> InsightResponse:
    """
    Get recent insights and recommendations.

    PATTERN: Prioritized insights list
    WHY: Surface actionable intelligence

    Parameters:
    - limit: Maximum number of insights (1-100)

    Returns:
    - Insights list with priority and category
    - Category distribution
    - Critical insight indicator

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("analytics_insights_requested", limit=limit)

        await dashboard_service.initialize()
        response = await dashboard_service.get_insights(limit)

        api_logger.info(
            "analytics_insights_generated",
            count=response.total_insights
        )

        return response

    except Exception as e:
        api_logger.error(
            "analytics_insights_error",
            limit=limit,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to get insights: {str(e)}")


@app.get("/api/analytics/export")
@limiter.limit("5/minute")
async def export_analytics_data(
    format: str = Query("json", regex="^(json|csv|pdf)$"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    http_request: Request = None
) -> ExportResponse:
    """
    Export analytics data.

    PATTERN: Data export with format options
    WHY: Enable data portability and offline analysis

    Parameters:
    - format: Export format (json, csv, pdf)
    - start_date: Start of export range (YYYY-MM-DD)
    - end_date: End of export range (YYYY-MM-DD)

    Returns:
    - Export data or download URL
    - Record count
    - Date range info

    Rate Limit: 5 requests per minute
    """
    try:
        api_logger.info(
            "analytics_export_requested",
            format=format,
            start_date=start_date,
            end_date=end_date
        )

        # Parse dates
        parsed_start = None
        parsed_end = None
        if start_date:
            parsed_start = date.fromisoformat(start_date)
        if end_date:
            parsed_end = date.fromisoformat(end_date)

        export_format = ExportFormat(format)

        await dashboard_service.initialize()
        response = await dashboard_service.export_data(
            format=export_format,
            start_date=parsed_start,
            end_date=parsed_end
        )

        api_logger.info(
            "analytics_export_generated",
            format=format,
            records=response.record_count
        )

        return response

    except ValueError as e:
        api_logger.warning("analytics_export_validation_error", error=str(e))
        raise HTTPException(400, f"Invalid date format: {str(e)}")
    except Exception as e:
        api_logger.error(
            "analytics_export_error",
            format=format,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to export data: {str(e)}")


# ============================================================================
# GOAL TRACKING ENDPOINTS (Phase 6)
# ============================================================================

# Import goal tracking components
from app.analytics.goal_tracker import goal_tracker, ProgressMetrics
from app.analytics.goal_models import (
    Goal, GoalType, GoalStatus,
    Achievement, GoalSuggestion,
    CreateGoalRequest, UpdateGoalRequest,
    GoalResponse, GoalListResponse, AchievementListResponse,
)
from app.analytics.achievement_system import achievement_system
from app.analytics.export_service import (
    export_service,
    ExportFormat as GoalExportFormat,
    ReportPeriod
)


@app.post("/api/goals")
@limiter.limit("30/minute")
async def create_goal(
    request: CreateGoalRequest,
    http_request: Request
) -> GoalResponse:
    """
    Create a new learning goal.

    PATTERN: Goal-based gamification
    WHY: Motivate learners with measurable objectives

    Parameters:
    - title: Goal title
    - description: Optional description
    - goal_type: Type (streak, sessions, topics, quality, exchanges, custom)
    - target_value: Target value to achieve
    - unit: Unit of measurement
    - deadline: Optional deadline (YYYY-MM-DD)

    Returns:
    - Created goal with auto-generated milestones

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info(
            "goal_creation_requested",
            title=request.title,
            goal_type=request.goal_type.value
        )

        await goal_tracker.initialize()
        goal = await goal_tracker.create_goal_from_request(request)

        api_logger.info(
            "goal_created",
            goal_id=goal.id,
            title=goal.title,
            goal_type=goal.goal_type.value
        )

        return GoalResponse(
            goal=goal,
            recently_completed_milestones=[],
            new_achievements=[]
        )

    except ValueError as e:
        api_logger.warning("goal_creation_validation_error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        api_logger.error(
            "goal_creation_error",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(500, f"Failed to create goal: {str(e)}")


@app.get("/api/goals")
@limiter.limit("30/minute")
async def list_goals(
    status: Optional[str] = Query(None, regex="^(active|completed|paused|abandoned|expired)$"),
    http_request: Request = None
) -> GoalListResponse:
    """
    List all goals or filter by status.

    PATTERN: Filtered goal listing
    WHY: Track active and historical goals

    Parameters:
    - status: Optional filter (active, completed, paused, abandoned, expired)

    Returns:
    - List of goals with counts

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("goals_list_requested", status=status)

        await goal_tracker.initialize()

        if status:
            goal_status = GoalStatus(status)
            if goal_status == GoalStatus.ACTIVE:
                goals = await goal_tracker.get_active_goals()
            elif goal_status == GoalStatus.COMPLETED:
                goals = await goal_tracker.get_completed_goals()
            else:
                all_goals = await goal_tracker.get_all_goals()
                goals = [g for g in all_goals if g.status == goal_status]
        else:
            goals = await goal_tracker.get_all_goals()

        active_count = len([g for g in goals if g.status == GoalStatus.ACTIVE])
        completed_count = len([g for g in goals if g.status == GoalStatus.COMPLETED])

        api_logger.info(
            "goals_listed",
            total=len(goals),
            active=active_count,
            completed=completed_count
        )

        return GoalListResponse(
            goals=goals,
            total_count=len(goals),
            active_count=active_count,
            completed_count=completed_count
        )

    except Exception as e:
        api_logger.error("goals_list_error", error=str(e))
        raise HTTPException(500, f"Failed to list goals: {str(e)}")


@app.get("/api/goals/suggestions")
@limiter.limit("10/minute")
async def get_goal_suggestions(
    limit: int = Query(5, ge=1, le=10),
    http_request: Request = None
) -> Dict:
    """
    Get AI-generated goal suggestions.

    PATTERN: Personalized recommendations
    WHY: Help users set appropriate goals

    Parameters:
    - limit: Maximum suggestions (1-10)

    Returns:
    - List of suggested goals with confidence scores

    Rate Limit: 10 requests per minute
    """
    try:
        api_logger.info("goal_suggestions_requested", limit=limit)

        await goal_tracker.initialize()

        # Create sample metrics (in production, fetch from actual data)
        metrics = ProgressMetrics(
            current_streak=0,
            longest_streak=0,
            total_sessions=0,
            total_exchanges=0,
            total_topics=0,
            avg_quality_score=0.0,
            total_duration_minutes=0,
            total_feedback=0
        )

        suggestions = await goal_tracker.get_goal_suggestions(metrics, limit)

        api_logger.info(
            "goal_suggestions_generated",
            count=len(suggestions)
        )

        return {
            "suggestions": [s.model_dump() for s in suggestions],
            "count": len(suggestions)
        }

    except Exception as e:
        api_logger.error("goal_suggestions_error", error=str(e))
        raise HTTPException(500, f"Failed to get suggestions: {str(e)}")


@app.get("/api/goals/{goal_id}")
@limiter.limit("60/minute")
async def get_goal(
    goal_id: str,
    http_request: Request = None
) -> GoalResponse:
    """
    Get a specific goal by ID.

    Parameters:
    - goal_id: Goal identifier

    Returns:
    - Goal details with milestones

    Rate Limit: 60 requests per minute
    """
    try:
        api_logger.info("goal_requested", goal_id=goal_id)

        await goal_tracker.initialize()
        goal = await goal_tracker.get_goal(goal_id)

        if not goal:
            raise HTTPException(404, f"Goal not found: {goal_id}")

        return GoalResponse(
            goal=goal,
            recently_completed_milestones=[],
            new_achievements=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("goal_get_error", goal_id=goal_id, error=str(e))
        raise HTTPException(500, f"Failed to get goal: {str(e)}")


@app.put("/api/goals/{goal_id}")
@limiter.limit("30/minute")
async def update_goal(
    goal_id: str,
    request: UpdateGoalRequest,
    http_request: Request = None
) -> GoalResponse:
    """
    Update an existing goal.

    Parameters:
    - goal_id: Goal identifier
    - title: New title (optional)
    - description: New description (optional)
    - target_value: New target (optional)
    - deadline: New deadline (optional)
    - status: New status (optional)

    Returns:
    - Updated goal

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("goal_update_requested", goal_id=goal_id)

        await goal_tracker.initialize()
        goal = await goal_tracker.update_goal(goal_id, request)

        if not goal:
            raise HTTPException(404, f"Goal not found: {goal_id}")

        api_logger.info("goal_updated", goal_id=goal_id)

        return GoalResponse(
            goal=goal,
            recently_completed_milestones=[],
            new_achievements=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("goal_update_error", goal_id=goal_id, error=str(e))
        raise HTTPException(500, f"Failed to update goal: {str(e)}")


@app.delete("/api/goals/{goal_id}")
@limiter.limit("30/minute")
async def delete_goal(
    goal_id: str,
    http_request: Request = None
) -> Dict:
    """
    Delete a goal.

    Parameters:
    - goal_id: Goal identifier

    Returns:
    - Success status

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("goal_deletion_requested", goal_id=goal_id)

        await goal_tracker.initialize()
        deleted = await goal_tracker.delete_goal(goal_id)

        if not deleted:
            raise HTTPException(404, f"Goal not found: {goal_id}")

        api_logger.info("goal_deleted", goal_id=goal_id)

        return {"status": "success", "message": "Goal deleted", "goal_id": goal_id}

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("goal_deletion_error", goal_id=goal_id, error=str(e))
        raise HTTPException(500, f"Failed to delete goal: {str(e)}")


@app.get("/api/goals/{goal_id}/progress")
@limiter.limit("30/minute")
async def get_goal_progress(
    goal_id: str,
    days: int = Query(30, ge=1, le=365),
    http_request: Request = None
) -> Dict:
    """
    Get progress history for a goal.

    Parameters:
    - goal_id: Goal identifier
    - days: Number of days of history (1-365)

    Returns:
    - Progress history with data points

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("goal_progress_requested", goal_id=goal_id, days=days)

        await goal_tracker.initialize()
        history = await goal_tracker.get_progress_history(goal_id, days)

        return {
            "goal_id": goal_id,
            "days": days,
            "history": [h.model_dump() for h in history],
            "count": len(history)
        }

    except Exception as e:
        api_logger.error("goal_progress_error", goal_id=goal_id, error=str(e))
        raise HTTPException(500, f"Failed to get progress: {str(e)}")


# ============================================================================
# ACHIEVEMENT ENDPOINTS (Phase 6)
# ============================================================================

@app.get("/api/achievements")
@limiter.limit("30/minute")
async def list_achievements(
    category: Optional[str] = Query(None, description="Filter by category"),
    http_request: Request = None
) -> AchievementListResponse:
    """
    List all achievements.

    PATTERN: Achievement/badge system
    WHY: Gamification to motivate learners

    Parameters:
    - category: Optional filter (beginner, streak, quality, exploration, engagement, mastery, milestone, social)

    Returns:
    - List of achievements with unlock status

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("achievements_list_requested", category=category)

        await achievement_system.initialize()

        if category:
            from app.analytics.goal_models import AchievementCategory
            try:
                cat = AchievementCategory(category)
                achievements = await achievement_system.get_achievements_by_category(cat)
            except ValueError:
                raise HTTPException(400, f"Invalid category: {category}")
        else:
            achievements = await achievement_system.get_all_achievements()

        unlocked = [a for a in achievements if a.unlocked]
        total_points = sum(a.points for a in unlocked)

        # Count by category
        categories = {}
        for a in achievements:
            cat = a.category.value
            if cat not in categories:
                categories[cat] = 0
            if a.unlocked:
                categories[cat] += 1

        api_logger.info(
            "achievements_listed",
            total=len(achievements),
            unlocked=len(unlocked)
        )

        return AchievementListResponse(
            achievements=achievements,
            total_count=len(achievements),
            unlocked_count=len(unlocked),
            total_points=total_points,
            categories=categories
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("achievements_list_error", error=str(e))
        raise HTTPException(500, f"Failed to list achievements: {str(e)}")


@app.get("/api/achievements/unlocked")
@limiter.limit("30/minute")
async def list_unlocked_achievements(
    http_request: Request = None
) -> Dict:
    """
    List unlocked achievements.

    Returns:
    - Only unlocked achievements

    Rate Limit: 30 requests per minute
    """
    try:
        api_logger.info("unlocked_achievements_requested")

        await achievement_system.initialize()
        achievements = await achievement_system.get_unlocked_achievements()
        total_points = sum(a.points for a in achievements)

        return {
            "achievements": [a.model_dump() for a in achievements],
            "count": len(achievements),
            "total_points": total_points
        }

    except Exception as e:
        api_logger.error("unlocked_achievements_error", error=str(e))
        raise HTTPException(500, f"Failed to list achievements: {str(e)}")


@app.get("/api/achievements/stats")
@limiter.limit("10/minute")
async def get_achievement_stats(
    http_request: Request = None
) -> Dict:
    """
    Get achievement statistics.

    Returns:
    - Completion stats by category and rarity

    Rate Limit: 10 requests per minute
    """
    try:
        api_logger.info("achievement_stats_requested")

        await achievement_system.initialize()
        stats = await achievement_system.get_achievement_stats()

        return stats

    except Exception as e:
        api_logger.error("achievement_stats_error", error=str(e))
        raise HTTPException(500, f"Failed to get stats: {str(e)}")


@app.get("/api/achievements/next")
@limiter.limit("10/minute")
async def get_next_achievements(
    limit: int = Query(5, ge=1, le=10),
    http_request: Request = None
) -> Dict:
    """
    Get achievements closest to being unlocked.

    Parameters:
    - limit: Maximum achievements (1-10)

    Returns:
    - List of near-completion achievements with progress

    Rate Limit: 10 requests per minute
    """
    try:
        api_logger.info("next_achievements_requested", limit=limit)

        await achievement_system.initialize()

        metrics = ProgressMetrics(
            current_streak=0,
            longest_streak=0,
            total_sessions=0,
            total_exchanges=0,
            total_topics=0,
            avg_quality_score=0.0,
            total_duration_minutes=0,
            total_feedback=0
        )

        next_achievements = await achievement_system.get_next_achievements(metrics, limit)

        return {
            "achievements": [
                {
                    "achievement": a.model_dump(),
                    "progress_percent": pct
                }
                for a, pct in next_achievements
            ],
            "count": len(next_achievements)
        }

    except Exception as e:
        api_logger.error("next_achievements_error", error=str(e))
        raise HTTPException(500, f"Failed to get next achievements: {str(e)}")


# ============================================================================
# EXPORT ENDPOINTS (Phase 6)
# ============================================================================

@app.get("/api/export")
@limiter.limit("5/minute")
async def export_data(
    format: str = Query("json", regex="^(json|csv)$"),
    period: str = Query("month", regex="^(week|month|quarter|year)$"),
    http_request: Request = None
) -> Response:
    """
    Export analytics data in various formats.

    PATTERN: Multi-format data export
    WHY: Enable data portability and analysis

    Parameters:
    - format: Export format (json, csv)
    - period: Time period (week, month, quarter, year)

    Returns:
    - Exported data in requested format

    Rate Limit: 5 requests per minute
    """
    try:
        api_logger.info(
            "data_export_requested",
            format=format,
            period=period
        )

        await export_service.initialize()

        export_period = ReportPeriod(period)

        if format == "json":
            content = await export_service.export_to_json(export_period)
            return Response(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=learning_export_{period}.json"
                }
            )
        elif format == "csv":
            content = await export_service.export_to_csv("goals", export_period)
            return Response(
                content=content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=learning_export_{period}.csv"
                }
            )

    except Exception as e:
        api_logger.error(
            "data_export_error",
            format=format,
            period=period,
            error=str(e)
        )
        raise HTTPException(500, f"Failed to export data: {str(e)}")


@app.get("/api/export/report")
@limiter.limit("3/minute")
async def generate_report(
    period: str = Query("week", regex="^(week|month)$"),
    http_request: Request = None
) -> Dict:
    """
    Generate a summary report.

    Parameters:
    - period: Report period (week, month)

    Returns:
    - Report data with insights and recommendations

    Rate Limit: 3 requests per minute
    """
    try:
        api_logger.info("report_generation_requested", period=period)

        await export_service.initialize()

        if period == "week":
            report = await export_service.generate_weekly_report()
            return {
                "period": "week",
                "week_start": str(report.week_start),
                "week_end": str(report.week_end),
                "summary": {
                    "total_sessions": report.summary.total_sessions,
                    "active_goals": report.summary.active_goals,
                    "completed_goals": report.summary.completed_goals,
                    "achievements_unlocked": report.summary.achievements_unlocked,
                    "total_points": report.summary.total_points
                },
                "goals_progress": report.goals_progress,
                "achievements_earned": report.achievements_earned,
                "insights": report.insights,
                "recommendations": report.recommendations
            }
        else:
            report = await export_service.generate_monthly_report()
            return {
                "period": "month",
                "month": report.month,
                "year": report.year,
                "summary": {
                    "total_sessions": report.summary.total_sessions,
                    "active_goals": report.summary.active_goals,
                    "completed_goals": report.summary.completed_goals,
                    "achievements_unlocked": report.summary.achievements_unlocked,
                    "total_points": report.summary.total_points
                },
                "goals_completed": report.goals_completed,
                "achievements_earned": report.achievements_earned,
                "insights": report.insights,
                "recommendations": report.recommendations
            }

    except Exception as e:
        api_logger.error("report_generation_error", period=period, error=str(e))
        raise HTTPException(500, f"Failed to generate report: {str(e)}")


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