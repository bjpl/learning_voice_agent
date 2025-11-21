"""
FastAPI Main Application - Integration Layer
PATTERN: Dependency injection with lifecycle management
"""
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, Request, UploadFile, File, Form
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