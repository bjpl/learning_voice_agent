"""
FastAPI Main Application - Integration Layer
PATTERN: Dependency injection with lifecycle management

Week 3 Enhancements:
- Vector store initialization for semantic search
- Semantic search API endpoints
- Enhanced offline support headers

Week 4 Enhancements:
- Admin dashboard for real-time monitoring
- CDN-friendly cache headers middleware
- Security headers middleware
- Request metrics tracking
- Production-ready observability
"""
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
from datetime import datetime
import uuid
import json
import logging

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

# Week 2: Import resilient Redis client
resilient_redis = None
try:
    from app.redis_client import resilient_redis as rr
    resilient_redis = rr
except ImportError:
    pass

# Week 3: Import vector store (optional)
vector_store = None
try:
    from app.vector_store import vector_store as vs
    vector_store = vs
except ImportError:
    pass

# Week 4: Import admin dashboard and production middleware
ADMIN_AVAILABLE = False
try:
    from app.admin import admin_router
    from app.middleware import (
        MetricsMiddleware,
        CDNHeadersMiddleware,
        SecurityHeadersMiddleware,
        RequestIDMiddleware
    )
    ADMIN_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    PATTERN: Application lifecycle management
    WHY: Proper initialization and cleanup
    """
    # Startup
    logger.info("Initializing application...")
    await db.initialize()
    await state_manager.initialize()

    # Week 3: Initialize vector store if enabled
    if settings.enable_vector_search and vector_store:
        try:
            await vector_store.initialize()
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")

    logger.info("Application ready!")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await state_manager.close()

    # Week 3: Close vector store
    if vector_store:
        try:
            await vector_store.close()
        except Exception:
            pass

    logger.info("Application shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="Learning Voice Agent",
    description="AI-powered voice conversation for learning capture",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Week 4: Add production middleware for monitoring and caching
if ADMIN_AVAILABLE:
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(CDNHeadersMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)
    # Include admin dashboard routes
    app.include_router(admin_router)
    logger.info("Week 4: Admin dashboard and production middleware enabled")

# API Routes

@app.get("/", tags=["health"])
async def root():
    """
    Health check and API information.

    Returns basic service status and available endpoints.
    Used by load balancers and monitoring systems.
    """
    return {
        "status": "healthy",
        "service": "Learning Voice Agent",
        "version": "4.0.0",
        "endpoints": {
            "websocket": "/ws/{session_id}",
            "twilio": "/twilio/voice",
            "search": "/api/search",
            "semantic_search": "/api/semantic-search",
            "stats": "/api/stats",
            "health": "/health",
            "health_detailed": "/health/detailed",
            "admin_dashboard": "/admin/dashboard" if ADMIN_AVAILABLE else None
        },
        "features": {
            "vector_search": settings.enable_vector_search,
            "offline_mode": settings.enable_offline_mode,
            "admin_dashboard": ADMIN_AVAILABLE,
            "cdn_caching": ADMIN_AVAILABLE,
            "metrics_tracking": ADMIN_AVAILABLE
        }
    }


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint for load balancers and monitoring.

    Week 2 Feature: Production health monitoring
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/detailed")
async def health_check_detailed():
    """
    Detailed health check with component status.

    Week 2 Feature: Comprehensive system health for debugging
    Returns status of:
    - Database connectivity
    - Redis connectivity and circuit breaker state
    - Vector store (if enabled)
    - Memory usage
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "components": {}
    }

    # Check database
    try:
        db_stats = await db.get_stats()
        health_status["components"]["database"] = {
            "status": "healthy",
            "type": "sqlite",
            "stats": db_stats
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Check Redis with resilient client info
    try:
        if resilient_redis and resilient_redis.is_connected:
            redis_info = await resilient_redis.info()
            health_status["components"]["redis"] = {
                "status": "healthy",
                "circuit_state": resilient_redis.circuit_state,
                "connected_clients": redis_info.get("connected_clients", "N/A"),
                "used_memory_human": redis_info.get("used_memory_human", "N/A"),
            }
        else:
            # Fallback to state_manager check
            active_sessions = await state_manager.get_active_sessions()
            health_status["components"]["redis"] = {
                "status": "healthy",
                "active_sessions": len(active_sessions)
            }
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e),
            "circuit_state": resilient_redis.circuit_state if resilient_redis else "unknown"
        }
        health_status["status"] = "degraded"

    # Check vector store if enabled
    if settings.enable_vector_search and vector_store:
        try:
            vs_stats = await vector_store.get_stats()
            health_status["components"]["vector_store"] = {
                "status": "healthy",
                "stats": vs_stats
            }
        except Exception as e:
            health_status["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e)
            }

    # Overall status determination
    unhealthy_components = [
        k for k, v in health_status["components"].items()
        if v.get("status") == "unhealthy"
    ]

    if unhealthy_components:
        health_status["status"] = "degraded"
        health_status["unhealthy_components"] = unhealthy_components

    return health_status


@app.get("/health/redis")
async def redis_health():
    """
    Redis-specific health endpoint with circuit breaker details.

    Week 2 Feature: Redis failover monitoring
    """
    if not resilient_redis:
        return {
            "status": "unavailable",
            "message": "Resilient Redis client not configured"
        }

    try:
        is_connected = resilient_redis.is_connected
        can_ping = await resilient_redis.ping() if is_connected else False

        return {
            "status": "healthy" if can_ping else "unhealthy",
            "connected": is_connected,
            "ping_successful": can_ping,
            "circuit_breaker": {
                "state": resilient_redis.circuit_state,
                "failures": resilient_redis.circuit_breaker.failures,
                "threshold": resilient_redis.circuit_breaker.failure_threshold
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "circuit_breaker": {
                "state": resilient_redis.circuit_state if resilient_redis else "unknown"
            }
        }

@app.post("/api/conversation")
async def handle_conversation(
    request: ConversationRequest,
    background_tasks: BackgroundTasks
) -> ConversationResponse:
    """
    PATTERN: REST endpoint for conversation handling
    WHY: Simple integration for various clients
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Transcribe if audio provided
        if request.audio_base64:
            user_text = await audio_pipeline.transcribe_base64(
                request.audio_base64,
                source="api"
            )
        else:
            user_text = request.text
        
        if not user_text:
            raise HTTPException(400, "No input provided")
        
        # Get conversation context
        context = await state_manager.get_conversation_context(session_id)
        
        # Generate response
        agent_response = await conversation_handler.generate_response(
            user_text,
            context
        )
        
        # Update state in background
        background_tasks.add_task(
            update_conversation_state,
            session_id,
            user_text,
            agent_response
        )
        
        return ConversationResponse(
            session_id=session_id,
            user_text=user_text,
            agent_text=agent_response,
            intent=conversation_handler.detect_intent(user_text)
        )
        
    except Exception as e:
        print(f"Conversation error: {e}")
        raise HTTPException(500, str(e))

async def update_conversation_state(
    session_id: str,
    user_text: str,
    agent_text: str
):
    """
    PATTERN: Background task for state updates
    WHY: Don't block the response for persistence

    Week 3: Now also indexes to vector store for semantic search
    """
    # Update Redis context
    await state_manager.update_conversation_context(
        session_id,
        user_text,
        agent_text
    )

    # Save to database
    exchange_id = await db.save_exchange(
        session_id,
        user_text,
        agent_text,
        metadata={"source": "api"}
    )

    # Week 3: Index to vector store for semantic search
    if settings.enable_vector_search and vector_store:
        try:
            await vector_store.add_conversation(
                conversation_id=str(exchange_id),
                user_text=user_text,
                agent_text=agent_text,
                session_id=session_id,
                metadata={"source": "api", "timestamp": datetime.utcnow().isoformat()}
            )
        except Exception as e:
            logger.warning(f"Failed to index to vector store: {e}")

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
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
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
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "user_text": user_text,
                    "agent_text": agent_response,
                    "intent": conversation_handler.detect_intent(user_text)
                })
                
            elif message["type"] == "end":
                # End conversation
                summary = conversation_handler.create_summary(
                    await state_manager.get_conversation_context(session_id)
                )
                await websocket.send_json({
                    "type": "summary",
                    "text": summary
                })
                await state_manager.end_session(session_id)
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.post("/api/search")
async def search_captures(request: SearchRequest) -> SearchResponse:
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


# Week 3: Semantic Search Endpoints

@app.post("/api/semantic-search")
async def semantic_search(
    query: str,
    limit: int = 10,
    threshold: float = 0.5,
    session_id: Optional[str] = None
) -> Dict:
    """
    PATTERN: Semantic similarity search using vector embeddings
    WHY: Find conceptually similar conversations, not just keyword matches

    Week 3 Feature: ChromaDB-powered semantic search
    """
    if not settings.enable_vector_search or not vector_store:
        # Fallback to FTS5 keyword search
        results = await db.search_captures(query, limit)
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "keyword"
        }

    try:
        results = await vector_store.semantic_search(
            query=query,
            limit=limit,
            similarity_threshold=threshold,
            session_filter=session_id
        )
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "semantic"
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        # Fallback to keyword search
        results = await db.search_captures(query, limit)
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "keyword",
            "fallback": True
        }


@app.get("/api/similar/{conversation_id}")
async def find_similar(conversation_id: str, limit: int = 5) -> Dict:
    """
    Find conversations similar to a given conversation.

    Week 3 Feature: "More like this" functionality
    """
    if not settings.enable_vector_search or not vector_store:
        return {
            "conversation_id": conversation_id,
            "similar": [],
            "available": False,
            "message": "Semantic search not enabled"
        }

    try:
        similar = await vector_store.find_similar_conversations(
            conversation_id=conversation_id,
            limit=limit
        )
        return {
            "conversation_id": conversation_id,
            "similar": similar,
            "count": len(similar)
        }
    except Exception as e:
        logger.error(f"Find similar failed: {e}")
        return {
            "conversation_id": conversation_id,
            "similar": [],
            "error": str(e)
        }


@app.get("/api/vector-stats")
async def get_vector_stats() -> Dict:
    """Get vector store statistics."""
    if not vector_store:
        return {
            "available": False,
            "message": "Vector store not configured"
        }

    try:
        stats = await vector_store.get_stats()
        return stats
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


@app.get("/api/offline-manifest")
async def get_offline_manifest() -> Dict:
    """
    Get manifest of resources for offline caching.

    Week 3 Feature: Enhanced PWA offline support
    """
    return {
        "version": "3.0.0",
        "cache_name": "voice-learn-v3",
        "static_resources": [
            "/static/index.html",
            "/static/manifest.json",
            "/static/sw.js"
        ],
        "api_endpoints": [
            "/api/stats",
            "/api/offline-manifest"
        ],
        "features": {
            "semantic_search": settings.enable_vector_search,
            "offline_mode": settings.enable_offline_mode,
            "max_cached_conversations": settings.offline_cache_max_entries
        }
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