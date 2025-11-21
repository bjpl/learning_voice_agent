"""
Redis State Management for Conversation Context
PATTERN: Cache-aside pattern with TTL
WHY: Fast context retrieval without database overhead
RESILIENCE: Connection retry and graceful degradation
"""
import json
import redis.asyncio as redis
from typing import List, Dict, Optional
from datetime import datetime
from app.config import settings
from app.resilience import with_retry
from app.logger import state_logger

class StateManager:
    def __init__(self):
        self.redis_client = None
        self.ttl = settings.redis_ttl
        self._degraded_mode = False  # Flag for operating without Redis

    @with_retry(max_attempts=3, initial_wait=0.5, max_wait=2.0)
    async def initialize(self):
        """
        CONCEPT: Connection pooling for Redis with retry
        WHY: Reuse connections for better performance
        RESILIENCE: Retry connection up to 3 times, gracefully degrade if fails
        """
        try:
            state_logger.info("redis_connection_initializing", url=settings.redis_url)

            self.redis_client = await redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self.redis_client.ping()
            self._degraded_mode = False

            state_logger.info(
                "redis_connection_established",
                url=settings.redis_url,
                max_connections=50,
                ttl_seconds=self.ttl
            )
        except Exception as e:
            state_logger.error(
                "redis_connection_failed_degraded_mode",
                url=settings.redis_url,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            self._degraded_mode = True
            self.redis_client = None
    
    async def get_conversation_context(
        self,
        session_id: str
    ) -> List[Dict]:
        """
        PATTERN: Sliding window of conversation history
        WHY: Claude needs context for coherent responses
        RESILIENCE: Graceful degradation when Redis unavailable
        """
        if self._degraded_mode or not self.redis_client:
            state_logger.debug("redis_unavailable_empty_context", session_id=session_id)
            return []

        try:
            key = f"session:{session_id}:context"
            data = await self.redis_client.get(key)

            if data:
                context = json.loads(data)
                state_logger.debug(
                    "context_retrieved",
                    session_id=session_id,
                    exchanges_count=len(context)
                )
                return context
            return []
        except Exception as e:
            state_logger.warning(
                "get_context_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return []
    
    async def update_conversation_context(
        self,
        session_id: str,
        user_text: str,
        agent_text: str
    ):
        """
        CONCEPT: FIFO queue with fixed size
        WHY: Maintain only relevant recent context
        RESILIENCE: Continue without Redis if unavailable
        """
        if self._degraded_mode or not self.redis_client:
            state_logger.debug("redis_unavailable_skip_update", session_id=session_id)
            return

        try:
            key = f"session:{session_id}:context"

            # Get existing context
            context = await self.get_conversation_context(session_id)

            # Add new exchange
            context.append({
                "timestamp": datetime.utcnow().isoformat(),
                "user": user_text,
                "agent": agent_text
            })

            # Keep only last N exchanges
            original_length = len(context)
            if len(context) > settings.max_context_exchanges:
                context = context[-settings.max_context_exchanges:]

            # Save with TTL
            await self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(context)
            )

            state_logger.debug(
                "context_updated",
                session_id=session_id,
                exchanges_count=len(context),
                pruned=(original_length - len(context)) if original_length > len(context) else 0,
                ttl_seconds=self.ttl
            )
        except Exception as e:
            state_logger.warning(
                "update_context_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def get_session_metadata(
        self,
        session_id: str
    ) -> Optional[Dict]:
        """Get session metadata like start time, exchange count"""
        if self._degraded_mode or not self.redis_client:
            return None

        try:
            key = f"session:{session_id}:metadata"
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            state_logger.warning(
                "get_metadata_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return None

    async def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict
    ):
        """Update session metadata with activity tracking"""
        if self._degraded_mode or not self.redis_client:
            return

        try:
            key = f"session:{session_id}:metadata"

            # Update last activity
            metadata["last_activity"] = datetime.utcnow().isoformat()

            await self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(metadata)
            )

            state_logger.debug(
                "metadata_updated",
                session_id=session_id,
                ttl_seconds=self.ttl
            )
        except Exception as e:
            state_logger.warning(
                "update_metadata_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def is_session_active(
        self,
        session_id: str
    ) -> bool:
        """
        PATTERN: Activity-based session validation
        WHY: Auto-end sessions after inactivity
        """
        if self._degraded_mode or not self.redis_client:
            return False

        try:
            metadata = await self.get_session_metadata(session_id)

            if not metadata:
                return False

            last_activity = datetime.fromisoformat(metadata["last_activity"])
            inactive_seconds = (datetime.utcnow() - last_activity).total_seconds()

            is_active = inactive_seconds < settings.session_timeout

            state_logger.debug(
                "session_activity_checked",
                session_id=session_id,
                is_active=is_active,
                inactive_seconds=inactive_seconds
            )

            return is_active
        except Exception as e:
            state_logger.warning(
                "check_session_active_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    async def end_session(self, session_id: str):
        """Clean up session data"""
        if self._degraded_mode or not self.redis_client:
            return

        try:
            keys = [
                f"session:{session_id}:context",
                f"session:{session_id}:metadata"
            ]

            deleted_count = 0
            for key in keys:
                result = await self.redis_client.delete(key)
                deleted_count += result

            state_logger.info(
                "session_ended",
                session_id=session_id,
                keys_deleted=deleted_count
            )
        except Exception as e:
            state_logger.warning(
                "end_session_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )

    async def get_active_sessions(self) -> List[str]:
        """Get all active session IDs for monitoring"""
        if self._degraded_mode or not self.redis_client:
            return []

        try:
            pattern = "session:*:metadata"
            keys = []

            async for key in self.redis_client.scan_iter(match=pattern):
                session_id = key.split(":")[1]
                if await self.is_session_active(session_id):
                    keys.append(session_id)

            state_logger.debug("active_sessions_retrieved", count=len(keys))
            return keys
        except Exception as e:
            state_logger.warning(
                "get_active_sessions_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return []
    
    async def close(self):
        """Cleanup Redis connection"""
        if self.redis_client:
            state_logger.info("redis_connection_closing")
            await self.redis_client.close()
            state_logger.info("redis_connection_closed")

# Global state manager instance
state_manager = StateManager()