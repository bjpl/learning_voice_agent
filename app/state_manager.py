"""
Redis State Management for Conversation Context
PATTERN: Cache-aside pattern with TTL
WHY: Fast context retrieval without database overhead
"""
import json
import redis.asyncio as redis
from typing import List, Dict, Optional
from datetime import datetime
from app.config import settings
from app.logger import state_logger as logger


class StateManager:
    def __init__(self):
        self.redis_client = None
        self.ttl = settings.redis_ttl
        logger.info(
            f"state_manager_created with redis_url={settings.redis_url}, ttl={self.ttl}"
        )

    async def initialize(self):
        """
        CONCEPT: Connection pooling for Redis
        WHY: Reuse connections for better performance

        PATTERN: Proper connection pool configuration
        - max_connections: Limits concurrent connections
        - health_check_interval: Auto-reconnect on failure
        - socket_timeout: Prevent hanging connections
        """
        logger.info("state_manager_initializing")

        try:
            self.redis_client = await redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
                health_check_interval=30,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("state_manager_initialized", status="connected")

        except redis.ConnectionError as e:
            logger.warning(
                "redis_connection_failed",
                error=str(e),
                fallback="in-memory"
            )
            # Could implement in-memory fallback here
            self.redis_client = None

    async def get_conversation_context(
        self,
        session_id: str
    ) -> List[Dict]:
        """
        PATTERN: Sliding window of conversation history
        WHY: Claude needs context for coherent responses
        """
        if not self.redis_client:
            logger.warning("redis_unavailable", operation="get_context")
            return []

        key = f"session:{session_id}:context"

        try:
            data = await self.redis_client.get(key)

            if data:
                context = json.loads(data)
                logger.debug(
                    "context_retrieved",
                    session_id=session_id,
                    exchange_count=len(context)
                )
                return context

            return []

        except redis.RedisError as e:
            logger.error(
                "redis_get_error",
                session_id=session_id,
                error=str(e)
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
        """
        if not self.redis_client:
            logger.warning("redis_unavailable", operation="update_context")
            return

        key = f"session:{session_id}:context"

        try:
            # Get existing context
            context = await self.get_conversation_context(session_id)

            # Add new exchange
            context.append({
                "timestamp": datetime.utcnow().isoformat(),
                "user": user_text,
                "agent": agent_text
            })

            # Keep only last N exchanges
            if len(context) > settings.max_context_exchanges:
                context = context[-settings.max_context_exchanges:]

            # Save with TTL
            await self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(context)
            )

            logger.debug(
                "context_updated",
                session_id=session_id,
                exchange_count=len(context)
            )

        except redis.RedisError as e:
            logger.error(
                "redis_update_error",
                session_id=session_id,
                error=str(e)
            )

    async def get_session_metadata(
        self,
        session_id: str
    ) -> Optional[Dict]:
        """Get session metadata like start time, exchange count"""
        if not self.redis_client:
            return None

        key = f"session:{session_id}:metadata"

        try:
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None

        except redis.RedisError as e:
            logger.error(
                "redis_metadata_get_error",
                session_id=session_id,
                error=str(e)
            )
            return None

    async def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict
    ):
        """Update session metadata with activity tracking"""
        if not self.redis_client:
            return

        key = f"session:{session_id}:metadata"

        try:
            # Update last activity
            metadata["last_activity"] = datetime.utcnow().isoformat()

            await self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(metadata)
            )

            logger.debug(
                "metadata_updated",
                session_id=session_id,
                exchange_count=metadata.get("exchange_count", 0)
            )

        except redis.RedisError as e:
            logger.error(
                "redis_metadata_update_error",
                session_id=session_id,
                error=str(e)
            )

    async def is_session_active(
        self,
        session_id: str
    ) -> bool:
        """
        PATTERN: Activity-based session validation
        WHY: Auto-end sessions after inactivity
        """
        metadata = await self.get_session_metadata(session_id)

        if not metadata:
            return False

        try:
            last_activity = datetime.fromisoformat(metadata["last_activity"])
            inactive_seconds = (datetime.utcnow() - last_activity).total_seconds()

            return inactive_seconds < settings.session_timeout

        except (KeyError, ValueError) as e:
            logger.warning(
                "invalid_session_metadata",
                session_id=session_id,
                error=str(e)
            )
            return False

    async def end_session(self, session_id: str):
        """Clean up session data"""
        if not self.redis_client:
            return

        keys = [
            f"session:{session_id}:context",
            f"session:{session_id}:metadata"
        ]

        try:
            for key in keys:
                await self.redis_client.delete(key)

            logger.info("session_ended", session_id=session_id)

        except redis.RedisError as e:
            logger.error(
                "redis_delete_error",
                session_id=session_id,
                error=str(e)
            )

    async def get_active_sessions(self) -> List[str]:
        """Get all active session IDs for monitoring"""
        if not self.redis_client:
            return []

        pattern = "session:*:metadata"
        keys = []

        try:
            async for key in self.redis_client.scan_iter(match=pattern):
                session_id = key.split(":")[1]
                if await self.is_session_active(session_id):
                    keys.append(session_id)

            logger.debug("active_sessions_fetched", count=len(keys))
            return keys

        except redis.RedisError as e:
            logger.error(
                "redis_scan_error",
                error=str(e)
            )
            return []

    async def close(self):
        """Cleanup Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("redis_connection_closed")


# Global state manager instance
state_manager = StateManager()
