"""
Database Layer with SQLite and PostgreSQL Support
PATTERN: Strategy pattern for multi-backend database support
WHY: Enables both local development (SQLite) and production deployment (PostgreSQL)
CRITICAL: Railway provides DATABASE_URL automatically for PostgreSQL
"""
import aiosqlite
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Protocol
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod

from app.logger import db_logger as logger
from app.config import settings

# Import PostgreSQL driver conditionally
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg_not_available", msg="PostgreSQL support disabled")


class DatabaseBackend(ABC):
    """Abstract base class for database backends"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database schema"""
        pass

    @abstractmethod
    async def save_exchange(
        self, session_id: str, user_text: str, agent_text: str, metadata: Optional[Dict[str, Any]]
    ) -> int:
        """Save conversation exchange"""
        pass

    @abstractmethod
    async def get_session_history(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get session history"""
        pass

    @abstractmethod
    async def search_captures(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search captures with full-text search"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> int:
        """Delete session data"""
        pass


class SQLiteBackend(DatabaseBackend):
    """SQLite backend with FTS5 full-text search"""

    def __init__(self, db_path: str = "learning_captures.db"):
        self.db_path = db_path
        self._initialized = False
        logger.info(f"sqlite_backend_created: {db_path}")

    async def initialize(self) -> None:
        """
        CONCEPT: Lazy initialization with FTS5 virtual table
        WHY: Efficient full-text search without external dependencies
        """
        if self._initialized:
            logger.debug("database_already_initialized")
            return

        logger.info("database_initializing", db_path=self.db_path)

        async with aiosqlite.connect(self.db_path) as db:
            # Main captures table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_text TEXT NOT NULL,
                    agent_text TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # FTS5 virtual table for search
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS captures_fts
                USING fts5(
                    session_id UNINDEXED,
                    user_text,
                    agent_text,
                    content=captures,
                    content_rowid=id
                )
            """)

            # Triggers to keep FTS index in sync
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS captures_ai
                AFTER INSERT ON captures BEGIN
                    INSERT INTO captures_fts(rowid, session_id, user_text, agent_text)
                    VALUES (new.id, new.session_id, new.user_text, new.agent_text);
                END
            """)

            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS captures_ad
                AFTER DELETE ON captures BEGIN
                    DELETE FROM captures_fts WHERE rowid = old.id;
                END
            """)

            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS captures_au
                AFTER UPDATE ON captures BEGIN
                    UPDATE captures_fts
                    SET user_text = new.user_text, agent_text = new.agent_text
                    WHERE rowid = new.id;
                END
            """)

            # Index for session queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp
                ON captures(session_id, timestamp DESC)
            """)

            await db.commit()

        self._initialized = True
        logger.info("database_initialized", db_path=self.db_path)

    @asynccontextmanager
    async def get_connection(self):
        """Connection pool manager"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def save_exchange(
        self,
        session_id: str,
        user_text: str,
        agent_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        PATTERN: Async write with automatic FTS indexing
        """
        logger.debug(
            "saving_exchange",
            session_id=session_id,
            user_text_length=len(user_text),
            agent_text_length=len(agent_text)
        )

        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                INSERT INTO captures (session_id, user_text, agent_text, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, user_text, agent_text, json.dumps(metadata or {}))
            )
            await db.commit()
            row_id = cursor.lastrowid

            logger.info(
                "exchange_saved",
                session_id=session_id,
                row_id=row_id
            )
            return row_id

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        CONCEPT: Efficient windowed retrieval
        WHY: Only fetch what's needed for context
        """
        logger.debug(
            "fetching_session_history",
            session_id=session_id,
            limit=limit
        )

        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT id, timestamp, user_text, agent_text, metadata
                FROM captures
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit)
            )
            rows = await cursor.fetchall()
            result = [dict(row) for row in reversed(rows)]

            logger.debug(
                "session_history_fetched",
                session_id=session_id,
                count=len(result)
            )
            return result

    async def search_captures(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        PATTERN: FTS5 search with BM25 ranking
        WHY: Better relevance than simple LIKE queries
        """
        logger.info(
            "searching_captures",
            query=query,
            limit=limit
        )

        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT
                    c.id, c.session_id, c.timestamp,
                    c.user_text, c.agent_text,
                    snippet(captures_fts, 1, '<mark>', '</mark>', '...', 32) as user_snippet,
                    snippet(captures_fts, 2, '<mark>', '</mark>', '...', 32) as agent_snippet
                FROM captures c
                JOIN captures_fts ON c.id = captures_fts.rowid
                WHERE captures_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit)
            )
            rows = await cursor.fetchall()
            result = [dict(row) for row in rows]

            logger.info(
                "search_completed",
                query=query,
                results_count=len(result)
            )
            return result

    async def get_stats(self) -> Dict[str, Any]:
        """Database statistics for monitoring"""
        logger.debug("fetching_database_stats")

        async with self.get_connection() as db:
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total_captures,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    MAX(timestamp) as last_capture
                FROM captures
            """)
            result = dict(await cursor.fetchone())

            logger.info(
                "database_stats",
                total_captures=result.get("total_captures", 0),
                unique_sessions=result.get("unique_sessions", 0)
            )
            return result

    async def delete_session(self, session_id: str) -> int:
        """
        Delete all captures for a session.
        Returns number of deleted rows.
        """
        logger.info("deleting_session", session_id=session_id)

        async with self.get_connection() as db:
            cursor = await db.execute(
                "DELETE FROM captures WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()
            deleted_count = cursor.rowcount

            logger.info(
                "session_deleted",
                session_id=session_id,
                deleted_count=deleted_count
            )
            return deleted_count


class PostgreSQLBackend(DatabaseBackend):
    """
    PostgreSQL backend with full-text search
    PATTERN: PostgreSQL-native tsvector for full-text search
    WHY: Better performance and scalability than SQLite FTS5
    """

    def __init__(self, database_url: str):
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg not installed - cannot use PostgreSQL backend")

        self.database_url = database_url
        self._initialized = False
        self._pool: Optional[asyncpg.Pool] = None
        logger.info("postgresql_backend_created", url=self._sanitize_url(database_url))

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove credentials from URL for logging"""
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)

    async def initialize(self) -> None:
        """
        CONCEPT: Connection pooling with PostgreSQL
        WHY: Efficient connection reuse for multi-instance deployment
        """
        if self._initialized:
            logger.debug("postgresql_already_initialized")
            return

        logger.info("postgresql_initializing")

        # Create connection pool
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )

        async with self._pool.acquire() as conn:
            # Main captures table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS captures (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_text TEXT NOT NULL,
                    agent_text TEXT NOT NULL,
                    metadata JSONB,
                    search_vector tsvector
                )
            """)

            # Index for full-text search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_captures_search
                ON captures USING GIN(search_vector)
            """)

            # Index for session queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp
                ON captures(session_id, timestamp DESC)
            """)

            # Trigger to maintain search_vector
            await conn.execute("""
                CREATE OR REPLACE FUNCTION captures_search_update() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector :=
                        setweight(to_tsvector('english', coalesce(NEW.user_text, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(NEW.agent_text, '')), 'B');
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """)

            await conn.execute("""
                DROP TRIGGER IF EXISTS captures_search_trigger ON captures
            """)

            await conn.execute("""
                CREATE TRIGGER captures_search_trigger
                BEFORE INSERT OR UPDATE ON captures
                FOR EACH ROW EXECUTE FUNCTION captures_search_update()
            """)

        self._initialized = True
        logger.info("postgresql_initialized")

    async def save_exchange(
        self,
        session_id: str,
        user_text: str,
        agent_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save exchange with automatic search vector update"""
        logger.debug(
            "saving_exchange_postgres",
            session_id=session_id,
            user_text_length=len(user_text),
            agent_text_length=len(agent_text)
        )

        async with self._pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO captures (session_id, user_text, agent_text, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                session_id, user_text, agent_text, json.dumps(metadata or {})
            )

            logger.info("exchange_saved_postgres", session_id=session_id, row_id=row_id)
            return row_id

    async def get_session_history(
        self, session_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch session history"""
        logger.debug("fetching_session_history_postgres", session_id=session_id, limit=limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, timestamp, user_text, agent_text, metadata
                FROM captures
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                session_id, limit
            )

            result = [dict(row) for row in reversed(rows)]
            logger.debug("session_history_fetched_postgres", session_id=session_id, count=len(result))
            return result

    async def search_captures(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        PATTERN: PostgreSQL full-text search with ts_rank
        WHY: Native PostgreSQL ranking for better relevance
        """
        logger.info("searching_captures_postgres", query=query, limit=limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, session_id, timestamp, user_text, agent_text,
                    ts_headline('english', user_text, plainto_tsquery('english', $1),
                        'MaxWords=10, MinWords=5, StartSel=<mark>, StopSel=</mark>') as user_snippet,
                    ts_headline('english', agent_text, plainto_tsquery('english', $1),
                        'MaxWords=10, MinWords=5, StartSel=<mark>, StopSel=</mark>') as agent_snippet,
                    ts_rank(search_vector, plainto_tsquery('english', $1)) as rank
                FROM captures
                WHERE search_vector @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $2
                """,
                query, limit
            )

            result = [dict(row) for row in rows]
            logger.info("search_completed_postgres", query=query, results_count=len(result))
            return result

    async def get_stats(self) -> Dict[str, Any]:
        """Database statistics"""
        logger.debug("fetching_database_stats_postgres")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_captures,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    MAX(timestamp) as last_capture
                FROM captures
            """)

            result = dict(row)
            logger.info(
                "database_stats_postgres",
                total_captures=result.get("total_captures", 0),
                unique_sessions=result.get("unique_sessions", 0)
            )
            return result

    async def delete_session(self, session_id: str) -> int:
        """Delete all captures for a session"""
        logger.info("deleting_session_postgres", session_id=session_id)

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM captures WHERE session_id = $1",
                session_id
            )
            # Extract count from "DELETE N" string
            deleted_count = int(result.split()[-1]) if result else 0

            logger.info("session_deleted_postgres", session_id=session_id, deleted_count=deleted_count)
            return deleted_count

    async def close(self) -> None:
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("postgresql_pool_closed")


class Database:
    """
    Database facade with automatic backend selection
    PATTERN: Strategy pattern with auto-detection
    WHY: Seamless development (SQLite) to production (PostgreSQL) transition
    """

    def __init__(self, database_url: Optional[str] = None):
        # Backward compatibility: accept raw paths like ":memory:" or "path/to/db.db"
        if database_url and "://" not in database_url:
            database_url = f"sqlite:///{database_url}"
        self.database_url = database_url or settings.database_url
        self._backend: DatabaseBackend = self._create_backend()
        self._initialized = False  # Backward compat: expose initialization state
        logger.info(
            "database_created",
            backend=type(self._backend).__name__,
            url=self._sanitize_url(self.database_url)
        )

    @property
    def db_path(self) -> str:
        """Backward compatibility: return db path for SQLite backends"""
        if hasattr(self._backend, 'db_path'):
            return self._backend.db_path
        return self.database_url

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove credentials from URL for logging"""
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)

    def _create_backend(self) -> DatabaseBackend:
        """
        CONCEPT: Automatic backend detection from DATABASE_URL
        WHY: Zero-config deployment to Railway (uses DATABASE_URL env var)
        """
        db_type = self.database_url.split("://")[0].lower()

        if db_type in ("postgresql", "postgres"):
            if not ASYNCPG_AVAILABLE:
                logger.error(
                    "asyncpg_required",
                    msg="PostgreSQL URL provided but asyncpg not installed. Falling back to SQLite."
                )
                return SQLiteBackend()
            return PostgreSQLBackend(self.database_url)

        elif db_type == "sqlite":
            # Extract path from sqlite:///path
            db_path = self.database_url.replace("sqlite:///", "").replace("sqlite://", "")
            return SQLiteBackend(db_path)

        else:
            logger.warning(
                "unknown_database_type",
                db_type=db_type,
                msg="Unknown database type, defaulting to SQLite"
            )
            return SQLiteBackend()

    # Delegate all methods to backend
    async def initialize(self) -> None:
        """Initialize database schema"""
        await self._backend.initialize()
        self._initialized = True  # Sync with backend state

    async def save_exchange(
        self, session_id: str, user_text: str, agent_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save conversation exchange"""
        return await self._backend.save_exchange(session_id, user_text, agent_text, metadata)

    async def get_session_history(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get session history"""
        return await self._backend.get_session_history(session_id, limit)

    async def search_captures(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search captures with full-text search"""
        return await self._backend.search_captures(query, limit)

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return await self._backend.get_stats()

    async def delete_session(self, session_id: str) -> int:
        """Delete session data"""
        return await self._backend.delete_session(session_id)

    async def close(self) -> None:
        """Close database connections"""
        if hasattr(self._backend, 'close'):
            await self._backend.close()


# Global database instance - auto-detects backend from DATABASE_URL
db = Database()
