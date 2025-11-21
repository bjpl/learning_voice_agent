"""
Database Layer with SQLite and FTS5 + Vector Search
PATTERN: Repository pattern with async operations
WHY: Separation of concerns and testability
RESILIENCE: Transaction retry and rollback on failures
"""
import aiosqlite
import json
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from app.resilience import with_retry
from app.logger import db_logger
from app.vector import VectorStore, vector_store

class Database:
    def __init__(self, db_path: str = "learning_captures.db", enable_vector: bool = True):
        self.db_path = db_path
        self.enable_vector = enable_vector
        self.vector_store: Optional[VectorStore] = None
        self._initialized = False
    
    @with_retry(max_attempts=3, initial_wait=0.5)
    async def initialize(self):
        """
        CONCEPT: Lazy initialization with FTS5 virtual table + vector store
        WHY: Efficient full-text search without external dependencies + semantic search
        RESILIENCE: Retry up to 3 times on initialization failures
        """
        if self._initialized:
            return

        try:
            db_logger.info("database_initialization_started", db_path=self.db_path, vector_enabled=self.enable_vector)

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
            db_logger.info("database_initialization_complete", db_path=self.db_path)
        except Exception as e:
            db_logger.error(
                "database_initialization_failed",
                db_path=self.db_path,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Connection pool manager"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db
    
    @with_retry(max_attempts=3, initial_wait=0.5)
    async def save_exchange(
        self,
        session_id: str,
        user_text: str,
        agent_text: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        PATTERN: Async write with automatic FTS indexing
        RESILIENCE: Retry on write failures with transaction rollback
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    INSERT INTO captures (session_id, user_text, agent_text, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, user_text, agent_text, json.dumps(metadata or {}))
                )
                await db.commit()
                exchange_id = cursor.lastrowid

                db_logger.info(
                    "exchange_saved",
                    session_id=session_id,
                    exchange_id=exchange_id,
                    user_text_length=len(user_text),
                    agent_text_length=len(agent_text),
                    source=metadata.get('source') if metadata else None
                )

                return exchange_id
        except Exception as e:
            db_logger.error(
                "save_exchange_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        CONCEPT: Efficient windowed retrieval
        WHY: Only fetch what's needed for context
        RESILIENCE: Graceful failure with empty result
        """
        try:
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
                results = [dict(row) for row in reversed(rows)]

                db_logger.debug(
                    "session_history_retrieved",
                    session_id=session_id,
                    results_count=len(results),
                    limit=limit
                )

                return results
        except Exception as e:
            db_logger.error(
                "get_session_history_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return []
    
    async def search_captures(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        PATTERN: FTS5 search with BM25 ranking
        WHY: Better relevance than simple LIKE queries
        RESILIENCE: Return empty results on search failures
        """
        try:
            db_logger.debug("fts_search_started", query=query, limit=limit)

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
                results = [dict(row) for row in rows]

                db_logger.info(
                    "fts_search_complete",
                    query=query,
                    results_count=len(results),
                    limit=limit
                )

                return results
        except Exception as e:
            db_logger.error(
                "fts_search_failed",
                query=query,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return []
    
    async def get_stats(self) -> Dict:
        """Database statistics for monitoring"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT
                        COUNT(*) as total_captures,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        MAX(timestamp) as last_capture
                    FROM captures
                """)
                stats = dict(await cursor.fetchone())
                db_logger.debug("database_stats_retrieved", **stats)
                return stats
        except Exception as e:
            db_logger.error(
                "get_stats_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return {
                "total_captures": 0,
                "unique_sessions": 0,
                "last_capture": None,
                "error": str(e)
            }

# Global database instance
db = Database()