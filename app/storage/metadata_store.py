"""
Metadata Store for Multi-Modal Files

SPECIFICATION:
- SQLite-based metadata storage
- Efficient queries with indexes
- Full-text search on analysis results
- File deduplication tracking
- Session and file relationship management

ARCHITECTURE:
- Async SQLite operations
- Repository pattern
- Transaction management
- Automatic schema migration
"""

import aiosqlite
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from pathlib import Path

from app.storage.config import storage_config
from app.resilience import with_retry
from app.logger import get_logger

logger = get_logger(__name__)


class MetadataStore:
    """
    SQLite-based metadata store for multi-modal files

    PATTERN: Repository pattern with async operations
    WHY: Separation of concerns, testability, non-blocking I/O

    Schema:
        - multimodal_files: File metadata and organization
        - file_analysis: Analysis results (vision, OCR, extraction)
        - file_links: Links between files and knowledge graph

    Example:
        store = MetadataStore()
        await store.initialize()

        file_id = await store.save_file_metadata(
            session_id="session_123",
            file_type="image",
            original_filename="photo.jpg",
            stored_path="/path/to/file",
            file_size=1024000,
            file_hash="abc123..."
        )

        await store.save_analysis(
            file_id=file_id,
            analysis_type="vision",
            result={"objects": ["cat", "tree"], "confidence": 0.95}
        )
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or storage_config.metadata_db_path
        self._initialized = False

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @with_retry(max_attempts=3, initial_wait=0.5)
    async def initialize(self):
        """
        Initialize database schema with tables and indexes

        PATTERN: Idempotent schema creation
        WHY: Safe to call multiple times, handles upgrades
        """
        if self._initialized:
            return

        try:
            logger.info("metadata_store_initialization_started", db_path=self.db_path)

            async with aiosqlite.connect(self.db_path) as db:
                # Main files table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS multimodal_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id TEXT UNIQUE NOT NULL,
                        session_id TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        original_filename TEXT,
                        stored_path TEXT NOT NULL,
                        file_size INTEGER,
                        mime_type TEXT,
                        file_hash TEXT,
                        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES captures(session_id)
                    )
                """)

                # Analysis results table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS file_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        analysis_result TEXT,
                        analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (file_id) REFERENCES multimodal_files(file_id) ON DELETE CASCADE
                    )
                """)

                # File links to knowledge graph concepts
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS file_concept_links (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id TEXT NOT NULL,
                        concept_name TEXT NOT NULL,
                        link_type TEXT,
                        confidence REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (file_id) REFERENCES multimodal_files(file_id) ON DELETE CASCADE
                    )
                """)

                # FTS5 virtual table for full-text search on analysis
                await db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS file_analysis_fts
                    USING fts5(
                        file_id UNINDEXED,
                        analysis_type UNINDEXED,
                        analysis_text,
                        content=file_analysis,
                        content_rowid=id
                    )
                """)

                # Triggers to keep FTS in sync
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS file_analysis_ai
                    AFTER INSERT ON file_analysis BEGIN
                        INSERT INTO file_analysis_fts(rowid, file_id, analysis_type, analysis_text)
                        VALUES (new.id, new.file_id, new.analysis_type, new.analysis_result);
                    END
                """)

                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS file_analysis_ad
                    AFTER DELETE ON file_analysis BEGIN
                        DELETE FROM file_analysis_fts WHERE rowid = old.id;
                    END
                """)

                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS file_analysis_au
                    AFTER UPDATE ON file_analysis BEGIN
                        UPDATE file_analysis_fts
                        SET analysis_text = new.analysis_result
                        WHERE rowid = new.id;
                    END
                """)

                # Indexes for performance
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_session
                    ON multimodal_files(session_id, uploaded_at DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_type
                    ON multimodal_files(file_type, uploaded_at DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_hash
                    ON multimodal_files(file_hash)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_analysis_file
                    ON file_analysis(file_id, analysis_type)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_concept_links
                    ON file_concept_links(file_id, concept_name)
                """)

                await db.commit()

            self._initialized = True
            logger.info("metadata_store_initialized", db_path=self.db_path)

        except Exception as e:
            logger.error(
                "metadata_store_initialization_failed",
                db_path=self.db_path,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Connection context manager"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    @with_retry(max_attempts=3, initial_wait=0.5)
    async def save_file_metadata(
        self,
        file_id: str,
        session_id: str,
        file_type: str,
        original_filename: str,
        stored_path: str,
        file_size: int,
        mime_type: Optional[str] = None,
        file_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save file metadata to database

        Returns:
            Database row ID
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    INSERT INTO multimodal_files
                    (file_id, session_id, file_type, original_filename, stored_path,
                     file_size, mime_type, file_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id, session_id, file_type, original_filename, stored_path,
                        file_size, mime_type, file_hash, json.dumps(metadata or {})
                    )
                )
                await db.commit()
                row_id = cursor.lastrowid

                logger.info(
                    "file_metadata_saved",
                    file_id=file_id,
                    session_id=session_id,
                    file_type=file_type,
                    size_kb=file_size // 1024
                )

                return row_id

        except aiosqlite.IntegrityError as e:
            if "UNIQUE constraint" in str(e):
                logger.warning("file_already_exists", file_id=file_id)
                raise ValueError(f"File ID {file_id} already exists")
            raise
        except Exception as e:
            logger.error(
                "save_file_metadata_failed",
                file_id=file_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    @with_retry(max_attempts=2, initial_wait=0.5)
    async def save_analysis(
        self,
        file_id: str,
        analysis_type: str,
        result: Dict[str, Any]
    ) -> int:
        """
        Save analysis results for a file

        Args:
            file_id: File identifier
            analysis_type: Type of analysis (vision, ocr, extraction)
            result: Analysis result dictionary

        Returns:
            Analysis row ID
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    INSERT INTO file_analysis (file_id, analysis_type, analysis_result)
                    VALUES (?, ?, ?)
                    """,
                    (file_id, analysis_type, json.dumps(result))
                )
                await db.commit()
                row_id = cursor.lastrowid

                logger.info(
                    "analysis_saved",
                    file_id=file_id,
                    analysis_type=analysis_type
                )

                return row_id

        except Exception as e:
            logger.error(
                "save_analysis_failed",
                file_id=file_id,
                analysis_type=analysis_type,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_file_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata by ID"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM multimodal_files WHERE file_id = ?
                    """,
                    (file_id,)
                )
                row = await cursor.fetchone()

                if row:
                    result = dict(row)
                    # Parse JSON metadata
                    if result.get("metadata"):
                        result["metadata"] = json.loads(result["metadata"])
                    return result
                return None

        except Exception as e:
            logger.error("get_file_by_id_failed", file_id=file_id, error=str(e))
            return None

    async def get_file_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Find existing file by hash (for deduplication)"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM multimodal_files
                    WHERE file_hash = ?
                    ORDER BY uploaded_at DESC
                    LIMIT 1
                    """,
                    (file_hash,)
                )
                row = await cursor.fetchone()

                if row:
                    result = dict(row)
                    if result.get("metadata"):
                        result["metadata"] = json.loads(result["metadata"])
                    return result
                return None

        except Exception as e:
            logger.error("get_file_by_hash_failed", error=str(e))
            return None

    async def list_files(
        self,
        session_id: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List files with optional filters

        Args:
            session_id: Filter by session
            file_type: Filter by file type
            limit: Maximum results
            offset: Pagination offset
        """
        try:
            query = "SELECT * FROM multimodal_files WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if file_type:
                query += " AND file_type = ?"
                params.append(file_type)

            query += " ORDER BY uploaded_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            async with self.get_connection() as db:
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()

                results = []
                for row in rows:
                    result = dict(row)
                    if result.get("metadata"):
                        result["metadata"] = json.loads(result["metadata"])
                    results.append(result)

                logger.debug(
                    "files_listed",
                    count=len(results),
                    session_id=session_id,
                    file_type=file_type
                )

                return results

        except Exception as e:
            logger.error("list_files_failed", error=str(e), exc_info=True)
            return []

    async def get_file_analysis(
        self,
        file_id: str,
        analysis_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get analysis results for a file"""
        try:
            query = "SELECT * FROM file_analysis WHERE file_id = ?"
            params = [file_id]

            if analysis_type:
                query += " AND analysis_type = ?"
                params.append(analysis_type)

            query += " ORDER BY analyzed_at DESC"

            async with self.get_connection() as db:
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()

                results = []
                for row in rows:
                    result = dict(row)
                    if result.get("analysis_result"):
                        result["analysis_result"] = json.loads(result["analysis_result"])
                    results.append(result)

                return results

        except Exception as e:
            logger.error(
                "get_file_analysis_failed",
                file_id=file_id,
                error=str(e)
            )
            return []

    async def search_analysis(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Full-text search on analysis results

        PATTERN: FTS5 search for semantic matching
        WHY: Fast text search across all analysis data
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT
                        fa.*,
                        mf.original_filename,
                        mf.file_type,
                        mf.session_id,
                        snippet(file_analysis_fts, 2, '<mark>', '</mark>', '...', 32) as snippet
                    FROM file_analysis fa
                    JOIN file_analysis_fts ON fa.id = file_analysis_fts.rowid
                    JOIN multimodal_files mf ON fa.file_id = mf.file_id
                    WHERE file_analysis_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit)
                )
                rows = await cursor.fetchall()

                results = []
                for row in rows:
                    result = dict(row)
                    if result.get("analysis_result"):
                        result["analysis_result"] = json.loads(result["analysis_result"])
                    results.append(result)

                logger.info("analysis_search_complete", query=query, results=len(results))
                return results

        except Exception as e:
            logger.error("search_analysis_failed", query=query, error=str(e))
            return []

    async def link_file_to_concept(
        self,
        file_id: str,
        concept_name: str,
        link_type: str = "mentioned",
        confidence: float = 1.0
    ):
        """Link file to knowledge graph concept"""
        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO file_concept_links
                    (file_id, concept_name, link_type, confidence)
                    VALUES (?, ?, ?, ?)
                    """,
                    (file_id, concept_name, link_type, confidence)
                )
                await db.commit()

                logger.info(
                    "file_concept_linked",
                    file_id=file_id,
                    concept=concept_name,
                    link_type=link_type
                )

        except Exception as e:
            logger.error("link_file_to_concept_failed", file_id=file_id, error=str(e))
            raise

    async def get_file_concepts(self, file_id: str) -> List[Dict[str, Any]]:
        """Get all concepts linked to a file"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM file_concept_links
                    WHERE file_id = ?
                    ORDER BY confidence DESC
                    """,
                    (file_id,)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error("get_file_concepts_failed", file_id=file_id, error=str(e))
            return []

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete file metadata and all related records

        Returns:
            True if deleted, False if not found
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "DELETE FROM multimodal_files WHERE file_id = ?",
                    (file_id,)
                )
                await db.commit()

                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info("file_deleted", file_id=file_id)

                return deleted

        except Exception as e:
            logger.error("delete_file_failed", file_id=file_id, error=str(e))
            raise

    async def get_old_files(self, days: int) -> List[Dict[str, Any]]:
        """Get files older than specified days for cleanup"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT file_id, stored_path, file_size, uploaded_at
                    FROM multimodal_files
                    WHERE uploaded_at < ?
                    ORDER BY uploaded_at ASC
                    """,
                    (cutoff.isoformat(),)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error("get_old_files_failed", days=days, error=str(e))
            return []

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT
                        COUNT(*) as total_files,
                        SUM(file_size) as total_bytes,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        COUNT(DISTINCT file_type) as file_types
                    FROM multimodal_files
                """)
                stats = dict(await cursor.fetchone())

                # Get per-type stats
                cursor = await db.execute("""
                    SELECT
                        file_type,
                        COUNT(*) as count,
                        SUM(file_size) as total_bytes
                    FROM multimodal_files
                    GROUP BY file_type
                """)
                type_stats = [dict(row) for row in await cursor.fetchall()]
                stats["by_type"] = type_stats

                logger.debug("storage_stats_retrieved", **stats)
                return stats

        except Exception as e:
            logger.error("get_storage_stats_failed", error=str(e))
            return {}


# Global metadata store instance
metadata_store = MetadataStore()
