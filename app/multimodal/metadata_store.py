"""
Metadata Store - File Metadata Persistence

SPECIFICATION:
- Store file metadata in SQLite database
- Track upload information, analysis results, indexing status
- Support queries by session, file type, date range
- Store relationships between files and conversations
- Track file access and usage statistics

ARCHITECTURE:
- SQLite database for metadata
- Async database operations
- Schema with proper indexes
- Migration support

PATTERN: Repository pattern for metadata persistence
WHY: Centralized metadata management with query capabilities
RESILIENCE: Transaction support, error handling, data validation
"""

import aiosqlite
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from app.logger import api_logger


class MetadataStore:
    """
    File metadata persistence layer

    PATTERN: Repository pattern with async SQLite
    WHY: Structured metadata storage with querying
    """

    def __init__(self, db_path: str = "file_metadata.db"):
        """
        Initialize metadata store

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.initialized = False

        api_logger.info(
            "metadata_store_initialized",
            db_path=str(self.db_path)
        )

    async def initialize(self):
        """Initialize database schema"""
        if self.initialized:
            return

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # File metadata table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS file_metadata (
                        file_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        original_filename TEXT NOT NULL,
                        stored_path TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        file_hash TEXT NOT NULL,
                        upload_timestamp TEXT NOT NULL,

                        -- Analysis results
                        analysis_status TEXT DEFAULT 'pending',
                        analysis_result TEXT,
                        analysis_timestamp TEXT,

                        -- Indexing status
                        indexed BOOLEAN DEFAULT 0,
                        index_timestamp TEXT,

                        -- Usage tracking
                        access_count INTEGER DEFAULT 0,
                        last_accessed TEXT,

                        -- Additional metadata
                        metadata TEXT,

                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                """)

                # Create indexes
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_session
                    ON file_metadata(session_id)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_type
                    ON file_metadata(file_type)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_upload_timestamp
                    ON file_metadata(upload_timestamp DESC)
                """)

                await db.commit()

            self.initialized = True
            api_logger.info("metadata_store_schema_initialized")

        except Exception as e:
            api_logger.error(
                "metadata_store_initialization_error",
                error=str(e),
                exc_info=True
            )
            raise

    async def save_file_metadata(self, metadata: Dict) -> bool:
        """
        Save file metadata

        Args:
            metadata: File metadata dictionary

        Returns:
            True if successful
        """
        if not self.initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO file_metadata (
                        file_id, session_id, file_type, original_filename,
                        stored_path, mime_type, file_size, file_hash,
                        upload_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata['file_id'],
                    metadata['session_id'],
                    metadata['file_type'],
                    metadata['original_filename'],
                    metadata['stored_path'],
                    metadata['mime_type'],
                    metadata['file_size'],
                    metadata['file_hash'],
                    metadata['upload_timestamp']
                ))

                await db.commit()

            api_logger.info(
                "file_metadata_saved",
                file_id=metadata['file_id'],
                file_type=metadata['file_type']
            )
            return True

        except Exception as e:
            api_logger.error(
                "file_metadata_save_error",
                file_id=metadata.get('file_id'),
                error=str(e),
                exc_info=True
            )
            return False

    async def save_analysis(
        self,
        file_id: str,
        analysis_type: str,
        analysis_result: Dict
    ) -> bool:
        """
        Save analysis results

        Args:
            file_id: File ID
            analysis_type: Type of analysis ('vision', 'document')
            analysis_result: Analysis results dictionary

        Returns:
            True if successful
        """
        if not self.initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE file_metadata
                    SET analysis_status = ?,
                        analysis_result = ?,
                        analysis_timestamp = ?
                    WHERE file_id = ?
                """, (
                    'completed' if analysis_result.get('success') else 'failed',
                    json.dumps(analysis_result),
                    datetime.utcnow().isoformat(),
                    file_id
                ))

                await db.commit()

            api_logger.info(
                "analysis_saved",
                file_id=file_id,
                analysis_type=analysis_type
            )
            return True

        except Exception as e:
            api_logger.error(
                "analysis_save_error",
                file_id=file_id,
                error=str(e),
                exc_info=True
            )
            return False

    async def mark_indexed(self, file_id: str) -> bool:
        """
        Mark file as indexed in vector store

        Args:
            file_id: File ID

        Returns:
            True if successful
        """
        if not self.initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE file_metadata
                    SET indexed = 1,
                        index_timestamp = ?
                    WHERE file_id = ?
                """, (
                    datetime.utcnow().isoformat(),
                    file_id
                ))

                await db.commit()

            return True

        except Exception as e:
            api_logger.error(
                "mark_indexed_error",
                file_id=file_id,
                error=str(e)
            )
            return False

    async def get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """
        Retrieve file metadata

        Args:
            file_id: File ID

        Returns:
            Metadata dictionary or None
        """
        if not self.initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("""
                    SELECT * FROM file_metadata WHERE file_id = ?
                """, (file_id,))

                row = await cursor.fetchone()
                if row:
                    return dict(row)

            return None

        except Exception as e:
            api_logger.error(
                "get_metadata_error",
                file_id=file_id,
                error=str(e)
            )
            return None

    async def get_session_files(
        self,
        session_id: str,
        file_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all files for a session

        Args:
            session_id: Session ID
            file_type: Optional file type filter

        Returns:
            List of file metadata dictionaries
        """
        if not self.initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                if file_type:
                    cursor = await db.execute("""
                        SELECT * FROM file_metadata
                        WHERE session_id = ? AND file_type = ?
                        ORDER BY upload_timestamp DESC
                    """, (session_id, file_type))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM file_metadata
                        WHERE session_id = ?
                        ORDER BY upload_timestamp DESC
                    """, (session_id,))

                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            api_logger.error(
                "get_session_files_error",
                session_id=session_id,
                error=str(e)
            )
            return []

    async def track_access(self, file_id: str):
        """
        Track file access

        Args:
            file_id: File ID
        """
        if not self.initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE file_metadata
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE file_id = ?
                """, (
                    datetime.utcnow().isoformat(),
                    file_id
                ))

                await db.commit()

        except Exception as e:
            api_logger.error(
                "track_access_error",
                file_id=file_id,
                error=str(e)
            )


# Singleton instance
metadata_store = MetadataStore()
