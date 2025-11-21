"""
Learning Data Store - SQLite Storage for Learning System

PATTERN: Repository pattern with async operations
WHY: Separation of concerns, testability, and data persistence

Stores:
- User preferences (JSON serialized)
- Feedback history
- Improvement experiments
- Quality metrics over time
"""

import aiosqlite
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
import asyncio

from app.learning.config import learning_config
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackRecord:
    """Record of user feedback"""
    id: Optional[int] = None
    session_id: str = ""
    timestamp: str = ""
    feedback_type: str = "explicit"  # explicit, implicit
    helpful: Optional[bool] = None
    rating: Optional[float] = None
    correction: Optional[str] = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    query_text: str = ""
    response_text: str = ""


@dataclass
class ImprovementRecord:
    """Record of an improvement experiment"""
    id: Optional[int] = None
    improvement_id: str = ""
    hypothesis: str = ""
    target_dimension: str = ""  # response_length, technical_depth, etc.
    change_description: str = ""
    status: str = "pending"  # pending, active, completed, rolled_back
    created_at: str = ""
    activated_at: Optional[str] = None
    completed_at: Optional[str] = None
    control_samples: int = 0
    treatment_samples: int = 0
    control_quality: float = 0.0
    treatment_quality: float = 0.0
    improvement_delta: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LearningStore:
    """
    SQLite-based storage for learning data

    PATTERN: Async repository with JSON serialization
    WHY: Flexible schema for evolving preferences
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or learning_config.learning_db_path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @asynccontextmanager
    async def get_connection(self):
        """Connection context manager"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def initialize(self):
        """
        Initialize database schema

        PATTERN: Idempotent initialization
        WHY: Safe to call multiple times
        """
        async with self._init_lock:
            if self._initialized:
                return

            try:
                async with self.get_connection() as db:
                    # User Preferences Table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS user_preferences (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT UNIQUE NOT NULL,
                            preferences_json TEXT NOT NULL,
                            interaction_count INTEGER DEFAULT 0,
                            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Feedback History Table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS feedback_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            feedback_type TEXT DEFAULT 'explicit',
                            helpful INTEGER,
                            rating REAL,
                            correction TEXT,
                            response_metadata TEXT,
                            query_text TEXT,
                            response_text TEXT
                        )
                    """)

                    # Improvement Experiments Table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS improvements (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            improvement_id TEXT UNIQUE NOT NULL,
                            hypothesis TEXT NOT NULL,
                            target_dimension TEXT NOT NULL,
                            change_description TEXT,
                            status TEXT DEFAULT 'pending',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            activated_at DATETIME,
                            completed_at DATETIME,
                            control_samples INTEGER DEFAULT 0,
                            treatment_samples INTEGER DEFAULT 0,
                            control_quality REAL DEFAULT 0.0,
                            treatment_quality REAL DEFAULT 0.0,
                            improvement_delta REAL DEFAULT 0.0,
                            metadata TEXT
                        )
                    """)

                    # Quality Metrics Table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS quality_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            response_id TEXT,
                            quality_score REAL,
                            relevance_score REAL,
                            coherence_score REAL,
                            helpfulness_score REAL,
                            word_count INTEGER,
                            response_time_ms REAL,
                            improvement_id TEXT,
                            is_treatment INTEGER DEFAULT 0
                        )
                    """)

                    # Vocabulary Adjustments Table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS vocabulary_adjustments (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            original_term TEXT NOT NULL,
                            preferred_term TEXT NOT NULL,
                            frequency INTEGER DEFAULT 1,
                            last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(session_id, original_term)
                        )
                    """)

                    # Indexes for efficient queries
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_feedback_session
                        ON feedback_history(session_id, timestamp DESC)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_quality_session
                        ON quality_metrics(session_id, timestamp DESC)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_improvements_status
                        ON improvements(status)
                    """)

                    await db.commit()

                self._initialized = True
                logger.info(
                    "learning_store_initialized",
                    db_path=self.db_path
                )

            except Exception as e:
                logger.error(
                    "learning_store_init_failed",
                    error=str(e),
                    exc_info=True
                )
                raise

    # =========================================================================
    # Preferences Operations
    # =========================================================================

    async def get_preferences(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user preferences for a session

        Returns:
            Preferences dict or None if not found
        """
        await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT preferences_json, interaction_count, last_updated
                    FROM user_preferences
                    WHERE session_id = ?
                    """,
                    (session_id,)
                )
                row = await cursor.fetchone()

                if row:
                    prefs = json.loads(row["preferences_json"])
                    prefs["_metadata"] = {
                        "interaction_count": row["interaction_count"],
                        "last_updated": row["last_updated"]
                    }
                    return prefs
                return None

        except Exception as e:
            logger.error(
                "get_preferences_failed",
                session_id=session_id,
                error=str(e)
            )
            return None

    async def save_preferences(
        self,
        session_id: str,
        preferences: Dict[str, Any],
        increment_count: bool = True
    ) -> bool:
        """
        Save or update user preferences

        PATTERN: Upsert with optional counter increment
        """
        await self.initialize()

        try:
            # Remove metadata before saving
            prefs_to_save = {k: v for k, v in preferences.items() if not k.startswith("_")}
            prefs_json = json.dumps(prefs_to_save)

            async with self.get_connection() as db:
                if increment_count:
                    await db.execute(
                        """
                        INSERT INTO user_preferences (session_id, preferences_json, interaction_count)
                        VALUES (?, ?, 1)
                        ON CONFLICT(session_id) DO UPDATE SET
                            preferences_json = excluded.preferences_json,
                            interaction_count = interaction_count + 1,
                            last_updated = CURRENT_TIMESTAMP
                        """,
                        (session_id, prefs_json)
                    )
                else:
                    await db.execute(
                        """
                        INSERT INTO user_preferences (session_id, preferences_json)
                        VALUES (?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            preferences_json = excluded.preferences_json,
                            last_updated = CURRENT_TIMESTAMP
                        """,
                        (session_id, prefs_json)
                    )
                await db.commit()

            logger.debug(
                "preferences_saved",
                session_id=session_id,
                keys=list(prefs_to_save.keys())
            )
            return True

        except Exception as e:
            logger.error(
                "save_preferences_failed",
                session_id=session_id,
                error=str(e)
            )
            return False

    async def get_interaction_count(self, session_id: str) -> int:
        """Get the number of interactions for a session"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "SELECT interaction_count FROM user_preferences WHERE session_id = ?",
                    (session_id,)
                )
                row = await cursor.fetchone()
                return row["interaction_count"] if row else 0
        except Exception as e:
            logger.error("get_interaction_count_failed", error=str(e))
            return 0

    # =========================================================================
    # Feedback Operations
    # =========================================================================

    async def save_feedback(self, feedback: FeedbackRecord) -> Optional[int]:
        """
        Save feedback record

        Returns:
            Feedback ID or None on failure
        """
        await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    INSERT INTO feedback_history
                    (session_id, feedback_type, helpful, rating, correction,
                     response_metadata, query_text, response_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.session_id,
                        feedback.feedback_type,
                        1 if feedback.helpful else (0 if feedback.helpful is False else None),
                        feedback.rating,
                        feedback.correction,
                        json.dumps(feedback.response_metadata),
                        feedback.query_text,
                        feedback.response_text
                    )
                )
                await db.commit()

                logger.info(
                    "feedback_saved",
                    session_id=feedback.session_id,
                    feedback_type=feedback.feedback_type,
                    helpful=feedback.helpful,
                    rating=feedback.rating
                )

                return cursor.lastrowid

        except Exception as e:
            logger.error(
                "save_feedback_failed",
                session_id=feedback.session_id,
                error=str(e)
            )
            return None

    async def get_feedback_history(
        self,
        session_id: str,
        limit: int = 100,
        feedback_type: Optional[str] = None
    ) -> List[FeedbackRecord]:
        """Get feedback history for a session"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                if feedback_type:
                    cursor = await db.execute(
                        """
                        SELECT * FROM feedback_history
                        WHERE session_id = ? AND feedback_type = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (session_id, feedback_type, limit)
                    )
                else:
                    cursor = await db.execute(
                        """
                        SELECT * FROM feedback_history
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (session_id, limit)
                    )

                rows = await cursor.fetchall()
                return [
                    FeedbackRecord(
                        id=row["id"],
                        session_id=row["session_id"],
                        timestamp=row["timestamp"],
                        feedback_type=row["feedback_type"],
                        helpful=bool(row["helpful"]) if row["helpful"] is not None else None,
                        rating=row["rating"],
                        correction=row["correction"],
                        response_metadata=json.loads(row["response_metadata"] or "{}"),
                        query_text=row["query_text"] or "",
                        response_text=row["response_text"] or ""
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(
                "get_feedback_history_failed",
                session_id=session_id,
                error=str(e)
            )
            return []

    async def get_feedback_stats(
        self,
        session_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get aggregated feedback statistics"""
        await self.initialize()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        try:
            async with self.get_connection() as db:
                if session_id:
                    cursor = await db.execute(
                        """
                        SELECT
                            COUNT(*) as total_feedback,
                            SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count,
                            SUM(CASE WHEN helpful = 0 THEN 1 ELSE 0 END) as not_helpful_count,
                            AVG(rating) as avg_rating,
                            COUNT(correction) as correction_count
                        FROM feedback_history
                        WHERE session_id = ? AND timestamp > ?
                        """,
                        (session_id, cutoff)
                    )
                else:
                    cursor = await db.execute(
                        """
                        SELECT
                            COUNT(*) as total_feedback,
                            SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count,
                            SUM(CASE WHEN helpful = 0 THEN 1 ELSE 0 END) as not_helpful_count,
                            AVG(rating) as avg_rating,
                            COUNT(correction) as correction_count
                        FROM feedback_history
                        WHERE timestamp > ?
                        """,
                        (cutoff,)
                    )

                row = await cursor.fetchone()
                total = row["total_feedback"] or 0
                helpful = row["helpful_count"] or 0

                return {
                    "total_feedback": total,
                    "helpful_count": helpful,
                    "not_helpful_count": row["not_helpful_count"] or 0,
                    "helpful_rate": helpful / total if total > 0 else 0.0,
                    "average_rating": row["avg_rating"],
                    "correction_count": row["correction_count"] or 0,
                    "period_days": days
                }

        except Exception as e:
            logger.error("get_feedback_stats_failed", error=str(e))
            return {"error": str(e)}

    # =========================================================================
    # Improvement Operations
    # =========================================================================

    async def save_improvement(self, improvement: ImprovementRecord) -> Optional[str]:
        """Save improvement experiment"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO improvements
                    (improvement_id, hypothesis, target_dimension, change_description,
                     status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(improvement_id) DO UPDATE SET
                        status = excluded.status,
                        control_samples = excluded.control_samples,
                        treatment_samples = excluded.treatment_samples,
                        control_quality = excluded.control_quality,
                        treatment_quality = excluded.treatment_quality,
                        improvement_delta = excluded.improvement_delta
                    """,
                    (
                        improvement.improvement_id,
                        improvement.hypothesis,
                        improvement.target_dimension,
                        improvement.change_description,
                        improvement.status,
                        json.dumps(improvement.metadata)
                    )
                )
                await db.commit()

                logger.info(
                    "improvement_saved",
                    improvement_id=improvement.improvement_id,
                    status=improvement.status
                )

                return improvement.improvement_id

        except Exception as e:
            logger.error("save_improvement_failed", error=str(e))
            return None

    async def get_improvement(self, improvement_id: str) -> Optional[ImprovementRecord]:
        """Get improvement by ID"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM improvements WHERE improvement_id = ?",
                    (improvement_id,)
                )
                row = await cursor.fetchone()

                if row:
                    return ImprovementRecord(
                        id=row["id"],
                        improvement_id=row["improvement_id"],
                        hypothesis=row["hypothesis"],
                        target_dimension=row["target_dimension"],
                        change_description=row["change_description"] or "",
                        status=row["status"],
                        created_at=row["created_at"],
                        activated_at=row["activated_at"],
                        completed_at=row["completed_at"],
                        control_samples=row["control_samples"],
                        treatment_samples=row["treatment_samples"],
                        control_quality=row["control_quality"],
                        treatment_quality=row["treatment_quality"],
                        improvement_delta=row["improvement_delta"],
                        metadata=json.loads(row["metadata"] or "{}")
                    )
                return None

        except Exception as e:
            logger.error("get_improvement_failed", error=str(e))
            return None

    async def get_active_improvements(self) -> List[ImprovementRecord]:
        """Get all active improvement experiments"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM improvements WHERE status = 'active'"
                )
                rows = await cursor.fetchall()

                return [
                    ImprovementRecord(
                        id=row["id"],
                        improvement_id=row["improvement_id"],
                        hypothesis=row["hypothesis"],
                        target_dimension=row["target_dimension"],
                        change_description=row["change_description"] or "",
                        status=row["status"],
                        created_at=row["created_at"],
                        activated_at=row["activated_at"],
                        control_samples=row["control_samples"],
                        treatment_samples=row["treatment_samples"],
                        control_quality=row["control_quality"],
                        treatment_quality=row["treatment_quality"],
                        improvement_delta=row["improvement_delta"],
                        metadata=json.loads(row["metadata"] or "{}")
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error("get_active_improvements_failed", error=str(e))
            return []

    async def update_improvement_status(
        self,
        improvement_id: str,
        status: str,
        **kwargs
    ) -> bool:
        """Update improvement status and metrics"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                # Build dynamic update query
                updates = ["status = ?"]
                values = [status]

                if status == "active":
                    updates.append("activated_at = CURRENT_TIMESTAMP")
                elif status in ("completed", "rolled_back"):
                    updates.append("completed_at = CURRENT_TIMESTAMP")

                for key, value in kwargs.items():
                    if key in ("control_samples", "treatment_samples",
                              "control_quality", "treatment_quality", "improvement_delta"):
                        updates.append(f"{key} = ?")
                        values.append(value)

                values.append(improvement_id)

                await db.execute(
                    f"UPDATE improvements SET {', '.join(updates)} WHERE improvement_id = ?",
                    tuple(values)
                )
                await db.commit()

                logger.info(
                    "improvement_status_updated",
                    improvement_id=improvement_id,
                    status=status
                )

                return True

        except Exception as e:
            logger.error("update_improvement_status_failed", error=str(e))
            return False

    # =========================================================================
    # Quality Metrics Operations
    # =========================================================================

    async def save_quality_metric(
        self,
        session_id: str,
        response_id: str,
        quality_score: float,
        relevance_score: Optional[float] = None,
        coherence_score: Optional[float] = None,
        helpfulness_score: Optional[float] = None,
        word_count: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        improvement_id: Optional[str] = None,
        is_treatment: bool = False
    ) -> bool:
        """Save quality metric for a response"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO quality_metrics
                    (session_id, response_id, quality_score, relevance_score,
                     coherence_score, helpfulness_score, word_count, response_time_ms,
                     improvement_id, is_treatment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        response_id,
                        quality_score,
                        relevance_score,
                        coherence_score,
                        helpfulness_score,
                        word_count,
                        response_time_ms,
                        improvement_id,
                        1 if is_treatment else 0
                    )
                )
                await db.commit()
                return True

        except Exception as e:
            logger.error("save_quality_metric_failed", error=str(e))
            return False

    async def get_quality_stats(
        self,
        session_id: Optional[str] = None,
        improvement_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get quality statistics"""
        await self.initialize()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        try:
            async with self.get_connection() as db:
                conditions = ["timestamp > ?"]
                params: List[Any] = [cutoff]

                if session_id:
                    conditions.append("session_id = ?")
                    params.append(session_id)

                if improvement_id:
                    conditions.append("improvement_id = ?")
                    params.append(improvement_id)

                where_clause = " AND ".join(conditions)

                cursor = await db.execute(
                    f"""
                    SELECT
                        COUNT(*) as total_responses,
                        AVG(quality_score) as avg_quality,
                        AVG(relevance_score) as avg_relevance,
                        AVG(coherence_score) as avg_coherence,
                        AVG(helpfulness_score) as avg_helpfulness,
                        AVG(word_count) as avg_word_count,
                        AVG(response_time_ms) as avg_response_time,
                        MIN(quality_score) as min_quality,
                        MAX(quality_score) as max_quality
                    FROM quality_metrics
                    WHERE {where_clause}
                    """,
                    tuple(params)
                )

                row = await cursor.fetchone()

                return {
                    "total_responses": row["total_responses"] or 0,
                    "avg_quality": row["avg_quality"],
                    "avg_relevance": row["avg_relevance"],
                    "avg_coherence": row["avg_coherence"],
                    "avg_helpfulness": row["avg_helpfulness"],
                    "avg_word_count": row["avg_word_count"],
                    "avg_response_time_ms": row["avg_response_time"],
                    "min_quality": row["min_quality"],
                    "max_quality": row["max_quality"],
                    "period_days": days
                }

        except Exception as e:
            logger.error("get_quality_stats_failed", error=str(e))
            return {"error": str(e)}

    # =========================================================================
    # Vocabulary Adjustments
    # =========================================================================

    async def save_vocabulary_adjustment(
        self,
        session_id: str,
        original_term: str,
        preferred_term: str
    ) -> bool:
        """Save or update vocabulary adjustment"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO vocabulary_adjustments
                    (session_id, original_term, preferred_term)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id, original_term) DO UPDATE SET
                        preferred_term = excluded.preferred_term,
                        frequency = frequency + 1,
                        last_used = CURRENT_TIMESTAMP
                    """,
                    (session_id, original_term.lower(), preferred_term)
                )
                await db.commit()
                return True

        except Exception as e:
            logger.error("save_vocabulary_adjustment_failed", error=str(e))
            return False

    async def get_vocabulary_adjustments(
        self,
        session_id: str,
        limit: int = 50
    ) -> Dict[str, str]:
        """Get vocabulary adjustments for a session"""
        await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT original_term, preferred_term
                    FROM vocabulary_adjustments
                    WHERE session_id = ?
                    ORDER BY frequency DESC, last_used DESC
                    LIMIT ?
                    """,
                    (session_id, limit)
                )
                rows = await cursor.fetchall()

                return {row["original_term"]: row["preferred_term"] for row in rows}

        except Exception as e:
            logger.error("get_vocabulary_adjustments_failed", error=str(e))
            return {}

    # =========================================================================
    # Cleanup Operations
    # =========================================================================

    async def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period"""
        await self.initialize()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        deleted = {}

        try:
            async with self.get_connection() as db:
                # Clean feedback history
                cursor = await db.execute(
                    "DELETE FROM feedback_history WHERE timestamp < ?",
                    (cutoff,)
                )
                deleted["feedback"] = cursor.rowcount

                # Clean quality metrics
                cursor = await db.execute(
                    "DELETE FROM quality_metrics WHERE timestamp < ?",
                    (cutoff,)
                )
                deleted["quality_metrics"] = cursor.rowcount

                # Clean completed improvements
                cursor = await db.execute(
                    """
                    DELETE FROM improvements
                    WHERE status IN ('completed', 'rolled_back')
                    AND completed_at < ?
                    """,
                    (cutoff,)
                )
                deleted["improvements"] = cursor.rowcount

                await db.commit()

                logger.info(
                    "cleanup_completed",
                    cutoff_date=cutoff,
                    deleted_counts=deleted
                )

                return deleted

        except Exception as e:
            logger.error("cleanup_failed", error=str(e))
            return {"error": str(e)}


# Global store instance
learning_store = LearningStore()
