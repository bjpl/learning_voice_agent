"""
Data Stores for Learning Analytics
PATTERN: Repository pattern with SQLite persistence
WHY: Centralized data access with efficient queries for analytics
RESILIENCE: Transaction management and error handling
"""
import aiosqlite
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from enum import Enum
import asyncio

from app.logger import get_logger

logger = get_logger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback"""
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    IMPLICIT_POSITIVE = "implicit_positive"
    IMPLICIT_NEGATIVE = "implicit_negative"
    CORRECTION = "correction"
    CLARIFICATION = "clarification"


@dataclass
class SessionData:
    """Session data model"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    exchange_count: int = 0
    correction_count: int = 0
    clarification_count: int = 0
    avg_response_time_ms: float = 0.0
    topics: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration(self) -> float:
        """Session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class FeedbackData:
    """Feedback data model"""
    feedback_id: str
    session_id: str
    exchange_id: int
    feedback_type: FeedbackType
    timestamp: datetime
    content: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QualityScore:
    """Quality score model"""
    score_id: str
    session_id: str
    exchange_id: int
    timestamp: datetime
    composite: float
    relevance: float = 0.0
    helpfulness: float = 0.0
    accuracy: float = 0.0
    clarity: float = 0.0
    completeness: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnalyticsStore:
    """
    Base store for learning analytics data
    PATTERN: Repository with async SQLite
    WHY: Unified data access layer for analytics
    """

    def __init__(self, db_path: str = "learning_analytics.db"):
        self.db_path = db_path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def initialize(self) -> None:
        """Initialize database schema"""
        async with self._init_lock:
            if self._initialized:
                return

            try:
                async with aiosqlite.connect(self.db_path) as db:
                    # Learning sessions table for analytics
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS analytics_sessions (
                            session_id TEXT PRIMARY KEY,
                            start_time DATETIME NOT NULL,
                            end_time DATETIME,
                            exchange_count INTEGER DEFAULT 0,
                            correction_count INTEGER DEFAULT 0,
                            clarification_count INTEGER DEFAULT 0,
                            avg_response_time_ms REAL DEFAULT 0,
                            topics TEXT DEFAULT '[]',
                            metadata TEXT DEFAULT '{}'
                        )
                    """)

                    # Feedback table for analytics
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS analytics_feedback (
                            feedback_id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            exchange_id INTEGER NOT NULL,
                            feedback_type TEXT NOT NULL,
                            timestamp DATETIME NOT NULL,
                            content TEXT,
                            metadata TEXT DEFAULT '{}'
                        )
                    """)

                    # Quality scores table for analytics
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS analytics_quality (
                            score_id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            exchange_id INTEGER NOT NULL,
                            timestamp DATETIME NOT NULL,
                            composite REAL NOT NULL,
                            relevance REAL DEFAULT 0,
                            helpfulness REAL DEFAULT 0,
                            accuracy REAL DEFAULT 0,
                            clarity REAL DEFAULT 0,
                            completeness REAL DEFAULT 0,
                            metadata TEXT DEFAULT '{}'
                        )
                    """)

                    # Detected patterns table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS analytics_patterns (
                            pattern_id TEXT PRIMARY KEY,
                            pattern_type TEXT NOT NULL,
                            first_detected DATETIME NOT NULL,
                            last_updated DATETIME NOT NULL,
                            frequency INTEGER DEFAULT 1,
                            confidence REAL DEFAULT 0,
                            representative_text TEXT,
                            examples TEXT DEFAULT '[]',
                            metadata TEXT DEFAULT '{}',
                            active INTEGER DEFAULT 1
                        )
                    """)

                    # Insights table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS analytics_insights (
                            insight_id TEXT PRIMARY KEY,
                            category TEXT NOT NULL,
                            title TEXT NOT NULL,
                            description TEXT NOT NULL,
                            evidence TEXT DEFAULT '{}',
                            confidence REAL DEFAULT 0,
                            actionable INTEGER DEFAULT 0,
                            recommendation TEXT,
                            created_at DATETIME NOT NULL,
                            valid_until DATETIME,
                            status TEXT DEFAULT 'active'
                        )
                    """)

                    # Aggregated metrics table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS aggregated_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            metric_date DATE NOT NULL,
                            interval_type TEXT NOT NULL,
                            metric_name TEXT NOT NULL,
                            metric_value REAL NOT NULL,
                            metadata TEXT DEFAULT '{}',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(metric_date, interval_type, metric_name)
                        )
                    """)

                    # Indexes
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_analytics_sessions_start
                        ON analytics_sessions(start_time DESC)
                    """)
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_analytics_feedback_session
                        ON analytics_feedback(session_id, timestamp DESC)
                    """)
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_analytics_quality_session
                        ON analytics_quality(session_id, timestamp DESC)
                    """)
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_aggregated_metrics_date
                        ON aggregated_metrics(metric_date DESC, interval_type)
                    """)

                    await db.commit()

                self._initialized = True
                logger.info("analytics_store_initialized", db_path=self.db_path)

            except Exception as e:
                logger.error(
                    "analytics_store_initialization_failed",
                    error=str(e),
                    exc_info=True
                )
                raise


class FeedbackStore(AnalyticsStore):
    """Store for feedback and session data"""

    async def save_feedback(self, feedback: FeedbackData) -> str:
        """Save feedback entry"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO analytics_feedback
                    (feedback_id, session_id, exchange_id, feedback_type,
                     timestamp, content, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    feedback.session_id,
                    feedback.exchange_id,
                    feedback.feedback_type.value if isinstance(feedback.feedback_type, FeedbackType) else feedback.feedback_type,
                    feedback.timestamp.isoformat(),
                    feedback.content,
                    json.dumps(feedback.metadata)
                ))
                await db.commit()
            return feedback.feedback_id
        except Exception as e:
            logger.error("save_feedback_failed", error=str(e), exc_info=True)
            raise

    async def get_feedback_by_session(
        self,
        session_id: str,
        feedback_types: Optional[List[FeedbackType]] = None
    ) -> List[FeedbackData]:
        """Get all feedback for a session"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                if feedback_types:
                    placeholders = ','.join('?' * len(feedback_types))
                    query = f"""
                        SELECT * FROM analytics_feedback
                        WHERE session_id = ? AND feedback_type IN ({placeholders})
                        ORDER BY timestamp ASC
                    """
                    params = [session_id] + [ft.value for ft in feedback_types]
                else:
                    query = """
                        SELECT * FROM analytics_feedback
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                    """
                    params = [session_id]

                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()

                return [self._row_to_feedback(row) for row in rows]
        except Exception as e:
            logger.error("get_feedback_failed", session_id=session_id, error=str(e))
            return []

    async def get_feedback_in_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[FeedbackData]:
        """Get feedback within date range"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM analytics_feedback
                    WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
                    ORDER BY timestamp ASC
                """, (start_date.isoformat(), end_date.isoformat()))
                rows = await cursor.fetchall()
                return [self._row_to_feedback(row) for row in rows]
        except Exception as e:
            logger.error("get_feedback_in_range_failed", error=str(e))
            return []

    async def get_sessions_in_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[SessionData]:
        """Get sessions within date range"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM analytics_sessions
                    WHERE DATE(start_time) >= ? AND DATE(start_time) < ?
                    ORDER BY start_time ASC
                """, (start_date.isoformat(), end_date.isoformat()))
                rows = await cursor.fetchall()
                return [self._row_to_session(row) for row in rows]
        except Exception as e:
            logger.error("get_sessions_in_range_failed", error=str(e))
            return []

    async def save_session(self, session: SessionData) -> str:
        """Save or update session data"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO analytics_sessions
                    (session_id, start_time, end_time, exchange_count,
                     correction_count, clarification_count, avg_response_time_ms,
                     topics, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.exchange_count,
                    session.correction_count,
                    session.clarification_count,
                    session.avg_response_time_ms,
                    json.dumps(session.topics),
                    json.dumps(session.metadata)
                ))
                await db.commit()
            return session.session_id
        except Exception as e:
            logger.error("save_session_failed", error=str(e), exc_info=True)
            raise

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM analytics_sessions WHERE session_id = ?",
                    (session_id,)
                )
                row = await cursor.fetchone()
                return self._row_to_session(row) if row else None
        except Exception as e:
            logger.error("get_session_failed", session_id=session_id, error=str(e))
            return None

    def _row_to_feedback(self, row) -> FeedbackData:
        """Convert row to FeedbackData"""
        return FeedbackData(
            feedback_id=row['feedback_id'],
            session_id=row['session_id'],
            exchange_id=row['exchange_id'],
            feedback_type=FeedbackType(row['feedback_type']),
            timestamp=datetime.fromisoformat(row['timestamp']),
            content=row['content'],
            metadata=json.loads(row['metadata'] or '{}')
        )

    def _row_to_session(self, row) -> SessionData:
        """Convert row to SessionData"""
        return SessionData(
            session_id=row['session_id'],
            start_time=datetime.fromisoformat(row['start_time']),
            end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
            exchange_count=row['exchange_count'],
            correction_count=row['correction_count'],
            clarification_count=row['clarification_count'],
            avg_response_time_ms=row['avg_response_time_ms'],
            topics=json.loads(row['topics'] or '[]'),
            metadata=json.loads(row['metadata'] or '{}')
        )


class QualityStore(AnalyticsStore):
    """Store for quality scores"""

    async def save_score(self, score: QualityScore) -> str:
        """Save quality score"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO analytics_quality
                    (score_id, session_id, exchange_id, timestamp,
                     composite, relevance, helpfulness, accuracy,
                     clarity, completeness, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    score.score_id,
                    score.session_id,
                    score.exchange_id,
                    score.timestamp.isoformat(),
                    score.composite,
                    score.relevance,
                    score.helpfulness,
                    score.accuracy,
                    score.clarity,
                    score.completeness,
                    json.dumps(score.metadata)
                ))
                await db.commit()
            return score.score_id
        except Exception as e:
            logger.error("save_quality_score_failed", error=str(e), exc_info=True)
            raise

    async def get_scores_by_session(self, session_id: str) -> List[QualityScore]:
        """Get all quality scores for a session"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM analytics_quality
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
                rows = await cursor.fetchall()
                return [self._row_to_score(row) for row in rows]
        except Exception as e:
            logger.error("get_scores_failed", session_id=session_id, error=str(e))
            return []

    async def get_scores_in_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[QualityScore]:
        """Get quality scores within date range"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM analytics_quality
                    WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
                    ORDER BY timestamp ASC
                """, (start_date.isoformat(), end_date.isoformat()))
                rows = await cursor.fetchall()
                return [self._row_to_score(row) for row in rows]
        except Exception as e:
            logger.error("get_scores_in_range_failed", error=str(e))
            return []

    async def get_daily_averages(
        self,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get daily average quality scores"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT
                        DATE(timestamp) as date,
                        COUNT(*) as count,
                        AVG(composite) as composite,
                        AVG(relevance) as relevance,
                        AVG(helpfulness) as helpfulness,
                        AVG(accuracy) as accuracy,
                        AVG(clarity) as clarity,
                        AVG(completeness) as completeness
                    FROM analytics_quality
                    WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date ASC
                """, (start_date.isoformat(), end_date.isoformat()))
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("get_daily_averages_failed", error=str(e))
            return []

    async def get_dimension_stats(self) -> Dict[str, Any]:
        """Get statistics for each quality dimension"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT
                        AVG(composite) as avg_composite,
                        AVG(relevance) as avg_relevance,
                        AVG(helpfulness) as avg_helpfulness,
                        AVG(accuracy) as avg_accuracy,
                        AVG(clarity) as avg_clarity,
                        AVG(completeness) as avg_completeness,
                        MIN(composite) as min_composite,
                        MAX(composite) as max_composite,
                        COUNT(*) as count
                    FROM analytics_quality
                """)
                row = await cursor.fetchone()

                if not row:
                    return {}

                return {
                    "composite": {
                        "avg": row['avg_composite'] or 0,
                        "min": row['min_composite'] or 0,
                        "max": row['max_composite'] or 0
                    },
                    "relevance": {"avg": row['avg_relevance'] or 0},
                    "helpfulness": {"avg": row['avg_helpfulness'] or 0},
                    "accuracy": {"avg": row['avg_accuracy'] or 0},
                    "clarity": {"avg": row['avg_clarity'] or 0},
                    "completeness": {"avg": row['avg_completeness'] or 0},
                    "count": row['count'] or 0
                }
        except Exception as e:
            logger.error("get_dimension_stats_failed", error=str(e))
            return {}

    def _row_to_score(self, row) -> QualityScore:
        """Convert database row to QualityScore"""
        return QualityScore(
            score_id=row['score_id'],
            session_id=row['session_id'],
            exchange_id=row['exchange_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            composite=row['composite'],
            relevance=row['relevance'],
            helpfulness=row['helpfulness'],
            accuracy=row['accuracy'],
            clarity=row['clarity'],
            completeness=row['completeness'],
            metadata=json.loads(row['metadata'] or '{}')
        )


class PatternStore(AnalyticsStore):
    """Store for detected patterns"""

    async def save_pattern(self, pattern: Dict[str, Any]) -> str:
        """Save or update detected pattern"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO analytics_patterns
                    (pattern_id, pattern_type, first_detected, last_updated,
                     frequency, confidence, representative_text, examples,
                     metadata, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern['pattern_id'],
                    pattern['pattern_type'],
                    pattern.get('first_detected', datetime.utcnow().isoformat()),
                    datetime.utcnow().isoformat(),
                    pattern.get('frequency', 1),
                    pattern.get('confidence', 0),
                    pattern.get('representative_text'),
                    json.dumps(pattern.get('examples', [])),
                    json.dumps(pattern.get('metadata', {})),
                    1 if pattern.get('active', True) else 0
                ))
                await db.commit()
            return pattern['pattern_id']
        except Exception as e:
            logger.error("save_pattern_failed", error=str(e), exc_info=True)
            raise

    async def get_active_patterns(
        self,
        pattern_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get active patterns"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                if pattern_type:
                    cursor = await db.execute("""
                        SELECT * FROM analytics_patterns
                        WHERE active = 1 AND pattern_type = ?
                        ORDER BY frequency DESC
                    """, (pattern_type,))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM analytics_patterns
                        WHERE active = 1
                        ORDER BY frequency DESC
                    """)
                rows = await cursor.fetchall()

                return [
                    {
                        'pattern_id': row['pattern_id'],
                        'pattern_type': row['pattern_type'],
                        'first_detected': row['first_detected'],
                        'last_updated': row['last_updated'],
                        'frequency': row['frequency'],
                        'confidence': row['confidence'],
                        'representative_text': row['representative_text'],
                        'examples': json.loads(row['examples'] or '[]'),
                        'metadata': json.loads(row['metadata'] or '{}'),
                        'active': row['active']
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("get_active_patterns_failed", error=str(e))
            return []


class InsightStore(AnalyticsStore):
    """Store for generated insights"""

    async def save_insight(self, insight: Dict[str, Any]) -> str:
        """Save insight"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO analytics_insights
                    (insight_id, category, title, description, evidence,
                     confidence, actionable, recommendation, created_at,
                     valid_until, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight['insight_id'],
                    insight['category'],
                    insight['title'],
                    insight['description'],
                    json.dumps(insight.get('evidence', {})),
                    insight.get('confidence', 0),
                    1 if insight.get('actionable', False) else 0,
                    insight.get('recommendation'),
                    insight.get('created_at', datetime.utcnow().isoformat()),
                    insight.get('valid_until'),
                    insight.get('status', 'active')
                ))
                await db.commit()
            return insight['insight_id']
        except Exception as e:
            logger.error("save_insight_failed", error=str(e), exc_info=True)
            raise

    async def get_active_insights(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get active insights"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                now = datetime.utcnow().isoformat()
                if category:
                    cursor = await db.execute("""
                        SELECT * FROM analytics_insights
                        WHERE status = 'active'
                        AND category = ?
                        AND confidence >= ?
                        AND (valid_until IS NULL OR valid_until > ?)
                        ORDER BY confidence DESC, created_at DESC
                    """, (category, min_confidence, now))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM analytics_insights
                        WHERE status = 'active'
                        AND confidence >= ?
                        AND (valid_until IS NULL OR valid_until > ?)
                        ORDER BY confidence DESC, created_at DESC
                    """, (min_confidence, now))

                rows = await cursor.fetchall()

                return [
                    {
                        'insight_id': row['insight_id'],
                        'category': row['category'],
                        'title': row['title'],
                        'description': row['description'],
                        'evidence': json.loads(row['evidence'] or '{}'),
                        'confidence': row['confidence'],
                        'actionable': bool(row['actionable']),
                        'recommendation': row['recommendation'],
                        'created_at': row['created_at'],
                        'valid_until': row['valid_until'],
                        'status': row['status']
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("get_active_insights_failed", error=str(e))
            return []

    async def expire_old_insights(self) -> int:
        """Mark expired insights as inactive"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    UPDATE analytics_insights
                    SET status = 'expired'
                    WHERE status = 'active'
                    AND valid_until IS NOT NULL
                    AND valid_until <= ?
                """, (datetime.utcnow().isoformat(),))
                await db.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error("expire_insights_failed", error=str(e))
            return 0


# Global store instances
feedback_store = FeedbackStore()
quality_store = QualityStore()
pattern_store = PatternStore()
insight_store = InsightStore()
