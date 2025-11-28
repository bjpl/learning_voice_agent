"""
Data Export Service - Full Data Export and Backup Creation
==========================================================

PATTERN: Comprehensive data export with compression and integrity validation
WHY: Enable users to export all their learning data for portability and backup

Features:
- Export all user data (conversations, feedback, goals, achievements, settings)
- Generate backup metadata (timestamp, version, checksum)
- Compress to gzip for smaller file size
- Support partial exports with date range filtering
- Checksum validation for data integrity
"""

import gzip
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field

from app.database import Database, db
from app.learning.feedback_store import FeedbackStore, feedback_store
from app.learning.feedback_models import ExplicitFeedback, ImplicitFeedback, CorrectionFeedback
from app.analytics.goal_store import GoalStore, goal_store
from app.analytics.goal_models import Goal, GoalStatus, Achievement
from app.analytics.achievement_system import AchievementSystem, achievement_system
from app.sync.serializers import (
    serialize_conversation,
    serialize_feedback,
    serialize_goal,
    serialize_achievement,
    serialize_conversations_batch,
    serialize_feedback_batch,
    serialize_goals_batch,
    serialize_achievements_batch,
    ConversationData,
    FeedbackData,
    SettingsData,
)
from app.logger import get_logger

logger = get_logger("sync.export_service")

# Export format version for compatibility tracking
EXPORT_VERSION = "1.0.0"
EXPORT_FORMAT = "learning_voice_agent_backup"


class ExportScope(str, Enum):
    """Scope of data export."""
    FULL = "full"
    CONVERSATIONS = "conversations"
    FEEDBACK = "feedback"
    GOALS = "goals"
    ACHIEVEMENTS = "achievements"
    SETTINGS = "settings"


@dataclass
class DateRange:
    """
    Date range for filtering exports.

    Attributes:
        start: Start date (inclusive)
        end: End date (inclusive)
    """
    start: Optional[date] = None
    end: Optional[date] = None

    @classmethod
    def last_n_days(cls, days: int) -> "DateRange":
        """Create a date range for the last N days."""
        end = date.today()
        start = end - timedelta(days=days)
        return cls(start=start, end=end)

    @classmethod
    def last_month(cls) -> "DateRange":
        """Create a date range for the last 30 days."""
        return cls.last_n_days(30)

    @classmethod
    def last_week(cls) -> "DateRange":
        """Create a date range for the last 7 days."""
        return cls.last_n_days(7)

    @classmethod
    def all_time(cls) -> "DateRange":
        """Create an unlimited date range."""
        return cls(start=None, end=None)

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary."""
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None
        }


@dataclass
class BackupMetadata:
    """
    Metadata for a backup file.

    Tracks version, timestamps, and integrity information.
    """
    format: str = EXPORT_FORMAT
    version: str = EXPORT_VERSION
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    export_scope: str = "full"
    date_range: Optional[Dict[str, Optional[str]]] = None
    record_counts: Dict[str, int] = field(default_factory=dict)
    checksum: Optional[str] = None
    compressed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BackupData:
    """
    Complete backup data structure.

    Contains all exportable data along with metadata.
    """
    metadata: BackupMetadata
    conversations: List[Dict[str, Any]] = field(default_factory=list)
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    goals: List[Dict[str, Any]] = field(default_factory=list)
    achievements: List[Dict[str, Any]] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "conversations": self.conversations,
            "feedback": self.feedback,
            "goals": self.goals,
            "achievements": self.achievements,
            "settings": self.settings
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class DataExportService:
    """
    Comprehensive data export service.

    PATTERN: Repository aggregation with serialization
    WHY: Centralized export logic with consistent formatting

    USAGE:
        service = DataExportService()
        await service.initialize()

        # Full export
        backup = await service.export_all_data()

        # Partial export (last 30 days)
        backup = await service.export_all_data(
            date_range=DateRange.last_month()
        )

        # Create compressed backup file
        compressed_bytes = await service.create_backup_file()

        # Verify integrity
        is_valid = service.verify_checksum(data, checksum)
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        feedback_store_instance: Optional[FeedbackStore] = None,
        goal_store_instance: Optional[GoalStore] = None,
        achievement_system_instance: Optional[AchievementSystem] = None
    ):
        """
        Initialize the export service.

        Args:
            database: Database instance for conversations
            feedback_store_instance: Feedback store instance
            goal_store_instance: Goal store instance
            achievement_system_instance: Achievement system instance
        """
        self.database = database or db
        self.feedback_store = feedback_store_instance or feedback_store
        self.goal_store = goal_store_instance or goal_store
        self.achievement_system = achievement_system_instance or achievement_system
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all data stores."""
        if self._initialized:
            return

        try:
            await self.database.initialize()
            await self.feedback_store.initialize()
            await self.goal_store.initialize()
            await self.achievement_system.initialize()

            self._initialized = True
            logger.info("data_export_service_initialized")

        except Exception as e:
            logger.error(
                "data_export_service_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    # ========================================================================
    # MAIN EXPORT METHODS
    # ========================================================================

    async def export_all_data(
        self,
        date_range: Optional[DateRange] = None,
        include_settings: bool = True
    ) -> BackupData:
        """
        Export all user data.

        Args:
            date_range: Optional date range filter
            include_settings: Whether to include settings

        Returns:
            BackupData with all exported data
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(
                "export_all_data_started",
                date_range=date_range.to_dict() if date_range else None
            )

            # Export each data type
            conversations = await self.export_conversations(date_range)
            feedback = await self.export_feedback(date_range)
            goals = await self.export_goals()
            achievements = await self.export_achievements()
            settings = await self.export_settings() if include_settings else {}

            # Build metadata
            metadata = BackupMetadata(
                export_scope="full",
                date_range=date_range.to_dict() if date_range else None,
                record_counts={
                    "conversations": len(conversations),
                    "feedback": len(feedback),
                    "goals": len(goals),
                    "achievements": len(achievements)
                }
            )

            backup = BackupData(
                metadata=metadata,
                conversations=conversations,
                feedback=feedback,
                goals=goals,
                achievements=achievements,
                settings=settings
            )

            # Calculate checksum
            backup.metadata.checksum = self.calculate_checksum(backup.to_json())

            logger.info(
                "export_all_data_complete",
                total_records=sum(metadata.record_counts.values()),
                checksum=backup.metadata.checksum[:16] + "..."
            )

            return backup

        except Exception as e:
            logger.error("export_all_data_failed", error=str(e), exc_info=True)
            raise

    async def export_conversations(
        self,
        date_range: Optional[DateRange] = None
    ) -> List[Dict[str, Any]]:
        """
        Export conversation exchanges.

        Args:
            date_range: Optional date range filter

        Returns:
            List of serialized conversation dicts
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.database.get_connection() as conn:
                # Build query with optional date filter
                if date_range and (date_range.start or date_range.end):
                    where_clauses = []
                    params = []

                    if date_range.start:
                        where_clauses.append("DATE(timestamp) >= ?")
                        params.append(date_range.start.isoformat())

                    if date_range.end:
                        where_clauses.append("DATE(timestamp) <= ?")
                        params.append(date_range.end.isoformat())

                    where_sql = " AND ".join(where_clauses)
                    query = f"""
                        SELECT id, session_id, timestamp, user_text, agent_text, metadata
                        FROM captures
                        WHERE {where_sql}
                        ORDER BY timestamp ASC
                    """
                    cursor = await conn.execute(query, params)
                else:
                    cursor = await conn.execute("""
                        SELECT id, session_id, timestamp, user_text, agent_text, metadata
                        FROM captures
                        ORDER BY timestamp ASC
                    """)

                rows = await cursor.fetchall()

            # Serialize all conversations
            conversations = serialize_conversations_batch(
                [dict(row) for row in rows],
                include_metadata=True
            )

            logger.info("export_conversations_complete", count=len(conversations))
            return conversations

        except Exception as e:
            logger.error("export_conversations_failed", error=str(e))
            return []

    async def export_feedback(
        self,
        date_range: Optional[DateRange] = None
    ) -> List[Dict[str, Any]]:
        """
        Export all feedback (explicit, implicit, corrections).

        Args:
            date_range: Optional date range filter

        Returns:
            List of serialized feedback dicts
        """
        if not self._initialized:
            await self.initialize()

        try:
            all_feedback = []

            # Get time range
            start_time = None
            end_time = None
            if date_range:
                if date_range.start:
                    start_time = datetime.combine(date_range.start, datetime.min.time())
                if date_range.end:
                    end_time = datetime.combine(date_range.end, datetime.max.time())

            # Export explicit feedback
            async with self.feedback_store.get_connection() as conn:
                if start_time and end_time:
                    cursor = await conn.execute(
                        """
                        SELECT * FROM explicit_feedback
                        WHERE timestamp BETWEEN ? AND ?
                        ORDER BY timestamp ASC
                        """,
                        (start_time.isoformat(), end_time.isoformat())
                    )
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM explicit_feedback ORDER BY timestamp ASC"
                    )
                explicit_rows = await cursor.fetchall()

                for row in explicit_rows:
                    fb = ExplicitFeedback(
                        id=row['id'],
                        session_id=row['session_id'],
                        exchange_id=row['exchange_id'],
                        rating=row['rating'],
                        helpful=bool(row['helpful']),
                        comment=row['comment'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    all_feedback.append(serialize_feedback(fb).to_dict())

                # Export implicit feedback
                if start_time and end_time:
                    cursor = await conn.execute(
                        """
                        SELECT * FROM implicit_feedback
                        WHERE timestamp BETWEEN ? AND ?
                        ORDER BY timestamp ASC
                        """,
                        (start_time.isoformat(), end_time.isoformat())
                    )
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM implicit_feedback ORDER BY timestamp ASC"
                    )
                implicit_rows = await cursor.fetchall()

                for row in implicit_rows:
                    fb = ImplicitFeedback(
                        id=row['id'],
                        session_id=row['session_id'],
                        response_time_ms=row['response_time_ms'],
                        follow_up_count=row['follow_up_count'] or 0,
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    all_feedback.append(serialize_feedback(fb).to_dict())

                # Export corrections
                if start_time and end_time:
                    cursor = await conn.execute(
                        """
                        SELECT * FROM corrections
                        WHERE timestamp BETWEEN ? AND ?
                        ORDER BY timestamp ASC
                        """,
                        (start_time.isoformat(), end_time.isoformat())
                    )
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM corrections ORDER BY timestamp ASC"
                    )
                correction_rows = await cursor.fetchall()

                from app.learning.feedback_models import CorrectionType
                for row in correction_rows:
                    fb = CorrectionFeedback(
                        id=row['id'],
                        session_id=row['session_id'],
                        original_text=row['original_text'],
                        corrected_text=row['corrected_text'],
                        correction_type=CorrectionType(row['correction_type']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    all_feedback.append(serialize_feedback(fb).to_dict())

            logger.info(
                "export_feedback_complete",
                count=len(all_feedback),
                explicit=len(explicit_rows),
                implicit=len(implicit_rows),
                corrections=len(correction_rows)
            )
            return all_feedback

        except Exception as e:
            logger.error("export_feedback_failed", error=str(e), exc_info=True)
            return []

    async def export_goals(self) -> List[Dict[str, Any]]:
        """
        Export all goals with their milestones.

        Returns:
            List of serialized goal dicts
        """
        if not self._initialized:
            await self.initialize()

        try:
            goals = await self.goal_store.get_all_goals(limit=1000)
            serialized = serialize_goals_batch(goals)

            logger.info("export_goals_complete", count=len(serialized))
            return serialized

        except Exception as e:
            logger.error("export_goals_failed", error=str(e))
            return []

    async def export_achievements(self) -> List[Dict[str, Any]]:
        """
        Export all achievements with unlock status.

        Returns:
            List of serialized achievement dicts
        """
        if not self._initialized:
            await self.initialize()

        try:
            achievements = await self.goal_store.get_all_achievements()
            serialized = serialize_achievements_batch(achievements)

            logger.info("export_achievements_complete", count=len(serialized))
            return serialized

        except Exception as e:
            logger.error("export_achievements_failed", error=str(e))
            return []

    async def export_settings(self) -> Dict[str, Any]:
        """
        Export user settings and preferences.

        Returns:
            Dictionary of settings
        """
        try:
            # Collect settings from various sources
            settings = SettingsData(
                version=EXPORT_VERSION,
                preferences={
                    # User preferences would be loaded from a settings store
                    "default_voice": "alloy",
                    "language": "en",
                    "theme": "light"
                },
                learning_config={
                    "session_duration_target_minutes": 15,
                    "daily_goal_sessions": 1,
                    "streak_reminder_enabled": True,
                    "quality_threshold": 0.75
                },
                notification_settings={
                    "streak_reminders": True,
                    "achievement_alerts": True,
                    "weekly_summary": True
                },
                privacy_settings={
                    "analytics_enabled": True,
                    "export_includes_metadata": True
                }
            )

            logger.info("export_settings_complete")
            return settings.dict()

        except Exception as e:
            logger.error("export_settings_failed", error=str(e))
            return {}

    # ========================================================================
    # BACKUP FILE CREATION
    # ========================================================================

    async def create_backup_file(
        self,
        date_range: Optional[DateRange] = None,
        compress: bool = True
    ) -> bytes:
        """
        Create a compressed backup file containing all data.

        Args:
            date_range: Optional date range filter
            compress: Whether to gzip compress the output

        Returns:
            Bytes of the backup file (gzip compressed JSON)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Export all data
            backup = await self.export_all_data(date_range)
            backup.metadata.compressed = compress

            # Convert to JSON
            json_data = backup.to_json(indent=2)
            json_bytes = json_data.encode('utf-8')

            if compress:
                # Compress with gzip
                compressed = gzip.compress(json_bytes, compresslevel=9)

                logger.info(
                    "backup_file_created",
                    original_size=len(json_bytes),
                    compressed_size=len(compressed),
                    compression_ratio=f"{(1 - len(compressed)/len(json_bytes))*100:.1f}%"
                )

                return compressed
            else:
                logger.info(
                    "backup_file_created",
                    size=len(json_bytes),
                    compressed=False
                )
                return json_bytes

        except Exception as e:
            logger.error("create_backup_file_failed", error=str(e), exc_info=True)
            raise

    # ========================================================================
    # CHECKSUM AND VALIDATION
    # ========================================================================

    def calculate_checksum(self, data: Union[str, bytes, dict]) -> str:
        """
        Calculate SHA-256 checksum of data.

        Args:
            data: Data to checksum (string, bytes, or dict)

        Returns:
            Hex-encoded SHA-256 hash
        """
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')

        return hashlib.sha256(data).hexdigest()

    def verify_checksum(self, data: Union[str, bytes, dict], expected_checksum: str) -> bool:
        """
        Verify data integrity using checksum.

        Args:
            data: Data to verify (string, bytes, or dict)
            expected_checksum: Expected SHA-256 hash

        Returns:
            True if checksum matches
        """
        calculated = self.calculate_checksum(data)
        return calculated == expected_checksum

    # ========================================================================
    # PARTIAL EXPORTS
    # ========================================================================

    async def export_partial(
        self,
        scope: ExportScope,
        date_range: Optional[DateRange] = None
    ) -> BackupData:
        """
        Export a partial backup with only specified data.

        Args:
            scope: What data to include
            date_range: Optional date range filter

        Returns:
            BackupData with requested data only
        """
        if not self._initialized:
            await self.initialize()

        try:
            conversations = []
            feedback = []
            goals = []
            achievements = []
            settings = {}

            if scope == ExportScope.FULL:
                return await self.export_all_data(date_range)

            elif scope == ExportScope.CONVERSATIONS:
                conversations = await self.export_conversations(date_range)

            elif scope == ExportScope.FEEDBACK:
                feedback = await self.export_feedback(date_range)

            elif scope == ExportScope.GOALS:
                goals = await self.export_goals()

            elif scope == ExportScope.ACHIEVEMENTS:
                achievements = await self.export_achievements()

            elif scope == ExportScope.SETTINGS:
                settings = await self.export_settings()

            metadata = BackupMetadata(
                export_scope=scope.value,
                date_range=date_range.to_dict() if date_range else None,
                record_counts={
                    "conversations": len(conversations),
                    "feedback": len(feedback),
                    "goals": len(goals),
                    "achievements": len(achievements)
                }
            )

            backup = BackupData(
                metadata=metadata,
                conversations=conversations,
                feedback=feedback,
                goals=goals,
                achievements=achievements,
                settings=settings
            )

            backup.metadata.checksum = self.calculate_checksum(backup.to_json())

            logger.info(
                "export_partial_complete",
                scope=scope.value,
                record_count=sum(metadata.record_counts.values())
            )

            return backup

        except Exception as e:
            logger.error(
                "export_partial_failed",
                scope=scope.value,
                error=str(e)
            )
            raise

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    async def get_export_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about exportable data.

        Returns:
            Dictionary with record counts and estimated sizes
        """
        if not self._initialized:
            await self.initialize()

        try:
            stats = {
                "conversations": 0,
                "feedback": {
                    "explicit": 0,
                    "implicit": 0,
                    "corrections": 0,
                    "total": 0
                },
                "goals": {
                    "active": 0,
                    "completed": 0,
                    "total": 0
                },
                "achievements": {
                    "unlocked": 0,
                    "total": 0
                }
            }

            # Count conversations
            async with self.database.get_connection() as conn:
                cursor = await conn.execute("SELECT COUNT(*) as count FROM captures")
                row = await cursor.fetchone()
                stats["conversations"] = row["count"] if row else 0

            # Count feedback
            async with self.feedback_store.get_connection() as conn:
                cursor = await conn.execute("SELECT COUNT(*) as count FROM explicit_feedback")
                row = await cursor.fetchone()
                stats["feedback"]["explicit"] = row["count"] if row else 0

                cursor = await conn.execute("SELECT COUNT(*) as count FROM implicit_feedback")
                row = await cursor.fetchone()
                stats["feedback"]["implicit"] = row["count"] if row else 0

                cursor = await conn.execute("SELECT COUNT(*) as count FROM corrections")
                row = await cursor.fetchone()
                stats["feedback"]["corrections"] = row["count"] if row else 0

            stats["feedback"]["total"] = (
                stats["feedback"]["explicit"] +
                stats["feedback"]["implicit"] +
                stats["feedback"]["corrections"]
            )

            # Count goals
            goals = await self.goal_store.get_all_goals()
            stats["goals"]["total"] = len(goals)
            stats["goals"]["active"] = len([g for g in goals if g.status == GoalStatus.ACTIVE])
            stats["goals"]["completed"] = len([g for g in goals if g.status == GoalStatus.COMPLETED])

            # Count achievements
            achievement_stats = await self.goal_store.get_achievement_stats()
            stats["achievements"]["total"] = achievement_stats.get("total", 0)
            stats["achievements"]["unlocked"] = achievement_stats.get("unlocked", 0)

            logger.info("export_statistics_retrieved", **stats)
            return stats

        except Exception as e:
            logger.error("get_export_statistics_failed", error=str(e))
            return {}


# Singleton instance
data_export_service = DataExportService()
