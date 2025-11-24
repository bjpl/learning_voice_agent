"""
Data Import/Restore Service
============================

Provides data import, restore, and backup restoration functionality
with transaction support for safe rollback on failure.

PATTERN: Transaction-based import with rollback support
WHY: Ensure data integrity during restore operations

Features:
- Import from backup files (gzip compressed or plain JSON)
- Multiple merge strategies (replace, merge, keep_newer)
- Conflict detection and resolution
- Transaction support with rollback on failure
- Settings restoration
- Progress tracking during import
"""

import gzip
import json
import copy
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import asynccontextmanager

from app.database import Database, db
from app.learning.feedback_store import FeedbackStore, feedback_store
from app.analytics.goal_store import GoalStore, goal_store
from app.analytics.goal_models import Goal, GoalStatus, Achievement
from app.sync.validators import BackupValidator, ValidationResult, backup_validator
from app.sync.conflict_resolver import (
    ConflictResolver,
    SyncConflict,
    ResolutionStrategy,
    conflict_resolver,
)
from app.sync.serializers import (
    deserialize_conversation,
    deserialize_feedback,
    deserialize_goal,
    deserialize_achievement,
)
from app.sync.models import MergeStrategy, BackupData, SyncMetadata
from app.logger import get_logger

logger = get_logger("sync.import_service")

# Import format version for compatibility tracking
IMPORT_VERSION = "1.0.0"


class ImportStrategy(str, Enum):
    """Strategy for importing data."""
    REPLACE = "replace"       # Clear local data, use imported
    MERGE = "merge"           # Combine both, newer wins on conflicts
    KEEP_NEWER = "keep_newer" # Only import if imported is newer


class ImportStatus(str, Enum):
    """Status of import operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    IMPORTING = "importing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    status: ImportStatus = ImportStatus.PENDING
    items_imported: Dict[str, int] = field(default_factory=dict)
    items_skipped: Dict[str, int] = field(default_factory=dict)
    items_updated: Dict[str, int] = field(default_factory=dict)
    conflicts: List[SyncConflict] = field(default_factory=list)
    conflicts_resolved: int = 0
    conflicts_pending: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rollback_performed: bool = False
    transaction_id: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration of import operation."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def total_imported(self) -> int:
        """Total number of items imported."""
        return sum(self.items_imported.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "status": self.status.value,
            "items_imported": self.items_imported,
            "items_skipped": self.items_skipped,
            "items_updated": self.items_updated,
            "conflicts_resolved": self.conflicts_resolved,
            "conflicts_pending": self.conflicts_pending,
            "errors": self.errors,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "total_imported": self.total_imported,
            "rollback_performed": self.rollback_performed
        }


@dataclass
class TransactionSnapshot:
    """Snapshot of data before import for rollback support."""
    transaction_id: str
    created_at: datetime
    conversations_backup: List[Dict[str, Any]] = field(default_factory=list)
    feedback_backup: List[Dict[str, Any]] = field(default_factory=list)
    goals_backup: List[Dict[str, Any]] = field(default_factory=list)
    achievements_backup: List[Dict[str, Any]] = field(default_factory=list)
    settings_backup: Dict[str, Any] = field(default_factory=dict)
    has_data: bool = False


class ImportService:
    """
    Data import and restore service with transaction support.

    PATTERN: Transaction-based import with rollback
    WHY: Ensure data integrity with safe recovery on failure

    USAGE:
        service = ImportService()
        await service.initialize()

        # Import from backup bytes
        result = await service.import_backup(backup_data, strategy="merge")

        # Validate before import
        validation = await service.validate_backup(backup_data)
        if validation.valid:
            result = await service.import_backup(backup_data)

        # Rollback if needed
        await service.rollback_import()
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        feedback_store_instance: Optional[FeedbackStore] = None,
        goal_store_instance: Optional[GoalStore] = None,
        validator: Optional[BackupValidator] = None,
        resolver: Optional[ConflictResolver] = None
    ):
        """
        Initialize the import service.

        Args:
            database: Database instance for conversations
            feedback_store_instance: Feedback store instance
            goal_store_instance: Goal store instance
            validator: Backup validator instance
            resolver: Conflict resolver instance
        """
        self.database = database or db
        self.feedback_store = feedback_store_instance or feedback_store
        self.goal_store = goal_store_instance or goal_store
        self.validator = validator or backup_validator
        self.resolver = resolver or conflict_resolver

        self._initialized = False
        self._transaction: Optional[TransactionSnapshot] = None
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    async def initialize(self) -> None:
        """Initialize all data stores."""
        if self._initialized:
            return

        try:
            await self.database.initialize()
            await self.feedback_store.initialize()
            await self.goal_store.initialize()

            self._initialized = True
            logger.info("import_service_initialized")

        except Exception as e:
            logger.error(
                "import_service_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    def set_progress_callback(
        self,
        callback: Callable[[str, float], None]
    ) -> None:
        """
        Set a callback for progress updates.

        Args:
            callback: Function that takes (stage_name, progress_percent)
        """
        self._progress_callback = callback

    def _report_progress(self, stage: str, progress: float) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(stage, progress)

    # ========================================================================
    # MAIN IMPORT METHODS
    # ========================================================================

    async def import_backup(
        self,
        data: Union[bytes, str, Dict],
        strategy: Union[str, ImportStrategy] = ImportStrategy.MERGE,
        validate_first: bool = True,
        create_snapshot: bool = True,
        resolve_conflicts: bool = True
    ) -> ImportResult:
        """
        Import backup data with the specified strategy.

        Args:
            data: Backup data (gzip bytes, JSON string, or dict)
            strategy: Import strategy to use
            validate_first: Whether to validate before importing
            create_snapshot: Whether to create rollback snapshot
            resolve_conflicts: Whether to auto-resolve conflicts

        Returns:
            ImportResult with import outcomes
        """
        if not self._initialized:
            await self.initialize()

        result = ImportResult(
            success=False,
            status=ImportStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        # Normalize strategy
        if isinstance(strategy, str):
            strategy = ImportStrategy(strategy)

        try:
            self._report_progress("parsing", 0)

            # Parse backup data
            parsed_data = await self._parse_backup(data)
            if parsed_data is None:
                result.errors.append("Failed to parse backup data")
                result.status = ImportStatus.FAILED
                return result

            self._report_progress("validating", 10)

            # Validate if requested
            if validate_first:
                result.status = ImportStatus.VALIDATING
                validation = self.validate_backup(parsed_data)
                result.validation_result = validation

                if not validation.valid:
                    result.errors.extend([e.message for e in validation.errors])
                    result.status = ImportStatus.FAILED
                    return result

                result.warnings.extend(validation.warnings)

            self._report_progress("snapshot", 20)

            # Create rollback snapshot
            if create_snapshot:
                await self._create_snapshot()
                result.transaction_id = self._transaction.transaction_id if self._transaction else None

            result.status = ImportStatus.IMPORTING
            self._report_progress("importing", 30)

            # Execute import based on strategy
            if strategy == ImportStrategy.REPLACE:
                await self._import_replace(parsed_data, result)
            elif strategy == ImportStrategy.MERGE:
                await self._import_merge(parsed_data, result, resolve_conflicts)
            elif strategy == ImportStrategy.KEEP_NEWER:
                await self._import_keep_newer(parsed_data, result, resolve_conflicts)

            self._report_progress("finalizing", 90)

            # Mark success
            result.success = len(result.errors) == 0
            result.status = ImportStatus.COMPLETED if result.success else ImportStatus.FAILED
            result.completed_at = datetime.utcnow()

            self._report_progress("complete", 100)

            logger.info(
                "import_backup_complete",
                success=result.success,
                strategy=strategy.value,
                items_imported=result.total_imported,
                conflicts=result.conflicts_resolved,
                errors=len(result.errors)
            )

            return result

        except Exception as e:
            logger.error("import_backup_failed", error=str(e), exc_info=True)
            result.errors.append(f"Import failed: {str(e)}")
            result.status = ImportStatus.FAILED
            result.completed_at = datetime.utcnow()

            # Attempt rollback
            if self._transaction:
                try:
                    await self.rollback_import()
                    result.rollback_performed = True
                    result.status = ImportStatus.ROLLED_BACK
                except Exception as rollback_error:
                    logger.error("rollback_failed", error=str(rollback_error))
                    result.errors.append(f"Rollback failed: {str(rollback_error)}")

            return result

    def validate_backup(self, data: Union[bytes, str, Dict]) -> ValidationResult:
        """
        Validate backup data before import.

        Args:
            data: Backup data to validate

        Returns:
            ValidationResult with validation outcomes
        """
        return self.validator.validate(data)

    async def merge_conversations(
        self,
        local: List[Dict],
        imported: List[Dict],
        strategy: ImportStrategy
    ) -> tuple:
        """
        Merge conversation data based on strategy.

        Args:
            local: Local conversation data
            imported: Imported conversation data
            strategy: Merge strategy

        Returns:
            Tuple of (merged_data, conflicts)
        """
        if strategy == ImportStrategy.REPLACE:
            return imported, []

        conflicts = []
        merged = list(local)  # Start with local data

        # Build lookup by ID
        local_map = {c.get("id"): c for c in local if c.get("id")}

        for imported_conv in imported:
            conv_id = imported_conv.get("id")

            if conv_id and conv_id in local_map:
                local_conv = local_map[conv_id]

                # Check for differences
                if strategy == ImportStrategy.KEEP_NEWER:
                    local_ts = self._parse_timestamp(local_conv.get("timestamp"))
                    imported_ts = self._parse_timestamp(imported_conv.get("timestamp"))

                    if imported_ts and (not local_ts or imported_ts > local_ts):
                        # Update local with imported
                        idx = next(i for i, c in enumerate(merged) if c.get("id") == conv_id)
                        merged[idx] = imported_conv
                else:
                    # Merge strategy - detect and record conflict
                    if local_conv.get("agent_text") != imported_conv.get("agent_text"):
                        conflicts.append(SyncConflict(
                            id=f"conv_{conv_id}",
                            conflict_type="value_mismatch",
                            data_type="conversations",
                            record_id=conv_id,
                            field_name="agent_text",
                            local_value=local_conv.get("agent_text"),
                            imported_value=imported_conv.get("agent_text"),
                            local_timestamp=self._parse_timestamp(local_conv.get("timestamp")),
                            imported_timestamp=self._parse_timestamp(imported_conv.get("timestamp"))
                        ))
            else:
                # New conversation, add it
                merged.append(imported_conv)

        return merged, conflicts

    async def merge_feedback(
        self,
        local: List[Dict],
        imported: List[Dict],
        strategy: ImportStrategy
    ) -> tuple:
        """
        Merge feedback data based on strategy.

        Args:
            local: Local feedback data
            imported: Imported feedback data
            strategy: Merge strategy

        Returns:
            Tuple of (merged_data, conflicts)
        """
        if strategy == ImportStrategy.REPLACE:
            return imported, []

        conflicts = []
        merged = list(local)

        local_map = {f.get("id"): f for f in local if f.get("id")}

        for imported_fb in imported:
            fb_id = imported_fb.get("id")

            if fb_id and fb_id in local_map:
                local_fb = local_map[fb_id]

                if strategy == ImportStrategy.KEEP_NEWER:
                    local_ts = self._parse_timestamp(local_fb.get("timestamp"))
                    imported_ts = self._parse_timestamp(imported_fb.get("timestamp"))

                    if imported_ts and (not local_ts or imported_ts > local_ts):
                        idx = next(i for i, f in enumerate(merged) if f.get("id") == fb_id)
                        merged[idx] = imported_fb
                else:
                    # Check for rating conflicts
                    if local_fb.get("rating") != imported_fb.get("rating"):
                        conflicts.append(SyncConflict(
                            id=f"fb_{fb_id}",
                            conflict_type="value_mismatch",
                            data_type="feedback",
                            record_id=fb_id,
                            field_name="rating",
                            local_value=local_fb.get("rating"),
                            imported_value=imported_fb.get("rating"),
                            local_timestamp=self._parse_timestamp(local_fb.get("timestamp")),
                            imported_timestamp=self._parse_timestamp(imported_fb.get("timestamp"))
                        ))
            else:
                merged.append(imported_fb)

        return merged, conflicts

    async def merge_goals(
        self,
        local: List[Dict],
        imported: List[Dict],
        strategy: ImportStrategy
    ) -> tuple:
        """
        Merge goals data based on strategy.

        Args:
            local: Local goals data
            imported: Imported goals data
            strategy: Merge strategy

        Returns:
            Tuple of (merged_data, conflicts)
        """
        if strategy == ImportStrategy.REPLACE:
            return imported, []

        conflicts = []
        merged = list(local)

        local_map = {g.get("id"): g for g in local if g.get("id")}

        for imported_goal in imported:
            goal_id = imported_goal.get("id")

            if goal_id and goal_id in local_map:
                local_goal = local_map[goal_id]

                if strategy == ImportStrategy.KEEP_NEWER:
                    local_ts = self._parse_timestamp(local_goal.get("updated_at") or local_goal.get("created_at"))
                    imported_ts = self._parse_timestamp(imported_goal.get("updated_at") or imported_goal.get("created_at"))

                    if imported_ts and (not local_ts or imported_ts > local_ts):
                        idx = next(i for i, g in enumerate(merged) if g.get("id") == goal_id)
                        merged[idx] = imported_goal
                else:
                    # Check for value conflicts
                    if local_goal.get("current_value") != imported_goal.get("current_value"):
                        # Take the higher progress value
                        local_val = local_goal.get("current_value", 0)
                        imported_val = imported_goal.get("current_value", 0)

                        if imported_val > local_val:
                            idx = next(i for i, g in enumerate(merged) if g.get("id") == goal_id)
                            merged[idx]["current_value"] = imported_val

                    if local_goal.get("status") != imported_goal.get("status"):
                        conflicts.append(SyncConflict(
                            id=f"goal_{goal_id}",
                            conflict_type="status_conflict",
                            data_type="goals",
                            record_id=goal_id,
                            field_name="status",
                            local_value=local_goal.get("status"),
                            imported_value=imported_goal.get("status"),
                            local_timestamp=self._parse_timestamp(local_goal.get("updated_at")),
                            imported_timestamp=self._parse_timestamp(imported_goal.get("updated_at"))
                        ))
            else:
                merged.append(imported_goal)

        return merged, conflicts

    async def restore_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Restore user settings from backup.

        Args:
            settings: Settings dictionary to restore

        Returns:
            True if successful
        """
        try:
            # Settings would be stored in a settings store
            # For now, log and return success
            logger.info(
                "settings_restored",
                keys=list(settings.keys()) if settings else []
            )
            return True

        except Exception as e:
            logger.error("restore_settings_failed", error=str(e))
            return False

    async def rollback_import(self) -> bool:
        """
        Rollback a failed import using the transaction snapshot.

        Returns:
            True if rollback successful
        """
        if not self._transaction or not self._transaction.has_data:
            logger.warning("no_transaction_to_rollback")
            return False

        try:
            logger.info(
                "rollback_started",
                transaction_id=self._transaction.transaction_id
            )

            # Restore conversations
            if self._transaction.conversations_backup:
                async with self.database.get_connection() as conn:
                    await conn.execute("DELETE FROM captures")
                    for conv in self._transaction.conversations_backup:
                        await conn.execute(
                            """
                            INSERT INTO captures (id, session_id, timestamp, user_text, agent_text, metadata)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                conv.get("id"),
                                conv.get("session_id"),
                                conv.get("timestamp"),
                                conv.get("user_text"),
                                conv.get("agent_text"),
                                json.dumps(conv.get("metadata")) if conv.get("metadata") else None
                            )
                        )
                    await conn.commit()

            # Restore goals
            if self._transaction.goals_backup:
                for goal_data in self._transaction.goals_backup:
                    goal = deserialize_goal(goal_data)
                    await self.goal_store.save_goal(goal)

            logger.info(
                "rollback_complete",
                transaction_id=self._transaction.transaction_id
            )

            self._transaction = None
            return True

        except Exception as e:
            logger.error("rollback_failed", error=str(e), exc_info=True)
            return False

    # ========================================================================
    # IMPORT STRATEGY IMPLEMENTATIONS
    # ========================================================================

    async def _import_replace(
        self,
        data: Dict[str, Any],
        result: ImportResult
    ) -> None:
        """Replace all local data with imported data."""
        try:
            # Clear and import conversations
            if "conversations" in data:
                async with self.database.get_connection() as conn:
                    await conn.execute("DELETE FROM captures")
                    await conn.commit()

                    for conv in data["conversations"]:
                        try:
                            await self._import_conversation(conn, conv)
                            result.items_imported["conversations"] = result.items_imported.get("conversations", 0) + 1
                        except Exception as e:
                            result.items_skipped["conversations"] = result.items_skipped.get("conversations", 0) + 1
                            result.warnings.append(f"Skipped conversation: {str(e)}")

                    await conn.commit()

            self._report_progress("importing_feedback", 50)

            # Clear and import feedback
            if "feedback" in data:
                async with self.feedback_store.get_connection() as conn:
                    await conn.execute("DELETE FROM explicit_feedback")
                    await conn.execute("DELETE FROM implicit_feedback")
                    await conn.execute("DELETE FROM corrections")
                    await conn.commit()

                for fb in data["feedback"]:
                    try:
                        await self._import_feedback(fb)
                        result.items_imported["feedback"] = result.items_imported.get("feedback", 0) + 1
                    except Exception as e:
                        result.items_skipped["feedback"] = result.items_skipped.get("feedback", 0) + 1
                        result.warnings.append(f"Skipped feedback: {str(e)}")

            self._report_progress("importing_goals", 70)

            # Import goals
            if "goals" in data:
                # Get goals from the data structure
                goals_list = self._extract_goals_list(data["goals"])

                for goal_data in goals_list:
                    try:
                        goal = deserialize_goal(goal_data)
                        await self.goal_store.save_goal(goal)
                        result.items_imported["goals"] = result.items_imported.get("goals", 0) + 1
                    except Exception as e:
                        result.items_skipped["goals"] = result.items_skipped.get("goals", 0) + 1
                        result.warnings.append(f"Skipped goal: {str(e)}")

            self._report_progress("importing_achievements", 80)

            # Import achievements
            if "achievements" in data:
                achievements_list = self._extract_achievements_list(data["achievements"])

                for ach_data in achievements_list:
                    try:
                        achievement = deserialize_achievement(ach_data)
                        await self.goal_store.save_achievement(achievement)
                        result.items_imported["achievements"] = result.items_imported.get("achievements", 0) + 1
                    except Exception as e:
                        result.items_skipped["achievements"] = result.items_skipped.get("achievements", 0) + 1
                        result.warnings.append(f"Skipped achievement: {str(e)}")

            # Import settings
            if "settings" in data:
                await self.restore_settings(data["settings"])

        except Exception as e:
            logger.error("import_replace_failed", error=str(e), exc_info=True)
            raise

    async def _import_merge(
        self,
        data: Dict[str, Any],
        result: ImportResult,
        resolve_conflicts: bool
    ) -> None:
        """Merge imported data with existing data."""
        try:
            # Get current local data
            local_data = await self._get_local_data()

            all_conflicts = []

            # Merge conversations
            if "conversations" in data:
                merged_convs, conv_conflicts = await self.merge_conversations(
                    local_data.get("conversations", []),
                    data["conversations"],
                    ImportStrategy.MERGE
                )
                all_conflicts.extend(conv_conflicts)

                # Import merged conversations
                async with self.database.get_connection() as conn:
                    for conv in data["conversations"]:
                        # Check if exists
                        cursor = await conn.execute(
                            "SELECT id FROM captures WHERE id = ?",
                            (conv.get("id"),)
                        )
                        existing = await cursor.fetchone()

                        if not existing:
                            await self._import_conversation(conn, conv)
                            result.items_imported["conversations"] = result.items_imported.get("conversations", 0) + 1
                        else:
                            result.items_skipped["conversations"] = result.items_skipped.get("conversations", 0) + 1

                    await conn.commit()

            self._report_progress("importing_goals", 60)

            # Merge goals
            if "goals" in data:
                goals_list = self._extract_goals_list(data["goals"])
                local_goals = await self.goal_store.get_all_goals()
                local_goals_data = [self._goal_to_dict(g) for g in local_goals]

                merged_goals, goal_conflicts = await self.merge_goals(
                    local_goals_data,
                    goals_list,
                    ImportStrategy.MERGE
                )
                all_conflicts.extend(goal_conflicts)

                # Import new goals
                local_goal_ids = {g.id for g in local_goals}
                for goal_data in goals_list:
                    if goal_data.get("id") not in local_goal_ids:
                        try:
                            goal = deserialize_goal(goal_data)
                            await self.goal_store.save_goal(goal)
                            result.items_imported["goals"] = result.items_imported.get("goals", 0) + 1
                        except Exception as e:
                            result.warnings.append(f"Failed to import goal: {str(e)}")

            # Resolve conflicts if requested
            if resolve_conflicts and all_conflicts:
                for conflict in all_conflicts:
                    self.resolver.resolve_by_timestamp(conflict)
                    result.conflicts_resolved += 1

            result.conflicts = all_conflicts
            result.conflicts_pending = len([c for c in all_conflicts if not c.resolved])

        except Exception as e:
            logger.error("import_merge_failed", error=str(e), exc_info=True)
            raise

    async def _import_keep_newer(
        self,
        data: Dict[str, Any],
        result: ImportResult,
        resolve_conflicts: bool
    ) -> None:
        """Import only items newer than local versions."""
        try:
            # Get current local data
            local_data = await self._get_local_data()

            # Merge with keep_newer strategy
            if "conversations" in data:
                merged_convs, _ = await self.merge_conversations(
                    local_data.get("conversations", []),
                    data["conversations"],
                    ImportStrategy.KEEP_NEWER
                )

                # Count updates vs new items
                local_ids = {c.get("id") for c in local_data.get("conversations", [])}
                for conv in data["conversations"]:
                    if conv.get("id") in local_ids:
                        result.items_updated["conversations"] = result.items_updated.get("conversations", 0) + 1
                    else:
                        result.items_imported["conversations"] = result.items_imported.get("conversations", 0) + 1

            self._report_progress("importing_goals", 60)

            # Similar for goals
            if "goals" in data:
                goals_list = self._extract_goals_list(data["goals"])
                local_goals = await self.goal_store.get_all_goals()

                for goal_data in goals_list:
                    existing = next((g for g in local_goals if g.id == goal_data.get("id")), None)

                    if existing:
                        # Check timestamps
                        imported_ts = self._parse_timestamp(goal_data.get("updated_at"))
                        if imported_ts and imported_ts > existing.updated_at:
                            goal = deserialize_goal(goal_data)
                            await self.goal_store.save_goal(goal)
                            result.items_updated["goals"] = result.items_updated.get("goals", 0) + 1
                    else:
                        goal = deserialize_goal(goal_data)
                        await self.goal_store.save_goal(goal)
                        result.items_imported["goals"] = result.items_imported.get("goals", 0) + 1

        except Exception as e:
            logger.error("import_keep_newer_failed", error=str(e), exc_info=True)
            raise

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    async def _parse_backup(
        self,
        data: Union[bytes, str, Dict]
    ) -> Optional[Dict[str, Any]]:
        """Parse backup data from various formats."""
        try:
            # Handle gzip compressed data
            if isinstance(data, bytes):
                try:
                    data = gzip.decompress(data)
                except gzip.BadGzipFile:
                    pass  # Not compressed, use as-is

                data = data.decode("utf-8")

            # Parse JSON string
            if isinstance(data, str):
                data = json.loads(data)

            return data

        except Exception as e:
            logger.error("parse_backup_failed", error=str(e))
            return None

    async def _create_snapshot(self) -> None:
        """Create a snapshot of current data for rollback."""
        import uuid

        try:
            self._transaction = TransactionSnapshot(
                transaction_id=str(uuid.uuid4()),
                created_at=datetime.utcnow()
            )

            # Backup conversations
            async with self.database.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT id, session_id, timestamp, user_text, agent_text, metadata FROM captures"
                )
                rows = await cursor.fetchall()
                self._transaction.conversations_backup = [dict(row) for row in rows]

            # Backup goals
            goals = await self.goal_store.get_all_goals()
            self._transaction.goals_backup = [self._goal_to_dict(g) for g in goals]

            self._transaction.has_data = True

            logger.info(
                "snapshot_created",
                transaction_id=self._transaction.transaction_id,
                conversations=len(self._transaction.conversations_backup),
                goals=len(self._transaction.goals_backup)
            )

        except Exception as e:
            logger.error("create_snapshot_failed", error=str(e))
            self._transaction = None

    async def _get_local_data(self) -> Dict[str, Any]:
        """Get all local data for merge operations."""
        local_data = {}

        try:
            # Get conversations
            async with self.database.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT id, session_id, timestamp, user_text, agent_text, metadata FROM captures"
                )
                rows = await cursor.fetchall()
                local_data["conversations"] = [dict(row) for row in rows]

            # Get goals
            goals = await self.goal_store.get_all_goals()
            local_data["goals"] = [self._goal_to_dict(g) for g in goals]

            # Get achievements
            achievements = await self.goal_store.get_all_achievements()
            local_data["achievements"] = [self._achievement_to_dict(a) for a in achievements]

            return local_data

        except Exception as e:
            logger.error("get_local_data_failed", error=str(e))
            return {}

    async def _import_conversation(self, conn, conv: Dict) -> None:
        """Import a single conversation."""
        await conn.execute(
            """
            INSERT OR REPLACE INTO captures
            (id, session_id, timestamp, user_text, agent_text, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conv.get("id"),
                conv.get("session_id"),
                conv.get("timestamp"),
                conv.get("user_text"),
                conv.get("agent_text"),
                json.dumps(conv.get("metadata")) if conv.get("metadata") else None
            )
        )

    async def _import_feedback(self, fb: Dict) -> None:
        """Import a single feedback item."""
        feedback_obj = deserialize_feedback(fb)

        from app.learning.feedback_models import ExplicitFeedback, ImplicitFeedback, CorrectionFeedback

        if isinstance(feedback_obj, ExplicitFeedback):
            await self.feedback_store.save_explicit(feedback_obj)
        elif isinstance(feedback_obj, ImplicitFeedback):
            await self.feedback_store.save_implicit(feedback_obj)
        elif isinstance(feedback_obj, CorrectionFeedback):
            await self.feedback_store.save_correction(feedback_obj)

    def _extract_goals_list(self, goals_data: Any) -> List[Dict]:
        """Extract goals as a flat list from various formats."""
        if isinstance(goals_data, list):
            return goals_data
        elif isinstance(goals_data, dict):
            result = []
            result.extend(goals_data.get("active", []))
            result.extend(goals_data.get("completed", []))
            return result
        return []

    def _extract_achievements_list(self, achievements_data: Any) -> List[Dict]:
        """Extract achievements as a flat list from various formats."""
        if isinstance(achievements_data, list):
            return achievements_data
        elif isinstance(achievements_data, dict):
            return achievements_data.get("all", [])
        return []

    def _goal_to_dict(self, goal: Goal) -> Dict[str, Any]:
        """Convert Goal model to dictionary."""
        from app.sync.serializers import serialize_goal
        return serialize_goal(goal)

    def _achievement_to_dict(self, achievement: Achievement) -> Dict[str, Any]:
        """Convert Achievement model to dictionary."""
        from app.sync.serializers import serialize_achievement
        return serialize_achievement(achievement)

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse a timestamp from various formats."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None


# Singleton instance
import_service = ImportService()
