"""
Conflict Resolution Module
==========================

Handles detection and resolution of merge conflicts during data import.

PATTERN: Strategy-based conflict resolution with manual override support
WHY: Provide flexible, predictable handling of data conflicts

Features:
- Automatic conflict detection
- Multiple resolution strategies
- Timestamp-based resolution
- Priority-based resolution
- Manual resolution support
- Batch conflict resolution
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import copy

from app.logger import get_logger

# Module logger
logger = get_logger("conflict_resolver")


class ConflictType(str, Enum):
    """Types of data conflicts."""
    VALUE_MISMATCH = "value_mismatch"
    MISSING_LOCAL = "missing_local"
    MISSING_IMPORTED = "missing_imported"
    TYPE_MISMATCH = "type_mismatch"
    TIMESTAMP_CONFLICT = "timestamp_conflict"
    STATUS_CONFLICT = "status_conflict"
    DELETION_CONFLICT = "deletion_conflict"


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""
    KEEP_LOCAL = "keep_local"
    KEEP_IMPORTED = "keep_imported"
    KEEP_NEWER = "keep_newer"
    KEEP_OLDER = "keep_older"
    MERGE_VALUES = "merge_values"
    MANUAL = "manual"


@dataclass
class SyncConflict:
    """Represents a data synchronization conflict."""
    id: str
    conflict_type: ConflictType
    data_type: str  # goals, achievements, feedback, conversations, settings
    record_id: str
    field_name: Optional[str] = None
    local_value: Any = None
    imported_value: Any = None
    local_timestamp: Optional[datetime] = None
    imported_timestamp: Optional[datetime] = None
    resolution: Optional[ResolutionStrategy] = None
    resolved_value: Any = None
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "conflict_type": self.conflict_type.value,
            "data_type": self.data_type,
            "record_id": self.record_id,
            "field_name": self.field_name,
            "local_value": self._serialize_value(self.local_value),
            "imported_value": self._serialize_value(self.imported_value),
            "local_timestamp": self.local_timestamp.isoformat() if self.local_timestamp else None,
            "imported_timestamp": self.imported_timestamp.isoformat() if self.imported_timestamp else None,
            "resolution": self.resolution.value if self.resolution else None,
            "resolved_value": self._serialize_value(self.resolved_value),
            "resolved": self.resolved,
            "metadata": self.metadata
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON output."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (list, dict)):
            return value
        return value


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    success: bool
    resolved_conflicts: List[SyncConflict] = field(default_factory=list)
    unresolved_conflicts: List[SyncConflict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def total_conflicts(self) -> int:
        return len(self.resolved_conflicts) + len(self.unresolved_conflicts)

    @property
    def resolution_rate(self) -> float:
        if self.total_conflicts == 0:
            return 1.0
        return len(self.resolved_conflicts) / self.total_conflicts


class ConflictResolver:
    """
    Detects and resolves data synchronization conflicts.

    PATTERN: Strategy pattern for flexible conflict resolution
    WHY: Different data types may require different resolution approaches

    USAGE:
        resolver = ConflictResolver()

        # Detect conflicts
        conflicts = resolver.detect_conflicts(local_data, imported_data)

        # Resolve by strategy
        for conflict in conflicts:
            resolver.resolve_by_timestamp(conflict)

        # Apply resolutions
        merged_data = resolver.apply_resolution(local_data, conflicts)
    """

    def __init__(self):
        """Initialize the conflict resolver."""
        self._custom_resolvers: Dict[str, Callable] = {}

    def register_resolver(
        self,
        data_type: str,
        resolver: Callable[[SyncConflict], Any]
    ) -> None:
        """
        Register a custom resolver for a data type.

        Args:
            data_type: Type of data (goals, achievements, etc.)
            resolver: Callable that takes a conflict and returns resolved value
        """
        self._custom_resolvers[data_type] = resolver
        logger.info("custom_resolver_registered", data_type=data_type)

    def detect_conflicts(
        self,
        local_data: Dict[str, Any],
        imported_data: Dict[str, Any]
    ) -> List[SyncConflict]:
        """
        Detect conflicts between local and imported data.

        Args:
            local_data: Current local data
            imported_data: Data being imported

        Returns:
            List of detected conflicts
        """
        conflicts = []
        conflict_counter = 0

        try:
            # Detect goal conflicts
            if "goals" in local_data or "goals" in imported_data:
                goal_conflicts = self._detect_goal_conflicts(
                    local_data.get("goals", {}),
                    imported_data.get("goals", {}),
                    conflict_counter
                )
                conflicts.extend(goal_conflicts)
                conflict_counter += len(goal_conflicts)

            # Detect achievement conflicts
            if "achievements" in local_data or "achievements" in imported_data:
                achievement_conflicts = self._detect_achievement_conflicts(
                    local_data.get("achievements", {}),
                    imported_data.get("achievements", {}),
                    conflict_counter
                )
                conflicts.extend(achievement_conflicts)
                conflict_counter += len(achievement_conflicts)

            # Detect feedback conflicts
            if "feedback" in local_data or "feedback" in imported_data:
                feedback_conflicts = self._detect_feedback_conflicts(
                    local_data.get("feedback", []),
                    imported_data.get("feedback", []),
                    conflict_counter
                )
                conflicts.extend(feedback_conflicts)
                conflict_counter += len(feedback_conflicts)

            # Detect conversation conflicts
            if "conversations" in local_data or "conversations" in imported_data:
                conv_conflicts = self._detect_conversation_conflicts(
                    local_data.get("conversations", []),
                    imported_data.get("conversations", []),
                    conflict_counter
                )
                conflicts.extend(conv_conflicts)
                conflict_counter += len(conv_conflicts)

            # Detect settings conflicts
            if "settings" in local_data or "settings" in imported_data:
                settings_conflicts = self._detect_settings_conflicts(
                    local_data.get("settings", {}),
                    imported_data.get("settings", {}),
                    conflict_counter
                )
                conflicts.extend(settings_conflicts)

            logger.info(
                "conflict_detection_complete",
                total_conflicts=len(conflicts),
                goal_conflicts=len([c for c in conflicts if c.data_type == "goals"]),
                achievement_conflicts=len([c for c in conflicts if c.data_type == "achievements"]),
                feedback_conflicts=len([c for c in conflicts if c.data_type == "feedback"]),
                conversation_conflicts=len([c for c in conflicts if c.data_type == "conversations"]),
                settings_conflicts=len([c for c in conflicts if c.data_type == "settings"])
            )

            return conflicts

        except Exception as e:
            logger.error("conflict_detection_failed", error=str(e), exc_info=True)
            return conflicts

    def resolve_by_timestamp(self, conflict: SyncConflict) -> Any:
        """
        Resolve conflict by keeping the newer value.

        Args:
            conflict: Conflict to resolve

        Returns:
            Resolved value
        """
        if conflict.local_timestamp and conflict.imported_timestamp:
            if conflict.local_timestamp >= conflict.imported_timestamp:
                conflict.resolution = ResolutionStrategy.KEEP_LOCAL
                conflict.resolved_value = conflict.local_value
            else:
                conflict.resolution = ResolutionStrategy.KEEP_IMPORTED
                conflict.resolved_value = conflict.imported_value
        elif conflict.imported_timestamp:
            # Only imported has timestamp, prefer it
            conflict.resolution = ResolutionStrategy.KEEP_IMPORTED
            conflict.resolved_value = conflict.imported_value
        else:
            # Only local has timestamp or neither, prefer local
            conflict.resolution = ResolutionStrategy.KEEP_LOCAL
            conflict.resolved_value = conflict.local_value

        conflict.resolved = True

        logger.debug(
            "conflict_resolved_by_timestamp",
            conflict_id=conflict.id,
            resolution=conflict.resolution.value
        )

        return conflict.resolved_value

    def resolve_by_priority(
        self,
        conflict: SyncConflict,
        priority: ResolutionStrategy
    ) -> Any:
        """
        Resolve conflict by a priority strategy.

        Args:
            conflict: Conflict to resolve
            priority: Which value to keep (KEEP_LOCAL or KEEP_IMPORTED)

        Returns:
            Resolved value
        """
        if priority == ResolutionStrategy.KEEP_LOCAL:
            conflict.resolved_value = conflict.local_value
        elif priority == ResolutionStrategy.KEEP_IMPORTED:
            conflict.resolved_value = conflict.imported_value
        elif priority == ResolutionStrategy.KEEP_NEWER:
            return self.resolve_by_timestamp(conflict)
        elif priority == ResolutionStrategy.KEEP_OLDER:
            # Inverse of timestamp resolution
            if conflict.local_timestamp and conflict.imported_timestamp:
                if conflict.local_timestamp <= conflict.imported_timestamp:
                    conflict.resolved_value = conflict.local_value
                else:
                    conflict.resolved_value = conflict.imported_value
            else:
                conflict.resolved_value = conflict.local_value
        elif priority == ResolutionStrategy.MERGE_VALUES:
            conflict.resolved_value = self._merge_values(
                conflict.local_value,
                conflict.imported_value
            )
        else:
            # Default to local
            conflict.resolved_value = conflict.local_value

        conflict.resolution = priority
        conflict.resolved = True

        logger.debug(
            "conflict_resolved_by_priority",
            conflict_id=conflict.id,
            priority=priority.value
        )

        return conflict.resolved_value

    def resolve_all(
        self,
        conflicts: List[SyncConflict],
        strategy: ResolutionStrategy
    ) -> ResolutionResult:
        """
        Resolve all conflicts using a single strategy.

        Args:
            conflicts: List of conflicts to resolve
            strategy: Resolution strategy to apply

        Returns:
            ResolutionResult with outcomes
        """
        result = ResolutionResult(success=True)

        for conflict in conflicts:
            try:
                if strategy == ResolutionStrategy.KEEP_NEWER:
                    self.resolve_by_timestamp(conflict)
                else:
                    self.resolve_by_priority(conflict, strategy)

                result.resolved_conflicts.append(conflict)

            except Exception as e:
                result.unresolved_conflicts.append(conflict)
                result.errors.append(f"Failed to resolve conflict {conflict.id}: {str(e)}")
                logger.error(
                    "conflict_resolution_failed",
                    conflict_id=conflict.id,
                    error=str(e)
                )

        result.success = len(result.unresolved_conflicts) == 0

        logger.info(
            "batch_resolution_complete",
            total=len(conflicts),
            resolved=len(result.resolved_conflicts),
            unresolved=len(result.unresolved_conflicts)
        )

        return result

    def apply_resolution(
        self,
        local_data: Dict[str, Any],
        conflicts: List[SyncConflict],
        imported_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply resolved conflicts to create merged data.

        Args:
            local_data: Original local data
            conflicts: List of resolved conflicts
            imported_data: Optional imported data for merge operations

        Returns:
            Merged data dictionary
        """
        # Deep copy to avoid modifying original
        merged = copy.deepcopy(local_data)

        try:
            for conflict in conflicts:
                if not conflict.resolved:
                    logger.warning(
                        "skipping_unresolved_conflict",
                        conflict_id=conflict.id
                    )
                    continue

                self._apply_single_resolution(merged, conflict, imported_data)

            logger.info(
                "resolutions_applied",
                total_conflicts=len(conflicts),
                applied=len([c for c in conflicts if c.resolved])
            )

            return merged

        except Exception as e:
            logger.error("apply_resolution_failed", error=str(e), exc_info=True)
            # Return local data if application fails
            return local_data

    def _detect_goal_conflicts(
        self,
        local_goals: Union[Dict, List],
        imported_goals: Union[Dict, List],
        start_id: int
    ) -> List[SyncConflict]:
        """Detect conflicts in goals data."""
        conflicts = []
        conflict_id = start_id

        # Normalize to lists
        local_list = self._normalize_goals(local_goals)
        imported_list = self._normalize_goals(imported_goals)

        # Build lookup maps
        local_map = {g.get("id"): g for g in local_list if g.get("id")}
        imported_map = {g.get("id"): g for g in imported_list if g.get("id")}

        # Check for conflicts in existing records
        for goal_id, local_goal in local_map.items():
            if goal_id in imported_map:
                imported_goal = imported_map[goal_id]

                # Check for value differences
                for field_name in ["title", "status", "current_value", "target_value"]:
                    local_val = local_goal.get(field_name)
                    imported_val = imported_goal.get(field_name)

                    if local_val != imported_val:
                        conflicts.append(SyncConflict(
                            id=f"conflict_{conflict_id}",
                            conflict_type=ConflictType.VALUE_MISMATCH,
                            data_type="goals",
                            record_id=goal_id,
                            field_name=field_name,
                            local_value=local_val,
                            imported_value=imported_val,
                            local_timestamp=self._parse_timestamp(local_goal.get("updated_at") or local_goal.get("created_at")),
                            imported_timestamp=self._parse_timestamp(imported_goal.get("updated_at") or imported_goal.get("created_at"))
                        ))
                        conflict_id += 1

        # Check for records only in imported
        for goal_id, imported_goal in imported_map.items():
            if goal_id not in local_map:
                conflicts.append(SyncConflict(
                    id=f"conflict_{conflict_id}",
                    conflict_type=ConflictType.MISSING_LOCAL,
                    data_type="goals",
                    record_id=goal_id,
                    local_value=None,
                    imported_value=imported_goal,
                    imported_timestamp=self._parse_timestamp(imported_goal.get("created_at"))
                ))
                conflict_id += 1

        return conflicts

    def _detect_achievement_conflicts(
        self,
        local_achievements: Union[Dict, List],
        imported_achievements: Union[Dict, List],
        start_id: int
    ) -> List[SyncConflict]:
        """Detect conflicts in achievements data."""
        conflicts = []
        conflict_id = start_id

        # Normalize to lists
        local_list = self._normalize_achievements(local_achievements)
        imported_list = self._normalize_achievements(imported_achievements)

        # Build lookup maps
        local_map = {a.get("id"): a for a in local_list if a.get("id")}
        imported_map = {a.get("id"): a for a in imported_list if a.get("id")}

        # Check for conflicts
        for ach_id, local_ach in local_map.items():
            if ach_id in imported_map:
                imported_ach = imported_map[ach_id]

                # Check unlock status conflict
                local_unlocked = local_ach.get("unlocked", False)
                imported_unlocked = imported_ach.get("unlocked", False)

                if local_unlocked != imported_unlocked:
                    conflicts.append(SyncConflict(
                        id=f"conflict_{conflict_id}",
                        conflict_type=ConflictType.STATUS_CONFLICT,
                        data_type="achievements",
                        record_id=ach_id,
                        field_name="unlocked",
                        local_value=local_unlocked,
                        imported_value=imported_unlocked,
                        local_timestamp=self._parse_timestamp(local_ach.get("unlocked_at")),
                        imported_timestamp=self._parse_timestamp(imported_ach.get("unlocked_at"))
                    ))
                    conflict_id += 1

                # Check progress conflict
                local_progress = local_ach.get("progress", 0)
                imported_progress = imported_ach.get("progress", 0)

                if local_progress != imported_progress:
                    conflicts.append(SyncConflict(
                        id=f"conflict_{conflict_id}",
                        conflict_type=ConflictType.VALUE_MISMATCH,
                        data_type="achievements",
                        record_id=ach_id,
                        field_name="progress",
                        local_value=local_progress,
                        imported_value=imported_progress
                    ))
                    conflict_id += 1

        return conflicts

    def _detect_feedback_conflicts(
        self,
        local_feedback: List,
        imported_feedback: List,
        start_id: int
    ) -> List[SyncConflict]:
        """Detect conflicts in feedback data."""
        conflicts = []
        conflict_id = start_id

        # Build lookup maps
        local_map = {f.get("id"): f for f in local_feedback if f.get("id")}
        imported_map = {f.get("id"): f for f in imported_feedback if f.get("id")}

        # Check for conflicts
        for fb_id, local_fb in local_map.items():
            if fb_id in imported_map:
                imported_fb = imported_map[fb_id]

                # Check rating conflict
                local_rating = local_fb.get("rating")
                imported_rating = imported_fb.get("rating")

                if local_rating != imported_rating:
                    conflicts.append(SyncConflict(
                        id=f"conflict_{conflict_id}",
                        conflict_type=ConflictType.VALUE_MISMATCH,
                        data_type="feedback",
                        record_id=fb_id,
                        field_name="rating",
                        local_value=local_rating,
                        imported_value=imported_rating,
                        local_timestamp=self._parse_timestamp(local_fb.get("timestamp")),
                        imported_timestamp=self._parse_timestamp(imported_fb.get("timestamp"))
                    ))
                    conflict_id += 1

        # Check for records only in imported
        for fb_id, imported_fb in imported_map.items():
            if fb_id not in local_map:
                conflicts.append(SyncConflict(
                    id=f"conflict_{conflict_id}",
                    conflict_type=ConflictType.MISSING_LOCAL,
                    data_type="feedback",
                    record_id=fb_id,
                    local_value=None,
                    imported_value=imported_fb,
                    imported_timestamp=self._parse_timestamp(imported_fb.get("timestamp"))
                ))
                conflict_id += 1

        return conflicts

    def _detect_conversation_conflicts(
        self,
        local_conversations: List,
        imported_conversations: List,
        start_id: int
    ) -> List[SyncConflict]:
        """Detect conflicts in conversation data."""
        conflicts = []
        conflict_id = start_id

        # Build lookup maps
        local_map = {c.get("id"): c for c in local_conversations if c.get("id")}
        imported_map = {c.get("id"): c for c in imported_conversations if c.get("id")}

        # Check for records only in imported (conversations shouldn't have value conflicts)
        for conv_id, imported_conv in imported_map.items():
            if conv_id not in local_map:
                conflicts.append(SyncConflict(
                    id=f"conflict_{conflict_id}",
                    conflict_type=ConflictType.MISSING_LOCAL,
                    data_type="conversations",
                    record_id=conv_id,
                    local_value=None,
                    imported_value=imported_conv,
                    imported_timestamp=self._parse_timestamp(imported_conv.get("timestamp"))
                ))
                conflict_id += 1

        return conflicts

    def _detect_settings_conflicts(
        self,
        local_settings: Dict,
        imported_settings: Dict,
        start_id: int
    ) -> List[SyncConflict]:
        """Detect conflicts in settings data."""
        conflicts = []
        conflict_id = start_id

        # Get all keys
        all_keys = set(local_settings.keys()) | set(imported_settings.keys())

        for key in all_keys:
            local_val = local_settings.get(key)
            imported_val = imported_settings.get(key)

            if local_val != imported_val:
                conflict_type = ConflictType.VALUE_MISMATCH
                if local_val is None:
                    conflict_type = ConflictType.MISSING_LOCAL
                elif imported_val is None:
                    conflict_type = ConflictType.MISSING_IMPORTED

                conflicts.append(SyncConflict(
                    id=f"conflict_{conflict_id}",
                    conflict_type=conflict_type,
                    data_type="settings",
                    record_id="settings",
                    field_name=key,
                    local_value=local_val,
                    imported_value=imported_val
                ))
                conflict_id += 1

        return conflicts

    def _normalize_goals(self, goals_data: Union[Dict, List]) -> List[Dict]:
        """Normalize goals data to a list format."""
        if isinstance(goals_data, list):
            return goals_data
        elif isinstance(goals_data, dict):
            result = []
            result.extend(goals_data.get("active", []))
            result.extend(goals_data.get("completed", []))
            return result
        return []

    def _normalize_achievements(self, achievements_data: Union[Dict, List]) -> List[Dict]:
        """Normalize achievements data to a list format."""
        if isinstance(achievements_data, list):
            return achievements_data
        elif isinstance(achievements_data, dict):
            return achievements_data.get("all", [])
        return []

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse a timestamp string to datetime."""
        if not timestamp_str:
            return None

        try:
            # Handle ISO format with timezone
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            return None

    def _merge_values(self, local_value: Any, imported_value: Any) -> Any:
        """Merge two values together."""
        # For lists, combine unique items
        if isinstance(local_value, list) and isinstance(imported_value, list):
            # Simple deduplication for primitive lists
            combined = list(local_value)
            for item in imported_value:
                if item not in combined:
                    combined.append(item)
            return combined

        # For dicts, merge keys
        if isinstance(local_value, dict) and isinstance(imported_value, dict):
            merged = dict(local_value)
            merged.update(imported_value)
            return merged

        # For numbers, take max (e.g., progress)
        if isinstance(local_value, (int, float)) and isinstance(imported_value, (int, float)):
            return max(local_value, imported_value)

        # Default: prefer imported
        return imported_value if imported_value is not None else local_value

    def _apply_single_resolution(
        self,
        data: Dict[str, Any],
        conflict: SyncConflict,
        imported_data: Optional[Dict[str, Any]]
    ) -> None:
        """Apply a single conflict resolution to the data."""
        data_type = conflict.data_type

        if data_type == "settings":
            if data_type not in data:
                data[data_type] = {}
            if conflict.field_name:
                data[data_type][conflict.field_name] = conflict.resolved_value

        elif data_type in ["goals", "achievements", "feedback", "conversations"]:
            if conflict.conflict_type == ConflictType.MISSING_LOCAL:
                # Add the imported record
                if data_type not in data:
                    data[data_type] = [] if data_type in ["feedback", "conversations"] else {}

                if isinstance(data[data_type], list):
                    data[data_type].append(conflict.resolved_value)
                elif isinstance(data[data_type], dict):
                    # For goals/achievements, add to appropriate list
                    if data_type == "goals":
                        status = conflict.resolved_value.get("status", "active")
                        key = "completed" if status == "completed" else "active"
                        if key not in data[data_type]:
                            data[data_type][key] = []
                        data[data_type][key].append(conflict.resolved_value)
                    elif data_type == "achievements":
                        if "all" not in data[data_type]:
                            data[data_type]["all"] = []
                        data[data_type]["all"].append(conflict.resolved_value)

            elif conflict.field_name:
                # Update specific field
                self._update_record_field(
                    data,
                    data_type,
                    conflict.record_id,
                    conflict.field_name,
                    conflict.resolved_value
                )

    def _update_record_field(
        self,
        data: Dict[str, Any],
        data_type: str,
        record_id: str,
        field_name: str,
        value: Any
    ) -> None:
        """Update a specific field in a record."""
        if data_type not in data:
            return

        container = data[data_type]

        # Handle list containers
        if isinstance(container, list):
            for record in container:
                if record.get("id") == record_id:
                    record[field_name] = value
                    return

        # Handle dict containers (goals, achievements)
        elif isinstance(container, dict):
            for key in ["active", "completed", "all"]:
                if key in container and isinstance(container[key], list):
                    for record in container[key]:
                        if record.get("id") == record_id:
                            record[field_name] = value
                            return


# Singleton instance
conflict_resolver = ConflictResolver()
