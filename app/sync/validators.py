"""
Backup Validation Module
========================

Validates backup data integrity, schema, and version compatibility.

PATTERN: Defensive validation with detailed error reporting
WHY: Prevent data corruption from invalid or malicious backups

Features:
- Checksum validation
- Version compatibility checking
- Schema validation
- Data integrity verification
- Detailed error reporting
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import re

from app.logger import get_logger

# Module logger
logger = get_logger("backup_validator")

# Current backup schema version
CURRENT_SCHEMA_VERSION = "1.0"
SUPPORTED_VERSIONS = ["1.0"]


class ValidationErrorCode(str, Enum):
    """Validation error codes for detailed reporting."""
    INVALID_FORMAT = "invalid_format"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    VERSION_UNSUPPORTED = "version_unsupported"
    SCHEMA_INVALID = "schema_invalid"
    DATA_CORRUPTED = "data_corrupted"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_DATA_TYPE = "invalid_data_type"
    INVALID_TIMESTAMP = "invalid_timestamp"
    INTEGRITY_FAILURE = "integrity_failure"


@dataclass
class ValidationError:
    """Detailed validation error."""
    code: ValidationErrorCode
    message: str
    path: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "code": self.code.value,
            "message": self.message
        }
        if self.path:
            result["path"] = self.path
        if self.expected:
            result["expected"] = self.expected
        if self.actual:
            result["actual"] = self.actual
        return result


@dataclass
class ValidationResult:
    """Result of backup validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    version: Optional[str] = None
    checksum_verified: bool = False
    schema_valid: bool = False
    data_integrity_valid: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(
        self,
        code: ValidationErrorCode,
        message: str,
        path: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None
    ) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(
            code=code,
            message=message,
            path=path,
            expected=expected,
            actual=actual
        ))
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "version": self.version,
            "checksum_verified": self.checksum_verified,
            "schema_valid": self.schema_valid,
            "data_integrity_valid": self.data_integrity_valid,
            "metadata": self.metadata
        }


# Schema definitions for validation
REQUIRED_METADATA_FIELDS = ["version", "export_date", "generated_by"]
REQUIRED_TOP_LEVEL_FIELDS = ["metadata"]

GOALS_SCHEMA = {
    "required": ["id", "title", "goal_type", "target_value", "current_value", "status"],
    "optional": ["description", "unit", "deadline", "created_at", "completed_at", "milestones", "metadata"],
    "types": {
        "id": str,
        "title": str,
        "goal_type": str,
        "target_value": (int, float),
        "current_value": (int, float),
        "status": str
    }
}

ACHIEVEMENTS_SCHEMA = {
    "required": ["id", "title", "description", "category", "rarity", "requirement_value"],
    "optional": ["icon", "requirement", "requirement_type", "points", "unlocked", "unlocked_at", "progress", "hidden", "metadata"],
    "types": {
        "id": str,
        "title": str,
        "description": str,
        "category": str,
        "rarity": str,
        "requirement_value": (int, float)
    }
}

FEEDBACK_SCHEMA = {
    "required": ["id", "session_id"],
    "optional": ["exchange_id", "rating", "helpful", "comment", "timestamp", "metadata", "feedback_type"],
    "types": {
        "id": str,
        "session_id": str
    }
}

CONVERSATIONS_SCHEMA = {
    "required": ["id", "session_id", "user_text", "agent_text"],
    "optional": ["timestamp", "intent", "metadata"],
    "types": {
        "id": str,
        "session_id": str,
        "user_text": str,
        "agent_text": str
    }
}

SETTINGS_SCHEMA = {
    "required": [],
    "optional": ["preferences", "theme", "language", "notifications", "privacy", "sync"],
    "types": {}
}


class BackupValidator:
    """
    Validates backup data for import operations.

    PATTERN: Multi-stage validation with detailed error reporting
    WHY: Ensure data safety before import operations

    USAGE:
        validator = BackupValidator()
        result = validator.validate(backup_data)
        if result.valid:
            # Proceed with import
        else:
            # Handle errors
    """

    def __init__(self):
        """Initialize the backup validator."""
        self.schemas = {
            "goals": GOALS_SCHEMA,
            "achievements": ACHIEVEMENTS_SCHEMA,
            "feedback": FEEDBACK_SCHEMA,
            "conversations": CONVERSATIONS_SCHEMA,
            "settings": SETTINGS_SCHEMA
        }

    def validate(self, data: Union[bytes, str, Dict]) -> ValidationResult:
        """
        Perform complete validation of backup data.

        Args:
            data: Raw backup data (bytes, JSON string, or dict)

        Returns:
            ValidationResult with all validation outcomes
        """
        result = ValidationResult(valid=True)

        try:
            # Parse data if needed
            parsed_data = self._parse_data(data, result)
            if not result.valid:
                return result

            # Validate metadata and version
            self._validate_metadata(parsed_data, result)
            if not result.valid:
                return result

            # Validate schema structure
            self._validate_schema(parsed_data, result)

            # Check data integrity
            self._check_data_integrity(parsed_data, result)

            logger.info(
                "backup_validation_complete",
                valid=result.valid,
                error_count=len(result.errors),
                warning_count=len(result.warnings)
            )

            return result

        except Exception as e:
            logger.error("backup_validation_failed", error=str(e), exc_info=True)
            result.add_error(
                ValidationErrorCode.INVALID_FORMAT,
                f"Unexpected validation error: {str(e)}"
            )
            return result

    def validate_checksum(
        self,
        data: bytes,
        expected_checksum: str,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Validate data checksum.

        Args:
            data: Raw data bytes
            expected_checksum: Expected checksum value
            algorithm: Hash algorithm (sha256, md5, sha1)

        Returns:
            True if checksum matches
        """
        try:
            if algorithm == "sha256":
                hasher = hashlib.sha256()
            elif algorithm == "md5":
                hasher = hashlib.md5()
            elif algorithm == "sha1":
                hasher = hashlib.sha1()
            else:
                logger.warning("unsupported_checksum_algorithm", algorithm=algorithm)
                return False

            hasher.update(data)
            actual_checksum = hasher.hexdigest()

            match = actual_checksum.lower() == expected_checksum.lower()

            if not match:
                logger.warning(
                    "checksum_mismatch",
                    expected=expected_checksum[:16] + "...",
                    actual=actual_checksum[:16] + "..."
                )

            return match

        except Exception as e:
            logger.error("checksum_validation_failed", error=str(e))
            return False

    def validate_version(self, version: str) -> tuple:
        """
        Validate backup version compatibility.

        Args:
            version: Version string from backup

        Returns:
            Tuple of (is_valid, is_current, migration_needed)
        """
        try:
            if not version:
                return (False, False, False)

            is_valid = version in SUPPORTED_VERSIONS
            is_current = version == CURRENT_SCHEMA_VERSION

            # Check if migration would be needed
            migration_needed = is_valid and not is_current

            logger.debug(
                "version_validation",
                version=version,
                is_valid=is_valid,
                is_current=is_current,
                migration_needed=migration_needed
            )

            return (is_valid, is_current, migration_needed)

        except Exception as e:
            logger.error("version_validation_failed", error=str(e))
            return (False, False, False)

    def validate_schema(self, data: Dict, data_type: str) -> List[ValidationError]:
        """
        Validate data against its schema.

        Args:
            data: Data to validate
            data_type: Type of data (goals, achievements, etc.)

        Returns:
            List of validation errors
        """
        errors = []

        if data_type not in self.schemas:
            errors.append(ValidationError(
                code=ValidationErrorCode.SCHEMA_INVALID,
                message=f"Unknown data type: {data_type}"
            ))
            return errors

        schema = self.schemas[data_type]

        # Check required fields
        for field_name in schema["required"]:
            if field_name not in data:
                errors.append(ValidationError(
                    code=ValidationErrorCode.MISSING_REQUIRED_FIELD,
                    message=f"Missing required field: {field_name}",
                    path=f"{data_type}.{field_name}",
                    expected=field_name
                ))

        # Check field types
        for field_name, expected_type in schema["types"].items():
            if field_name in data and data[field_name] is not None:
                if not isinstance(data[field_name], expected_type):
                    errors.append(ValidationError(
                        code=ValidationErrorCode.INVALID_DATA_TYPE,
                        message=f"Invalid type for field: {field_name}",
                        path=f"{data_type}.{field_name}",
                        expected=str(expected_type),
                        actual=str(type(data[field_name]))
                    ))

        return errors

    def check_data_integrity(self, backup: Dict) -> List[ValidationError]:
        """
        Check overall data integrity of backup.

        Args:
            backup: Parsed backup data

        Returns:
            List of integrity errors
        """
        errors = []

        try:
            # Check goal-milestone relationships
            if "goals" in backup:
                goals_data = backup["goals"]
                if isinstance(goals_data, dict):
                    all_goals = goals_data.get("active", []) + goals_data.get("completed", [])
                else:
                    all_goals = goals_data if isinstance(goals_data, list) else []

                goal_ids = set()
                for goal in all_goals:
                    if isinstance(goal, dict):
                        goal_id = goal.get("id")
                        if goal_id:
                            if goal_id in goal_ids:
                                errors.append(ValidationError(
                                    code=ValidationErrorCode.INTEGRITY_FAILURE,
                                    message=f"Duplicate goal ID: {goal_id}",
                                    path="goals"
                                ))
                            goal_ids.add(goal_id)

            # Check achievement uniqueness
            if "achievements" in backup:
                achievements_data = backup["achievements"]
                if isinstance(achievements_data, dict):
                    all_achievements = achievements_data.get("all", [])
                else:
                    all_achievements = achievements_data if isinstance(achievements_data, list) else []

                achievement_ids = set()
                for achievement in all_achievements:
                    if isinstance(achievement, dict):
                        ach_id = achievement.get("id")
                        if ach_id:
                            if ach_id in achievement_ids:
                                errors.append(ValidationError(
                                    code=ValidationErrorCode.INTEGRITY_FAILURE,
                                    message=f"Duplicate achievement ID: {ach_id}",
                                    path="achievements"
                                ))
                            achievement_ids.add(ach_id)

            # Check timestamp validity
            self._validate_timestamps(backup, errors)

            return errors

        except Exception as e:
            logger.error("integrity_check_failed", error=str(e))
            errors.append(ValidationError(
                code=ValidationErrorCode.DATA_CORRUPTED,
                message=f"Integrity check failed: {str(e)}"
            ))
            return errors

    def _parse_data(self, data: Union[bytes, str, Dict], result: ValidationResult) -> Optional[Dict]:
        """Parse raw data into dictionary."""
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            if isinstance(data, str):
                parsed = json.loads(data)
            elif isinstance(data, dict):
                parsed = data
            else:
                result.add_error(
                    ValidationErrorCode.INVALID_FORMAT,
                    f"Unsupported data type: {type(data).__name__}",
                    expected="bytes, str, or dict"
                )
                return None

            result.metadata["parsed_type"] = type(parsed).__name__
            return parsed

        except json.JSONDecodeError as e:
            result.add_error(
                ValidationErrorCode.INVALID_FORMAT,
                f"Invalid JSON format: {str(e)}",
                path=f"line {e.lineno}, column {e.colno}" if hasattr(e, 'lineno') else None
            )
            return None
        except UnicodeDecodeError as e:
            result.add_error(
                ValidationErrorCode.INVALID_FORMAT,
                f"Invalid character encoding: {str(e)}"
            )
            return None

    def _validate_metadata(self, data: Dict, result: ValidationResult) -> None:
        """Validate backup metadata."""
        # Check top-level structure
        if not isinstance(data, dict):
            result.add_error(
                ValidationErrorCode.INVALID_FORMAT,
                "Backup must be a JSON object",
                expected="object",
                actual=type(data).__name__
            )
            return

        # Check for metadata section
        if "metadata" not in data:
            result.add_error(
                ValidationErrorCode.MISSING_REQUIRED_FIELD,
                "Missing metadata section",
                path="metadata"
            )
            return

        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            result.add_error(
                ValidationErrorCode.INVALID_DATA_TYPE,
                "Metadata must be an object",
                path="metadata",
                expected="object",
                actual=type(metadata).__name__
            )
            return

        # Check required metadata fields
        for field_name in REQUIRED_METADATA_FIELDS:
            if field_name not in metadata:
                result.add_error(
                    ValidationErrorCode.MISSING_REQUIRED_FIELD,
                    f"Missing required metadata field: {field_name}",
                    path=f"metadata.{field_name}"
                )

        # Validate version
        version = metadata.get("version")
        if version:
            result.version = version
            is_valid, is_current, migration_needed = self.validate_version(version)

            if not is_valid:
                result.add_error(
                    ValidationErrorCode.VERSION_UNSUPPORTED,
                    f"Unsupported backup version: {version}",
                    path="metadata.version",
                    expected=f"one of {SUPPORTED_VERSIONS}",
                    actual=version
                )
            elif migration_needed:
                result.add_warning(
                    f"Backup version {version} may require migration to {CURRENT_SCHEMA_VERSION}"
                )

    def _validate_schema(self, data: Dict, result: ValidationResult) -> None:
        """Validate data schemas."""
        schema_errors = []

        # Validate goals
        if "goals" in data:
            goals_data = data["goals"]
            if isinstance(goals_data, dict):
                for goal in goals_data.get("active", []):
                    schema_errors.extend(self.validate_schema(goal, "goals"))
                for goal in goals_data.get("completed", []):
                    schema_errors.extend(self.validate_schema(goal, "goals"))
            elif isinstance(goals_data, list):
                for goal in goals_data:
                    schema_errors.extend(self.validate_schema(goal, "goals"))

        # Validate achievements
        if "achievements" in data:
            achievements_data = data["achievements"]
            if isinstance(achievements_data, dict):
                for achievement in achievements_data.get("all", []):
                    schema_errors.extend(self.validate_schema(achievement, "achievements"))
            elif isinstance(achievements_data, list):
                for achievement in achievements_data:
                    schema_errors.extend(self.validate_schema(achievement, "achievements"))

        # Validate feedback
        if "feedback" in data:
            feedback_data = data["feedback"]
            if isinstance(feedback_data, list):
                for fb in feedback_data:
                    schema_errors.extend(self.validate_schema(fb, "feedback"))

        # Validate conversations
        if "conversations" in data:
            conversations_data = data["conversations"]
            if isinstance(conversations_data, list):
                for conv in conversations_data:
                    schema_errors.extend(self.validate_schema(conv, "conversations"))

        # Validate settings
        if "settings" in data:
            schema_errors.extend(self.validate_schema(data["settings"], "settings"))

        # Add errors to result
        for error in schema_errors:
            result.errors.append(error)
            result.valid = False

        result.schema_valid = len(schema_errors) == 0

    def _check_data_integrity(self, data: Dict, result: ValidationResult) -> None:
        """Check data integrity."""
        integrity_errors = self.check_data_integrity(data)

        for error in integrity_errors:
            result.errors.append(error)
            result.valid = False

        result.data_integrity_valid = len(integrity_errors) == 0

    def _validate_timestamps(self, data: Dict, errors: List[ValidationError]) -> None:
        """Validate timestamp formats in data."""
        timestamp_pattern = re.compile(
            r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
        )

        def check_timestamp(value: str, path: str) -> None:
            if value and isinstance(value, str):
                if not timestamp_pattern.match(value):
                    # Try to parse as datetime
                    try:
                        datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        errors.append(ValidationError(
                            code=ValidationErrorCode.INVALID_TIMESTAMP,
                            message=f"Invalid timestamp format",
                            path=path,
                            actual=value
                        ))

        # Check metadata timestamps
        if "metadata" in data:
            metadata = data["metadata"]
            if "export_date" in metadata:
                check_timestamp(metadata["export_date"], "metadata.export_date")

        # Check goal timestamps
        if "goals" in data:
            goals_data = data["goals"]
            all_goals = []
            if isinstance(goals_data, dict):
                all_goals = goals_data.get("active", []) + goals_data.get("completed", [])
            elif isinstance(goals_data, list):
                all_goals = goals_data

            for i, goal in enumerate(all_goals):
                if isinstance(goal, dict):
                    if "created_at" in goal:
                        check_timestamp(goal["created_at"], f"goals[{i}].created_at")
                    if "completed_at" in goal:
                        check_timestamp(goal["completed_at"], f"goals[{i}].completed_at")

        # Check achievement timestamps
        if "achievements" in data:
            achievements_data = data["achievements"]
            all_achievements = []
            if isinstance(achievements_data, dict):
                all_achievements = achievements_data.get("all", [])
            elif isinstance(achievements_data, list):
                all_achievements = achievements_data

            for i, achievement in enumerate(all_achievements):
                if isinstance(achievement, dict) and "unlocked_at" in achievement:
                    check_timestamp(achievement["unlocked_at"], f"achievements[{i}].unlocked_at")


# Singleton instance
backup_validator = BackupValidator()
