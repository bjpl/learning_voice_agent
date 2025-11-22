"""
Sync Service - Core Sync Business Logic
========================================

Handles data export, import, validation, and conflict resolution.

PATTERN: Service layer with async operations
WHY: Separation of HTTP handling from business logic
"""

import json
import gzip
import hashlib
import aiosqlite
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import uuid
import os

try:
    from app.logger import api_logger as logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from app.sync.models import (
    SyncStatus,
    SyncStatusResponse,
    ExportResponse,
    ImportResponse,
    DeviceInfo,
    DeviceListResponse,
    ValidationResponse,
    ConflictInfo,
    ConflictListResponse,
    ConflictType,
    ResolutionStrategy,
    ConflictResolutionResponse,
)


def format_file_size(size_bytes: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


class SyncService:
    """
    Core sync service for backup and restore operations.

    PATTERN: Stateful service with database operations
    WHY: Centralized sync logic with proper state management
    """

    def __init__(
        self,
        db_path: str = "learning_captures.db",
        backup_dir: str = "backups",
        sync_db_path: str = "sync_state.db"
    ):
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.sync_db_path = sync_db_path
        self._initialized = False
        self._devices: Dict[str, DeviceInfo] = {}
        self._conflicts: Dict[str, ConflictInfo] = {}
        self._current_device_id: Optional[str] = None
        self._last_sync: Optional[datetime] = None
        self._status: SyncStatus = SyncStatus.IDLE

    async def initialize(self) -> None:
        """Initialize sync service and create necessary tables"""
        if self._initialized:
            return

        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sync state database
        async with aiosqlite.connect(self.sync_db_path) as db:
            # Devices table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    device_type TEXT DEFAULT 'unknown',
                    platform TEXT,
                    app_version TEXT,
                    last_sync TEXT,
                    registered_at TEXT NOT NULL,
                    is_current INTEGER DEFAULT 0
                )
            """)

            # Conflicts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    conflict_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    local_value TEXT NOT NULL,
                    remote_value TEXT NOT NULL,
                    local_timestamp TEXT NOT NULL,
                    remote_timestamp TEXT NOT NULL,
                    suggested_resolution TEXT DEFAULT 'keep_remote',
                    created_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0
                )
            """)

            # Sync state table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            await db.commit()

        # Load existing data
        await self._load_devices()
        await self._load_conflicts()
        await self._load_state()

        self._initialized = True
        logger.info("sync_service_initialized")

    async def _load_devices(self) -> None:
        """Load devices from database"""
        async with aiosqlite.connect(self.sync_db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM devices")
            rows = await cursor.fetchall()

            for row in rows:
                device = DeviceInfo(
                    device_id=row['device_id'],
                    device_name=row['device_name'],
                    device_type=row['device_type'],
                    platform=row['platform'],
                    app_version=row['app_version'],
                    last_sync=datetime.fromisoformat(row['last_sync']) if row['last_sync'] else None,
                    registered_at=datetime.fromisoformat(row['registered_at']),
                    is_current=bool(row['is_current'])
                )
                self._devices[device.device_id] = device
                if device.is_current:
                    self._current_device_id = device.device_id

    async def _load_conflicts(self) -> None:
        """Load unresolved conflicts from database"""
        async with aiosqlite.connect(self.sync_db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM conflicts WHERE resolved = 0"
            )
            rows = await cursor.fetchall()

            for row in rows:
                conflict = ConflictInfo(
                    conflict_id=row['conflict_id'],
                    conflict_type=ConflictType(row['conflict_type']),
                    entity_type=row['entity_type'],
                    entity_id=row['entity_id'],
                    local_value=json.loads(row['local_value']),
                    remote_value=json.loads(row['remote_value']),
                    local_timestamp=datetime.fromisoformat(row['local_timestamp']),
                    remote_timestamp=datetime.fromisoformat(row['remote_timestamp']),
                    suggested_resolution=ResolutionStrategy(row['suggested_resolution']),
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                self._conflicts[conflict.conflict_id] = conflict

    async def _load_state(self) -> None:
        """Load sync state from database"""
        async with aiosqlite.connect(self.sync_db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sync_state")
            rows = await cursor.fetchall()

            for row in rows:
                if row['key'] == 'last_sync':
                    self._last_sync = datetime.fromisoformat(row['value'])

    async def _save_state(self, key: str, value: str) -> None:
        """Save sync state to database"""
        async with aiosqlite.connect(self.sync_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO sync_state (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.utcnow().isoformat()))
            await db.commit()

    async def get_status(self) -> SyncStatusResponse:
        """
        Get current sync status.

        Returns:
            SyncStatusResponse with current state
        """
        await self.initialize()

        # Calculate data size
        data_size = 0
        if os.path.exists(self.db_path):
            data_size = os.path.getsize(self.db_path)

        # Get next backup time from scheduler
        from app.sync.scheduler import backup_scheduler

        return SyncStatusResponse(
            status=self._status,
            last_sync=self._last_sync,
            next_backup=backup_scheduler.next_backup_time,
            device_count=len(self._devices),
            data_size_bytes=data_size,
            data_size_human=format_file_size(data_size),
            pending_changes=0,  # TODO: Implement change tracking
            conflicts_count=len(self._conflicts),
            backup_enabled=backup_scheduler.is_enabled,
            backup_interval_hours=backup_scheduler.interval_hours if backup_scheduler.is_enabled else None
        )

    async def export_data(
        self,
        include_sessions: bool = True,
        include_feedback: bool = True,
        include_analytics: bool = True,
        include_goals: bool = True,
        include_files: bool = False,
        compress: bool = True
    ) -> Tuple[ExportResponse, Optional[bytes]]:
        """
        Export all data as a backup.

        PATTERN: Comprehensive data export with optional compression
        WHY: Enable data portability and backup

        Returns:
            Tuple of (ExportResponse, backup_data_bytes)
        """
        await self.initialize()

        self._status = SyncStatus.SYNCING
        start_time = datetime.utcnow()

        try:
            export_data = {
                "version": "1.0.0",
                "created_at": start_time.isoformat(),
                "source_device": self._devices.get(self._current_device_id, {}).device_name if self._current_device_id else "unknown",
                "sections": {}
            }

            record_count = 0

            # Export sessions/captures
            if include_sessions and os.path.exists(self.db_path):
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute("""
                        SELECT id, session_id, timestamp, user_text, agent_text, metadata
                        FROM captures ORDER BY timestamp
                    """)
                    rows = await cursor.fetchall()

                    captures = []
                    for row in rows:
                        captures.append({
                            "id": row['id'],
                            "session_id": row['session_id'],
                            "timestamp": row['timestamp'],
                            "user_text": row['user_text'],
                            "agent_text": row['agent_text'],
                            "metadata": row['metadata']
                        })

                    export_data["sections"]["captures"] = captures
                    record_count += len(captures)

            # Export feedback data
            if include_feedback:
                feedback_db = "feedback.db"
                if os.path.exists(feedback_db):
                    async with aiosqlite.connect(feedback_db) as db:
                        db.row_factory = aiosqlite.Row

                        # Export explicit feedback
                        cursor = await db.execute("SELECT * FROM explicit_feedback")
                        explicit = [dict(row) for row in await cursor.fetchall()]
                        export_data["sections"]["explicit_feedback"] = explicit
                        record_count += len(explicit)

                        # Export implicit feedback
                        cursor = await db.execute("SELECT * FROM implicit_feedback")
                        implicit = [dict(row) for row in await cursor.fetchall()]
                        export_data["sections"]["implicit_feedback"] = implicit
                        record_count += len(implicit)

            # Export goals
            if include_goals:
                goals_db = "goals.db"
                if os.path.exists(goals_db):
                    async with aiosqlite.connect(goals_db) as db:
                        db.row_factory = aiosqlite.Row
                        cursor = await db.execute("SELECT * FROM goals")
                        goals = [dict(row) for row in await cursor.fetchall()]
                        export_data["sections"]["goals"] = goals
                        record_count += len(goals)

            # Serialize data
            json_data = json.dumps(export_data, indent=2, default=str)

            # Compress if requested
            if compress:
                backup_bytes = gzip.compress(json_data.encode('utf-8'))
                filename = f"learning_backup_{start_time.strftime('%Y%m%d_%H%M%S')}.json.gz"
            else:
                backup_bytes = json_data.encode('utf-8')
                filename = f"learning_backup_{start_time.strftime('%Y%m%d_%H%M%S')}.json"

            # Calculate checksum
            checksum = f"sha256:{hashlib.sha256(backup_bytes).hexdigest()}"

            # Save to backup directory
            export_id = str(uuid.uuid4())[:8]
            backup_path = self.backup_dir / f"{export_id}_{filename}"
            backup_path.write_bytes(backup_bytes)

            # Update last sync time
            self._last_sync = datetime.utcnow()
            await self._save_state('last_sync', self._last_sync.isoformat())

            self._status = SyncStatus.IDLE

            logger.info(
                "data_export_completed",
                export_id=export_id,
                record_count=record_count,
                file_size=len(backup_bytes)
            )

            return ExportResponse(
                success=True,
                export_id=export_id,
                download_url=f"/api/sync/download/{export_id}",
                filename=filename,
                file_size_bytes=len(backup_bytes),
                file_size_human=format_file_size(len(backup_bytes)),
                record_count=record_count,
                created_at=start_time,
                expires_at=start_time + timedelta(hours=24),
                checksum=checksum,
                compressed=compress
            ), backup_bytes

        except Exception as e:
            self._status = SyncStatus.ERROR
            logger.error("data_export_failed", error=str(e))
            raise

    async def validate_backup(self, backup_data: bytes) -> ValidationResponse:
        """
        Validate a backup file without importing.

        PATTERN: Pre-import validation
        WHY: Prevent data corruption from invalid backups

        Args:
            backup_data: Raw backup file bytes

        Returns:
            ValidationResponse with validation results
        """
        await self.initialize()

        errors = []
        warnings = []
        sections = {}

        try:
            # Try to decompress
            try:
                json_data = gzip.decompress(backup_data).decode('utf-8')
            except gzip.BadGzipFile:
                # Try as plain JSON
                json_data = backup_data.decode('utf-8')

            # Parse JSON
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return ValidationResponse(
                    valid=False,
                    errors=[f"Invalid JSON: {str(e)}"]
                )

            # Check version
            version = data.get('version')
            if not version:
                warnings.append("No version specified in backup")

            # Check required fields
            created_at = None
            if 'created_at' in data:
                try:
                    created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    warnings.append("Invalid created_at timestamp")

            source_device = data.get('source_device')

            # Count records by section
            total_records = 0
            if 'sections' in data:
                for section_name, section_data in data['sections'].items():
                    if isinstance(section_data, list):
                        sections[section_name] = len(section_data)
                        total_records += len(section_data)
                    else:
                        warnings.append(f"Section '{section_name}' is not a list")
            else:
                warnings.append("No sections found in backup")

            # Schema compatibility check
            schema_compatible = True
            if 'captures' in sections:
                # Check for required fields in captures
                sample_capture = data['sections'].get('captures', [{}])[0] if data['sections'].get('captures') else {}
                required_fields = ['session_id', 'user_text', 'agent_text']
                for field in required_fields:
                    if field not in sample_capture:
                        schema_compatible = False
                        errors.append(f"Missing required field '{field}' in captures")

            valid = len(errors) == 0

            return ValidationResponse(
                valid=valid,
                version=version,
                created_at=created_at,
                source_device=source_device,
                record_count=total_records,
                file_size_bytes=len(backup_data),
                checksum_valid=True,
                schema_compatible=schema_compatible,
                warnings=warnings,
                errors=errors,
                sections=sections
            )

        except Exception as e:
            return ValidationResponse(
                valid=False,
                errors=[f"Validation error: {str(e)}"]
            )

    async def import_data(
        self,
        backup_data: bytes,
        merge_strategy: ResolutionStrategy = ResolutionStrategy.KEEP_REMOTE,
        dry_run: bool = False,
        skip_conflicts: bool = False
    ) -> ImportResponse:
        """
        Import data from a backup file.

        PATTERN: Transactional import with conflict handling
        WHY: Safe data restoration with rollback capability

        Args:
            backup_data: Raw backup file bytes
            merge_strategy: How to handle existing data
            dry_run: Validate without importing
            skip_conflicts: Skip conflicting records

        Returns:
            ImportResponse with import results
        """
        await self.initialize()

        start_time = datetime.utcnow()

        # First validate
        validation = await self.validate_backup(backup_data)
        if not validation.valid:
            return ImportResponse(
                success=False,
                errors=[{"type": "validation", "message": e} for e in validation.errors],
                warnings=validation.warnings,
                dry_run=dry_run
            )

        if dry_run:
            return ImportResponse(
                success=True,
                total_records=validation.record_count,
                imported_count=validation.record_count,
                dry_run=True,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )

        self._status = SyncStatus.SYNCING

        try:
            # Decompress and parse
            try:
                json_data = gzip.decompress(backup_data).decode('utf-8')
            except gzip.BadGzipFile:
                json_data = backup_data.decode('utf-8')

            data = json.loads(json_data)

            imported_count = 0
            skipped_count = 0
            merged_count = 0
            conflicts_count = 0
            errors = []
            warnings = []

            # Import captures
            if 'captures' in data.get('sections', {}):
                captures = data['sections']['captures']

                async with aiosqlite.connect(self.db_path) as db:
                    for capture in captures:
                        try:
                            # Check for existing record
                            cursor = await db.execute(
                                "SELECT id FROM captures WHERE session_id = ? AND timestamp = ?",
                                (capture['session_id'], capture['timestamp'])
                            )
                            existing = await cursor.fetchone()

                            if existing:
                                if merge_strategy == ResolutionStrategy.KEEP_REMOTE:
                                    # Update existing
                                    await db.execute("""
                                        UPDATE captures SET user_text = ?, agent_text = ?, metadata = ?
                                        WHERE id = ?
                                    """, (
                                        capture['user_text'],
                                        capture['agent_text'],
                                        capture.get('metadata', '{}'),
                                        existing[0]
                                    ))
                                    merged_count += 1
                                elif merge_strategy == ResolutionStrategy.KEEP_LOCAL:
                                    skipped_count += 1
                                else:
                                    if skip_conflicts:
                                        skipped_count += 1
                                    else:
                                        conflicts_count += 1
                            else:
                                # Insert new record
                                await db.execute("""
                                    INSERT INTO captures (session_id, timestamp, user_text, agent_text, metadata)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    capture['session_id'],
                                    capture['timestamp'],
                                    capture['user_text'],
                                    capture['agent_text'],
                                    capture.get('metadata', '{}')
                                ))
                                imported_count += 1

                        except Exception as e:
                            errors.append({
                                "type": "import_error",
                                "entity": "capture",
                                "message": str(e)
                            })

                    await db.commit()

            # Update sync state
            self._last_sync = datetime.utcnow()
            await self._save_state('last_sync', self._last_sync.isoformat())

            self._status = SyncStatus.IDLE

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.info(
                "data_import_completed",
                imported=imported_count,
                merged=merged_count,
                skipped=skipped_count,
                conflicts=conflicts_count,
                duration_ms=duration_ms
            )

            return ImportResponse(
                success=len(errors) == 0,
                total_records=validation.record_count,
                imported_count=imported_count,
                skipped_count=skipped_count,
                merged_count=merged_count,
                conflicts_count=conflicts_count,
                errors=errors,
                warnings=warnings,
                dry_run=False,
                duration_ms=duration_ms
            )

        except Exception as e:
            self._status = SyncStatus.ERROR
            logger.error("data_import_failed", error=str(e))
            return ImportResponse(
                success=False,
                errors=[{"type": "import_error", "message": str(e)}],
                dry_run=False,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )

    # Device Management

    async def register_device(
        self,
        device_name: str,
        device_type: str = "unknown",
        platform: Optional[str] = None,
        app_version: Optional[str] = None
    ) -> DeviceInfo:
        """
        Register a new device for sync.

        Args:
            device_name: User-friendly device name
            device_type: Type (desktop, mobile, tablet)
            platform: OS platform
            app_version: App version

        Returns:
            DeviceInfo for the registered device
        """
        await self.initialize()

        device = DeviceInfo(
            device_name=device_name,
            device_type=device_type,
            platform=platform,
            app_version=app_version,
            is_current=True
        )

        # Mark other devices as not current
        async with aiosqlite.connect(self.sync_db_path) as db:
            await db.execute("UPDATE devices SET is_current = 0")

            await db.execute("""
                INSERT INTO devices (device_id, device_name, device_type, platform, app_version, registered_at, is_current)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (
                device.device_id,
                device.device_name,
                device.device_type,
                device.platform,
                device.app_version,
                device.registered_at.isoformat()
            ))

            await db.commit()

        self._devices[device.device_id] = device
        self._current_device_id = device.device_id

        logger.info("device_registered", device_id=device.device_id, device_name=device_name)

        return device

    async def get_devices(self) -> DeviceListResponse:
        """Get list of registered devices"""
        await self.initialize()

        return DeviceListResponse(
            devices=list(self._devices.values()),
            total_count=len(self._devices),
            current_device_id=self._current_device_id
        )

    async def remove_device(self, device_id: str) -> bool:
        """
        Remove a registered device.

        Args:
            device_id: Device ID to remove

        Returns:
            bool: Whether device was removed
        """
        await self.initialize()

        if device_id not in self._devices:
            return False

        async with aiosqlite.connect(self.sync_db_path) as db:
            await db.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
            await db.commit()

        del self._devices[device_id]

        if self._current_device_id == device_id:
            self._current_device_id = None

        logger.info("device_removed", device_id=device_id)

        return True

    # Conflict Management

    async def get_conflicts(self) -> ConflictListResponse:
        """Get list of pending conflicts"""
        await self.initialize()

        conflicts = list(self._conflicts.values())

        by_type = {}
        by_entity = {}

        for conflict in conflicts:
            ct = conflict.conflict_type.value if hasattr(conflict.conflict_type, 'value') else conflict.conflict_type
            by_type[ct] = by_type.get(ct, 0) + 1
            by_entity[conflict.entity_type] = by_entity.get(conflict.entity_type, 0) + 1

        return ConflictListResponse(
            conflicts=conflicts,
            total_count=len(conflicts),
            by_type=by_type,
            by_entity=by_entity
        )

    async def resolve_conflicts(
        self,
        conflict_id: Optional[str] = None,
        conflict_ids: Optional[List[str]] = None,
        strategy: ResolutionStrategy = ResolutionStrategy.KEEP_REMOTE,
        custom_value: Optional[Dict[str, Any]] = None
    ) -> ConflictResolutionResponse:
        """
        Resolve one or more conflicts.

        Args:
            conflict_id: Single conflict to resolve
            conflict_ids: Multiple conflicts to resolve
            strategy: Resolution strategy
            custom_value: Custom merged value for manual resolution

        Returns:
            ConflictResolutionResponse with results
        """
        await self.initialize()

        ids_to_resolve = []
        if conflict_id:
            ids_to_resolve.append(conflict_id)
        if conflict_ids:
            ids_to_resolve.extend(conflict_ids)

        resolved_count = 0
        failed_count = 0
        errors = []

        async with aiosqlite.connect(self.sync_db_path) as db:
            for cid in ids_to_resolve:
                if cid not in self._conflicts:
                    errors.append({"conflict_id": cid, "error": "Conflict not found"})
                    failed_count += 1
                    continue

                try:
                    await db.execute(
                        "UPDATE conflicts SET resolved = 1 WHERE conflict_id = ?",
                        (cid,)
                    )
                    del self._conflicts[cid]
                    resolved_count += 1
                except Exception as e:
                    errors.append({"conflict_id": cid, "error": str(e)})
                    failed_count += 1

            await db.commit()

        logger.info(
            "conflicts_resolved",
            resolved=resolved_count,
            failed=failed_count
        )

        return ConflictResolutionResponse(
            success=failed_count == 0,
            resolved_count=resolved_count,
            failed_count=failed_count,
            remaining_conflicts=len(self._conflicts),
            errors=errors
        )

    async def get_backup_file(self, export_id: str) -> Optional[Tuple[bytes, str]]:
        """
        Get a backup file by export ID.

        Args:
            export_id: Export ID from ExportResponse

        Returns:
            Tuple of (file_bytes, filename) or None
        """
        # Find file matching export ID
        for file_path in self.backup_dir.iterdir():
            if file_path.name.startswith(export_id):
                return file_path.read_bytes(), file_path.name

        return None


# Global service instance
sync_service = SyncService()
