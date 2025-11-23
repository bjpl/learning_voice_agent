"""
Sync Store - SQLite Persistence Layer for Sync Metadata
PATTERN: Repository pattern with async operations
WHY: Efficient data access for sync state and device management

Tables:
- sync_metadata: Sync state tracking per device
- devices: Registered device information
- sync_conflicts: Conflict history and resolution
- sync_events: Audit log of sync operations

Features:
- Async SQLite operations with aiosqlite
- Device registration and management
- Conflict tracking and resolution
- Sync history and audit trail
"""
import aiosqlite
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from app.sync.config import sync_config
from app.sync.models import (
    SyncMetadata,
    DeviceInfo,
    SyncConflict,
    SyncStatus,
    ConflictResolution,
)
from app.logger import get_logger

# Module logger
logger = get_logger("sync_store")


class SyncStore:
    """
    SQLite persistence layer for sync metadata and device management.

    PATTERN: Repository with async operations
    WHY: Centralized sync state management with audit trail
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the sync store.

        Args:
            db_path: Path to SQLite database. Uses config default if not provided.
        """
        self.db_path = db_path or sync_config.sync_database_path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection with row factory."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates all required tables and indexes if they don't exist.
        """
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("sync_store_initialization_started", db_path=self.db_path)

            try:
                async with aiosqlite.connect(self.db_path) as db:
                    # Sync metadata table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS sync_metadata (
                            id TEXT PRIMARY KEY,
                            device_id TEXT NOT NULL,
                            version TEXT NOT NULL,
                            checksum TEXT NOT NULL,
                            sync_status TEXT DEFAULT 'pending',
                            data_size_bytes INTEGER DEFAULT 0,
                            item_counts TEXT DEFAULT '{}',
                            last_sync DATETIME,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                            UNIQUE(device_id)
                        )
                    """)

                    # Devices table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS devices (
                            device_id TEXT PRIMARY KEY,
                            device_name TEXT NOT NULL,
                            platform TEXT NOT NULL,
                            platform_version TEXT,
                            app_version TEXT,
                            last_seen DATETIME,
                            last_sync_status TEXT DEFAULT 'pending',
                            registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            is_primary INTEGER DEFAULT 0,
                            push_enabled INTEGER DEFAULT 0,
                            metadata TEXT DEFAULT '{}'
                        )
                    """)

                    # Sync conflicts table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS sync_conflicts (
                            id TEXT PRIMARY KEY,
                            field TEXT NOT NULL,
                            item_type TEXT NOT NULL,
                            item_id TEXT NOT NULL,
                            local_value TEXT,
                            remote_value TEXT,
                            local_timestamp DATETIME,
                            remote_timestamp DATETIME,
                            resolved_value TEXT,
                            resolution TEXT,
                            resolved_at DATETIME,
                            auto_resolved INTEGER DEFAULT 0,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Sync events table (audit log)
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS sync_events (
                            id TEXT PRIMARY KEY,
                            device_id TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            event_data TEXT DEFAULT '{}',
                            success INTEGER DEFAULT 1,
                            error_message TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Create indexes
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_sync_metadata_device
                        ON sync_metadata(device_id)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_sync_metadata_last_sync
                        ON sync_metadata(last_sync DESC)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_devices_last_seen
                        ON devices(last_seen DESC)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_sync_conflicts_item
                        ON sync_conflicts(item_type, item_id)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_sync_conflicts_unresolved
                        ON sync_conflicts(resolved_at) WHERE resolved_at IS NULL
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_sync_events_device
                        ON sync_events(device_id, timestamp DESC)
                    """)

                    await db.commit()

                self._initialized = True
                logger.info("sync_store_initialization_complete", db_path=self.db_path)

            except Exception as e:
                logger.error(
                    "sync_store_initialization_failed",
                    db_path=self.db_path,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise

    # ========================================================================
    # SYNC METADATA OPERATIONS
    # ========================================================================

    async def save_sync_metadata(self, metadata: SyncMetadata) -> str:
        """
        Save or update sync metadata for a device.

        Args:
            metadata: SyncMetadata model instance

        Returns:
            Metadata ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO sync_metadata
                    (id, device_id, version, checksum, sync_status, data_size_bytes,
                     item_counts, last_sync, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.id,
                    metadata.device_id,
                    metadata.version,
                    metadata.checksum,
                    metadata.sync_status.value,
                    metadata.data_size_bytes,
                    json.dumps(metadata.item_counts),
                    metadata.last_sync.isoformat(),
                    datetime.utcnow().isoformat()
                ))
                await db.commit()

            logger.debug(
                "sync_metadata_saved",
                device_id=metadata.device_id,
                checksum=metadata.checksum[:16] + "..."
            )
            return metadata.id

        except Exception as e:
            logger.error(
                "save_sync_metadata_failed",
                device_id=metadata.device_id,
                error=str(e)
            )
            raise

    async def get_sync_metadata(self, device_id: str) -> Optional[SyncMetadata]:
        """
        Get sync metadata for a device.

        Args:
            device_id: Device identifier

        Returns:
            SyncMetadata instance or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM sync_metadata WHERE device_id = ?
                """, (device_id,))
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_sync_metadata(row)

        except Exception as e:
            logger.error(
                "get_sync_metadata_failed",
                device_id=device_id,
                error=str(e)
            )
            return None

    async def get_last_sync(self, device_id: str) -> Optional[datetime]:
        """
        Get the timestamp of the last successful sync for a device.

        Args:
            device_id: Device identifier

        Returns:
            Last sync datetime or None if never synced
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT last_sync FROM sync_metadata
                    WHERE device_id = ? AND sync_status = 'completed'
                    ORDER BY last_sync DESC
                    LIMIT 1
                """, (device_id,))
                row = await cursor.fetchone()

                if not row or not row['last_sync']:
                    return None

                return datetime.fromisoformat(row['last_sync'])

        except Exception as e:
            logger.error(
                "get_last_sync_failed",
                device_id=device_id,
                error=str(e)
            )
            return None

    async def update_sync_status(
        self,
        device_id: str,
        status: SyncStatus,
        checksum: Optional[str] = None
    ) -> bool:
        """
        Update sync status for a device.

        Args:
            device_id: Device identifier
            status: New sync status
            checksum: Optional new checksum

        Returns:
            True if updated successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                if checksum:
                    await db.execute("""
                        UPDATE sync_metadata
                        SET sync_status = ?, checksum = ?, updated_at = ?,
                            last_sync = CASE WHEN ? = 'completed' THEN ? ELSE last_sync END
                        WHERE device_id = ?
                    """, (
                        status.value,
                        checksum,
                        datetime.utcnow().isoformat(),
                        status.value,
                        datetime.utcnow().isoformat(),
                        device_id
                    ))
                else:
                    await db.execute("""
                        UPDATE sync_metadata
                        SET sync_status = ?, updated_at = ?,
                            last_sync = CASE WHEN ? = 'completed' THEN ? ELSE last_sync END
                        WHERE device_id = ?
                    """, (
                        status.value,
                        datetime.utcnow().isoformat(),
                        status.value,
                        datetime.utcnow().isoformat(),
                        device_id
                    ))
                await db.commit()

            logger.debug(
                "sync_status_updated",
                device_id=device_id,
                status=status.value
            )
            return True

        except Exception as e:
            logger.error(
                "update_sync_status_failed",
                device_id=device_id,
                error=str(e)
            )
            return False

    # ========================================================================
    # DEVICE OPERATIONS
    # ========================================================================

    async def register_device(self, device: DeviceInfo) -> str:
        """
        Register a new device or update existing device info.

        Args:
            device: DeviceInfo model instance

        Returns:
            Device ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                # Check device count limit
                cursor = await db.execute(
                    "SELECT COUNT(*) as count FROM devices"
                )
                row = await cursor.fetchone()

                if row['count'] >= sync_config.max_devices_per_user:
                    # Check if this is an update to existing device
                    cursor = await db.execute(
                        "SELECT 1 FROM devices WHERE device_id = ?",
                        (device.device_id,)
                    )
                    if not await cursor.fetchone():
                        raise ValueError(
                            f"Maximum device limit ({sync_config.max_devices_per_user}) reached"
                        )

                await db.execute("""
                    INSERT OR REPLACE INTO devices
                    (device_id, device_name, platform, platform_version, app_version,
                     last_seen, last_sync_status, registered_at, is_primary,
                     push_enabled, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    device.device_id,
                    device.device_name,
                    device.platform,
                    device.platform_version,
                    device.app_version,
                    device.last_seen.isoformat(),
                    device.last_sync_status.value,
                    device.registered_at.isoformat(),
                    1 if device.is_primary else 0,
                    1 if device.push_enabled else 0,
                    json.dumps(device.metadata or {})
                ))
                await db.commit()

            logger.info(
                "device_registered",
                device_id=device.device_id,
                device_name=device.device_name,
                platform=device.platform
            )
            return device.device_id

        except Exception as e:
            logger.error(
                "register_device_failed",
                device_id=device.device_id,
                error=str(e)
            )
            raise

    async def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """
        Get device information by ID.

        Args:
            device_id: Device identifier

        Returns:
            DeviceInfo instance or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM devices WHERE device_id = ?
                """, (device_id,))
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_device_info(row)

        except Exception as e:
            logger.error(
                "get_device_failed",
                device_id=device_id,
                error=str(e)
            )
            return None

    async def list_devices(self, limit: int = 50) -> List[DeviceInfo]:
        """
        List all registered devices.

        Args:
            limit: Maximum number of devices to return

        Returns:
            List of DeviceInfo instances
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM devices
                    ORDER BY last_seen DESC
                    LIMIT ?
                """, (limit,))
                rows = await cursor.fetchall()

                return [self._row_to_device_info(row) for row in rows]

        except Exception as e:
            logger.error("list_devices_failed", error=str(e))
            return []

    async def update_device_last_seen(
        self,
        device_id: str,
        status: SyncStatus = SyncStatus.COMPLETED
    ) -> bool:
        """
        Update device last seen timestamp.

        Args:
            device_id: Device identifier
            status: Sync status

        Returns:
            True if updated successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    UPDATE devices
                    SET last_seen = ?, last_sync_status = ?
                    WHERE device_id = ?
                """, (
                    datetime.utcnow().isoformat(),
                    status.value,
                    device_id
                ))
                await db.commit()

            return True

        except Exception as e:
            logger.error(
                "update_device_last_seen_failed",
                device_id=device_id,
                error=str(e)
            )
            return False

    async def remove_device(self, device_id: str) -> bool:
        """
        Remove a device registration.

        Args:
            device_id: Device identifier

        Returns:
            True if removed successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute(
                    "DELETE FROM devices WHERE device_id = ?",
                    (device_id,)
                )
                await db.execute(
                    "DELETE FROM sync_metadata WHERE device_id = ?",
                    (device_id,)
                )
                await db.commit()

            logger.info("device_removed", device_id=device_id)
            return True

        except Exception as e:
            logger.error(
                "remove_device_failed",
                device_id=device_id,
                error=str(e)
            )
            return False

    async def set_primary_device(self, device_id: str) -> bool:
        """
        Set a device as the primary device.

        Args:
            device_id: Device identifier

        Returns:
            True if set successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                # Clear existing primary
                await db.execute("UPDATE devices SET is_primary = 0")
                # Set new primary
                await db.execute(
                    "UPDATE devices SET is_primary = 1 WHERE device_id = ?",
                    (device_id,)
                )
                await db.commit()

            logger.info("primary_device_set", device_id=device_id)
            return True

        except Exception as e:
            logger.error(
                "set_primary_device_failed",
                device_id=device_id,
                error=str(e)
            )
            return False

    # ========================================================================
    # CONFLICT OPERATIONS
    # ========================================================================

    async def save_conflict(self, conflict: SyncConflict) -> str:
        """
        Save a sync conflict.

        Args:
            conflict: SyncConflict model instance

        Returns:
            Conflict ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO sync_conflicts
                    (id, field, item_type, item_id, local_value, remote_value,
                     local_timestamp, remote_timestamp, resolved_value, resolution,
                     resolved_at, auto_resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conflict.id,
                    conflict.field,
                    conflict.item_type,
                    conflict.item_id,
                    json.dumps(conflict.local_value),
                    json.dumps(conflict.remote_value),
                    conflict.local_timestamp.isoformat() if conflict.local_timestamp else None,
                    conflict.remote_timestamp.isoformat() if conflict.remote_timestamp else None,
                    json.dumps(conflict.resolved_value) if conflict.resolved_value else None,
                    conflict.resolution.value if conflict.resolution else None,
                    conflict.resolved_at.isoformat() if conflict.resolved_at else None,
                    1 if conflict.auto_resolved else 0
                ))
                await db.commit()

            logger.debug(
                "conflict_saved",
                conflict_id=conflict.id,
                item_type=conflict.item_type,
                field=conflict.field
            )
            return conflict.id

        except Exception as e:
            logger.error("save_conflict_failed", error=str(e))
            raise

    async def get_unresolved_conflicts(self, limit: int = 100) -> List[SyncConflict]:
        """
        Get all unresolved conflicts.

        Args:
            limit: Maximum number of conflicts to return

        Returns:
            List of unresolved SyncConflict instances
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM sync_conflicts
                    WHERE resolved_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                rows = await cursor.fetchall()

                return [self._row_to_sync_conflict(row) for row in rows]

        except Exception as e:
            logger.error("get_unresolved_conflicts_failed", error=str(e))
            return []

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolved_value: Any,
        resolution: ConflictResolution,
        auto: bool = False
    ) -> bool:
        """
        Resolve a sync conflict.

        Args:
            conflict_id: Conflict identifier
            resolved_value: The resolved value
            resolution: How the conflict was resolved
            auto: Whether it was auto-resolved

        Returns:
            True if resolved successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    UPDATE sync_conflicts
                    SET resolved_value = ?, resolution = ?, resolved_at = ?, auto_resolved = ?
                    WHERE id = ?
                """, (
                    json.dumps(resolved_value),
                    resolution.value,
                    datetime.utcnow().isoformat(),
                    1 if auto else 0,
                    conflict_id
                ))
                await db.commit()

            logger.info(
                "conflict_resolved",
                conflict_id=conflict_id,
                resolution=resolution.value
            )
            return True

        except Exception as e:
            logger.error(
                "resolve_conflict_failed",
                conflict_id=conflict_id,
                error=str(e)
            )
            return False

    # ========================================================================
    # SYNC EVENT OPERATIONS
    # ========================================================================

    async def log_sync_event(
        self,
        device_id: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log a sync event for audit trail.

        Args:
            device_id: Device identifier
            event_type: Type of sync event
            event_data: Optional event data
            success: Whether the event was successful
            error_message: Error message if failed

        Returns:
            Event ID
        """
        if not self._initialized:
            await self.initialize()

        event_id = str(uuid.uuid4())

        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO sync_events
                    (id, device_id, event_type, event_data, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    device_id,
                    event_type,
                    json.dumps(event_data or {}),
                    1 if success else 0,
                    error_message
                ))
                await db.commit()

            return event_id

        except Exception as e:
            logger.error("log_sync_event_failed", error=str(e))
            return event_id

    async def get_sync_history(
        self,
        device_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get sync event history for a device.

        Args:
            device_id: Device identifier
            limit: Maximum number of events to return

        Returns:
            List of sync events
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM sync_events
                    WHERE device_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (device_id, limit))
                rows = await cursor.fetchall()

                return [
                    {
                        "id": row['id'],
                        "device_id": row['device_id'],
                        "event_type": row['event_type'],
                        "event_data": json.loads(row['event_data'] or '{}'),
                        "success": bool(row['success']),
                        "error_message": row['error_message'],
                        "timestamp": row['timestamp']
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error("get_sync_history_failed", error=str(e))
            return []

    # ========================================================================
    # ROW CONVERSION HELPERS
    # ========================================================================

    def _row_to_sync_metadata(self, row) -> SyncMetadata:
        """Convert database row to SyncMetadata model."""
        return SyncMetadata(
            id=row['id'],
            device_id=row['device_id'],
            version=row['version'],
            checksum=row['checksum'],
            sync_status=SyncStatus(row['sync_status']),
            data_size_bytes=row['data_size_bytes'],
            item_counts=json.loads(row['item_counts'] or '{}'),
            last_sync=datetime.fromisoformat(row['last_sync']) if row['last_sync'] else datetime.utcnow(),
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.utcnow(),
        )

    def _row_to_device_info(self, row) -> DeviceInfo:
        """Convert database row to DeviceInfo model."""
        return DeviceInfo(
            device_id=row['device_id'],
            device_name=row['device_name'],
            platform=row['platform'],
            platform_version=row['platform_version'],
            app_version=row['app_version'],
            last_seen=datetime.fromisoformat(row['last_seen']) if row['last_seen'] else datetime.utcnow(),
            last_sync_status=SyncStatus(row['last_sync_status']),
            registered_at=datetime.fromisoformat(row['registered_at']) if row['registered_at'] else datetime.utcnow(),
            is_primary=bool(row['is_primary']),
            push_enabled=bool(row['push_enabled']),
            metadata=json.loads(row['metadata'] or '{}'),
        )

    def _row_to_sync_conflict(self, row) -> SyncConflict:
        """Convert database row to SyncConflict model."""
        return SyncConflict(
            id=row['id'],
            field=row['field'],
            item_type=row['item_type'],
            item_id=row['item_id'],
            local_value=json.loads(row['local_value']) if row['local_value'] else None,
            remote_value=json.loads(row['remote_value']) if row['remote_value'] else None,
            local_timestamp=datetime.fromisoformat(row['local_timestamp']) if row['local_timestamp'] else None,
            remote_timestamp=datetime.fromisoformat(row['remote_timestamp']) if row['remote_timestamp'] else None,
            resolved_value=json.loads(row['resolved_value']) if row['resolved_value'] else None,
            resolution=ConflictResolution(row['resolution']) if row['resolution'] else None,
            resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
            auto_resolved=bool(row['auto_resolved']),
        )

    async def close(self) -> None:
        """Close the store (cleanup if needed)."""
        self._initialized = False
        logger.info("sync_store_closed", db_path=self.db_path)


# Global store instance
sync_store = SyncStore()
