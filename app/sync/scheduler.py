"""
Backup Scheduler - Auto-backup Scheduling
==========================================

Handles automatic backup scheduling and execution.

PATTERN: Background task scheduling
WHY: Ensure data is regularly backed up without user intervention
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
import logging

# Use standard logging if app logger not available
try:
    from app.logger import api_logger as logger
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class BackupScheduler:
    """
    Manages automatic backup scheduling.

    PATTERN: Singleton scheduler with async task management
    WHY: Centralized backup scheduling across the application
    """
    _interval_hours: int = 12
    _enabled: bool = False
    _next_backup: Optional[datetime] = None
    _last_backup: Optional[datetime] = None
    _task: Optional[asyncio.Task] = None
    _backup_callback: Optional[Callable[[], Awaitable[bool]]] = None
    _running: bool = False

    def __post_init__(self):
        """Initialize scheduler state"""
        self._lock = asyncio.Lock()

    @property
    def is_enabled(self) -> bool:
        """Check if auto-backup is enabled"""
        return self._enabled

    @property
    def interval_hours(self) -> int:
        """Get current backup interval in hours"""
        return self._interval_hours

    @property
    def next_backup_time(self) -> Optional[datetime]:
        """Get next scheduled backup time"""
        return self._next_backup

    @property
    def last_backup_time(self) -> Optional[datetime]:
        """Get last successful backup time"""
        return self._last_backup

    def set_backup_callback(self, callback: Callable[[], Awaitable[bool]]) -> None:
        """
        Set the callback function for backup execution.

        Args:
            callback: Async function that performs the backup and returns success status
        """
        self._backup_callback = callback

    async def schedule_auto_backup(
        self,
        interval_hours: int = 12,
        enabled: bool = True
    ) -> dict:
        """
        Schedule automatic backups at specified interval.

        PATTERN: Configurable interval scheduling
        WHY: Allow users to customize backup frequency

        Args:
            interval_hours: Hours between backups (1-168)
            enabled: Whether to enable auto-backup

        Returns:
            dict with scheduling status
        """
        async with self._lock:
            # Validate interval
            if interval_hours < 1 or interval_hours > 168:
                raise ValueError("Interval must be between 1 and 168 hours")

            # Cancel existing task if any
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            self._interval_hours = interval_hours
            self._enabled = enabled

            if enabled:
                # Calculate next backup time
                self._next_backup = datetime.utcnow() + timedelta(hours=interval_hours)

                # Start background task
                self._task = asyncio.create_task(self._backup_loop())

                logger.info(
                    "auto_backup_scheduled",
                    interval_hours=interval_hours,
                    next_backup=self._next_backup.isoformat()
                )

                return {
                    "success": True,
                    "enabled": True,
                    "interval_hours": interval_hours,
                    "next_backup": self._next_backup,
                    "message": f"Auto-backup scheduled every {interval_hours} hours"
                }
            else:
                self._next_backup = None

                logger.info("auto_backup_disabled")

                return {
                    "success": True,
                    "enabled": False,
                    "interval_hours": interval_hours,
                    "next_backup": None,
                    "message": "Auto-backup disabled"
                }

    async def _backup_loop(self) -> None:
        """
        Background loop that executes backups at scheduled intervals.

        PATTERN: Async background task with error handling
        WHY: Resilient backup execution that doesn't block main app
        """
        self._running = True

        try:
            while self._enabled and self._running:
                # Calculate sleep time until next backup
                if self._next_backup:
                    now = datetime.utcnow()
                    sleep_seconds = (self._next_backup - now).total_seconds()

                    if sleep_seconds > 0:
                        logger.debug(
                            "backup_scheduler_waiting",
                            sleep_seconds=sleep_seconds,
                            next_backup=self._next_backup.isoformat()
                        )
                        await asyncio.sleep(sleep_seconds)

                    # Execute backup
                    if self._enabled:  # Check again after sleep
                        await self.run_scheduled_backup()

                        # Schedule next backup
                        self._next_backup = datetime.utcnow() + timedelta(
                            hours=self._interval_hours
                        )
                else:
                    # No next backup scheduled, wait and check again
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info("backup_scheduler_cancelled")
            raise
        except Exception as e:
            logger.error(
                "backup_scheduler_error",
                error=str(e),
                error_type=type(e).__name__
            )
        finally:
            self._running = False

    async def run_scheduled_backup(self) -> bool:
        """
        Execute a scheduled backup.

        PATTERN: Callback-based backup execution
        WHY: Decouple scheduler from backup implementation

        Returns:
            bool: Whether backup was successful
        """
        logger.info("scheduled_backup_started")

        try:
            if self._backup_callback:
                success = await self._backup_callback()
            else:
                # Default backup behavior - just log
                logger.warning("backup_callback_not_set")
                success = True

            if success:
                self._last_backup = datetime.utcnow()
                logger.info(
                    "scheduled_backup_completed",
                    last_backup=self._last_backup.isoformat()
                )
            else:
                logger.error("scheduled_backup_failed")

            return success

        except Exception as e:
            logger.error(
                "scheduled_backup_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    def get_next_backup_time(self) -> Optional[datetime]:
        """
        Get the next scheduled backup time.

        Returns:
            datetime or None if not scheduled
        """
        return self._next_backup

    async def cancel_scheduled_backup(self) -> dict:
        """
        Cancel scheduled auto-backup.

        PATTERN: Graceful task cancellation
        WHY: Clean shutdown without orphaned tasks

        Returns:
            dict with cancellation status
        """
        async with self._lock:
            self._enabled = False
            self._running = False

            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            self._next_backup = None

            logger.info("scheduled_backup_cancelled")

            return {
                "success": True,
                "enabled": False,
                "message": "Auto-backup cancelled"
            }

    def get_status(self) -> dict:
        """
        Get current scheduler status.

        Returns:
            dict with scheduler state
        """
        return {
            "enabled": self._enabled,
            "interval_hours": self._interval_hours,
            "next_backup": self._next_backup,
            "last_backup": self._last_backup,
            "running": self._running
        }


# Global scheduler instance
backup_scheduler = BackupScheduler()


# Module-level convenience functions
async def schedule_auto_backup(interval_hours: int = 12, enabled: bool = True) -> dict:
    """
    Schedule automatic backups.

    Args:
        interval_hours: Hours between backups
        enabled: Whether to enable auto-backup

    Returns:
        dict with scheduling status
    """
    return await backup_scheduler.schedule_auto_backup(interval_hours, enabled)


async def run_scheduled_backup() -> bool:
    """
    Execute a scheduled backup.

    Returns:
        bool: Whether backup was successful
    """
    return await backup_scheduler.run_scheduled_backup()


def get_next_backup_time() -> Optional[datetime]:
    """
    Get the next scheduled backup time.

    Returns:
        datetime or None if not scheduled
    """
    return backup_scheduler.get_next_backup_time()


async def cancel_scheduled_backup() -> dict:
    """
    Cancel scheduled auto-backup.

    Returns:
        dict with cancellation status
    """
    return await backup_scheduler.cancel_scheduled_backup()
