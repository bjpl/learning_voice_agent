#!/usr/bin/env python3
"""
Session Cleanup Background Job
PATTERN: Scheduled maintenance task with graceful handling
WHY: Prevent resource exhaustion from stale sessions

Usage:
    # Run once
    python scripts/session_cleanup.py

    # Run as cron job (every 15 minutes)
    */15 * * * * cd /app && python scripts/session_cleanup.py >> /var/log/cleanup.log 2>&1

    # Run in daemon mode
    python scripts/session_cleanup.py --daemon --interval 900
"""
import asyncio
import argparse
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.redis_client import ResilientRedisClient, RedisConnectionConfig
import aiosqlite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("session_cleanup")


class SessionCleanupJob:
    """
    Cleanup job for expired sessions in both Redis and SQLite.

    Responsibilities:
    - Identify expired Redis sessions
    - Mark sessions as expired/cleaned in database
    - Remove stale Redis keys
    - Generate cleanup reports
    """

    def __init__(
        self,
        redis_client: ResilientRedisClient,
        db_path: str = "learning_captures.db",
        session_timeout: int = None,
        batch_size: int = 100,
    ):
        self.redis = redis_client
        self.db_path = db_path
        self.session_timeout = session_timeout or settings.session_timeout
        self.batch_size = batch_size
        self.stats = {
            "redis_keys_scanned": 0,
            "redis_keys_deleted": 0,
            "db_sessions_expired": 0,
            "db_sessions_cleaned": 0,
            "errors": 0,
        }

    async def run(self) -> dict:
        """
        Execute the cleanup job.

        Returns:
            dict: Cleanup statistics
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting session cleanup job at {start_time.isoformat()}")

        self.stats = {
            "redis_keys_scanned": 0,
            "redis_keys_deleted": 0,
            "db_sessions_expired": 0,
            "db_sessions_cleaned": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
        }

        try:
            # Phase 1: Clean up Redis sessions
            await self._cleanup_redis_sessions()

            # Phase 2: Update database session statuses
            await self._update_database_sessions()

            # Phase 3: Clean up old database records (optional, >30 days)
            await self._cleanup_old_records()

        except Exception as e:
            logger.error(f"Cleanup job failed: {e}")
            self.stats["errors"] += 1
            raise

        finally:
            end_time = datetime.utcnow()
            self.stats["end_time"] = end_time.isoformat()
            self.stats["duration_seconds"] = (end_time - start_time).total_seconds()

            logger.info(f"Cleanup job completed: {self.stats}")

        return self.stats

    async def _cleanup_redis_sessions(self):
        """Scan and clean expired Redis session keys."""
        logger.info("Phase 1: Cleaning Redis sessions")

        expired_keys: List[str] = []
        timeout_threshold = datetime.utcnow() - timedelta(seconds=self.session_timeout)

        # Scan for session metadata keys
        pattern = "session:*:metadata"
        async for key in self.redis.scan_iter(match=pattern, count=self.batch_size):
            self.stats["redis_keys_scanned"] += 1

            try:
                data = await self.redis.get(key)
                if not data:
                    expired_keys.append(key)
                    continue

                import json
                metadata = json.loads(data)

                if "last_activity" in metadata:
                    last_activity = datetime.fromisoformat(metadata["last_activity"])
                    if last_activity < timeout_threshold:
                        # Extract session_id and add related keys
                        session_id = key.split(":")[1]
                        expired_keys.extend([
                            f"session:{session_id}:metadata",
                            f"session:{session_id}:context",
                        ])

            except Exception as e:
                logger.warning(f"Error processing key {key}: {e}")
                self.stats["errors"] += 1

        # Delete expired keys in batches
        if expired_keys:
            unique_keys = list(set(expired_keys))
            for i in range(0, len(unique_keys), self.batch_size):
                batch = unique_keys[i:i + self.batch_size]
                deleted = await self.redis.delete(*batch)
                self.stats["redis_keys_deleted"] += deleted
                logger.info(f"Deleted {deleted} Redis keys")

    async def _update_database_sessions(self):
        """Update session statuses in the database."""
        logger.info("Phase 2: Updating database sessions")

        timeout_threshold = datetime.utcnow() - timedelta(seconds=self.session_timeout)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Mark active sessions as expired if past timeout
                cursor = await db.execute("""
                    UPDATE sessions
                    SET status = 'expired', ended_at = CURRENT_TIMESTAMP
                    WHERE status = 'active'
                    AND last_activity < ?
                """, (timeout_threshold.isoformat(),))

                self.stats["db_sessions_expired"] = cursor.rowcount
                await db.commit()

                logger.info(f"Marked {cursor.rowcount} sessions as expired")

        except Exception as e:
            logger.error(f"Database session update failed: {e}")
            self.stats["errors"] += 1

    async def _cleanup_old_records(self, retention_days: int = 30):
        """
        Clean up old database records beyond retention period.

        Note: This is optional and can be configured based on data retention policy.
        """
        logger.info(f"Phase 3: Cleaning records older than {retention_days} days")

        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Update expired sessions to cleaned status
                cursor = await db.execute("""
                    UPDATE sessions
                    SET status = 'cleaned'
                    WHERE status = 'expired'
                    AND ended_at < ?
                """, (cutoff_date.isoformat(),))

                self.stats["db_sessions_cleaned"] = cursor.rowcount
                await db.commit()

                logger.info(f"Marked {cursor.rowcount} old sessions as cleaned")

        except Exception as e:
            logger.error(f"Old records cleanup failed: {e}")
            self.stats["errors"] += 1


class CleanupDaemon:
    """
    Daemon process for continuous cleanup scheduling.

    Features:
    - Graceful shutdown on SIGTERM/SIGINT
    - Configurable interval
    - Health check endpoint support
    """

    def __init__(self, interval_seconds: int = 900):
        self.interval = interval_seconds
        self.running = False
        self.redis: ResilientRedisClient = None
        self.job: SessionCleanupJob = None

    async def start(self):
        """Start the cleanup daemon."""
        logger.info(f"Starting cleanup daemon with {self.interval}s interval")

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        # Initialize Redis connection
        self.redis = ResilientRedisClient(RedisConnectionConfig())
        await self.redis.connect()

        self.job = SessionCleanupJob(self.redis)
        self.running = True

        while self.running:
            try:
                await self.job.run()
            except Exception as e:
                logger.error(f"Cleanup job error: {e}")

            # Wait for next interval
            for _ in range(self.interval):
                if not self.running:
                    break
                await asyncio.sleep(1)

        await self.redis.close()
        logger.info("Cleanup daemon stopped")

    async def stop(self):
        """Stop the cleanup daemon gracefully."""
        logger.info("Stopping cleanup daemon...")
        self.running = False


async def run_once():
    """Run cleanup job once and exit."""
    redis = ResilientRedisClient(RedisConnectionConfig())
    connected = await redis.connect()

    if not connected:
        logger.warning("Could not connect to Redis - running database cleanup only")

    try:
        job = SessionCleanupJob(redis)
        stats = await job.run()
        return stats
    finally:
        await redis.close()


async def run_daemon(interval: int):
    """Run cleanup as a daemon process."""
    daemon = CleanupDaemon(interval_seconds=interval)
    await daemon.start()


def main():
    parser = argparse.ArgumentParser(
        description="Session cleanup job for Learning Voice Agent"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a daemon process"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Cleanup interval in seconds (daemon mode only, default: 900)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="learning_captures.db",
        help="Path to SQLite database"
    )

    args = parser.parse_args()

    if args.daemon:
        asyncio.run(run_daemon(args.interval))
    else:
        asyncio.run(run_once())


if __name__ == "__main__":
    main()
