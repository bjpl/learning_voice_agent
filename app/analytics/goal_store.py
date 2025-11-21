"""
Goal Store - SQLite Persistence Layer for Goals and Achievements
PATTERN: Repository pattern with async operations
WHY: Efficient storage and retrieval of goal tracking data

Tables:
- goals: Learning goals with progress
- milestones: Goal milestones/checkpoints
- goal_progress: Progress history
- achievements: Achievement definitions and unlock status
- achievement_unlocks: User achievement unlocks

Features:
- Async SQLite operations with aiosqlite
- Efficient indexing for common queries
- JSON metadata columns for flexibility
- Transaction support for complex operations
"""
import aiosqlite
import json
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager

from app.analytics.goal_models import (
    Goal, GoalType, GoalStatus,
    Milestone, GoalProgress,
    Achievement, AchievementRarity, AchievementCategory
)
from app.logger import get_logger

# Module logger
logger = get_logger("goal_store")


class GoalStore:
    """
    SQLite persistence layer for goals and achievements.

    PATTERN: Repository with async operations
    WHY: Clean data access with proper error handling
    """

    def __init__(self, db_path: str = "data/goals.db"):
        """
        Initialize the goal store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._initialized = False

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
        if self._initialized:
            return

        logger.info("goal_store_initialization_started", db_path=self.db_path)

        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)

            async with aiosqlite.connect(self.db_path) as db:
                # Goals table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS goals (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        goal_type TEXT NOT NULL,
                        target_value REAL NOT NULL,
                        current_value REAL DEFAULT 0,
                        unit TEXT DEFAULT '',
                        status TEXT DEFAULT 'active',
                        deadline DATE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        completed_at DATETIME,
                        metadata TEXT
                    )
                """)

                # Milestones table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS milestones (
                        id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        target_value REAL NOT NULL,
                        current_value REAL DEFAULT 0,
                        order_num INTEGER NOT NULL,
                        completed INTEGER DEFAULT 0,
                        completed_at DATETIME,
                        reward_message TEXT,
                        FOREIGN KEY (goal_id) REFERENCES goals(id) ON DELETE CASCADE
                    )
                """)

                # Goal progress history
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS goal_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        goal_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        value REAL NOT NULL,
                        progress_percent REAL NOT NULL,
                        delta REAL DEFAULT 0,
                        source TEXT,
                        FOREIGN KEY (goal_id) REFERENCES goals(id) ON DELETE CASCADE
                    )
                """)

                # Achievements table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS achievements (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        icon TEXT NOT NULL,
                        category TEXT NOT NULL,
                        rarity TEXT DEFAULT 'common',
                        requirement TEXT NOT NULL,
                        requirement_type TEXT NOT NULL,
                        requirement_value REAL NOT NULL,
                        points INTEGER DEFAULT 10,
                        unlocked INTEGER DEFAULT 0,
                        unlocked_at DATETIME,
                        progress REAL DEFAULT 0,
                        hidden INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)

                # Create indexes
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_goals_status
                    ON goals(status)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_goals_type
                    ON goals(goal_type)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_goals_deadline
                    ON goals(deadline)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_milestones_goal
                    ON milestones(goal_id, order_num)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_progress_goal
                    ON goal_progress(goal_id, timestamp DESC)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_achievements_category
                    ON achievements(category)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_achievements_unlocked
                    ON achievements(unlocked)
                """)

                await db.commit()

            self._initialized = True
            logger.info("goal_store_initialization_complete", db_path=self.db_path)

        except Exception as e:
            logger.error(
                "goal_store_initialization_failed",
                db_path=self.db_path,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    # ========================================================================
    # GOAL OPERATIONS
    # ========================================================================

    async def save_goal(self, goal: Goal) -> str:
        """
        Save or update a goal.

        Args:
            goal: Goal model instance

        Returns:
            Goal ID
        """
        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO goals
                    (id, title, description, goal_type, target_value, current_value,
                     unit, status, deadline, created_at, updated_at, completed_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal.id,
                        goal.title,
                        goal.description,
                        goal.goal_type.value,
                        goal.target_value,
                        goal.current_value,
                        goal.unit,
                        goal.status.value,
                        goal.deadline.isoformat() if goal.deadline else None,
                        goal.created_at.isoformat(),
                        datetime.utcnow().isoformat(),
                        goal.completed_at.isoformat() if goal.completed_at else None,
                        json.dumps(goal.metadata) if goal.metadata else None
                    )
                )

                # Save milestones
                for milestone in goal.milestones:
                    await self._save_milestone(db, milestone)

                await db.commit()

            logger.info(
                "goal_saved",
                goal_id=goal.id,
                title=goal.title,
                goal_type=goal.goal_type.value
            )

            return goal.id

        except Exception as e:
            logger.error(
                "goal_save_failed",
                goal_id=goal.id,
                error=str(e)
            )
            raise

    async def _save_milestone(self, db, milestone: Milestone) -> None:
        """Save a milestone within a transaction."""
        await db.execute(
            """
            INSERT OR REPLACE INTO milestones
            (id, goal_id, title, description, target_value, current_value,
             order_num, completed, completed_at, reward_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                milestone.id,
                milestone.goal_id,
                milestone.title,
                milestone.description,
                milestone.target_value,
                milestone.current_value,
                milestone.order,
                1 if milestone.completed else 0,
                milestone.completed_at.isoformat() if milestone.completed_at else None,
                milestone.reward_message
            )
        )

    async def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Get a goal by ID.

        Args:
            goal_id: Goal ID

        Returns:
            Goal instance or None
        """
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM goals WHERE id = ?",
                    (goal_id,)
                )
                row = await cursor.fetchone()

                if not row:
                    return None

                # Get milestones
                cursor = await db.execute(
                    """
                    SELECT * FROM milestones
                    WHERE goal_id = ?
                    ORDER BY order_num ASC
                    """,
                    (goal_id,)
                )
                milestone_rows = await cursor.fetchall()

                return self._row_to_goal(row, milestone_rows)

        except Exception as e:
            logger.error("get_goal_failed", goal_id=goal_id, error=str(e))
            return None

    async def get_goals_by_status(
        self,
        status: GoalStatus,
        limit: int = 100
    ) -> List[Goal]:
        """Get goals by status."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM goals
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status.value, limit)
                )
                rows = await cursor.fetchall()

                goals = []
                for row in rows:
                    cursor = await db.execute(
                        "SELECT * FROM milestones WHERE goal_id = ? ORDER BY order_num",
                        (row['id'],)
                    )
                    milestone_rows = await cursor.fetchall()
                    goals.append(self._row_to_goal(row, milestone_rows))

                return goals

        except Exception as e:
            logger.error("get_goals_by_status_failed", status=status.value, error=str(e))
            return []

    async def get_all_goals(self, limit: int = 100) -> List[Goal]:
        """Get all goals."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM goals
                    ORDER BY
                        CASE status
                            WHEN 'active' THEN 1
                            WHEN 'completed' THEN 2
                            ELSE 3
                        END,
                        created_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
                rows = await cursor.fetchall()

                goals = []
                for row in rows:
                    cursor = await db.execute(
                        "SELECT * FROM milestones WHERE goal_id = ? ORDER BY order_num",
                        (row['id'],)
                    )
                    milestone_rows = await cursor.fetchall()
                    goals.append(self._row_to_goal(row, milestone_rows))

                return goals

        except Exception as e:
            logger.error("get_all_goals_failed", error=str(e))
            return []

    async def update_goal_progress(
        self,
        goal_id: str,
        new_value: float,
        source: Optional[str] = None
    ) -> Optional[Goal]:
        """
        Update goal progress and record history.

        Args:
            goal_id: Goal ID
            new_value: New progress value
            source: What triggered the update

        Returns:
            Updated goal or None
        """
        try:
            async with self.get_connection() as db:
                # Get current goal
                cursor = await db.execute(
                    "SELECT * FROM goals WHERE id = ?",
                    (goal_id,)
                )
                row = await cursor.fetchone()

                if not row:
                    return None

                old_value = row['current_value']
                delta = new_value - old_value
                target = row['target_value']
                progress_percent = min(100.0, (new_value / target) * 100) if target > 0 else 100.0

                # Update goal
                completed_at = None
                status = row['status']

                if new_value >= target and status == 'active':
                    completed_at = datetime.utcnow().isoformat()
                    status = 'completed'

                await db.execute(
                    """
                    UPDATE goals
                    SET current_value = ?, updated_at = ?, completed_at = ?, status = ?
                    WHERE id = ?
                    """,
                    (new_value, datetime.utcnow().isoformat(), completed_at, status, goal_id)
                )

                # Record progress history
                await db.execute(
                    """
                    INSERT INTO goal_progress
                    (goal_id, value, progress_percent, delta, source)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (goal_id, new_value, progress_percent, delta, source)
                )

                # Update milestones
                await db.execute(
                    """
                    UPDATE milestones
                    SET current_value = ?,
                        completed = CASE WHEN ? >= target_value THEN 1 ELSE 0 END,
                        completed_at = CASE
                            WHEN ? >= target_value AND completed = 0
                            THEN ?
                            ELSE completed_at
                        END
                    WHERE goal_id = ?
                    """,
                    (new_value, new_value, new_value, datetime.utcnow().isoformat(), goal_id)
                )

                await db.commit()

                logger.info(
                    "goal_progress_updated",
                    goal_id=goal_id,
                    old_value=old_value,
                    new_value=new_value,
                    progress_percent=progress_percent
                )

                return await self.get_goal(goal_id)

        except Exception as e:
            logger.error("update_goal_progress_failed", goal_id=goal_id, error=str(e))
            return None

    async def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal and its milestones."""
        try:
            async with self.get_connection() as db:
                await db.execute("DELETE FROM goal_progress WHERE goal_id = ?", (goal_id,))
                await db.execute("DELETE FROM milestones WHERE goal_id = ?", (goal_id,))
                cursor = await db.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
                await db.commit()

                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info("goal_deleted", goal_id=goal_id)

                return deleted

        except Exception as e:
            logger.error("delete_goal_failed", goal_id=goal_id, error=str(e))
            return False

    async def get_goal_progress_history(
        self,
        goal_id: str,
        days: int = 30
    ) -> List[GoalProgress]:
        """Get progress history for a goal."""
        try:
            async with self.get_connection() as db:
                since = datetime.utcnow() - timedelta(days=days)
                cursor = await db.execute(
                    """
                    SELECT * FROM goal_progress
                    WHERE goal_id = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                    """,
                    (goal_id, since.isoformat())
                )
                rows = await cursor.fetchall()

                return [
                    GoalProgress(
                        goal_id=row['goal_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        value=row['value'],
                        progress_percent=row['progress_percent'],
                        delta=row['delta'],
                        source=row['source']
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error("get_goal_progress_history_failed", goal_id=goal_id, error=str(e))
            return []

    # ========================================================================
    # ACHIEVEMENT OPERATIONS
    # ========================================================================

    async def save_achievement(self, achievement: Achievement) -> str:
        """Save or update an achievement."""
        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO achievements
                    (id, title, description, icon, category, rarity, requirement,
                     requirement_type, requirement_value, points, unlocked, unlocked_at,
                     progress, hidden, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        achievement.id,
                        achievement.title,
                        achievement.description,
                        achievement.icon,
                        achievement.category.value,
                        achievement.rarity.value,
                        achievement.requirement,
                        achievement.requirement_type,
                        achievement.requirement_value,
                        achievement.points,
                        1 if achievement.unlocked else 0,
                        achievement.unlocked_at.isoformat() if achievement.unlocked_at else None,
                        achievement.progress,
                        1 if achievement.hidden else 0,
                        json.dumps(achievement.metadata) if achievement.metadata else None
                    )
                )
                await db.commit()

            return achievement.id

        except Exception as e:
            logger.error("save_achievement_failed", achievement_id=achievement.id, error=str(e))
            raise

    async def get_achievement(self, achievement_id: str) -> Optional[Achievement]:
        """Get an achievement by ID."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM achievements WHERE id = ?",
                    (achievement_id,)
                )
                row = await cursor.fetchone()

                if not row:
                    return None

                return self._row_to_achievement(row)

        except Exception as e:
            logger.error("get_achievement_failed", achievement_id=achievement_id, error=str(e))
            return None

    async def get_all_achievements(self) -> List[Achievement]:
        """Get all achievements."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM achievements
                    ORDER BY
                        CASE rarity
                            WHEN 'legendary' THEN 1
                            WHEN 'epic' THEN 2
                            WHEN 'rare' THEN 3
                            WHEN 'uncommon' THEN 4
                            ELSE 5
                        END,
                        unlocked DESC,
                        title ASC
                    """
                )
                rows = await cursor.fetchall()

                return [self._row_to_achievement(row) for row in rows]

        except Exception as e:
            logger.error("get_all_achievements_failed", error=str(e))
            return []

    async def get_unlocked_achievements(self) -> List[Achievement]:
        """Get all unlocked achievements."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM achievements
                    WHERE unlocked = 1
                    ORDER BY unlocked_at DESC
                    """
                )
                rows = await cursor.fetchall()

                return [self._row_to_achievement(row) for row in rows]

        except Exception as e:
            logger.error("get_unlocked_achievements_failed", error=str(e))
            return []

    async def unlock_achievement(
        self,
        achievement_id: str,
        progress: float = 0
    ) -> Optional[Achievement]:
        """Unlock an achievement."""
        try:
            async with self.get_connection() as db:
                now = datetime.utcnow()
                await db.execute(
                    """
                    UPDATE achievements
                    SET unlocked = 1, unlocked_at = ?, progress = ?
                    WHERE id = ? AND unlocked = 0
                    """,
                    (now.isoformat(), progress, achievement_id)
                )
                await db.commit()

            achievement = await self.get_achievement(achievement_id)
            if achievement and achievement.unlocked:
                logger.info(
                    "achievement_unlocked",
                    achievement_id=achievement_id,
                    title=achievement.title
                )

            return achievement

        except Exception as e:
            logger.error("unlock_achievement_failed", achievement_id=achievement_id, error=str(e))
            return None

    async def update_achievement_progress(
        self,
        achievement_id: str,
        progress: float
    ) -> Optional[Achievement]:
        """Update achievement progress."""
        try:
            async with self.get_connection() as db:
                # Get achievement
                cursor = await db.execute(
                    "SELECT * FROM achievements WHERE id = ?",
                    (achievement_id,)
                )
                row = await cursor.fetchone()

                if not row or row['unlocked']:
                    return await self.get_achievement(achievement_id)

                # Check if should unlock
                if progress >= row['requirement_value']:
                    return await self.unlock_achievement(achievement_id, progress)

                # Update progress
                await db.execute(
                    "UPDATE achievements SET progress = ? WHERE id = ?",
                    (progress, achievement_id)
                )
                await db.commit()

                return await self.get_achievement(achievement_id)

        except Exception as e:
            logger.error("update_achievement_progress_failed", achievement_id=achievement_id, error=str(e))
            return None

    async def get_achievements_by_category(
        self,
        category: AchievementCategory
    ) -> List[Achievement]:
        """Get achievements by category."""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT * FROM achievements
                    WHERE category = ?
                    ORDER BY unlocked DESC, rarity DESC
                    """,
                    (category.value,)
                )
                rows = await cursor.fetchall()

                return [self._row_to_achievement(row) for row in rows]

        except Exception as e:
            logger.error("get_achievements_by_category_failed", category=category.value, error=str(e))
            return []

    async def get_achievement_stats(self) -> Dict[str, Any]:
        """Get achievement statistics."""
        try:
            async with self.get_connection() as db:
                # Total counts
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN unlocked = 1 THEN 1 ELSE 0 END) as unlocked,
                        SUM(CASE WHEN unlocked = 1 THEN points ELSE 0 END) as total_points
                    FROM achievements
                    """
                )
                stats = dict(await cursor.fetchone())

                # By category
                cursor = await db.execute(
                    """
                    SELECT category,
                           COUNT(*) as total,
                           SUM(CASE WHEN unlocked = 1 THEN 1 ELSE 0 END) as unlocked
                    FROM achievements
                    GROUP BY category
                    """
                )
                by_category = {row['category']: {'total': row['total'], 'unlocked': row['unlocked']}
                              for row in await cursor.fetchall()}

                # By rarity
                cursor = await db.execute(
                    """
                    SELECT rarity,
                           COUNT(*) as total,
                           SUM(CASE WHEN unlocked = 1 THEN 1 ELSE 0 END) as unlocked
                    FROM achievements
                    GROUP BY rarity
                    """
                )
                by_rarity = {row['rarity']: {'total': row['total'], 'unlocked': row['unlocked']}
                            for row in await cursor.fetchall()}

                return {
                    'total': stats['total'] or 0,
                    'unlocked': stats['unlocked'] or 0,
                    'total_points': stats['total_points'] or 0,
                    'completion_percent': (
                        (stats['unlocked'] / stats['total'] * 100)
                        if stats['total'] > 0 else 0
                    ),
                    'by_category': by_category,
                    'by_rarity': by_rarity
                }

        except Exception as e:
            logger.error("get_achievement_stats_failed", error=str(e))
            return {}

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _row_to_goal(self, row, milestone_rows: List = None) -> Goal:
        """Convert database row to Goal model."""
        milestones = []
        if milestone_rows:
            for ms_row in milestone_rows:
                milestones.append(Milestone(
                    id=ms_row['id'],
                    goal_id=ms_row['goal_id'],
                    title=ms_row['title'],
                    description=ms_row['description'],
                    target_value=ms_row['target_value'],
                    current_value=ms_row['current_value'],
                    order=ms_row['order_num'],
                    completed=bool(ms_row['completed']),
                    completed_at=datetime.fromisoformat(ms_row['completed_at']) if ms_row['completed_at'] else None,
                    reward_message=ms_row['reward_message']
                ))

        return Goal(
            id=row['id'],
            title=row['title'],
            description=row['description'],
            goal_type=GoalType(row['goal_type']),
            target_value=row['target_value'],
            current_value=row['current_value'],
            unit=row['unit'] or '',
            status=GoalStatus(row['status']),
            deadline=date.fromisoformat(row['deadline']) if row['deadline'] else None,
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            milestones=milestones,
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )

    def _row_to_achievement(self, row) -> Achievement:
        """Convert database row to Achievement model."""
        return Achievement(
            id=row['id'],
            title=row['title'],
            description=row['description'],
            icon=row['icon'],
            category=AchievementCategory(row['category']),
            rarity=AchievementRarity(row['rarity']),
            requirement=row['requirement'],
            requirement_type=row['requirement_type'],
            requirement_value=row['requirement_value'],
            points=row['points'],
            unlocked=bool(row['unlocked']),
            unlocked_at=datetime.fromisoformat(row['unlocked_at']) if row['unlocked_at'] else None,
            progress=row['progress'],
            hidden=bool(row['hidden']),
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )

    async def close(self) -> None:
        """Close the store."""
        self._initialized = False
        logger.info("goal_store_closed", db_path=self.db_path)


# Singleton instance
goal_store = GoalStore()
