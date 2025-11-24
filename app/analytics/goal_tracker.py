"""
Goal Tracker - Learning Goal Management System
PATTERN: Goal-based gamification with automatic progress tracking
WHY: Motivate learners with measurable, achievable goals

Features:
- Create and manage learning goals
- Automatic progress updates from metrics
- Milestone tracking for large goals
- AI-powered goal suggestions
- Integration with achievement system
"""
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from app.analytics.goal_models import (
    Goal, GoalType, GoalStatus,
    Milestone, GoalProgress, GoalSuggestion,
    CreateGoalRequest, UpdateGoalRequest
)
from app.analytics.goal_store import GoalStore, goal_store
from app.logger import get_logger

# Module logger
logger = get_logger("goal_tracker")


@dataclass
class ProgressMetrics:
    """Current progress metrics from the learning system."""
    current_streak: int = 0
    longest_streak: int = 0
    total_sessions: int = 0
    total_exchanges: int = 0
    total_topics: int = 0
    avg_quality_score: float = 0.0
    total_duration_minutes: int = 0
    total_feedback: int = 0
    sessions_today: int = 0
    sessions_this_week: int = 0
    sessions_this_month: int = 0


class GoalTracker:
    """
    Learning goal management and tracking system.

    PATTERN: Goal-oriented gamification
    WHY: Increase engagement through measurable progress

    USAGE:
        tracker = GoalTracker()
        await tracker.initialize()

        # Create a goal
        goal = await tracker.create_goal(
            title="Build a 7-Day Streak",
            goal_type=GoalType.STREAK,
            target_value=7
        )

        # Update progress
        await tracker.update_all_goals(metrics)

        # Get suggestions
        suggestions = await tracker.get_goal_suggestions(metrics)
    """

    def __init__(self, store: Optional[GoalStore] = None):
        """
        Initialize the goal tracker.

        Args:
            store: Goal store instance (uses global if not provided)
        """
        self.store = store or goal_store
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the goal tracker and dependencies."""
        if self._initialized:
            return

        try:
            await self.store.initialize()
            self._initialized = True
            logger.info("goal_tracker_initialized")

        except Exception as e:
            logger.error(
                "goal_tracker_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    # ========================================================================
    # GOAL MANAGEMENT
    # ========================================================================

    async def create_goal(
        self,
        title: str,
        goal_type: GoalType,
        target_value: float,
        description: Optional[str] = None,
        unit: Optional[str] = None,
        deadline: Optional[date] = None,
        milestones: Optional[List[Dict[str, Any]]] = None,
        initial_value: Optional[float] = None
    ) -> Goal:
        """
        Create a new learning goal.

        Args:
            title: Goal title
            goal_type: Type of goal (streak, sessions, etc.)
            target_value: Target value to achieve
            description: Optional description
            unit: Unit of measurement
            deadline: Optional deadline date
            milestones: Optional milestone definitions
            initial_value: Optional starting value

        Returns:
            Created Goal instance
        """
        if not self._initialized:
            await self.initialize()

        try:
            goal_id = str(uuid.uuid4())

            # Set default unit based on goal type
            if unit is None:
                unit = self._get_default_unit(goal_type)

            # Create goal
            goal = Goal(
                id=goal_id,
                title=title,
                description=description,
                goal_type=goal_type,
                target_value=target_value,
                current_value=initial_value or 0,
                unit=unit,
                status=GoalStatus.ACTIVE,
                deadline=deadline,
                milestones=[]
            )

            # Add milestones if provided
            if milestones:
                for i, ms_def in enumerate(milestones):
                    milestone = Milestone(
                        goal_id=goal_id,
                        title=ms_def.get('title', f'Milestone {i+1}'),
                        description=ms_def.get('description'),
                        target_value=ms_def.get('target_value', target_value * (i + 1) / len(milestones)),
                        order=i,
                        reward_message=ms_def.get('reward_message')
                    )
                    goal.milestones.append(milestone)
            else:
                # Auto-generate milestones for large goals
                goal.milestones = self._generate_milestones(goal_id, target_value, goal_type)

            await self.store.save_goal(goal)

            logger.info(
                "goal_created",
                goal_id=goal.id,
                title=title,
                goal_type=goal_type.value,
                target=target_value
            )

            return goal

        except Exception as e:
            logger.error(
                "create_goal_failed",
                title=title,
                error=str(e),
                exc_info=True
            )
            raise

    async def create_goal_from_request(self, request: CreateGoalRequest) -> Goal:
        """Create a goal from an API request."""
        return await self.create_goal(
            title=request.title,
            description=request.description,
            goal_type=request.goal_type,
            target_value=request.target_value,
            unit=request.unit,
            deadline=request.deadline
        )

    async def update_goal(
        self,
        goal_id: str,
        request: UpdateGoalRequest
    ) -> Optional[Goal]:
        """
        Update an existing goal.

        Args:
            goal_id: Goal ID to update
            request: Update request with new values

        Returns:
            Updated goal or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            goal = await self.store.get_goal(goal_id)
            if not goal:
                return None

            # Apply updates
            if request.title is not None:
                goal.title = request.title
            if request.description is not None:
                goal.description = request.description
            if request.target_value is not None:
                goal.target_value = request.target_value
            if request.deadline is not None:
                goal.deadline = request.deadline
            if request.status is not None:
                goal.status = request.status
                if request.status == GoalStatus.COMPLETED and not goal.completed_at:
                    goal.completed_at = datetime.utcnow()

            goal.updated_at = datetime.utcnow()

            await self.store.save_goal(goal)

            logger.info("goal_updated", goal_id=goal_id)

            return goal

        except Exception as e:
            logger.error("update_goal_failed", goal_id=goal_id, error=str(e))
            return None

    async def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal."""
        if not self._initialized:
            await self.initialize()

        return await self.store.delete_goal(goal_id)

    async def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        if not self._initialized:
            await self.initialize()

        return await self.store.get_goal(goal_id)

    async def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        if not self._initialized:
            await self.initialize()

        return await self.store.get_goals_by_status(GoalStatus.ACTIVE)

    async def get_completed_goals(self) -> List[Goal]:
        """Get all completed goals."""
        if not self._initialized:
            await self.initialize()

        return await self.store.get_goals_by_status(GoalStatus.COMPLETED)

    async def get_all_goals(self) -> List[Goal]:
        """Get all goals."""
        if not self._initialized:
            await self.initialize()

        return await self.store.get_all_goals()

    # ========================================================================
    # PROGRESS TRACKING
    # ========================================================================

    async def update_progress(
        self,
        goal_id: str,
        new_value: float,
        source: Optional[str] = None
    ) -> Tuple[Optional[Goal], List[Milestone]]:
        """
        Update progress for a specific goal.

        Args:
            goal_id: Goal ID to update
            new_value: New progress value
            source: What triggered the update

        Returns:
            Tuple of (updated goal, newly completed milestones)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get current goal state
            old_goal = await self.store.get_goal(goal_id)
            if not old_goal:
                return None, []

            # Track which milestones are about to complete
            old_completed = {m.id for m in old_goal.milestones if m.completed}

            # Update progress
            updated_goal = await self.store.update_goal_progress(goal_id, new_value, source)

            if not updated_goal:
                return None, []

            # Find newly completed milestones
            new_completed = [
                m for m in updated_goal.milestones
                if m.completed and m.id not in old_completed
            ]

            if new_completed:
                logger.info(
                    "milestones_completed",
                    goal_id=goal_id,
                    milestones=[m.title for m in new_completed]
                )

            # Check if goal just completed
            if updated_goal.is_completed and not old_goal.is_completed:
                logger.info(
                    "goal_completed",
                    goal_id=goal_id,
                    title=updated_goal.title
                )

            return updated_goal, new_completed

        except Exception as e:
            logger.error("update_progress_failed", goal_id=goal_id, error=str(e))
            return None, []

    async def update_all_goals(
        self,
        metrics: ProgressMetrics
    ) -> Dict[str, Any]:
        """
        Update progress for all active goals based on current metrics.

        Args:
            metrics: Current progress metrics

        Returns:
            Summary of updates performed
        """
        if not self._initialized:
            await self.initialize()

        try:
            active_goals = await self.get_active_goals()

            updates = {
                'goals_updated': 0,
                'goals_completed': 0,
                'milestones_completed': 0,
                'details': []
            }

            for goal in active_goals:
                # Get current value based on goal type
                current_value = self._get_metric_for_goal_type(goal.goal_type, metrics)

                # Skip if no change
                if current_value == goal.current_value:
                    continue

                # Update progress
                updated_goal, new_milestones = await self.update_progress(
                    goal.id,
                    current_value,
                    source="metrics_update"
                )

                if updated_goal:
                    updates['goals_updated'] += 1
                    updates['milestones_completed'] += len(new_milestones)

                    if updated_goal.status == GoalStatus.COMPLETED:
                        updates['goals_completed'] += 1

                    updates['details'].append({
                        'goal_id': goal.id,
                        'title': goal.title,
                        'old_value': goal.current_value,
                        'new_value': current_value,
                        'completed': updated_goal.is_completed,
                        'new_milestones': [m.title for m in new_milestones]
                    })

            # Check for expired goals
            await self._check_expired_goals()

            logger.info(
                "all_goals_updated",
                goals_updated=updates['goals_updated'],
                goals_completed=updates['goals_completed']
            )

            return updates

        except Exception as e:
            logger.error("update_all_goals_failed", error=str(e), exc_info=True)
            return {'error': str(e)}

    async def _check_expired_goals(self) -> List[Goal]:
        """Check for and mark expired goals."""
        try:
            active_goals = await self.get_active_goals()
            expired = []

            for goal in active_goals:
                if goal.is_expired and not goal.is_completed:
                    goal.status = GoalStatus.EXPIRED
                    await self.store.save_goal(goal)
                    expired.append(goal)
                    logger.info("goal_expired", goal_id=goal.id, title=goal.title)

            return expired

        except Exception as e:
            logger.error("check_expired_goals_failed", error=str(e))
            return []

    # ========================================================================
    # GOAL SUGGESTIONS
    # ========================================================================

    async def get_goal_suggestions(
        self,
        metrics: ProgressMetrics,
        limit: int = 5
    ) -> List[GoalSuggestion]:
        """
        Generate AI-powered goal suggestions based on user patterns.

        Args:
            metrics: Current progress metrics
            limit: Maximum suggestions to return

        Returns:
            List of GoalSuggestion instances
        """
        if not self._initialized:
            await self.initialize()

        try:
            suggestions = []
            existing_goals = await self.get_active_goals()
            existing_types = {g.goal_type for g in existing_goals}

            # Streak suggestions
            if GoalType.STREAK not in existing_types:
                if metrics.current_streak > 0 and metrics.current_streak < 7:
                    suggestions.append(GoalSuggestion(
                        title="Build a 7-Day Streak",
                        description=f"You're on a {metrics.current_streak}-day streak! Keep it going for a full week.",
                        goal_type=GoalType.STREAK,
                        suggested_target=7,
                        suggested_deadline_days=14,
                        confidence=0.9,
                        reason=f"Based on your current {metrics.current_streak}-day streak",
                        based_on="streak_history",
                        difficulty="moderate" if metrics.current_streak >= 3 else "challenging"
                    ))
                elif metrics.current_streak >= 7 and metrics.longest_streak < 30:
                    suggestions.append(GoalSuggestion(
                        title="Month-Long Streak Challenge",
                        description="You've mastered weekly streaks. Try for a full month!",
                        goal_type=GoalType.STREAK,
                        suggested_target=30,
                        suggested_deadline_days=45,
                        confidence=0.85,
                        reason="You've demonstrated consistency with weekly streaks",
                        based_on="streak_history",
                        difficulty="challenging"
                    ))

            # Session count suggestions
            if GoalType.SESSIONS not in existing_types:
                if metrics.total_sessions < 10:
                    suggestions.append(GoalSuggestion(
                        title="Complete 10 Sessions",
                        description="Build a learning habit with your first 10 sessions.",
                        goal_type=GoalType.SESSIONS,
                        suggested_target=10,
                        suggested_deadline_days=14,
                        confidence=0.95,
                        reason="Great starting goal for new learners",
                        based_on="session_count",
                        difficulty="easy"
                    ))
                elif metrics.total_sessions < 50:
                    suggestions.append(GoalSuggestion(
                        title="Reach 50 Sessions",
                        description="You're making progress! Aim for 50 total sessions.",
                        goal_type=GoalType.SESSIONS,
                        suggested_target=50,
                        suggested_deadline_days=30,
                        confidence=0.85,
                        reason=f"You've completed {metrics.total_sessions} sessions",
                        based_on="session_count",
                        difficulty="moderate"
                    ))

            # Topic exploration suggestions
            if GoalType.TOPICS not in existing_types and metrics.total_topics < 10:
                suggestions.append(GoalSuggestion(
                    title="Explore 10 Topics",
                    description="Broaden your learning by exploring different topics.",
                    goal_type=GoalType.TOPICS,
                    suggested_target=10,
                    suggested_deadline_days=21,
                    confidence=0.8,
                    reason=f"You've explored {metrics.total_topics} topics so far",
                    based_on="topic_diversity",
                    difficulty="moderate"
                ))

            # Quality improvement suggestions
            if GoalType.QUALITY not in existing_types and metrics.avg_quality_score > 0:
                if metrics.avg_quality_score < 0.75:
                    target = 0.80
                    difficulty = "moderate"
                elif metrics.avg_quality_score < 0.85:
                    target = 0.90
                    difficulty = "challenging"
                else:
                    target = 0.95
                    difficulty = "challenging"

                suggestions.append(GoalSuggestion(
                    title=f"Achieve {int(target*100)}% Quality Score",
                    description=f"Improve your conversation quality from {metrics.avg_quality_score:.0%} to {target:.0%}.",
                    goal_type=GoalType.QUALITY,
                    suggested_target=target,
                    suggested_deadline_days=30,
                    confidence=0.75,
                    reason=f"Current average quality: {metrics.avg_quality_score:.0%}",
                    based_on="quality_scores",
                    difficulty=difficulty
                ))

            # Exchange volume suggestions
            if GoalType.EXCHANGES not in existing_types:
                if metrics.total_exchanges < 100:
                    suggestions.append(GoalSuggestion(
                        title="Reach 100 Exchanges",
                        description="Engage deeply with 100 conversation exchanges.",
                        goal_type=GoalType.EXCHANGES,
                        suggested_target=100,
                        suggested_deadline_days=14,
                        confidence=0.85,
                        reason=f"You've had {metrics.total_exchanges} exchanges",
                        based_on="exchange_count",
                        difficulty="easy" if metrics.total_exchanges > 50 else "moderate"
                    ))

            # Sort by confidence and return top suggestions
            suggestions.sort(key=lambda s: s.confidence, reverse=True)

            logger.info(
                "goal_suggestions_generated",
                count=len(suggestions[:limit])
            )

            return suggestions[:limit]

        except Exception as e:
            logger.error("get_goal_suggestions_failed", error=str(e), exc_info=True)
            return []

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_default_unit(self, goal_type: GoalType) -> str:
        """Get default unit for a goal type."""
        units = {
            GoalType.STREAK: "days",
            GoalType.SESSIONS: "sessions",
            GoalType.TOPICS: "topics",
            GoalType.QUALITY: "%",
            GoalType.EXCHANGES: "exchanges",
            GoalType.DURATION: "minutes",
            GoalType.FEEDBACK: "feedback",
            GoalType.CUSTOM: ""
        }
        return units.get(goal_type, "")

    def _get_metric_for_goal_type(
        self,
        goal_type: GoalType,
        metrics: ProgressMetrics
    ) -> float:
        """Get the current metric value for a goal type."""
        metric_map = {
            GoalType.STREAK: metrics.current_streak,
            GoalType.SESSIONS: metrics.total_sessions,
            GoalType.TOPICS: metrics.total_topics,
            GoalType.QUALITY: metrics.avg_quality_score,
            GoalType.EXCHANGES: metrics.total_exchanges,
            GoalType.DURATION: metrics.total_duration_minutes,
            GoalType.FEEDBACK: metrics.total_feedback,
        }
        return metric_map.get(goal_type, 0)

    def _generate_milestones(
        self,
        goal_id: str,
        target: float,
        goal_type: GoalType
    ) -> List[Milestone]:
        """Auto-generate milestones for a goal."""
        milestones = []

        # Determine milestone count based on target
        if target <= 5:
            return []  # No milestones for small goals
        elif target <= 10:
            checkpoints = [0.5, 1.0]
        elif target <= 30:
            checkpoints = [0.25, 0.5, 0.75, 1.0]
        else:
            checkpoints = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        # Create milestones
        milestone_names = self._get_milestone_names(goal_type, len(checkpoints))

        for i, (pct, name) in enumerate(zip(checkpoints, milestone_names)):
            milestone_target = target * pct
            milestones.append(Milestone(
                goal_id=goal_id,
                title=name,
                target_value=milestone_target,
                order=i,
                reward_message=f"Great progress! You've reached {int(pct*100)}% of your goal!"
            ))

        return milestones

    def _get_milestone_names(self, goal_type: GoalType, count: int) -> List[str]:
        """Get milestone names based on goal type and count."""
        templates = {
            2: ["Halfway There", "Goal Achieved!"],
            4: ["Getting Started", "Building Momentum", "Almost There", "Goal Achieved!"],
            6: ["First Steps", "Gaining Traction", "Halfway Point", "Strong Progress", "Final Push", "Goal Achieved!"]
        }

        if count in templates:
            return templates[count]

        # Generic names
        return [f"Milestone {i+1}" for i in range(count)]

    async def get_progress_history(
        self,
        goal_id: str,
        days: int = 30
    ) -> List[GoalProgress]:
        """Get progress history for a goal."""
        if not self._initialized:
            await self.initialize()

        return await self.store.get_goal_progress_history(goal_id, days)


# Singleton instance
goal_tracker = GoalTracker()
