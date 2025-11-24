"""
Export Service - Analytics Data Export and Reporting
PATTERN: Data export with multiple format support
WHY: Enable users to download and analyze their learning data

Features:
- JSON export for data analysis
- CSV export for spreadsheet compatibility
- PDF report generation (optional)
- Customizable date ranges and metrics
- Pre-built report templates
"""
import json
import csv
import io
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from app.analytics.goal_models import Goal, GoalStatus, Achievement
from app.analytics.goal_store import GoalStore, goal_store
from app.analytics.goal_tracker import GoalTracker, goal_tracker, ProgressMetrics
from app.analytics.achievement_system import AchievementSystem, achievement_system
from app.logger import get_logger

# Module logger
logger = get_logger("export_service")


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"


class ReportPeriod(str, Enum):
    """Pre-defined report periods."""
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


@dataclass
class ExportMetadata:
    """Metadata for exported data."""
    export_date: datetime
    period_start: date
    period_end: date
    format: str
    version: str = "1.0"
    generated_by: str = "Learning Voice Agent"


@dataclass
class ProgressSummary:
    """Summary of learning progress."""
    total_sessions: int
    total_exchanges: int
    total_topics: int
    avg_quality_score: float
    current_streak: int
    longest_streak: int
    total_duration_minutes: int
    active_goals: int
    completed_goals: int
    achievements_unlocked: int
    total_points: int


@dataclass
class DailyProgressEntry:
    """Daily progress data point."""
    date: str
    sessions: int
    exchanges: int
    quality_score: float
    duration_minutes: int
    topics: List[str]


@dataclass
class WeeklyReport:
    """Weekly summary report."""
    week_start: date
    week_end: date
    summary: ProgressSummary
    daily_progress: List[DailyProgressEntry]
    goals_progress: List[Dict[str, Any]]
    achievements_earned: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]


@dataclass
class MonthlyReport:
    """Monthly summary report."""
    month: int
    year: int
    summary: ProgressSummary
    weekly_breakdown: List[Dict[str, Any]]
    goals_completed: List[Dict[str, Any]]
    achievements_earned: List[Dict[str, Any]]
    top_topics: List[Dict[str, Any]]
    quality_trend: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]


class ExportService:
    """
    Analytics data export and reporting service.

    PATTERN: Multi-format data export
    WHY: Enable data portability and external analysis

    USAGE:
        service = ExportService()
        await service.initialize()

        # Export to JSON
        json_data = await service.export_to_json(period="month")

        # Export to CSV
        csv_data = await service.export_to_csv(period="week")

        # Generate report
        report = await service.generate_weekly_report()
    """

    def __init__(
        self,
        store: Optional[GoalStore] = None,
        tracker: Optional[GoalTracker] = None,
        achievements: Optional[AchievementSystem] = None
    ):
        """
        Initialize the export service.

        Args:
            store: Goal store instance
            tracker: Goal tracker instance
            achievements: Achievement system instance
        """
        self.store = store or goal_store
        self.tracker = tracker or goal_tracker
        self.achievements = achievements or achievement_system
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the export service and dependencies."""
        if self._initialized:
            return

        try:
            await self.store.initialize()
            await self.tracker.initialize()
            await self.achievements.initialize()

            self._initialized = True
            logger.info("export_service_initialized")

        except Exception as e:
            logger.error(
                "export_service_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    # ========================================================================
    # DATA GATHERING
    # ========================================================================

    async def get_export_data(
        self,
        period: ReportPeriod = ReportPeriod.MONTH,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        include_goals: bool = True,
        include_achievements: bool = True,
        include_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Gather data for export.

        Args:
            period: Pre-defined period or custom
            start_date: Start date for custom period
            end_date: End date for custom period
            include_goals: Include goals data
            include_achievements: Include achievements data
            include_progress: Include progress history

        Returns:
            Dictionary with all requested data
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Determine date range
            start, end = self._get_date_range(period, start_date, end_date)

            data = {
                'metadata': asdict(ExportMetadata(
                    export_date=datetime.utcnow(),
                    period_start=start,
                    period_end=end,
                    format="json"
                )),
                'summary': {}
            }

            # Get goals data
            if include_goals:
                all_goals = await self.tracker.get_all_goals()
                data['goals'] = {
                    'active': [self._goal_to_dict(g) for g in all_goals if g.status == GoalStatus.ACTIVE],
                    'completed': [self._goal_to_dict(g) for g in all_goals if g.status == GoalStatus.COMPLETED],
                    'total_count': len(all_goals),
                    'active_count': len([g for g in all_goals if g.status == GoalStatus.ACTIVE]),
                    'completed_count': len([g for g in all_goals if g.status == GoalStatus.COMPLETED])
                }

            # Get achievements data
            if include_achievements:
                all_achievements = await self.achievements.get_all_achievements()
                unlocked = await self.achievements.get_unlocked_achievements()
                stats = await self.achievements.get_achievement_stats()

                data['achievements'] = {
                    'all': [self._achievement_to_dict(a) for a in all_achievements],
                    'unlocked': [self._achievement_to_dict(a) for a in unlocked],
                    'stats': stats
                }

            # Build summary
            data['summary'] = await self._build_summary(data)

            logger.info(
                "export_data_gathered",
                period=period.value,
                start_date=str(start),
                end_date=str(end)
            )

            return data

        except Exception as e:
            logger.error("get_export_data_failed", error=str(e), exc_info=True)
            raise

    # ========================================================================
    # JSON EXPORT
    # ========================================================================

    async def export_to_json(
        self,
        period: ReportPeriod = ReportPeriod.MONTH,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        pretty: bool = True
    ) -> str:
        """
        Export analytics data as JSON.

        Args:
            period: Report period
            start_date: Custom start date
            end_date: Custom end date
            pretty: Pretty-print JSON

        Returns:
            JSON string
        """
        if not self._initialized:
            await self.initialize()

        try:
            data = await self.get_export_data(period, start_date, end_date)

            # Convert dates to ISO strings
            data = self._serialize_dates(data)

            if pretty:
                return json.dumps(data, indent=2, ensure_ascii=False)
            return json.dumps(data, ensure_ascii=False)

        except Exception as e:
            logger.error("export_to_json_failed", error=str(e))
            raise

    # ========================================================================
    # CSV EXPORT
    # ========================================================================

    async def export_to_csv(
        self,
        data_type: str = "goals",
        period: ReportPeriod = ReportPeriod.MONTH,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> str:
        """
        Export analytics data as CSV.

        Args:
            data_type: Type of data to export (goals, achievements, progress)
            period: Report period
            start_date: Custom start date
            end_date: Custom end date

        Returns:
            CSV string
        """
        if not self._initialized:
            await self.initialize()

        try:
            output = io.StringIO()

            if data_type == "goals":
                await self._export_goals_csv(output)
            elif data_type == "achievements":
                await self._export_achievements_csv(output)
            elif data_type == "progress":
                await self._export_progress_csv(output, period, start_date, end_date)
            else:
                raise ValueError(f"Unknown data type: {data_type}")

            result = output.getvalue()
            output.close()

            logger.info("export_to_csv_complete", data_type=data_type)

            return result

        except Exception as e:
            logger.error("export_to_csv_failed", data_type=data_type, error=str(e))
            raise

    async def _export_goals_csv(self, output: io.StringIO) -> None:
        """Export goals to CSV."""
        goals = await self.tracker.get_all_goals()

        writer = csv.writer(output)
        writer.writerow([
            'ID', 'Title', 'Type', 'Status', 'Target', 'Current', 'Progress %',
            'Unit', 'Deadline', 'Created At', 'Completed At'
        ])

        for goal in goals:
            writer.writerow([
                goal.id,
                goal.title,
                goal.goal_type.value,
                goal.status.value,
                goal.target_value,
                goal.current_value,
                f"{goal.progress_percent:.1f}",
                goal.unit,
                str(goal.deadline) if goal.deadline else '',
                goal.created_at.isoformat(),
                goal.completed_at.isoformat() if goal.completed_at else ''
            ])

    async def _export_achievements_csv(self, output: io.StringIO) -> None:
        """Export achievements to CSV."""
        achievements = await self.achievements.get_all_achievements()

        writer = csv.writer(output)
        writer.writerow([
            'ID', 'Title', 'Description', 'Category', 'Rarity', 'Points',
            'Unlocked', 'Progress %', 'Unlocked At'
        ])

        for achievement in achievements:
            writer.writerow([
                achievement.id,
                achievement.title,
                achievement.description,
                achievement.category.value,
                achievement.rarity.value,
                achievement.points,
                'Yes' if achievement.unlocked else 'No',
                f"{achievement.progress_percent:.1f}",
                achievement.unlocked_at.isoformat() if achievement.unlocked_at else ''
            ])

    async def _export_progress_csv(
        self,
        output: io.StringIO,
        period: ReportPeriod,
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> None:
        """Export progress history to CSV."""
        # Get all goals and their progress history
        goals = await self.tracker.get_all_goals()

        writer = csv.writer(output)
        writer.writerow([
            'Goal ID', 'Goal Title', 'Date', 'Value', 'Progress %', 'Delta', 'Source'
        ])

        for goal in goals:
            history = await self.tracker.get_progress_history(goal.id)

            for entry in history:
                writer.writerow([
                    goal.id,
                    goal.title,
                    entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    entry.value,
                    f"{entry.progress_percent:.1f}",
                    entry.delta,
                    entry.source or ''
                ])

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    async def generate_weekly_report(
        self,
        week_start: Optional[date] = None
    ) -> WeeklyReport:
        """
        Generate a weekly summary report.

        Args:
            week_start: Start of the week (defaults to last Monday)

        Returns:
            WeeklyReport instance
        """
        if not self._initialized:
            await self.initialize()

        try:
            if week_start is None:
                today = date.today()
                week_start = today - timedelta(days=today.weekday() + 7)

            week_end = week_start + timedelta(days=6)

            # Get data
            goals = await self.tracker.get_all_goals()
            achievements = await self.achievements.get_unlocked_achievements()
            stats = await self.achievements.get_achievement_stats()

            # Build summary
            summary = ProgressSummary(
                total_sessions=0,  # Would come from session store
                total_exchanges=0,
                total_topics=0,
                avg_quality_score=0.0,
                current_streak=0,
                longest_streak=0,
                total_duration_minutes=0,
                active_goals=len([g for g in goals if g.status == GoalStatus.ACTIVE]),
                completed_goals=len([g for g in goals if g.status == GoalStatus.COMPLETED]),
                achievements_unlocked=stats.get('unlocked', 0),
                total_points=stats.get('total_points', 0)
            )

            # Generate insights
            insights = self._generate_weekly_insights(goals, achievements, summary)

            # Generate recommendations
            recommendations = self._generate_recommendations(goals, summary)

            report = WeeklyReport(
                week_start=week_start,
                week_end=week_end,
                summary=summary,
                daily_progress=[],  # Would be populated from session data
                goals_progress=[self._goal_to_dict(g) for g in goals],
                achievements_earned=[
                    self._achievement_to_dict(a) for a in achievements
                    if a.unlocked_at and a.unlocked_at.date() >= week_start
                ],
                insights=insights,
                recommendations=recommendations
            )

            logger.info(
                "weekly_report_generated",
                week_start=str(week_start),
                week_end=str(week_end)
            )

            return report

        except Exception as e:
            logger.error("generate_weekly_report_failed", error=str(e), exc_info=True)
            raise

    async def generate_monthly_report(
        self,
        month: Optional[int] = None,
        year: Optional[int] = None
    ) -> MonthlyReport:
        """
        Generate a monthly summary report.

        Args:
            month: Month number (1-12)
            year: Year

        Returns:
            MonthlyReport instance
        """
        if not self._initialized:
            await self.initialize()

        try:
            today = date.today()
            if month is None:
                month = today.month - 1 if today.month > 1 else 12
            if year is None:
                year = today.year if today.month > 1 else today.year - 1

            # Get data
            goals = await self.tracker.get_all_goals()
            achievements = await self.achievements.get_unlocked_achievements()
            stats = await self.achievements.get_achievement_stats()

            # Build summary
            summary = ProgressSummary(
                total_sessions=0,
                total_exchanges=0,
                total_topics=0,
                avg_quality_score=0.0,
                current_streak=0,
                longest_streak=0,
                total_duration_minutes=0,
                active_goals=len([g for g in goals if g.status == GoalStatus.ACTIVE]),
                completed_goals=len([g for g in goals if g.status == GoalStatus.COMPLETED]),
                achievements_unlocked=stats.get('unlocked', 0),
                total_points=stats.get('total_points', 0)
            )

            # Filter goals completed this month
            completed_this_month = [
                g for g in goals
                if g.completed_at and g.completed_at.month == month and g.completed_at.year == year
            ]

            # Generate insights and recommendations
            insights = self._generate_monthly_insights(goals, achievements, summary, month)
            recommendations = self._generate_recommendations(goals, summary)

            report = MonthlyReport(
                month=month,
                year=year,
                summary=summary,
                weekly_breakdown=[],  # Would be calculated from daily data
                goals_completed=[self._goal_to_dict(g) for g in completed_this_month],
                achievements_earned=[
                    self._achievement_to_dict(a) for a in achievements
                    if a.unlocked_at and a.unlocked_at.month == month and a.unlocked_at.year == year
                ],
                top_topics=[],  # Would come from session data
                quality_trend=[],  # Would come from quality scores
                insights=insights,
                recommendations=recommendations
            )

            logger.info(
                "monthly_report_generated",
                month=month,
                year=year
            )

            return report

        except Exception as e:
            logger.error("generate_monthly_report_failed", error=str(e), exc_info=True)
            raise

    async def generate_custom_report(
        self,
        start_date: date,
        end_date: date,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a custom report for a date range.

        Args:
            start_date: Report start date
            end_date: Report end date
            metrics: List of metrics to include

        Returns:
            Custom report dictionary
        """
        if not self._initialized:
            await self.initialize()

        try:
            report = {
                'period': {
                    'start': str(start_date),
                    'end': str(end_date),
                    'days': (end_date - start_date).days
                },
                'generated_at': datetime.utcnow().isoformat(),
                'metrics': {}
            }

            if 'goals' in metrics:
                goals = await self.tracker.get_all_goals()
                report['metrics']['goals'] = {
                    'active': len([g for g in goals if g.status == GoalStatus.ACTIVE]),
                    'completed': len([g for g in goals if g.status == GoalStatus.COMPLETED]),
                    'completion_rate': (
                        len([g for g in goals if g.status == GoalStatus.COMPLETED]) / len(goals) * 100
                        if goals else 0
                    )
                }

            if 'achievements' in metrics:
                stats = await self.achievements.get_achievement_stats()
                report['metrics']['achievements'] = stats

            logger.info(
                "custom_report_generated",
                start_date=str(start_date),
                end_date=str(end_date),
                metrics=metrics
            )

            return report

        except Exception as e:
            logger.error("generate_custom_report_failed", error=str(e))
            raise

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_date_range(
        self,
        period: ReportPeriod,
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> tuple:
        """Get date range for a period."""
        today = date.today()

        if period == ReportPeriod.CUSTOM:
            return (start_date or today - timedelta(days=30), end_date or today)
        elif period == ReportPeriod.WEEK:
            return (today - timedelta(days=7), today)
        elif period == ReportPeriod.MONTH:
            return (today - timedelta(days=30), today)
        elif period == ReportPeriod.QUARTER:
            return (today - timedelta(days=90), today)
        elif period == ReportPeriod.YEAR:
            return (today - timedelta(days=365), today)

        return (today - timedelta(days=30), today)

    def _goal_to_dict(self, goal: Goal) -> Dict[str, Any]:
        """Convert goal to dictionary."""
        return {
            'id': goal.id,
            'title': goal.title,
            'description': goal.description,
            'type': goal.goal_type.value,
            'status': goal.status.value,
            'target_value': goal.target_value,
            'current_value': goal.current_value,
            'progress_percent': goal.progress_percent,
            'unit': goal.unit,
            'deadline': str(goal.deadline) if goal.deadline else None,
            'days_remaining': goal.days_remaining,
            'created_at': goal.created_at.isoformat(),
            'completed_at': goal.completed_at.isoformat() if goal.completed_at else None,
            'milestones': [
                {
                    'title': m.title,
                    'target': m.target_value,
                    'current': m.current_value,
                    'completed': m.completed
                }
                for m in goal.milestones
            ]
        }

    def _achievement_to_dict(self, achievement: Achievement) -> Dict[str, Any]:
        """Convert achievement to dictionary."""
        return {
            'id': achievement.id,
            'title': achievement.title,
            'description': achievement.description,
            'icon': achievement.icon,
            'category': achievement.category.value,
            'rarity': achievement.rarity.value,
            'points': achievement.points,
            'requirement': achievement.requirement,
            'unlocked': achievement.unlocked,
            'progress_percent': achievement.progress_percent,
            'unlocked_at': achievement.unlocked_at.isoformat() if achievement.unlocked_at else None
        }

    async def _build_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build summary from gathered data."""
        summary = {
            'export_date': datetime.utcnow().isoformat()
        }

        if 'goals' in data:
            summary['goals'] = {
                'total': data['goals']['total_count'],
                'active': data['goals']['active_count'],
                'completed': data['goals']['completed_count'],
                'completion_rate': (
                    data['goals']['completed_count'] / data['goals']['total_count'] * 100
                    if data['goals']['total_count'] > 0 else 0
                )
            }

        if 'achievements' in data:
            summary['achievements'] = {
                'total': data['achievements']['stats'].get('total', 0),
                'unlocked': data['achievements']['stats'].get('unlocked', 0),
                'points': data['achievements']['stats'].get('total_points', 0),
                'completion_rate': data['achievements']['stats'].get('completion_percent', 0)
            }

        return summary

    def _serialize_dates(self, data: Any) -> Any:
        """Recursively convert dates to ISO strings."""
        if isinstance(data, dict):
            return {k: self._serialize_dates(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_dates(item) for item in data]
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        return data

    def _generate_weekly_insights(
        self,
        goals: List[Goal],
        achievements: List[Achievement],
        summary: ProgressSummary
    ) -> List[str]:
        """Generate insights for weekly report."""
        insights = []

        # Goals progress
        active_goals = [g for g in goals if g.status == GoalStatus.ACTIVE]
        if active_goals:
            avg_progress = sum(g.progress_percent for g in active_goals) / len(active_goals)
            insights.append(f"Average goal progress: {avg_progress:.1f}%")

            close_goals = [g for g in active_goals if g.progress_percent >= 75]
            if close_goals:
                insights.append(f"{len(close_goals)} goal(s) are close to completion (75%+)")

        # Achievement progress
        if summary.achievements_unlocked > 0:
            insights.append(f"Total achievements unlocked: {summary.achievements_unlocked}")
            insights.append(f"Total points earned: {summary.total_points}")

        return insights

    def _generate_monthly_insights(
        self,
        goals: List[Goal],
        achievements: List[Achievement],
        summary: ProgressSummary,
        month: int
    ) -> List[str]:
        """Generate insights for monthly report."""
        insights = []

        month_names = [
            '', 'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        insights.append(f"Monthly summary for {month_names[month]}")

        # Goals analysis
        completed_this_month = [
            g for g in goals
            if g.completed_at and g.completed_at.month == month
        ]
        if completed_this_month:
            insights.append(f"Completed {len(completed_this_month)} goal(s) this month")

        # Achievement analysis
        if summary.achievements_unlocked > 0:
            insights.append(f"Earned {summary.total_points} total points from achievements")

        return insights

    def _generate_recommendations(
        self,
        goals: List[Goal],
        summary: ProgressSummary
    ) -> List[str]:
        """Generate recommendations based on data."""
        recommendations = []

        # No active goals
        if summary.active_goals == 0:
            recommendations.append("Consider setting a new learning goal to maintain momentum")

        # Goals close to deadline
        active_goals = [g for g in goals if g.status == GoalStatus.ACTIVE]
        expiring_soon = [g for g in active_goals if g.days_remaining and g.days_remaining <= 7]
        if expiring_soon:
            recommendations.append(
                f"{len(expiring_soon)} goal(s) expiring within a week - focus on completing them"
            )

        # Low progress goals
        low_progress = [g for g in active_goals if g.progress_percent < 25]
        if low_progress:
            recommendations.append(
                f"Consider revising {len(low_progress)} goal(s) with less than 25% progress"
            )

        return recommendations


# Singleton instance
export_service = ExportService()
