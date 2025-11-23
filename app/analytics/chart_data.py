"""
Chart Data Formatters
=====================

Formatters for producing Chart.js compatible data structures.
Supports line, bar, pie, and heatmap chart types.

PATTERN: Chart.js data adapter
WHY: Consistent data format for frontend charting library
"""

import statistics
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

from app.logger import db_logger


# =============================================================================
# Base Chart Data Structures
# =============================================================================

@dataclass
class TimeSeriesPoint:
    """
    Single point in a time series.

    PATTERN: Standardized time-series data point
    WHY: Consistent format for line/area charts
    """
    date: str  # ISO format YYYY-MM-DD
    value: float
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {"date": self.date, "value": self.value}
        if self.label:
            result["label"] = self.label
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ChartDataset:
    """
    Chart.js dataset structure.

    PATTERN: Chart.js compatible dataset
    WHY: Direct consumption by Chart.js frontend
    """
    label: str
    data: List[float] = field(default_factory=list)
    backgroundColor: Optional[Union[str, List[str]]] = None
    borderColor: Optional[str] = None
    borderWidth: int = 2
    fill: bool = False
    tension: float = 0.4  # Line smoothing
    pointRadius: int = 3
    pointHoverRadius: int = 5
    yAxisID: Optional[str] = None
    type: Optional[str] = None  # 'line', 'bar', etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Chart.js compatible dictionary"""
        result = {
            "label": self.label,
            "data": self.data,
            "borderWidth": self.borderWidth,
            "fill": self.fill,
            "tension": self.tension,
            "pointRadius": self.pointRadius,
            "pointHoverRadius": self.pointHoverRadius,
        }
        if self.backgroundColor:
            result["backgroundColor"] = self.backgroundColor
        if self.borderColor:
            result["borderColor"] = self.borderColor
        if self.yAxisID:
            result["yAxisID"] = self.yAxisID
        if self.type:
            result["type"] = self.type
        return result


@dataclass
class ChartJSData:
    """
    Complete Chart.js data object.

    PATTERN: Chart.js data structure
    WHY: Ready-to-use format for Chart.js
    """
    labels: List[str] = field(default_factory=list)
    datasets: List[ChartDataset] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Chart.js compatible dictionary"""
        return {
            "labels": self.labels,
            "datasets": [ds.to_dict() for ds in self.datasets]
        }

    def add_dataset(
        self,
        label: str,
        data: List[float],
        color: str = "#4F46E5",
        fill: bool = False
    ) -> None:
        """Add a new dataset to the chart"""
        self.datasets.append(ChartDataset(
            label=label,
            data=data,
            borderColor=color,
            backgroundColor=color if fill else f"{color}20",
            fill=fill
        ))


# =============================================================================
# Specialized Chart Data Structures
# =============================================================================

@dataclass
class BarChartData:
    """
    Bar chart specific data structure.

    PATTERN: Categorical comparison chart
    WHY: Optimal format for bar chart visualizations
    """
    labels: List[str] = field(default_factory=list)
    datasets: List[ChartDataset] = field(default_factory=list)
    horizontal: bool = False
    stacked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Chart.js bar chart format"""
        return {
            "labels": self.labels,
            "datasets": [ds.to_dict() for ds in self.datasets],
            "options": {
                "indexAxis": "y" if self.horizontal else "x",
                "scales": {
                    "x": {"stacked": self.stacked},
                    "y": {"stacked": self.stacked}
                }
            }
        }

    @classmethod
    def from_dict_data(
        cls,
        data: Dict[str, float],
        label: str = "Value",
        colors: Optional[List[str]] = None
    ) -> "BarChartData":
        """Create bar chart from dictionary of label -> value"""
        default_colors = [
            "#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
            "#06B6D4", "#84CC16", "#F97316", "#EC4899", "#6366F1"
        ]

        labels = list(data.keys())
        values = list(data.values())
        bar_colors = colors or default_colors[:len(labels)]

        # Extend colors if needed
        while len(bar_colors) < len(labels):
            bar_colors.extend(default_colors)
        bar_colors = bar_colors[:len(labels)]

        return cls(
            labels=labels,
            datasets=[ChartDataset(
                label=label,
                data=values,
                backgroundColor=bar_colors,
                borderColor=bar_colors,
                borderWidth=1
            )]
        )


@dataclass
class PieChartData:
    """
    Pie/Doughnut chart data structure.

    PATTERN: Distribution visualization
    WHY: Show proportional data clearly
    """
    labels: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    is_doughnut: bool = True
    cutout: str = "60%"  # Doughnut hole size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Chart.js pie/doughnut format"""
        default_colors = [
            "#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
            "#06B6D4", "#84CC16", "#F97316", "#EC4899", "#6366F1"
        ]

        colors = self.colors if self.colors else default_colors[:len(self.labels)]

        return {
            "labels": self.labels,
            "datasets": [{
                "data": self.values,
                "backgroundColor": colors,
                "borderColor": "#ffffff",
                "borderWidth": 2
            }],
            "options": {
                "cutout": self.cutout if self.is_doughnut else "0%"
            }
        }

    @classmethod
    def from_dict_data(
        cls,
        data: Dict[str, float],
        colors: Optional[List[str]] = None,
        is_doughnut: bool = True
    ) -> "PieChartData":
        """Create pie chart from dictionary of label -> value"""
        return cls(
            labels=list(data.keys()),
            values=list(data.values()),
            colors=colors or [],
            is_doughnut=is_doughnut
        )


@dataclass
class HeatmapCell:
    """Single cell in a heatmap"""
    x: int  # Column index (day of week: 0-6)
    y: int  # Row index (week number)
    value: float
    date_str: str
    intensity: int = 0  # 0-4 for color intensity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "x": self.x,
            "y": self.y,
            "value": self.value,
            "date": self.date_str,
            "intensity": self.intensity
        }


@dataclass
class HeatmapData:
    """
    Heatmap data structure (GitHub-style activity calendar).

    PATTERN: Calendar heatmap visualization
    WHY: Show activity density over time
    """
    cells: List[HeatmapCell] = field(default_factory=list)
    x_labels: List[str] = field(default_factory=lambda: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    y_labels: List[str] = field(default_factory=list)  # Week numbers or month labels
    min_value: float = 0
    max_value: float = 0
    color_scale: List[str] = field(default_factory=lambda: [
        "#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to heatmap format"""
        return {
            "cells": [c.to_dict() for c in self.cells],
            "xLabels": self.x_labels,
            "yLabels": self.y_labels,
            "minValue": self.min_value,
            "maxValue": self.max_value,
            "colorScale": self.color_scale
        }

    @classmethod
    def from_daily_counts(
        cls,
        daily_counts: Dict[str, int],
        year: int
    ) -> "HeatmapData":
        """
        Create heatmap from dictionary of date -> count.

        Args:
            daily_counts: Dict mapping 'YYYY-MM-DD' to count
            year: Year to generate heatmap for
        """
        cells = []
        max_value = max(daily_counts.values()) if daily_counts else 0

        # Start from first Sunday of the year or last Sunday of previous year
        start_date = date(year, 1, 1)
        # Adjust to previous Sunday if Jan 1 is not Sunday
        days_to_subtract = start_date.weekday() + 1 if start_date.weekday() != 6 else 0
        start_date = start_date - timedelta(days=days_to_subtract)

        current_date = start_date
        end_date = date(year, 12, 31)

        week = 0
        while current_date <= end_date:
            day_of_week = current_date.weekday()
            # Convert to Sunday=0 format
            day_index = (day_of_week + 1) % 7

            date_str = current_date.isoformat()
            count = daily_counts.get(date_str, 0)

            # Calculate intensity (0-4)
            if max_value > 0:
                intensity = min(4, int((count / max_value) * 4) + (1 if count > 0 else 0))
            else:
                intensity = 0

            cells.append(HeatmapCell(
                x=day_index,
                y=week,
                value=count,
                date_str=date_str,
                intensity=intensity
            ))

            # Move to next day
            current_date += timedelta(days=1)
            if day_index == 6:  # Saturday, move to next week
                week += 1

        # Generate week labels (can be customized)
        y_labels = [str(i) for i in range(week + 1)]

        return cls(
            cells=cells,
            y_labels=y_labels,
            min_value=0,
            max_value=max_value
        )


# =============================================================================
# Chart Data Formatter Utility Class
# =============================================================================

class ChartDataFormatter:
    """
    Utility class for formatting data into chart-ready structures.

    PATTERN: Data transformation utility
    WHY: Centralized chart data formatting logic

    USAGE:
        formatter = ChartDataFormatter()

        # Line chart
        line_data = formatter.format_time_series(
            data_points=[{"date": "2024-01-01", "value": 10}, ...],
            label="Sessions"
        )

        # Multi-series line chart
        multi_line = formatter.format_multi_series([
            {"label": "Quality", "data": quality_data},
            {"label": "Engagement", "data": engagement_data}
        ])
    """

    # Default color palette
    COLORS = {
        "primary": "#4F46E5",     # Indigo
        "success": "#10B981",     # Emerald
        "warning": "#F59E0B",     # Amber
        "danger": "#EF4444",      # Red
        "info": "#06B6D4",        # Cyan
        "purple": "#8B5CF6",      # Purple
        "pink": "#EC4899",        # Pink
        "orange": "#F97316",      # Orange
        "lime": "#84CC16",        # Lime
        "blue": "#3B82F6",        # Blue
    }

    COLOR_PALETTE = [
        "#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
        "#06B6D4", "#84CC16", "#F97316", "#EC4899", "#6366F1"
    ]

    @classmethod
    def format_time_series(
        cls,
        data_points: List[Dict[str, Any]],
        date_key: str = "date",
        value_key: str = "value",
        label: str = "Value",
        color: str = None,
        fill: bool = False,
        date_format: str = "%b %d"
    ) -> ChartJSData:
        """
        Format time series data for Chart.js line chart.

        Args:
            data_points: List of dicts with date and value
            date_key: Key for date field
            value_key: Key for value field
            label: Dataset label
            color: Line color (default: primary)
            fill: Fill area under line
            date_format: Format for x-axis labels

        Returns:
            ChartJSData ready for Chart.js
        """
        color = color or cls.COLORS["primary"]

        labels = []
        values = []

        for point in data_points:
            date_val = point.get(date_key)
            if isinstance(date_val, str):
                try:
                    dt = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
                    labels.append(dt.strftime(date_format))
                except ValueError:
                    labels.append(date_val)
            elif isinstance(date_val, (date, datetime)):
                labels.append(date_val.strftime(date_format))
            else:
                labels.append(str(date_val))

            values.append(float(point.get(value_key, 0)))

        chart_data = ChartJSData(labels=labels)
        chart_data.add_dataset(label, values, color, fill)

        return chart_data

    @classmethod
    def format_multi_series(
        cls,
        series_list: List[Dict[str, Any]],
        shared_labels: Optional[List[str]] = None,
        date_format: str = "%b %d"
    ) -> ChartJSData:
        """
        Format multiple time series for comparison chart.

        Args:
            series_list: List of {"label": str, "data": List[dict], "color": str (optional)}
            shared_labels: Optional shared x-axis labels
            date_format: Format for date labels

        Returns:
            ChartJSData with multiple datasets
        """
        chart_data = ChartJSData()

        for i, series in enumerate(series_list):
            label = series.get("label", f"Series {i+1}")
            data = series.get("data", [])
            color = series.get("color") or cls.COLOR_PALETTE[i % len(cls.COLOR_PALETTE)]
            fill = series.get("fill", False)

            # Extract labels from first series if not provided
            if not chart_data.labels and not shared_labels:
                for point in data:
                    date_val = point.get("date")
                    if isinstance(date_val, str):
                        try:
                            dt = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
                            chart_data.labels.append(dt.strftime(date_format))
                        except ValueError:
                            chart_data.labels.append(date_val)
                    elif isinstance(date_val, (date, datetime)):
                        chart_data.labels.append(date_val.strftime(date_format))
                    else:
                        chart_data.labels.append(str(date_val))
            elif shared_labels and not chart_data.labels:
                chart_data.labels = shared_labels

            values = [float(point.get("value", 0)) for point in data]
            chart_data.add_dataset(label, values, color, fill)

        return chart_data

    @classmethod
    def format_quality_dimensions(
        cls,
        dimensions: Dict[str, float]
    ) -> BarChartData:
        """
        Format quality dimension scores for radar/bar chart.

        Args:
            dimensions: Dict of dimension name -> score (0-1)

        Returns:
            BarChartData for dimension comparison
        """
        dimension_colors = {
            "relevance": cls.COLORS["primary"],
            "helpfulness": cls.COLORS["success"],
            "accuracy": cls.COLORS["warning"],
            "clarity": cls.COLORS["info"],
            "completeness": cls.COLORS["purple"]
        }

        labels = []
        values = []
        colors = []

        for dim, score in dimensions.items():
            display_name = dim.replace("_", " ").title()
            labels.append(display_name)
            values.append(round(score * 100, 1))  # Convert to percentage
            colors.append(dimension_colors.get(dim.lower(), cls.COLORS["primary"]))

        return BarChartData(
            labels=labels,
            datasets=[ChartDataset(
                label="Quality Score (%)",
                data=values,
                backgroundColor=colors,
                borderColor=colors,
                borderWidth=1
            )],
            horizontal=True
        )

    @classmethod
    def format_topic_distribution(
        cls,
        topics: Dict[str, int],
        max_topics: int = 10
    ) -> PieChartData:
        """
        Format topic distribution for pie chart.

        Args:
            topics: Dict of topic name -> count
            max_topics: Maximum topics to show (rest grouped as "Other")

        Returns:
            PieChartData for topic distribution
        """
        # Sort by count
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

        labels = []
        values = []

        # Take top topics
        for topic, count in sorted_topics[:max_topics]:
            labels.append(topic)
            values.append(count)

        # Group remaining as "Other"
        if len(sorted_topics) > max_topics:
            other_count = sum(count for _, count in sorted_topics[max_topics:])
            if other_count > 0:
                labels.append("Other")
                values.append(other_count)

        return PieChartData(
            labels=labels,
            values=values,
            is_doughnut=True
        )

    @classmethod
    def format_hourly_distribution(
        cls,
        hourly_counts: Dict[int, int]
    ) -> BarChartData:
        """
        Format hourly activity distribution for bar chart.

        Args:
            hourly_counts: Dict of hour (0-23) -> count

        Returns:
            BarChartData for hourly distribution
        """
        labels = []
        values = []

        for hour in range(24):
            labels.append(f"{hour:02d}:00")
            values.append(hourly_counts.get(hour, 0))

        return BarChartData(
            labels=labels,
            datasets=[ChartDataset(
                label="Sessions",
                data=values,
                backgroundColor=cls.COLORS["primary"],
                borderColor=cls.COLORS["primary"],
                borderWidth=1
            )]
        )

    @classmethod
    def format_daily_distribution(
        cls,
        daily_counts: Dict[str, int]
    ) -> BarChartData:
        """
        Format daily distribution for bar chart.

        Args:
            daily_counts: Dict of day name -> count

        Returns:
            BarChartData for daily distribution
        """
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        labels = day_order
        values = [daily_counts.get(day, 0) for day in day_order]

        return BarChartData(
            labels=labels,
            datasets=[ChartDataset(
                label="Sessions",
                data=values,
                backgroundColor=cls.COLORS["success"],
                borderColor=cls.COLORS["success"],
                borderWidth=1
            )]
        )

    @classmethod
    def format_trend_comparison(
        cls,
        current_period: List[float],
        previous_period: List[float],
        labels: List[str],
        current_label: str = "Current Period",
        previous_label: str = "Previous Period"
    ) -> ChartJSData:
        """
        Format trend comparison between two periods.

        Args:
            current_period: Current period values
            previous_period: Previous period values
            labels: X-axis labels
            current_label: Label for current period
            previous_label: Label for previous period

        Returns:
            ChartJSData with comparison datasets
        """
        chart_data = ChartJSData(labels=labels)
        chart_data.add_dataset(current_label, current_period, cls.COLORS["primary"], fill=True)
        chart_data.add_dataset(previous_label, previous_period, cls.COLORS["success"], fill=False)

        # Make previous period dashed
        if len(chart_data.datasets) > 1:
            chart_data.datasets[1].borderWidth = 2

        return chart_data

    @classmethod
    def calculate_trend_direction(
        cls,
        values: List[float],
        threshold: float = 0.05
    ) -> Tuple[str, float]:
        """
        Calculate trend direction from values.

        Args:
            values: List of numeric values
            threshold: Minimum change threshold for trend

        Returns:
            Tuple of (direction: "up"|"down"|"stable", change_percent: float)
        """
        if len(values) < 2:
            return "stable", 0.0

        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        first_avg = statistics.mean(first_half) if first_half else 0
        second_avg = statistics.mean(second_half) if second_half else 0

        if first_avg == 0:
            return "stable", 0.0

        change = (second_avg - first_avg) / first_avg

        if change > threshold:
            return "up", change * 100
        elif change < -threshold:
            return "down", change * 100
        return "stable", change * 100

    @classmethod
    def generate_chart_config(
        cls,
        chart_type: str,
        title: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        show_legend: bool = True,
        responsive: bool = True
    ) -> Dict[str, Any]:
        """
        Generate Chart.js configuration options.

        Args:
            chart_type: Type of chart ('line', 'bar', 'pie', 'doughnut')
            title: Chart title
            y_axis_label: Y-axis label
            show_legend: Show legend
            responsive: Responsive sizing

        Returns:
            Chart.js options configuration
        """
        config = {
            "responsive": responsive,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {
                    "display": show_legend,
                    "position": "top"
                }
            }
        }

        if title:
            config["plugins"]["title"] = {
                "display": True,
                "text": title
            }

        if chart_type in ("line", "bar"):
            config["scales"] = {
                "x": {
                    "grid": {"display": False}
                },
                "y": {
                    "beginAtZero": True,
                    "grid": {"color": "rgba(0,0,0,0.05)"}
                }
            }
            if y_axis_label:
                config["scales"]["y"]["title"] = {
                    "display": True,
                    "text": y_axis_label
                }

        return config
