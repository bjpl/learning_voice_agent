"""
Trend Analyzer - Statistical Trend Analysis for Learning Metrics
PATTERN: Statistical rigor with practical interpretation
WHY: Generate reliable trend insights from learning data
SPARC: Linear regression, smoothing, and forecasting algorithms

Methods:
- calculate_trend(): Compute trend direction and magnitude
- get_rolling_average(): Calculate smoothed metric values
- detect_seasonality(): Identify time-based patterns
- forecast_progress(): Simple exponential smoothing prediction
- compare_periods(): Statistical period comparison
"""
import statistics
import math
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from app.logger import get_logger
from app.analytics.insights_models import (
    TrendData, TrendDirection, PeriodComparison, ForecastResult, MetricType
)
from app.learning.config import LearningConfig, learning_config
from app.learning.stores import (
    QualityStore, FeedbackStore, PatternStore,
    quality_store, feedback_store, pattern_store
)

logger = get_logger(__name__)


@dataclass
class SeasonalityResult:
    """Result of seasonality detection"""
    has_seasonality: bool
    period_type: Optional[str] = None  # "weekly", "monthly", "daily"
    peak_periods: List[int] = field(default_factory=list)
    trough_periods: List[int] = field(default_factory=list)
    amplitude: float = 0.0  # Strength of seasonal effect
    confidence: float = 0.0
    pattern_description: Optional[str] = None


@dataclass
class RollingAverageResult:
    """Result of rolling average calculation"""
    metric: str
    window_size: int
    values: List[float]
    dates: List[date]
    original_values: List[float]
    smoothing_factor: float = 0.0


class TrendAnalyzer:
    """
    Statistical trend analysis engine for learning metrics.

    PATTERN: Statistical analysis with domain interpretation
    WHY: Convert raw data into actionable trend insights

    USAGE:
        analyzer = TrendAnalyzer()
        await analyzer.initialize()

        # Calculate trend
        trend = await analyzer.calculate_trend("quality_score", days=7)

        # Get rolling average
        rolling = await analyzer.get_rolling_average("quality_score", window=7)

        # Forecast future values
        forecast = await analyzer.forecast_progress("quality_score", days=14)

        # Compare periods
        comparison = await analyzer.compare_periods(
            "quality_score", "last_week", "this_week"
        )
    """

    # Minimum data points for reliable analysis
    MIN_DATA_POINTS = 3
    MIN_FORECAST_DATA = 7

    # Trend detection thresholds
    STABLE_THRESHOLD = 5.0  # Less than 5% change = stable
    VOLATILITY_THRESHOLD = 0.3  # CV > 0.3 = volatile

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        quality: Optional[QualityStore] = None,
        feedback: Optional[FeedbackStore] = None,
        patterns: Optional[PatternStore] = None
    ):
        """
        Initialize trend analyzer.

        Args:
            config: Learning configuration
            quality: Quality store instance
            feedback: Feedback store instance
            patterns: Pattern store instance
        """
        self.config = config or learning_config
        self.quality = quality or quality_store
        self.feedback = feedback or feedback_store
        self.patterns = patterns or pattern_store
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize analyzer and dependencies"""
        if self._initialized:
            return

        try:
            await self.quality.initialize()
            await self.feedback.initialize()
            await self.patterns.initialize()
            self._initialized = True
            logger.info("trend_analyzer_initialized")
        except Exception as e:
            logger.error("trend_analyzer_init_failed", error=str(e), exc_info=True)
            raise

    # =========================================================================
    # CORE TREND ANALYSIS
    # =========================================================================

    async def calculate_trend(
        self,
        metric: str,
        days: int = 7,
        data: Optional[List[float]] = None
    ) -> TrendData:
        """
        Calculate trend for a metric over a specified period.

        PATTERN: Linear regression with statistical confidence
        WHY: Quantify trend direction and strength

        Args:
            metric: Name of metric to analyze
            days: Number of days to analyze
            data: Optional pre-loaded data (overrides metric lookup)

        Returns:
            TrendData with direction, magnitude, and confidence
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.debug("calculating_trend", metric=metric, days=days)

            # Get data if not provided
            if data is None:
                data = await self._get_metric_history(metric, days)

            if len(data) < self.MIN_DATA_POINTS:
                logger.debug(
                    "insufficient_data_for_trend",
                    metric=metric,
                    data_points=len(data)
                )
                return TrendData(
                    metric=metric,
                    direction=TrendDirection.STABLE,
                    magnitude=0.0,
                    confidence=0.0,
                    period_days=days,
                    data_points=data
                )

            # Calculate linear regression
            x = list(range(len(data)))
            slope, intercept = self._linear_regression(x, data)

            # Calculate R-squared (coefficient of determination)
            r_squared = self._calculate_r_squared(x, data, slope, intercept)

            # Calculate magnitude (percentage change over period)
            start_value = data[0] if data[0] != 0 else 1.0
            predicted_change = slope * len(data)
            magnitude = (predicted_change / abs(start_value)) * 100

            # Determine direction based on magnitude and confidence
            direction = self._determine_direction(magnitude, r_squared, data)

            # Calculate additional statistics
            std_dev = statistics.stdev(data) if len(data) > 1 else 0.0
            variance = std_dev ** 2

            trend = TrendData(
                metric=metric,
                direction=direction,
                magnitude=abs(magnitude),
                confidence=r_squared,
                period_days=days,
                data_points=data,
                slope=slope,
                intercept=intercept,
                start_value=data[0],
                end_value=data[-1],
                min_value=min(data),
                max_value=max(data),
                avg_value=statistics.mean(data),
                std_dev=std_dev,
                variance=variance
            )

            logger.info(
                "trend_calculated",
                metric=metric,
                direction=direction.value,
                magnitude=round(magnitude, 2),
                confidence=round(r_squared, 4)
            )

            return trend

        except Exception as e:
            logger.error(
                "calculate_trend_failed",
                metric=metric,
                error=str(e),
                exc_info=True
            )
            return TrendData(
                metric=metric,
                direction=TrendDirection.STABLE,
                magnitude=0.0,
                confidence=0.0,
                period_days=days
            )

    async def get_rolling_average(
        self,
        metric: str,
        window: int = 7,
        days: int = 30,
        data: Optional[List[Tuple[date, float]]] = None
    ) -> RollingAverageResult:
        """
        Calculate rolling average for smoothed trend visualization.

        PATTERN: Simple moving average with configurable window
        WHY: Reduce noise for clearer trend visualization

        Args:
            metric: Metric to smooth
            window: Window size in days
            days: Total days to analyze
            data: Optional pre-loaded data as (date, value) tuples

        Returns:
            RollingAverageResult with smoothed values
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.debug(
                "calculating_rolling_average",
                metric=metric,
                window=window,
                days=days
            )

            # Get data if not provided
            if data is None:
                raw_data = await self._get_metric_history_with_dates(metric, days)
            else:
                raw_data = data

            if len(raw_data) < window:
                logger.debug(
                    "insufficient_data_for_rolling",
                    metric=metric,
                    data_points=len(raw_data)
                )
                dates = [d for d, _ in raw_data]
                values = [v for _, v in raw_data]
                return RollingAverageResult(
                    metric=metric,
                    window_size=window,
                    values=values,
                    dates=dates,
                    original_values=values
                )

            dates = [d for d, _ in raw_data]
            original_values = [v for _, v in raw_data]

            # Calculate rolling average
            smoothed_values = []
            for i in range(len(original_values)):
                window_start = max(0, i - window + 1)
                window_values = original_values[window_start:i + 1]
                smoothed_values.append(statistics.mean(window_values))

            # Calculate smoothing factor (reduction in variance)
            original_var = statistics.variance(original_values) if len(original_values) > 1 else 0
            smoothed_var = statistics.variance(smoothed_values) if len(smoothed_values) > 1 else 0
            smoothing_factor = 1 - (smoothed_var / original_var) if original_var > 0 else 0

            result = RollingAverageResult(
                metric=metric,
                window_size=window,
                values=smoothed_values,
                dates=dates,
                original_values=original_values,
                smoothing_factor=smoothing_factor
            )

            logger.info(
                "rolling_average_calculated",
                metric=metric,
                window=window,
                data_points=len(smoothed_values),
                smoothing_factor=round(smoothing_factor, 4)
            )

            return result

        except Exception as e:
            logger.error(
                "get_rolling_average_failed",
                metric=metric,
                error=str(e),
                exc_info=True
            )
            return RollingAverageResult(
                metric=metric,
                window_size=window,
                values=[],
                dates=[],
                original_values=[]
            )

    async def detect_seasonality(
        self,
        metric: str,
        days: int = 30
    ) -> SeasonalityResult:
        """
        Detect time-based patterns in metric data.

        PATTERN: Periodicity analysis with autocorrelation
        WHY: Identify recurring patterns for prediction and insights

        Args:
            metric: Metric to analyze
            days: Days of data to analyze

        Returns:
            SeasonalityResult with pattern details
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.debug("detecting_seasonality", metric=metric, days=days)

            # Get data with dates
            data = await self._get_metric_history_with_dates(metric, days)

            if len(data) < 14:  # Need at least 2 weeks
                return SeasonalityResult(has_seasonality=False)

            # Analyze weekly patterns (day of week)
            weekly_pattern = self._analyze_weekly_pattern(data)

            # Analyze hourly patterns if we have intraday data
            hourly_pattern = await self._analyze_hourly_pattern(metric, days)

            # Determine dominant pattern
            if weekly_pattern['confidence'] > 0.5:
                result = SeasonalityResult(
                    has_seasonality=True,
                    period_type="weekly",
                    peak_periods=weekly_pattern['peak_days'],
                    trough_periods=weekly_pattern['trough_days'],
                    amplitude=weekly_pattern['amplitude'],
                    confidence=weekly_pattern['confidence'],
                    pattern_description=weekly_pattern['description']
                )
            elif hourly_pattern and hourly_pattern['confidence'] > 0.5:
                result = SeasonalityResult(
                    has_seasonality=True,
                    period_type="daily",
                    peak_periods=hourly_pattern['peak_hours'],
                    trough_periods=hourly_pattern['trough_hours'],
                    amplitude=hourly_pattern['amplitude'],
                    confidence=hourly_pattern['confidence'],
                    pattern_description=hourly_pattern['description']
                )
            else:
                result = SeasonalityResult(has_seasonality=False)

            logger.info(
                "seasonality_detected",
                metric=metric,
                has_seasonality=result.has_seasonality,
                period_type=result.period_type
            )

            return result

        except Exception as e:
            logger.error(
                "detect_seasonality_failed",
                metric=metric,
                error=str(e),
                exc_info=True
            )
            return SeasonalityResult(has_seasonality=False)

    async def forecast_progress(
        self,
        metric: str,
        forecast_days: int = 14,
        history_days: int = 30
    ) -> ForecastResult:
        """
        Forecast future metric values using exponential smoothing.

        PATTERN: Simple exponential smoothing (SES)
        WHY: Practical prediction for short-term planning

        Args:
            metric: Metric to forecast
            forecast_days: Days to forecast ahead
            history_days: Days of history to use

        Returns:
            ForecastResult with predictions and confidence interval
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.debug(
                "forecasting_progress",
                metric=metric,
                forecast_days=forecast_days,
                history_days=history_days
            )

            # Get historical data
            data = await self._get_metric_history(metric, history_days)

            if len(data) < self.MIN_FORECAST_DATA:
                logger.debug(
                    "insufficient_data_for_forecast",
                    metric=metric,
                    data_points=len(data)
                )
                base_value = data[-1] if data else 0.5
                return ForecastResult(
                    metric=metric,
                    forecast_days=forecast_days,
                    predicted_values=[base_value] * forecast_days,
                    confidence_interval=(base_value * 0.8, base_value * 1.2),
                    base_value=base_value,
                    predicted_direction=TrendDirection.STABLE,
                    confidence=0.0
                )

            # Apply simple exponential smoothing
            alpha = 0.3  # Smoothing factor
            predicted_values = self._exponential_smoothing(data, alpha, forecast_days)

            # Calculate confidence interval based on historical variance
            std_dev = statistics.stdev(data) if len(data) > 1 else 0.0
            confidence_margin = 1.96 * std_dev  # 95% CI

            final_prediction = predicted_values[-1]
            confidence_interval = (
                max(0, final_prediction - confidence_margin),
                final_prediction + confidence_margin
            )

            # Determine direction
            if final_prediction > data[-1] * 1.05:
                direction = TrendDirection.INCREASING
            elif final_prediction < data[-1] * 0.95:
                direction = TrendDirection.DECREASING
            else:
                direction = TrendDirection.STABLE

            # Calculate forecast confidence (based on historical stability)
            cv = std_dev / statistics.mean(data) if statistics.mean(data) > 0 else 1.0
            confidence = max(0, 1 - cv)

            result = ForecastResult(
                metric=metric,
                forecast_days=forecast_days,
                predicted_values=predicted_values,
                confidence_interval=confidence_interval,
                base_value=data[-1],
                predicted_direction=direction,
                confidence=confidence
            )

            logger.info(
                "forecast_generated",
                metric=metric,
                forecast_days=forecast_days,
                direction=direction.value,
                confidence=round(confidence, 4)
            )

            return result

        except Exception as e:
            logger.error(
                "forecast_progress_failed",
                metric=metric,
                error=str(e),
                exc_info=True
            )
            return ForecastResult(
                metric=metric,
                forecast_days=forecast_days,
                predicted_values=[0.5] * forecast_days,
                confidence_interval=(0.3, 0.7),
                base_value=0.5,
                predicted_direction=TrendDirection.STABLE,
                confidence=0.0
            )

    async def compare_periods(
        self,
        metric: str,
        period1_type: str,
        period2_type: str
    ) -> PeriodComparison:
        """
        Compare metric between two time periods.

        PATTERN: Period-over-period statistical comparison
        WHY: Enable week-over-week, month-over-month analysis

        Args:
            metric: Metric to compare
            period1_type: First period ("last_week", "last_month", etc.)
            period2_type: Second period ("this_week", "this_month", etc.)

        Returns:
            PeriodComparison with change statistics
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.debug(
                "comparing_periods",
                metric=metric,
                period1=period1_type,
                period2=period2_type
            )

            # Calculate date ranges
            period1_start, period1_end = self._get_period_dates(period1_type)
            period2_start, period2_end = self._get_period_dates(period2_type)

            # Get data for each period
            period1_data = await self._get_metric_history_in_range(
                metric, period1_start, period1_end
            )
            period2_data = await self._get_metric_history_in_range(
                metric, period2_start, period2_end
            )

            # Calculate averages
            period1_value = statistics.mean(period1_data) if period1_data else 0.0
            period2_value = statistics.mean(period2_data) if period2_data else 0.0

            # Calculate statistical significance using Welch's t-test approximation
            significance = self._calculate_significance(period1_data, period2_data)

            # Create comparison
            comparison = PeriodComparison(
                metric=metric,
                period1_label=self._get_period_label(period1_type),
                period2_label=self._get_period_label(period2_type),
                period1_value=period1_value,
                period2_value=period2_value,
                significance=significance
            )

            logger.info(
                "periods_compared",
                metric=metric,
                period1=period1_type,
                period2=period2_type,
                change_pct=round(comparison.percent_change, 2)
            )

            return comparison

        except Exception as e:
            logger.error(
                "compare_periods_failed",
                metric=metric,
                error=str(e),
                exc_info=True
            )
            return PeriodComparison(
                metric=metric,
                period1_label=period1_type,
                period2_label=period2_type,
                period1_value=0.0,
                period2_value=0.0
            )

    # =========================================================================
    # ADVANCED ANALYSIS
    # =========================================================================

    async def calculate_multiple_trends(
        self,
        metrics: List[str],
        days: int = 7
    ) -> Dict[str, TrendData]:
        """
        Calculate trends for multiple metrics efficiently.

        Args:
            metrics: List of metric names
            days: Analysis period

        Returns:
            Dictionary of metric -> TrendData
        """
        if not self._initialized:
            await self.initialize()

        results = {}
        for metric in metrics:
            results[metric] = await self.calculate_trend(metric, days)

        return results

    async def detect_trend_reversal(
        self,
        metric: str,
        window1_days: int = 7,
        window2_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if a trend has recently reversed direction.

        Args:
            metric: Metric to analyze
            window1_days: Recent period
            window2_days: Previous period

        Returns:
            Reversal details if detected, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get data for both windows
            total_days = window1_days + window2_days
            all_data = await self._get_metric_history(metric, total_days)

            if len(all_data) < total_days:
                return None

            # Split into periods
            recent_data = all_data[-window1_days:]
            previous_data = all_data[-(window1_days + window2_days):-window1_days]

            # Calculate trends for each period
            recent_trend = await self.calculate_trend(metric, window1_days, recent_data)
            previous_trend = await self.calculate_trend(metric, window2_days, previous_data)

            # Check for reversal
            is_reversal = (
                recent_trend.direction != previous_trend.direction and
                recent_trend.direction != TrendDirection.STABLE and
                previous_trend.direction != TrendDirection.STABLE
            )

            if is_reversal:
                return {
                    "detected": True,
                    "previous_direction": previous_trend.direction.value,
                    "current_direction": recent_trend.direction.value,
                    "previous_magnitude": previous_trend.magnitude,
                    "current_magnitude": recent_trend.magnitude,
                    "confidence": min(recent_trend.confidence, previous_trend.confidence)
                }

            return None

        except Exception as e:
            logger.error(
                "detect_trend_reversal_failed",
                metric=metric,
                error=str(e)
            )
            return None

    async def calculate_momentum(
        self,
        metric: str,
        short_window: int = 3,
        long_window: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate trend momentum (rate of change of the trend).

        PATTERN: Moving average convergence/divergence concept
        WHY: Identify trend strength and potential reversals

        Args:
            metric: Metric to analyze
            short_window: Short-term window
            long_window: Long-term window

        Returns:
            Momentum analysis results
        """
        if not self._initialized:
            await self.initialize()

        try:
            data = await self._get_metric_history(metric, long_window + 7)

            if len(data) < long_window:
                return {
                    "momentum": 0.0,
                    "signal": "neutral",
                    "strength": 0.0
                }

            # Calculate short and long moving averages
            short_ma = statistics.mean(data[-short_window:])
            long_ma = statistics.mean(data[-long_window:])

            # Calculate momentum
            momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0

            # Determine signal
            if momentum > 0.05:
                signal = "bullish"  # Strong positive momentum
            elif momentum > 0.02:
                signal = "positive"
            elif momentum < -0.05:
                signal = "bearish"  # Strong negative momentum
            elif momentum < -0.02:
                signal = "negative"
            else:
                signal = "neutral"

            # Calculate momentum strength
            strength = min(1.0, abs(momentum) / 0.1)

            return {
                "momentum": momentum,
                "signal": signal,
                "strength": strength,
                "short_ma": short_ma,
                "long_ma": long_ma
            }

        except Exception as e:
            logger.error("calculate_momentum_failed", metric=metric, error=str(e))
            return {"momentum": 0.0, "signal": "neutral", "strength": 0.0}

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    async def _get_metric_history(self, metric: str, days: int) -> List[float]:
        """Retrieve metric history from appropriate store"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        try:
            if metric in ["quality_score", "composite", "relevance", "helpfulness",
                         "accuracy", "clarity", "completeness"]:
                # Quality metrics from quality store
                daily_avgs = await self.quality.get_daily_averages(start_date, end_date)
                field_name = "composite" if metric == "quality_score" else metric
                return [d.get(field_name, 0.5) or 0.5 for d in daily_avgs]

            elif metric in ["session_count", "exchange_count", "duration"]:
                # Session metrics from feedback store
                sessions = await self.feedback.get_sessions_in_range(start_date, end_date)
                if metric == "session_count":
                    # Group by date and count
                    by_date = defaultdict(int)
                    for s in sessions:
                        day = s.start_time.date()
                        by_date[day] += 1
                    return list(by_date.values()) if by_date else [0]
                elif metric == "exchange_count":
                    by_date = defaultdict(int)
                    for s in sessions:
                        day = s.start_time.date()
                        by_date[day] += s.exchange_count
                    return list(by_date.values()) if by_date else [0]
                elif metric == "duration":
                    by_date = defaultdict(list)
                    for s in sessions:
                        day = s.start_time.date()
                        by_date[day].append(s.duration / 60)  # Minutes
                    return [statistics.mean(v) for v in by_date.values()] if by_date else [0]

            else:
                # Default: try aggregated metrics
                logger.debug("unknown_metric_type", metric=metric)
                return []

        except Exception as e:
            logger.error("get_metric_history_failed", metric=metric, error=str(e))
            return []

    async def _get_metric_history_with_dates(
        self,
        metric: str,
        days: int
    ) -> List[Tuple[date, float]]:
        """Retrieve metric history with dates"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        try:
            if metric in ["quality_score", "composite"]:
                daily_avgs = await self.quality.get_daily_averages(start_date, end_date)
                return [
                    (date.fromisoformat(d['date']), d.get('composite', 0.5) or 0.5)
                    for d in daily_avgs
                ]
            else:
                # Fallback to values only
                values = await self._get_metric_history(metric, days)
                dates = [start_date + timedelta(days=i) for i in range(len(values))]
                return list(zip(dates, values))

        except Exception as e:
            logger.error("get_metric_history_with_dates_failed", error=str(e))
            return []

    async def _get_metric_history_in_range(
        self,
        metric: str,
        start: date,
        end: date
    ) -> List[float]:
        """Get metric values for a specific date range"""
        try:
            if metric in ["quality_score", "composite"]:
                daily_avgs = await self.quality.get_daily_averages(start, end)
                return [d.get('composite', 0.5) or 0.5 for d in daily_avgs]
            else:
                days = (end - start).days
                return await self._get_metric_history(metric, days)
        except Exception as e:
            logger.error("get_metric_in_range_failed", error=str(e))
            return []

    def _linear_regression(
        self,
        x: List[float],
        y: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate linear regression slope and intercept.

        Uses ordinary least squares method.
        """
        n = len(x)
        if n == 0:
            return 0.0, 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi ** 2 for xi in x)

        denominator = n * sum_xx - sum_x ** 2
        if denominator == 0:
            return 0.0, sum_y / n if n > 0 else 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        return slope, intercept

    def _calculate_r_squared(
        self,
        x: List[float],
        y: List[float],
        slope: float,
        intercept: float
    ) -> float:
        """
        Calculate R-squared (coefficient of determination).

        Measures how well the linear model fits the data.
        """
        if len(y) < 2:
            return 0.0

        y_mean = statistics.mean(y)

        # Total sum of squares
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        if ss_tot == 0:
            return 1.0  # Perfect fit if no variance

        # Residual sum of squares
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

        r_squared = 1 - (ss_res / ss_tot)
        return max(0, min(1, r_squared))  # Clamp to [0, 1]

    def _determine_direction(
        self,
        magnitude: float,
        confidence: float,
        data: List[float]
    ) -> TrendDirection:
        """Determine trend direction based on magnitude, confidence, and volatility"""
        # Check volatility
        if len(data) > 1:
            cv = statistics.stdev(data) / statistics.mean(data) if statistics.mean(data) > 0 else 0
            if cv > self.VOLATILITY_THRESHOLD and confidence < 0.3:
                return TrendDirection.VOLATILE

        # Low confidence = stable
        if confidence < 0.2:
            return TrendDirection.STABLE

        # Determine direction by magnitude
        if magnitude > self.STABLE_THRESHOLD:
            return TrendDirection.INCREASING
        elif magnitude < -self.STABLE_THRESHOLD:
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE

    def _exponential_smoothing(
        self,
        data: List[float],
        alpha: float,
        forecast_steps: int
    ) -> List[float]:
        """
        Apply simple exponential smoothing and forecast.

        Args:
            data: Historical data
            alpha: Smoothing factor (0-1)
            forecast_steps: Steps to forecast

        Returns:
            List of forecasted values
        """
        if not data:
            return [0.5] * forecast_steps

        # Initialize with first value
        smoothed = data[0]

        # Apply smoothing to get final level
        for value in data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed

        # Also calculate trend component using Holt's method
        if len(data) > 1:
            trend = statistics.mean([
                data[i] - data[i-1] for i in range(1, len(data))
            ])
        else:
            trend = 0

        # Generate forecasts
        forecasts = []
        for step in range(1, forecast_steps + 1):
            forecast = smoothed + step * trend * 0.5  # Damped trend
            forecasts.append(forecast)

        return forecasts

    def _analyze_weekly_pattern(
        self,
        data: List[Tuple[date, float]]
    ) -> Dict[str, Any]:
        """Analyze data for weekly seasonality patterns"""
        if len(data) < 7:
            return {'confidence': 0.0}

        # Group by day of week
        by_day = defaultdict(list)
        for d, value in data:
            by_day[d.weekday()].append(value)

        # Calculate averages per day
        day_avgs = {}
        for day, values in by_day.items():
            if values:
                day_avgs[day] = statistics.mean(values)

        if len(day_avgs) < 5:  # Need most days
            return {'confidence': 0.0}

        # Find peaks and troughs
        if not day_avgs:
            return {'confidence': 0.0}

        overall_avg = statistics.mean(day_avgs.values())
        peaks = [d for d, v in day_avgs.items() if v > overall_avg * 1.1]
        troughs = [d for d, v in day_avgs.items() if v < overall_avg * 0.9]

        # Calculate amplitude
        max_val = max(day_avgs.values())
        min_val = min(day_avgs.values())
        amplitude = (max_val - min_val) / overall_avg if overall_avg > 0 else 0

        # Calculate confidence based on consistency
        day_stds = [
            statistics.stdev(values) / statistics.mean(values)
            if len(values) > 1 and statistics.mean(values) > 0 else 1.0
            for values in by_day.values()
        ]
        avg_cv = statistics.mean(day_stds) if day_stds else 1.0
        confidence = max(0, 1 - avg_cv) * min(1, amplitude * 2)

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        description = None
        if peaks:
            peak_names = [days[p] for p in peaks]
            description = f"Activity peaks on {', '.join(peak_names)}"

        return {
            'confidence': confidence,
            'peak_days': peaks,
            'trough_days': troughs,
            'amplitude': amplitude,
            'description': description
        }

    async def _analyze_hourly_pattern(
        self,
        metric: str,
        days: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze data for daily (hourly) patterns"""
        # This would require hourly data which we don't currently store
        # Returning None for now - can be implemented when hourly data is available
        return None

    def _get_period_dates(self, period_type: str) -> Tuple[date, date]:
        """Convert period type string to date range"""
        today = date.today()

        if period_type == "this_week":
            start = today - timedelta(days=today.weekday())
            end = today + timedelta(days=1)
        elif period_type == "last_week":
            start = today - timedelta(days=today.weekday() + 7)
            end = today - timedelta(days=today.weekday())
        elif period_type == "this_month":
            start = today.replace(day=1)
            end = today + timedelta(days=1)
        elif period_type == "last_month":
            first_of_month = today.replace(day=1)
            end = first_of_month
            start = (first_of_month - timedelta(days=1)).replace(day=1)
        elif period_type == "last_7_days":
            start = today - timedelta(days=7)
            end = today + timedelta(days=1)
        elif period_type == "last_30_days":
            start = today - timedelta(days=30)
            end = today + timedelta(days=1)
        else:
            # Default to last 7 days
            start = today - timedelta(days=7)
            end = today + timedelta(days=1)

        return start, end

    def _get_period_label(self, period_type: str) -> str:
        """Convert period type to human-readable label"""
        labels = {
            "this_week": "This Week",
            "last_week": "Last Week",
            "this_month": "This Month",
            "last_month": "Last Month",
            "last_7_days": "Last 7 Days",
            "last_30_days": "Last 30 Days"
        }
        return labels.get(period_type, period_type.replace("_", " ").title())

    def _calculate_significance(
        self,
        data1: List[float],
        data2: List[float]
    ) -> float:
        """
        Calculate approximate statistical significance of difference.

        Uses simplified Welch's t-test approach.
        Returns p-value approximation (lower = more significant).
        """
        if len(data1) < 2 or len(data2) < 2:
            return 1.0  # Not significant

        mean1 = statistics.mean(data1)
        mean2 = statistics.mean(data2)
        var1 = statistics.variance(data1)
        var2 = statistics.variance(data2)
        n1 = len(data1)
        n2 = len(data2)

        # Pooled standard error
        se = math.sqrt(var1/n1 + var2/n2) if (var1/n1 + var2/n2) > 0 else 1.0

        # T-statistic
        t = abs(mean1 - mean2) / se if se > 0 else 0

        # Simplified p-value approximation
        # Higher t = lower p-value = more significant
        # This is a rough approximation
        if t > 3:
            return 0.01
        elif t > 2:
            return 0.05
        elif t > 1.5:
            return 0.1
        elif t > 1:
            return 0.2
        else:
            return 0.5


# Global trend analyzer instance
trend_analyzer = TrendAnalyzer()
