"""
Metrics service for comprehensive tracking and analysis.
Provides advanced metrics collection and reporting capabilities.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as redis

from app.core.config import get_settings
from app.core.optimizations import CircuitBreaker, cache_warmer

logger = logging.getLogger(__name__)
settings = get_settings()


class MetricsCollector:
    """Advanced metrics collection system."""

    def __init__(self):
        """Initialize metrics collector."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        self.retention_periods = {
            "realtime": 3600,  # 1 hour
            "hourly": 86400,  # 1 day
            "daily": 2592000,  # 30 days
            "monthly": 31536000,  # 1 year
        }

    async def record_metric(
        self, metric_name: str, value: float, tags: Dict[str, str]
    ):
        """Record metric value with tags."""
        timestamp = datetime.now(timezone.utc)

        # Store raw metric
        await self._store_raw_metric(metric_name, value, tags, timestamp)

        # Update aggregations
        await self._update_aggregations(metric_name, value, tags, timestamp)

    async def get_metrics(
        self,
        metric_names: List[str],
        tags: Optional[Dict[str, str]] = None,
        timeframe: str = "realtime",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get metrics data."""
        if not start_time:
            start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        if not end_time:
            end_time = datetime.now(timezone.utc)

        results = {}
        for metric_name in metric_names:
            results[metric_name] = await self._get_metric_data(
                metric_name, tags, timeframe, start_time, end_time
            )

        return results

    async def _store_raw_metric(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str],
        timestamp: datetime,
    ):
        """Store raw metric data."""
        key = f"metrics:raw:{metric_name}:{timestamp.strftime('%Y%m%d%H')}"
        data = {
            "value": value,
            "tags": tags,
            "timestamp": timestamp.isoformat(),
        }

        await self.redis.rpush(key, json.dumps(data))
        await self.redis.expire(key, self.retention_periods["realtime"])

    async def _update_aggregations(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str],
        timestamp: datetime,
    ):
        """Update metric aggregations."""
        # Update hourly aggregation
        await self._update_hourly_aggregation(
            metric_name, value, tags, timestamp
        )

        # Update daily aggregation
        await self._update_daily_aggregation(
            metric_name, value, tags, timestamp
        )

        # Update monthly aggregation
        await self._update_monthly_aggregation(
            metric_name, value, tags, timestamp
        )

    async def _update_hourly_aggregation(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str],
        timestamp: datetime,
    ):
        """Update hourly metric aggregation."""
        key = f"metrics:hourly:{metric_name}:{timestamp.strftime('%Y%m%d%H')}"

        async with self.redis.pipeline() as pipe:
            await pipe.hincrby(key, "count", 1)
            await pipe.hincrbyfloat(key, "sum", value)
            await pipe.hset(key, "tags", json.dumps(tags))
            await pipe.expire(key, self.retention_periods["hourly"])
            await pipe.execute()

    async def _update_daily_aggregation(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str],
        timestamp: datetime,
    ):
        """Update daily metric aggregation."""
        key = f"metrics:daily:{metric_name}:{timestamp.strftime('%Y%m%d')}"

        async with self.redis.pipeline() as pipe:
            await pipe.hincrby(key, "count", 1)
            await pipe.hincrbyfloat(key, "sum", value)
            await pipe.hset(key, "tags", json.dumps(tags))
            await pipe.expire(key, self.retention_periods["daily"])
            await pipe.execute()

    async def _update_monthly_aggregation(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str],
        timestamp: datetime,
    ):
        """Update monthly metric aggregation."""
        key = f"metrics:monthly:{metric_name}:{timestamp.strftime('%Y%m')}"

        async with self.redis.pipeline() as pipe:
            await pipe.hincrby(key, "count", 1)
            await pipe.hincrbyfloat(key, "sum", value)
            await pipe.hset(key, "tags", json.dumps(tags))
            await pipe.expire(key, self.retention_periods["monthly"])
            await pipe.execute()

    async def _get_metric_data(
        self,
        metric_name: str,
        tags: Optional[Dict[str, str]],
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get metric data for specified parameters."""
        if timeframe == "realtime":
            return await self._get_realtime_data(
                metric_name, tags, start_time, end_time
            )
        elif timeframe == "hourly":
            return await self._get_hourly_data(
                metric_name, tags, start_time, end_time
            )
        elif timeframe == "daily":
            return await self._get_daily_data(
                metric_name, tags, start_time, end_time
            )
        elif timeframe == "monthly":
            return await self._get_monthly_data(
                metric_name, tags, start_time, end_time
            )

        raise ValueError(f"Invalid timeframe: {timeframe}")


class MetricsAnalyzer:
    """Advanced metrics analysis system."""

    def __init__(self):
        """Initialize metrics analyzer."""
        self.collector = MetricsCollector()

    async def analyze_metrics(
        self,
        metric_names: List[str],
        timeframe: str = "hourly",
        window: int = 24,
    ) -> Dict[str, Any]:
        """Analyze metrics over time window."""
        end_time = datetime.now(timezone.utc)
        start_time = self._get_start_time(end_time, timeframe, window)

        # Get metrics data
        metrics_data = await self.collector.get_metrics(
            metric_names,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )

        # Perform analysis
        analysis = {}
        for metric_name, data in metrics_data.items():
            analysis[metric_name] = {
                "basic_stats": self._calculate_basic_stats(data),
                "trends": self._analyze_trends(data),
                "patterns": self._detect_patterns(data),
            }

        return analysis

    def _get_start_time(
        self, end_time: datetime, timeframe: str, window: int
    ) -> datetime:
        """Calculate start time based on timeframe and window."""
        if timeframe == "hourly":
            return end_time - timedelta(hours=window)
        elif timeframe == "daily":
            return end_time - timedelta(days=window)
        elif timeframe == "monthly":
            return end_time - timedelta(days=window * 30)

        return end_time - timedelta(hours=1)

    def _calculate_basic_stats(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate basic statistical measures."""
        values = [d["value"] for d in data]

        if not values:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def _analyze_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metric trends."""
        if not data:
            return {"trend": "no_data"}

        values = [d["value"] for d in data]
        times = [datetime.fromisoformat(d["timestamp"]) for d in data]

        # Calculate trend
        if len(values) < 2:
            return {"trend": "insufficient_data"}

        slope = (values[-1] - values[0]) / (
            (times[-1] - times[0]).total_seconds()
        )

        trend = "increasing" if slope > 0 else "decreasing"
        if abs(slope) < 0.01:
            trend = "stable"

        return {
            "trend": trend,
            "slope": float(slope),
            "change_rate": float((values[-1] - values[0]) / values[0])
            if values[0] != 0
            else 0,
        }

    def _detect_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in metric data."""
        if not data:
            return {"patterns": []}

        values = [d["value"] for d in data]
        times = [datetime.fromisoformat(d["timestamp"]) for d in data]

        patterns = []

        # Detect spikes
        mean = np.mean(values)
        std = np.std(values)
        spikes = [i for i, v in enumerate(values) if abs(v - mean) > 2 * std]

        if spikes:
            patterns.append(
                {
                    "type": "spike",
                    "count": len(spikes),
                    "timestamps": [times[i].isoformat() for i in spikes],
                }
            )

        # Detect seasonality
        if len(values) >= 24:
            hourly_values = []
            for hour in range(24):
                hour_values = [
                    v for t, v in zip(times, values) if t.hour == hour
                ]
                if hour_values:
                    hourly_values.append(np.mean(hour_values))

            if len(hourly_values) == 24:
                patterns.append(
                    {"type": "hourly_pattern", "values": hourly_values}
                )

        return {"patterns": patterns}


class MetricsService:
    """Comprehensive metrics service."""

    def __init__(self):
        """Initialize metrics service."""
        self.collector = MetricsCollector()
        self.analyzer = MetricsAnalyzer()

    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def track_metrics(
        self, metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Track multiple metrics."""
        try:
            for metric in metrics:
                await self.collector.record_metric(
                    metric["name"], metric["value"], metric.get("tags", {})
                )

            return {
                "status": "success",
                "count": len(metrics),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Error tracking metrics: %s", e)
            return {"status": "error", "message": str(e)}

    @cache_warmer(["metrics_analysis"])
    async def analyze_metrics(
        self,
        metric_names: List[str],
        timeframe: str = "hourly",
        window: int = 24,
    ) -> Dict[str, Any]:
        """Analyze metrics data."""
        try:
            analysis = await self.analyzer.analyze_metrics(
                metric_names, timeframe, window
            )

            return {
                "status": "success",
                "analysis": analysis,
                "timeframe": timeframe,
                "window": window,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Error analyzing metrics: %s", e)
            return {"status": "error", "message": str(e)}

    async def get_metrics_summary(
        self,
        metric_names: List[str],
        tags: Optional[Dict[str, str]] = None,
        timeframe: str = "daily",
    ) -> Dict[str, Any]:
        """Get summary of metrics."""
        try:
            metrics_data = await self.collector.get_metrics(
                metric_names, tags, timeframe
            )

            summary = {}
            for metric_name, data in metrics_data.items():
                summary[metric_name] = {
                    "current_value": data[-1]["value"] if data else None,
                    "stats": self.analyzer._calculate_basic_stats(data),
                    "trend": self.analyzer._analyze_trends(data),
                }

            return {
                "status": "success",
                "summary": summary,
                "timeframe": timeframe,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Error getting metrics summary: %s", e)
            return {"status": "error", "message": str(e)}
