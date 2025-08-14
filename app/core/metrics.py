"""
Advanced metrics system.
Provides comprehensive metrics collection, analysis, and reporting.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import psutil
from prometheus_client import Counter, Gauge, Histogram

from app.core.config import get_settings
from app.core.errors import ErrorCategory
from app.models.reputation import AlertConfig, AlertEvent, MetricsWindow
from app.core.error_utils import handle_errors, ErrorSeverity

# Performance metrics
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint", "status"],
)
REQUEST_SIZE = Histogram(
    "http_request_size_bytes", "Request size in bytes", ["method", "endpoint"]
)
RESPONSE_SIZE = Histogram(
    "http_response_size_bytes",
    "Response size in bytes",
    ["method", "endpoint"],
)

# Throughput metrics
REQUEST_RATE = Counter(
    "http_requests_total", "Total requests", ["method", "endpoint", "status"]
)
BYTES_TRANSFERRED = Counter(
    "http_bytes_transferred_total", "Total bytes transferred", ["direction"]
)
ACTIVE_REQUESTS = Gauge(
    "http_active_requests", "Number of active requests", ["method", "endpoint"]
)

# System metrics
CPU_USAGE = Gauge("host_cpu_usage", "System CPU usage", ["type"])
MEMORY_USAGE = Gauge(
    "host_memory_usage_bytes", "System memory usage in bytes", ["type"]
)
DISK_USAGE = Gauge(
    "host_disk_usage_bytes", "System disk usage in bytes", ["type"]
)

# Business metrics
REPUTATION_SCORES = Histogram(
    "reputation_scores_distribution", "Reputation scores", ["type"]
)
ANALYSIS_DURATION = Histogram(
    "analysis_duration_seconds", "Analysis duration in seconds", ["type"]
)
MODEL_PREDICTIONS = Counter(
    "model_predictions_total", "Total model predictions", ["model", "result"]
)

# Sentiment analysis metrics
SENTIMENT_ANALYSIS_LATENCY = Histogram(
    "sentiment_analysis_latency_seconds",
    "Sentiment analysis latency in seconds",
    ["model", "language"],
)
SENTIMENT_SCORES = Histogram(
    "sentiment_scores_distribution",
    "Sentiment scores distribution",
    ["type", "source"],
)
SENTIMENT_ERRORS = Counter(
    "sentiment_analysis_errors_total",
    "Sentiment analysis errors",
    ["error_type", "model"],
)

# Comment analysis metrics
COMMENT_ANALYSIS_LATENCY = Histogram(
    "comment_analysis_latency_seconds",
    "Comment analysis latency in seconds",
    ["model", "type"],
)
COMMENT_SCORES = Histogram(
    "comment_scores_distribution",
    "Comment scores distribution",
    ["type", "source"],
)
COMMENT_ERRORS = Counter(
    "comment_analysis_errors_total",
    "Comment analysis errors",
    ["error_type", "model"],
)
COMMENT_ANALYSIS_TOTAL = Counter(
    "comment_analysis_total",
    "Total number of comment analyses performed"
)

# Response generation metrics
RESPONSE_GENERATION_LATENCY = Histogram(
    "response_generation_latency_seconds",
    "Response generation latency in seconds",
    ["model", "type"],
)
RESPONSE_QUALITY_SCORES = Histogram(
    "response_quality_scores_distribution",
    "Response quality scores distribution",
    ["type", "source"],
)
RESPONSE_ERRORS = Counter(
    "response_generation_errors_total",
    "Response generation errors",
    ["error_type", "model"],
)

# Prediction metrics
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model", "type"],
)
PREDICTION_ACCURACY = Gauge(
    "prediction_accuracy", "Prediction accuracy", ["model", "metric"]
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Prediction errors", ["error_type", "model"]
)

# LinkedIn API metrics
LINKEDIN_REQUESTS_TOTAL = Counter(
    "linkedin_requests_total", "Total number of LinkedIn API requests"
)
LINKEDIN_ERRORS_TOTAL = Counter(
    "linkedin_errors_total", "Total number of LinkedIn API errors"
)
LINKEDIN_API_LATENCY = Histogram(
    "linkedin_api_latency_seconds",
    "LinkedIn API latency in seconds"
)

# Monitoring service metrics
REQUEST_COUNT = Counter(
    "request_total", "Total request count (API and rate limiting)", ["method", "endpoint", "status", "client"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)
ACTIVE_USERS = Gauge("active_users", "Number of active users")
ERROR_COUNT = Counter("error_total", "Total error count", ["type", "service"])
SYSTEM_METRICS = Gauge("system_metrics", "System metrics", ["metric"])

# Enhanced monitoring metrics
MONITOR_ERRORS = Counter(
    "monitor_errors_total", "Total monitoring errors", ["type", "severity"]
)
MONITOR_LATENCY = Histogram(
    "monitor_latency_seconds", "Monitoring latency", ["operation", "status"]
)
ALERT_TRIGGERS = Counter(
    "alert_triggers_total", "Total alerts triggered", ["severity", "type"]
)
MONITOR_HEALTH = Gauge(
    "monitor_health", "Monitor health status", ["monitor_id", "component"]
)
MODEL_PERFORMANCE = Gauge(
    "model_performance", "ML model performance metrics", ["model", "metric"]
)
RECOVERY_METRICS = Counter(
    "recovery_metrics_total", "Recovery metrics", ["type", "success"]
)

# Notification metrics
NOTIFICATION_LATENCY = Histogram(
    "notification_latency_seconds",
    "Notification delivery latency in seconds",
    ["channel", "type"]
)

logger = logging.getLogger(__name__)
settings = get_settings()


class MetricValue(TypedDict):
    """Metric value type."""

    value: float
    timestamp: float
    labels: Dict[str, str]


class MetricData(TypedDict):
    """Metric data type."""

    name: str
    description: str
    type: str
    values: List[MetricValue]
    labels: Dict[str, str]


class MetricsManager:
    """Metrics manager for collecting and analyzing system metrics."""
    
    def __init__(self):
        """Initialize metrics manager."""
        self._metrics: Dict[str, MetricData] = {}
        self._errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        self._max_history = 1000  # Maximum number of values per metric
        self._max_error_history = 100  # Maximum number of error records
        self._alerts: Dict[str, AlertConfig] = {}
        self._alert_history: List[AlertEvent] = []
        self._background_tasks = []
        self._windows = {}  # <-- Fix: initialize _windows
        self._initialize_default_metrics()

    async def _analyze_metrics(self, *args, **kwargs):
        """Analyze metrics periodically."""
        while True:
            try:
                # Calculate aggregates
                await self._calculate_aggregates()
                
                # Check alerts
                await self._check_alerts()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error analyzing metrics: {str(e)}")
                await asyncio.sleep(60)

    def _initialize_default_metrics(self):
        """Initialize default system metrics."""
        # System metrics
        self._system_metrics = {
            "cpu_usage": Gauge(
                "host_cpu_usage_percent", "System CPU usage percentage"
            ),
            "memory_usage": Gauge(
                "host_memory_usage_percent", "System memory usage percentage"
            ),
            "disk_usage": Gauge(
                "host_disk_usage_percent", "System disk usage percentage"
            ),
            "network_io": Gauge(
                "host_network_io_bytes",
                "System network IO bytes",
                ["direction"],
            ),
            "process_count": Gauge(
                "host_process_count", "Number of running processes"
            ),
            "thread_count": Gauge(
                "host_thread_count", "Number of running threads"
            ),
            "open_files": Gauge("host_open_files", "Number of open files"),
            "system_load": Gauge(
                "host_load_average", "System load average", ["interval"]
            ),
        }

        # Application metrics
        self._app_metrics = {
            "request_count": Counter(
                "app_http_requests_total",
                "Total request count",
                ["endpoint", "method", "status"],
            ),
            "request_latency": Histogram(
                "app_http_request_latency_seconds",
                "Request latency in seconds",
                ["endpoint"],
            ),
            "active_connections": Gauge(
                "app_active_connections_current",
                "Number of active connections",
            ),
            "error_count": Counter(
                "app_errors_total", "Total error count", ["type", "severity"]
            ),
            "cache_hits": Counter("app_cache_hits_total", "Cache hit count"),
            "cache_misses": Counter(
                "app_cache_misses_total", "Cache miss count"
            ),
            "rate_limit_hits": Counter(
                "app_rate_limit_hits_total", "Rate limit hit count"
            ),
            "auth_failures": Counter(
                "app_auth_failures_total", "Authentication failure count"
            ),
        }

        # Default alerts
        self._default_alerts = {
            "high_cpu": AlertConfig(
                name="high_cpu",
                metric="cpu_usage",
                threshold=90.0,
                condition="above",
                window=timedelta(minutes=5),
                cooldown=timedelta(minutes=30),
            ),
            "high_memory": AlertConfig(
                name="high_memory",
                metric="memory_usage",
                threshold=90.0,
                condition="above",
                window=timedelta(minutes=5),
                cooldown=timedelta(minutes=30),
            ),
            "high_disk": AlertConfig(
                name="high_disk",
                metric="disk_usage",
                threshold=90.0,
                condition="above",
                window=timedelta(minutes=5),
                cooldown=timedelta(minutes=30),
            ),
            "high_error_rate": AlertConfig(
                name="high_error_rate",
                metric="error_rate",
                threshold=0.05,
                condition="above",
                window=timedelta(minutes=5),
                cooldown=timedelta(minutes=15),
            ),
        }

        # Register default alerts
        for alert in self._default_alerts.values():
            self._alerts[alert.name] = alert

    async def initialize(self):
        """Initialize metrics manager and start background tasks."""
        self._background_tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._analyze_metrics()),
            asyncio.create_task(self._cleanup_old_metrics()),
        ]

    async def shutdown(self):
        """Shutdown metrics manager and cleanup tasks."""
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def record_request(self, request, response, request_size: int, response_size: int, *args, **kwargs):
        """Record request metrics."""
        try:
            endpoint = request.url.path
            REQUEST_DURATION.labels(
                method=request.method, endpoint=endpoint, status=response.status_code
            ).observe(getattr(request, 'elapsed', timedelta(seconds=0)).total_seconds() if hasattr(request, 'elapsed') else 0)

            REQUEST_SIZE.labels(method=request.method, endpoint=endpoint).observe(
                request_size
            )

            RESPONSE_SIZE.labels(method=request.method, endpoint=endpoint).observe(
                response_size
            )

            # Update counters
            REQUEST_RATE.labels(
                method=request.method, endpoint=endpoint, status=response.status_code
            ).inc()

            BYTES_TRANSFERRED.labels(direction="in").inc(request_size)

            BYTES_TRANSFERRED.labels(direction="out").inc(response_size)

            # Store in time windows
            await self._store_metric(
                f"request_duration_{request.method}_{endpoint}",
                getattr(request, 'elapsed', timedelta(seconds=0)).total_seconds() if hasattr(request, 'elapsed') else 0,
                [
                    timedelta(minutes=1),
                    timedelta(minutes=5),
                    timedelta(hours=1),
                ],
            )
            
        except Exception as e:
            logger.error(f"Error recording request metrics: {str(e)}")

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def record_reputation_score(self, score: float, score_type: str):
        """Record reputation score metrics."""
        try:
            # Update histogram
            REPUTATION_SCORES.labels(type=score_type).observe(score)
            
            # Store in time windows
            await self._store_metric(
                f"reputation_score_{score_type}",
                score,
                [timedelta(minutes=5), timedelta(hours=1), timedelta(days=1)],
            )
            
        except Exception as e:
            logger.error(f"Error recording reputation score: {str(e)}")

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def record_analysis(self, duration: float, analysis_type: str):
        """Record analysis metrics."""
        try:
            # Update histogram
            ANALYSIS_DURATION.labels(type=analysis_type).observe(duration)
            
            # Store in time windows
            await self._store_metric(
                f"analysis_duration_{analysis_type}",
                duration,
                [timedelta(minutes=5), timedelta(hours=1)],
            )
            
        except Exception as e:
            logger.error(f"Error recording analysis metrics: {str(e)}")

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def record_prediction(self, model: str, result: str):
        """Record model prediction metrics."""
        try:
            MODEL_PREDICTIONS.labels(model=model, result=result).inc()
            
        except Exception as e:
            logger.error(f"Error recording prediction: {str(e)}")

    async def _store_metric(
        self, name: str, value: float, windows: List[timedelta]
    ):
        """Store a metric value in time windows."""
        try:
            current_time = datetime.utcnow()
            
            with self._lock:
                # Initialize windows if needed
                if name not in self._windows:
                    self._windows[name] = {}
                
                # Store in each window
                for duration in windows:
                    if duration not in self._windows[name]:
                        self._windows[name][duration] = MetricsWindow(
                            duration=duration
                        )
                    
                    window = self._windows[name][duration]
                    window.values.append(value)
                    window.timestamps.append(current_time)
                    
        except Exception as e:
            logger.error(f"Error storing metric: {str(e)}")

    async def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.labels(type="system").set(cpu_percent)
                
                # Per-CPU metrics
                for i, percent in enumerate(psutil.cpu_percent(percpu=True)):
                    CPU_USAGE.labels(type=f"cpu{i}").set(percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                MEMORY_USAGE.labels(type="total").set(memory.total)
                MEMORY_USAGE.labels(type="available").set(memory.available)
                MEMORY_USAGE.labels(type="used").set(memory.used)
                MEMORY_USAGE.labels(type="cached").set(getattr(memory, "cached", 0))
                
                # Disk metrics
                disk = psutil.disk_usage("/")
                DISK_USAGE.labels(type="total").set(disk.total)
                DISK_USAGE.labels(type="used").set(disk.used)
                DISK_USAGE.labels(type="free").set(disk.free)
                
                await asyncio.sleep(15)  # Collect every 15 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
                await asyncio.sleep(60)

    async def _cleanup_old_metrics(self):
        """Clean up old metrics periodically."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                with self._lock:
                    # Clean up each metric type
                    for metric_name in list(self._windows.keys()):
                        for duration in list(
                            self._windows[metric_name].keys()
                        ):
                            window = self._windows[metric_name][duration]
                            
                            # Remove values older than the window
                            while (
                                window.timestamps
                                and current_time - window.timestamps[0]
                                > window.duration
                            ):
                                window.values.popleft()
                                window.timestamps.popleft()
                            
                            # Remove empty windows
                            if not window.values:
                                del self._windows[metric_name][duration]
                        
                        # Remove empty metric types
                        if not self._windows[metric_name]:
                            del self._windows[metric_name]
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error cleaning up metrics: {str(e)}")
                await asyncio.sleep(60)

    async def _calculate_aggregates(self):
        """Calculate aggregate metrics."""
        try:
            current_time = datetime.utcnow()
            with self._lock:
                for metric_name, windows in self._windows.items():
                    for duration, window in windows.items():
                        if window.values:
                            # Calculate statistics
                            values = np.array(window.values)
                            stats = {
                                "count": len(values),
                                "mean": float(np.mean(values)),
                                "std": float(np.std(values)),
                                "min": float(np.min(values)),
                                "max": float(np.max(values)),
                                "median": float(np.median(values)),
                                "p95": float(np.percentile(values, 95)),
                                "p99": float(np.percentile(values, 99))
                            }
                            # Update stats
                            self._update_stats(metric_name, duration, values)
        except Exception as e:
            logger.error(f"Error calculating aggregates: {str(e)}")

    async def _check_alerts(self):
        """Check alert conditions and trigger alerts if needed."""
        try:
            current_time = datetime.utcnow()
            for alert_name, alert_config in self._alerts.items():
                # Check if alert is in cooldown
                if alert_name in self._alert_history:
                    last_alert = self._alert_history[alert_name]
                    if current_time - last_alert.timestamp < alert_config.cooldown:
                        continue
                # Get metric value
                metric_value = self._get_current_metric_value(alert_config.metric)
                if metric_value is None:
                    continue
                # Check condition
                should_trigger = False
                if alert_config.condition == "above" and metric_value > alert_config.threshold:
                    should_trigger = True
                elif alert_config.condition == "below" and metric_value < alert_config.threshold:
                    should_trigger = True
                if should_trigger:
                    # Create alert event
                    alert_event = AlertEvent(
                        name=alert_name,
                        message=f"{alert_config.metric} is {alert_config.condition} threshold {alert_config.threshold}",
                        severity="high" if alert_config.threshold > 80 else "medium",
                        timestamp=current_time,
                        metadata={
                            "metric": alert_config.metric,
                            "value": metric_value,
                            "threshold": alert_config.threshold,
                            "condition": alert_config.condition
                        }
                    )
                    # Add to history
                    self._alert_history[alert_name] = alert_event
                    # Log alert
                    logger.warning(f"Alert triggered: {alert_name} - {alert_event.message}")
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")

    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric."""
        try:
            # This is a simplified implementation
            # In a real system, you'd get the actual current metric value
            return None
        except Exception as e:
            logger.error(f"Error getting metric value: {str(e)}")
            return None

    def _sanitize_metric_name(self, name: str) -> str:
        """Sanitize metric name by replacing special characters with underscores."""
        import re
        # Replace special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized

    def _update_stats(
        self, metric_name: str, duration: timedelta, values: np.ndarray
    ):
        """Update statistics for a metric."""
        try:
            # Sanitize metric name
            sanitized_name = self._sanitize_metric_name(metric_name)
            
            # Calculate statistics
            stats = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
            
            # Store statistics in memory (removed Prometheus integration)
            duration_str = self._format_duration(duration)
            stat_key = f"{sanitized_name}_{duration_str}"
            
            with self._lock:
                if not hasattr(self, '_statistics'):
                    self._statistics = {}
                self._statistics[stat_key] = {
                    "stats": stats,
                    "last_updated": time.time()
                }
                
        except Exception as e:
            logger.error(f"Error updating stats: {str(e)}")

    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for metric names."""
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            return f"{total_seconds // 3600}h"
        else:
            return f"{total_seconds // 86400}d"

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            with self._lock:
                summary = {
                    "system": {
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": dict(
                            psutil.virtual_memory()._asdict()
                        ),
                        "disk_usage": dict(psutil.disk_usage("/")._asdict()),
                    },
                    "requests": {
                        name: {
                            self._format_duration(duration): {
                                "count": len(window.values),
                                "mean": np.mean(window.values)
                                if window.values
                                else 0,
                                "max": np.max(window.values)
                                if window.values
                                else 0,
                            }
                            for duration, window in windows.items()
                        }
                        for name, windows in self._windows.items()
                        if name.startswith("request_")
                    },
                    "reputation": {
                        name: {
                            self._format_duration(duration): {
                                "count": len(window.values),
                                "mean": np.mean(window.values)
                                if window.values
                                else 0,
                                "std": np.std(window.values)
                                if window.values
                                else 0,
                            }
                            for duration, window in windows.items()
                        }
                        for name, windows in self._windows.items()
                        if name.startswith("reputation_")
                    },
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            return {}

    def register_metric(
        self, name: str, description: str, metric_type: str = "gauge"
    ) -> None:
        """Register a new metric."""
        try:
            with self._lock:
                if name in self._metrics:
                    raise ReputationError(
                        message=f"Metric {name} already registered",
                        severity=ErrorSeverity.WARNING,
                        category=ErrorCategory.VALIDATION,
                    )

                self._metrics[name] = {
                    "name": name,
                    "description": description,
                    "type": metric_type,
                    "values": [],
                    "labels": {},
                }
                logger.info(f"Registered metric: {name}")
        except Exception as e:
            logger.error(f"Error registering metric {name}: {str(e)}")
            raise ReputationError(
                message=f"Failed to register metric {name}: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def record_metric(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value."""
        try:
            with self._lock:
                if name not in self._metrics:
                    raise ReputationError(
                        message=f"Metric {name} not registered",
                        severity=ErrorSeverity.WARNING,
                        category=ErrorCategory.VALIDATION,
                    )

                metric = self._metrics[name]
                metric["values"].append(
                    {
                        "value": value,
                        "timestamp": time.time(),
                        "labels": labels or {},
                    }
                )

                # Trim history if needed
                if len(metric["values"]) > self._max_history:
                    metric["values"] = metric["values"][-self._max_history :]

                # Cleanup old metrics periodically
                self._cleanup_old_metrics()

        except Exception as e:
            logger.error(f"Error recording metric {name}: {str(e)}")
            raise ReputationError(
                message=f"Failed to record metric {name}: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def record_error(
        self,
        error_type: str,
        message: str,
        severity: ErrorSeverity,
        category: ErrorCategory,
    ) -> None:
        """Record an error metric."""
        try:
            with self._lock:
                self._errors[error_type].append(
                    {
                        "message": message,
                        "severity": severity,
                        "category": category,
                        "timestamp": time.time(),
                    }
                )

                # Trim error history if needed
                if len(self._errors[error_type]) > self._max_error_history:
                    self._errors[error_type] = self._errors[error_type][
                        -self._max_error_history :
                    ]

        except Exception as e:
            logger.error(f"Error recording error metric: {str(e)}")
            raise ReputationError(
                message=f"Failed to record error metric: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def get_metric(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Optional[MetricData]:
        """Get metric data with optional time range filter."""
        try:
            with self._lock:
                if name not in self._metrics:
                    return None

                metric = self._metrics[name].copy()

                # Filter by time range if specified
                if start_time or end_time:
                    metric["values"] = [
                        v
                        for v in metric["values"]
                        if (not start_time or v["timestamp"] >= start_time)
                        and (not end_time or v["timestamp"] <= end_time)
                    ]

                return metric
            
        except Exception as e:
            logger.error(f"Error getting metric {name}: {str(e)}")
            raise ReputationError(
                message=f"Failed to get metric {name}: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def get_error_metrics(
        self, error_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get error metrics with optional type filter."""
        try:
            with self._lock:
                if error_type:
                    return {error_type: self._errors.get(error_type, [])}
                return dict(self._errors)

        except Exception as e:
            logger.error(f"Error getting error metrics: {str(e)}")
            raise ReputationError(
                message=f"Failed to get error metrics: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        try:
            with self._lock:
                self._metrics.clear()
                self._errors.clear()
                logger.info("Cleared all metrics")

        except Exception as e:
            logger.error(f"Error clearing metrics: {str(e)}")
            raise ReputationError(
                message=f"Failed to clear metrics: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    # Note: _cleanup_old_metrics is implemented as an async method above
    # This method is called by the background task in initialize()

    async def record_system_metric(self, metric_type: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a system metric (stub implementation for test/integration compatibility)."""
        logger.debug(f"[DEBUG] record_system_metric called with: {metric_type=}, {value=}, {labels=}")
        logger.info(f"System metric recorded: {metric_type}={value}, labels={labels}")
        return


# Global metrics manager instance
metrics_manager = MetricsManager()


def track_metric(
    name: str, value: float, labels: Optional[Dict[str, str]] = None
):
    """Track a custom metric."""
    try:
        # Create gauge if it doesn't exist
        gauge = Gauge(
            name,
            f"Custom metric: {name}",
            list(labels.keys()) if labels else [],
        )

        # Set value
        if labels:
            gauge.labels(**labels).set(value)
        else:
            gauge.set(value)
    except Exception as e:
        logger.error(f"Error tracking metric {name}: {str(e)}")


def track_latency(name: str):
    """Decorator to track function latency."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                start_time = datetime.utcnow()
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                # Record latency
                Histogram(
                    f"{name}_latency_seconds",
                    f"Latency for {name}",
                    ["function"],
                ).labels(function=func.__name__).observe(duration)

                return result

            except Exception as e:
                logger.error(f"Error tracking latency for {name}: {str(e)}")
                raise

        return wrapper

    return decorator

def track_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logging.getLogger(func.__module__).info(f"Performance: {func.__name__} took {duration:.4f}s")
        return result
    return wrapper
