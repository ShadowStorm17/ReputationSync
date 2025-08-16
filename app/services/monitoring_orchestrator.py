import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Dict, List

from circuitbreaker import circuit
from fastapi import WebSocket
from prometheus_client import Counter, Histogram
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.telemetry import log_error, monitor_execution_time
from app.services.comment_service import CommentService
from app.services.predictive_service import PredictiveService
from app.services.response_service import ResponseService
from app.services.sentiment_service import SentimentService
from app.services.enhanced_monitoring import MONITOR_ERRORS, MONITOR_LATENCY, ALERT_TRIGGERS

logger = logging.getLogger(__name__)
settings = get_settings()


class MonitoringOrchestrator:
    """Advanced monitoring orchestration service with ML-powered analytics
    and high resilience."""

    def __init__(self):
        self.sentiment_service = SentimentService()
        self.predictive_service = PredictiveService()
        self.comment_service = CommentService()
        self.response_service = ResponseService()
        self.active_monitors: Dict[str, Dict] = {}
        self.websocket_clients: Dict[str, List[WebSocket]] = {}

        # Initialize ML models with error handling
        self._initialize_ml_models()

        # Enhanced alert thresholds with adaptive learning
        self.alert_thresholds = {
            "sentiment": {
                "critical": -0.7,
                "warning": -0.4,
                "adaptive": True,
                "learning_rate": 0.1,
                "min_threshold": -0.9,
                "max_threshold": -0.2
            },
            "engagement": {
                "drop_percent": 30,
                "window_hours": 24,
                "adaptive": True,
                "min_threshold": 10,
                "max_threshold": 50,
                "recovery_threshold": 15
            },
            "velocity": {
                "mention_spike": 200,
                "timeframe_minutes": 30,
                "adaptive": True,
                "sensitivity": 0.8,
                "cooldown_period": 60
            },
            "anomaly": {
                "score_threshold": -0.7,
                "min_samples": 10,
                "adaptive": True,
                "false_positive_rate": 0.01
            }
        }

        # Initialize performance metrics with detailed tracking
        self.performance_metrics = {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "alert_accuracy": 1.0,
            "response_times": [],
            "error_rates": {},
            "recovery_success": 0,
            "total_recoveries": 0
        }

        # Initialize recovery strategies
        self.recovery_strategies = {
            "data_collection": self._recover_data_collection,
            "analysis": self._recover_analysis,
            "alerting": self._recover_alerting
        }

    def _initialize_ml_models(self):
        """Initialize ML models with error handling and validation."""
        try:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=200,
                max_samples='auto'
            )
            self.scaler = StandardScaler()

            # (Validation step skipped: _validate_ml_models not implemented)

        except Exception as e:
            logger.error("Error initializing ML models: %s", e)
            # Fallback to simpler models if necessary
            self.anomaly_detector = None
            self.scaler = None
            logger.warning("Falling back: ML models could not be initialized. Using None.")

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    @circuit(failure_threshold=5, recovery_timeout=60)
    @monitor_execution_time(MONITOR_LATENCY)
    async def start_advanced_monitoring(
        self,
        entity_id: str,
        config: Dict
    ) -> Dict:
        """Start comprehensive monitoring with advanced ML features and error
        recovery."""
        try:
            monitor_id = f"monitor_{entity_id}_{datetime.now(timezone.utc).timestamp()}"

            # Validate configuration
            self._validate_monitor_config(config)

            # Initialize monitor with enhanced error tracking
            monitor_config = await self._initialize_monitor(entity_id, config)

            # Store monitor configuration with backup
            await self._store_monitor_config(monitor_id, monitor_config)

            # Start monitoring tasks with supervision
            await self._start_supervised_monitoring(monitor_id)

            return {
                "monitor_id": monitor_id,
                "status": "started",
                "config": config,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "health_check": await self._perform_health_check(monitor_id)
            }

        except Exception as e:
            MONITOR_ERRORS.inc()
            log_error("start_monitoring_error", str(e))
            raise

    async def _initialize_monitor(self, entity_id: str, config: Dict) -> Dict:
        """Initialize monitor with comprehensive configuration."""
        try:
            return {
                "entity_id": entity_id,
                "config": config,
                "metrics": {},
                "baselines": await self._calculate_baselines(entity_id),
                "active_alerts": [],
                "last_check": datetime.now(timezone.utc),
                "status": "active",
                "health": {
                    "last_successful_run": None,
                    "error_count": 0,
                    "recovery_attempts": 0
                },
                "ml_state": {
                    "anomaly_scores": [],
                    "trend_patterns": [],
                    "adaptive_thresholds": (
                        self._initialize_adaptive_thresholds()
                    )
                }
            }
        except Exception as e:
            logger.error("Error initializing monitor: %s", e)
            # Attempt recovery or use fallback configuration
            return await self._get_fallback_configuration(entity_id, config)

    @monitor_execution_time(MONITOR_LATENCY)
    async def _run_monitoring_loop(self, monitor_id: str):
        """Enhanced monitoring loop with error recovery and performance
        optimization."""
        try:
            while monitor_id in self.active_monitors:
                monitor = self.active_monitors[monitor_id]

                try:
                    # Collect data with timeout protection
                    async with asyncio.timeout(30):
                        data = await self._collect_monitoring_data(
                            monitor["entity_id"]
                        )

                    # Perform analysis with error isolation
                    analysis = await self._analyze_monitoring_data(
                        data,
                        monitor["baselines"]
                    )

                    # Check for issues with enhanced detection
                    alerts = await self._check_alert_conditions(
                        analysis,
                        monitor["baselines"]
                    )

                    # Update monitor state with validation
                    await self._update_monitor_state(
                        monitor_id, analysis, alerts)

                    # Update health metrics
                    monitor["health"]["last_successful_run"] = (
                        datetime.now(timezone.utc))

                except Exception as e:
                    await self._handle_monitoring_error(monitor_id, e)
                    continue

                # Adaptive sleep based on system load and priority
                await self._adaptive_sleep(monitor_id)

        except Exception as e:
            MONITOR_ERRORS.inc()
            log_error("monitoring_loop_error", str(e))
            await self._initiate_recovery(monitor_id)

    async def _handle_monitoring_error(
            self, monitor_id: str, error: Exception):
        """Enhanced error handling with recovery strategies."""
        monitor = self.active_monitors[monitor_id]
        monitor["health"]["error_count"] += 1

        logger.error("Monitoring error for %s: %s", monitor_id, error)
        MONITOR_ERRORS.inc()

        # Update error statistics
        error_type = type(error).__name__
        self.performance_metrics["error_rates"][error_type] = \
            self.performance_metrics["error_rates"].get(error_type, 0) + 1

        # Attempt recovery based on error type
        if monitor["health"]["error_count"] <= 3:
            await self._attempt_recovery(monitor_id, error)
        else:
            await self._initiate_failover(monitor_id)

    async def _attempt_recovery(self, monitor_id: str, error: Exception):
        """Attempt to recover from monitoring errors."""
        monitor = self.active_monitors[monitor_id]
        monitor["health"]["recovery_attempts"] += 1

        try:
            # Identify appropriate recovery strategy
            strategy = self._identify_recovery_strategy(error)

            # Execute recovery
            if strategy in self.recovery_strategies:
                success = await self.recovery_strategies[strategy](monitor_id)

                # Update recovery statistics
                self.performance_metrics["total_recoveries"] += 1
                if success:
                    self.performance_metrics["recovery_success"] += 1

                return success

        except Exception as recovery_error:
            logger.error("Recovery failed for %s: %s", monitor_id, recovery_error)
            return False

    async def _initiate_failover(self, monitor_id: str):
        """Initiate failover to backup monitoring system."""
        try:
            # Notify stakeholders
            await self._notify_stakeholders(monitor_id, "failover_initiated")

            # Switch to backup monitoring
            await self._switch_to_backup_monitoring(monitor_id)

            # Update monitor status
            self.active_monitors[monitor_id]["status"] = "failover"

        except Exception as e:
            logger.error("Failover failed for %s: %s", monitor_id, e)
            # Last resort: suspend monitoring
            await self._suspend_monitoring(monitor_id)

    @monitor_execution_time(MONITOR_LATENCY)
    async def _analyze_monitoring_data(
            self, data: Dict, baselines: Dict) -> Dict:
        """Perform comprehensive data analysis with enhanced accuracy."""
        try:
            analysis = {
                "metrics": await self._analyze_core_metrics(data),
                "anomalies": await self._detect_advanced_anomalies(
                    data, baselines
                ),
                "predictions": await self._generate_enhanced_predictions(data),
                "patterns": await self._analyze_complex_patterns(data),
                "correlations": self._analyze_metric_correlations(data)
            }

            # Validate analysis results
            self._validate_analysis_results(analysis)

            return analysis

        except Exception as e:
            logger.error("Analysis error: %s", e)
            # Return fallback analysis
            return await self._generate_fallback_analysis(data)

    async def _check_alert_conditions(
        self,
        analysis: Dict,
        baselines: Dict
    ) -> List[Dict]:
        """Check for alert conditions with advanced detection and
        validation."""
        try:
            alerts = []

            # Check each alert type with enhanced detection
            sentiment_alerts = await self._check_sentiment_alerts(
                analysis,
                baselines
            )
            engagement_alerts = await self._check_engagement_alerts(
                analysis,
                baselines
            )
            anomaly_alerts = await self._check_anomaly_alerts(
                analysis,
                baselines
            )

            alerts.extend(sentiment_alerts)
            alerts.extend(engagement_alerts)
            alerts.extend(anomaly_alerts)

            # Deduplicate and prioritize alerts
            alerts = self._deduplicate_alerts(alerts)
            alerts = self._prioritize_alerts(alerts)

            # Track alert metrics
            for alert in alerts:
                ALERT_TRIGGERS.inc()

            return alerts

        except Exception as e:
            logger.error("Error checking alerts: %s", e)
            return []

    def _validate_analysis_results(self, analysis: Dict):
        """Validate analysis results for consistency and accuracy."""
        if not analysis.get("metrics"):
            raise ValueError("Missing core metrics in analysis")

        if not isinstance(analysis.get("anomalies"), list):
            raise ValueError("Invalid anomalies format")

        if not isinstance(analysis.get("predictions"), dict):
            raise ValueError("Invalid predictions format")

        # Validate metric ranges
        for metric, value in analysis["metrics"].items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid metric value for {metric}")
            if value < 0 and metric not in ["sentiment"]:
                raise ValueError(f"Negative value not allowed for {metric}")

    async def _adaptive_sleep(self, monitor_id: str):
        """Implement adaptive sleep based on system load and priority."""
        monitor = self.active_monitors[monitor_id]
        base_interval = settings.MONITORING_INTERVAL

        # Adjust interval based on priority
        priority_factor = self._get_priority_factor(monitor)

        # Adjust for system load
        load_factor = await self._get_system_load_factor()

        # Calculate final interval
        adjusted_interval = base_interval * priority_factor * load_factor

        # Add small random jitter to prevent thundering herd
        jitter = random.uniform(0.9, 1.1)
        final_interval = adjusted_interval * jitter

        await asyncio.sleep(final_interval)

    def _get_priority_factor(self, monitor: Dict) -> float:
        """Calculate priority factor for monitoring frequency."""
        base_factor = 1.0

        # Adjust based on active alerts
        if monitor["active_alerts"]:
            base_factor *= 0.5

        # Adjust based on recent anomalies
        if monitor["ml_state"]["anomaly_scores"]:
            recent_anomalies = [
                score for score in monitor["ml_state"]["anomaly_scores"][-5:]
                if score < -0.5
            ]
            if recent_anomalies:
                base_factor *= 0.7

        return max(0.2, min(base_factor, 2.0))  # Clamp between 0.2 and 2.0

    async def _collect_monitoring_data(self, entity_id: str) -> Dict:
        """Collect comprehensive monitoring data."""
        try:
            # Collect data from all relevant sources
            tasks = [
                self._get_sentiment_metrics(entity_id),
                self._get_engagement_metrics(entity_id),
                self._get_mention_metrics(entity_id),
                self._get_comment_metrics(entity_id),
                self._get_competitor_metrics(entity_id)
            ]

            results = await asyncio.gather(*tasks)

            return {
                "sentiment": results[0],
                "engagement": results[1],
                "mentions": results[2],
                "comments": results[3],
                "competitor": results[4],
                "collected_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("Error collecting monitoring data: %s", e)
            raise

    async def _analyze_core_metrics(self, data: Dict) -> Dict:
        """Analyze core metrics from monitoring data."""
        try:
            # Implement core metrics analysis logic here
            return {}
        except Exception as e:
            logger.error("Error analyzing core metrics: %s", e)
            raise

    async def _detect_advanced_anomalies(
            self, data: Dict, baselines: Dict) -> List[Dict]:
        """Detect advanced anomalies using ML models."""
        try:
            # Implement advanced anomaly detection logic here
            return []
        except Exception as e:
            logger.error("Error detecting advanced anomalies: %s", e)
            raise

    async def _generate_enhanced_predictions(self, data: Dict) -> Dict:
        """Generate enhanced predictions using ML models."""
        try:
            # Implement enhanced prediction logic here
            return {}
        except Exception as e:
            logger.error("Error generating enhanced predictions: %s", e)
            raise

    async def _analyze_complex_patterns(self, data: Dict) -> Dict:
        """Analyze complex patterns in monitoring data."""
        try:
            # Implement complex pattern analysis logic here
            return {}
        except Exception as e:
            logger.error("Error analyzing complex patterns: %s", e)
            raise

    async def _analyze_metric_correlations(self, data: Dict) -> Dict:
        """Analyze metric correlations in monitoring data."""
        try:
            # Implement metric correlation analysis logic here
            return {}
        except Exception as e:
            logger.error("Error analyzing metric correlations: %s", e)
            raise

    async def _generate_fallback_analysis(self, data: Dict) -> Dict:
        """Generate fallback analysis when primary analysis fails."""
        try:
            # Implement fallback analysis logic here
            return {}
        except Exception as e:
            logger.error("Error generating fallback analysis: %s", e)
            raise

    def _initialize_adaptive_thresholds(self) -> Dict:
        """Initialize adaptive thresholds with default values."""
        return {
            "sentiment": {
                "current": self.alert_thresholds["sentiment"]["warning"],
                "history": [],
                "adaptation_rate": 0.1
            },
            "engagement": {
                "current": self.alert_thresholds["engagement"]["drop_percent"],
                "history": [],
                "adaptation_rate": 0.1
            },
            "velocity": {
                "current": self.alert_thresholds["velocity"]["mention_spike"],
                "history": [],
                "adaptation_rate": 0.1
            }
        }

    def _recover_data_collection(self, monitor_id):
        """Stub: Recover data collection."""
        logger.warning("Stub: Recovering data collection for %s", monitor_id)
        return True

    def _recover_analysis(self, monitor_id):
        """Stub: Recover analysis."""
        logger.warning("Stub: Recovering analysis for %s", monitor_id)
        return True

    def _recover_alerting(self, monitor_id):
        """Stub: Recover alerting."""
        logger.warning("Stub: Recovering alerting for %s", monitor_id)
        return True
