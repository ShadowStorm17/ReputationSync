"""
Enhanced analytics service with advanced reputation metrics.
Provides ML-powered insights and predictive analytics.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from app.core.metrics import ANALYTICS_LATENCY
from app.core.optimizations import CircuitBreaker, cache_warmer


class ReputationScore:
    """Advanced reputation scoring system."""

    def __init__(self):
        """Initialize reputation scoring."""
        self.weights = {
            "sentiment": 0.3,
            "engagement": 0.25,
            "reach": 0.2,
            "influence": 0.15,
            "consistency": 0.1,
        }

    async def calculate_score(
            self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate weighted reputation score."""
        score = 0
        subscores = {}

        for metric, weight in self.weights.items():
            if metric in metrics:
                subscores[metric] = metrics[metric]
                score += metrics[metric] * weight

        return {
            "overall_score": score,
            "subscores": subscores,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class TrendAnalyzer:
    """Advanced trend analysis system."""

    def __init__(self):
        """Initialize trend analyzer."""
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100)

    async def analyze_trends(
        self, historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze reputation trends and predict future values."""
        df = pd.DataFrame(historical_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Calculate trend indicators
        trends = {
            "short_term": self._calculate_trend(df, days=7),
            "medium_term": self._calculate_trend(df, days=30),
            "long_term": self._calculate_trend(df, days=90),
        }

        # Predict future values
        predictions = await self._predict_future_values(df)

        return {
            "trends": trends,
            "predictions": predictions,
            "volatility": self._calculate_volatility(df),
            "seasonality": self._detect_seasonality(df),
        }

    def _calculate_trend(self, df: pd.DataFrame,
                         days: int) -> Dict[str, float]:
        """Calculate trend over specified period."""
        recent_data = df[df["timestamp"] >
                         datetime.now(timezone.utc) - timedelta(days=days)]

        if len(recent_data) < 2:
            return {"slope": 0, "change_rate": 0}

        values = recent_data["score"].values
        slope = np.polyfit(range(len(values)), values, 1)[0]
        change_rate = (values[-1] - values[0]) / \
            values[0] if values[0] != 0 else 0

        return {"slope": float(slope), "change_rate": float(change_rate)}

    async def _predict_future_values(
            self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict future reputation scores."""
        if len(df) < 30:  # Need sufficient historical data
            return []

        # Prepare features
        features = self._prepare_features(df)

        # Train model on historical data
        X = features[:-30]  # Use all but last 30 days for training
        y = df["score"].values[:-30]

        if len(X) < 30:
            return []

        self.model.fit(X, y)

        # Predict next 30 days
        future_features = self._prepare_future_features(df)
        predictions = self.model.predict(future_features)

        return [
            {
                "date": (datetime.now(timezone.utc) + timedelta(days=i)).isoformat(),
                "predicted_score": float(score),
            }
            for i, score in enumerate(predictions)
        ]

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        features = []

        for i in range(len(df)):
            row_features = [
                df["sentiment_score"].values[i],
                df["engagement_rate"].values[i],
                df["reach"].values[i],
                df["influence_score"].values[i],
                df["hour"].values[i],
                df["day_of_week"].values[i],
            ]
            features.append(row_features)

        return np.array(features)

    def _prepare_future_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for future predictions."""
        last_values = df.iloc[-1]
        future_features = []

        for i in range(30):  # Predict 30 days ahead
            future_date = datetime.now(timezone.utc) + timedelta(days=i)
            features = [
                last_values["sentiment_score"],
                last_values["engagement_rate"],
                last_values["reach"],
                last_values["influence_score"],
                future_date.hour,
                future_date.weekday(),
            ]
            future_features.append(features)

        return np.array(future_features)

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate reputation score volatility."""
        if len(df) < 2:
            return 0.0

        returns = df["score"].pct_change().dropna()
        return float(returns.std())

    def _detect_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in reputation scores."""
        if len(df) < 14:  # Need at least 2 weeks of data
            return {"has_seasonality": False}

        # Group by different time periods
        hourly = df.groupby(df["timestamp"].dt.hour)["score"].mean()
        daily = df.groupby(df["timestamp"].dt.dayofweek)["score"].mean()

        return {
            "has_seasonality": True,
            "patterns": {"hourly": hourly.to_dict(), "daily": daily.to_dict()},
        }


class CompetitorAnalysis:
    """Competitor reputation analysis system."""

    async def analyze_competitors(
        self, competitor_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze competitor reputation metrics."""
        if not competitor_data:
            return {"status": "no_data"}

        df = pd.DataFrame(competitor_data)

        analysis = {
            "rankings": self._calculate_rankings(df),
            "strengths_weaknesses": self._analyze_strengths_weaknesses(df),
            "market_position": self._analyze_market_position(df),
            "recommendations": self._generate_recommendations(df),
        }

        return analysis

    def _calculate_rankings(
            self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate rankings for different metrics."""
        rankings = {}

        for metric in [
            "reputation_score",
            "engagement_rate",
                "sentiment_score"]:
            if metric in df.columns:
                sorted_df = df.sort_values(metric, ascending=False)
                rankings[metric] = [
                    {"competitor": row["name"], "score": float(row[metric])}
                    for _, row in sorted_df.iterrows()
                ]

        return rankings

    def _analyze_strengths_weaknesses(
            self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze competitor strengths and weaknesses."""
        analysis = {}

        for _, row in df.iterrows():
            strengths = []
            weaknesses = []

            # Analyze each metric
            metrics = {
                "engagement_rate": "Engagement",
                "sentiment_score": "Sentiment",
                "reach": "Reach",
                "response_rate": "Response Rate",
            }

            for metric, label in metrics.items():
                if metric in row:
                    if row[metric] > df[metric].mean():
                        strengths.append(f"Strong {label}")
                    else:
                        weaknesses.append(f"Weak {label}")

            analysis[row["name"]] = {
                "strengths": strengths, "weaknesses": weaknesses}

        return analysis

    def _analyze_market_position(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market position and share."""
        total_engagement = df["engagement_rate"].sum()
        total_reach = df["reach"].sum()

        positions = []
        for _, row in df.iterrows():
            positions.append(
                {
                    "competitor": row["name"],
                    "market_share": float(
                        row["engagement_rate"] /
                        total_engagement),
                    "reach_share": float(
                        row["reach"] /
                        total_reach),
                })

        return {
            "positions": positions,
            "market_concentration": self._calculate_concentration(df),
        }

    def _calculate_concentration(self, df: pd.DataFrame) -> float:
        """Calculate market concentration (HHI)."""
        total = df["engagement_rate"].sum()
        shares = df["engagement_rate"] / total
        return float((shares**2).sum())

    def _generate_recommendations(
            self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate strategic recommendations."""
        recommendations = []

        # Analyze gaps and opportunities
        market_leader = df.loc[df["reputation_score"].idxmax()]
        our_position = df[df["is_self"]
                          ].iloc[0] if "is_self" in df.columns else None

        if our_position is not None:
            gap = market_leader["reputation_score"] - \
                our_position["reputation_score"]

            if gap > 0:
                recommendations.append(
                    {
                        "type": "improvement",
                        "area": "reputation_score",
                        "description": f"Close {gap:.2f} point gap with market leader",
                        "priority": "high",
                    })

        # Add more specific recommendations based on metrics
        metrics = ["engagement_rate", "sentiment_score", "response_rate"]
        for metric in metrics:
            if metric in df.columns and our_position is not None:
                if our_position[metric] < df[metric].mean():
                    recommendations.append(
                        {
                            "type": "improvement",
                            "area": metric,
                            "description": f"Improve {metric.replace('_', ' ')} to meet market average",
                            "priority": "medium",
                        })

        return recommendations


class AnomalyDetector:
    """Advanced anomaly detection system."""

    def __init__(self):
        """Initialize anomaly detector."""
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()

    async def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in reputation metrics."""
        if len(data) < 10:  # Need sufficient data
            return {"status": "insufficient_data"}

        # Prepare features
        features = self._prepare_features(data)
        scaled_features = self.scaler.fit_transform(features)

        # Detect anomalies
        predictions = self.model.fit_predict(scaled_features)
        anomaly_indices = np.where(predictions == -1)[0]

        # Analyze anomalies
        anomalies = []
        for idx in anomaly_indices:
            anomaly = {
                "timestamp": data.iloc[idx]["timestamp"],
                "metrics": {
                    col: data.iloc[idx][col]
                    for col in data.columns
                    if col != "timestamp"
                },
                "severity": self._calculate_anomaly_severity(data.iloc[idx], data),
            }
            anomalies.append(anomaly)

        return {
            "anomalies": anomalies,
            "total_count": len(anomalies),
            "severity_distribution": self._get_severity_distribution(anomalies),
        }

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection."""
        feature_columns = [
            "reputation_score",
            "sentiment_score",
            "engagement_rate",
            "response_rate",
        ]

        return df[feature_columns].values

    def _calculate_anomaly_severity(
            self,
            anomaly: pd.Series,
            df: pd.DataFrame) -> str:
        """Calculate severity of anomaly."""
        z_scores = {}
        for col in df.columns:
            if col != "timestamp":
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    z_scores[col] = abs((anomaly[col] - mean) / std)

        max_z_score = max(z_scores.values())

        if max_z_score > 3:
            return "critical"
        elif max_z_score > 2:
            return "severe"
        else:
            return "moderate"

    def _get_severity_distribution(
        self, anomalies: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get distribution of anomaly severities."""
        distribution = {"critical": 0, "severe": 0, "moderate": 0}

        for anomaly in anomalies:
            distribution[anomaly["severity"]] += 1

        return distribution


class RealtimeTrendAnalyzer:
    """Real-time trend analysis system."""

    def __init__(self):
        """Initialize real-time analyzer."""
        self.window_sizes = {
            "short": timedelta(hours=1),
            "medium": timedelta(hours=6),
            "long": timedelta(hours=24),
        }

    async def analyze_realtime_trends(
            self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze real-time trends in reputation metrics."""
        now = datetime.now(timezone.utc)
        trends = {}

        for window_name, window_size in self.window_sizes.items():
            window_data = data[data["timestamp"] >= now - window_size]

            if len(window_data) < 2:
                continue

            trends[window_name] = {
                "metrics": self._calculate_window_metrics(window_data),
                "changes": self._calculate_changes(window_data),
                "velocity": self._calculate_velocity(window_data),
            }

        return {
            "trends": trends,
            "alerts": self._generate_alerts(trends),
            "summary": self._generate_trend_summary(trends),
        }

    def _calculate_window_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for time window."""
        metrics = {}

        for col in df.columns:
            if col != "timestamp":
                metrics[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

        return metrics

    def _calculate_changes(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metric changes in window."""
        changes = {}

        for col in df.columns:
            if col != "timestamp":
                first_value = df[col].iloc[0]
                last_value = df[col].iloc[-1]

                if first_value != 0:
                    change = (last_value - first_value) / first_value
                    changes[col] = float(change)

        return changes

    def _calculate_velocity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate rate of change for metrics."""
        velocity = {}

        for col in df.columns:
            if col != "timestamp":
                # Calculate rate of change per hour
                time_diff = (
                    df["timestamp"].max() - df["timestamp"].min()
                ).total_seconds() / 3600
                if time_diff > 0:
                    value_diff = df[col].max() - df[col].min()
                    velocity[col] = float(value_diff / time_diff)

        return velocity

    def _generate_alerts(self, trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on trends."""
        alerts = []

        for window_name, window_data in trends.items():
            changes = window_data["changes"]

            for metric, change in changes.items():
                if abs(change) > 0.2:  # 20% change threshold
                    alerts.append({"type": "significant_change",
                                   "metric": metric,
                                   "window": window_name,
                                   "change": change,
                                   "priority": "high" if abs(change) > 0.5 else "medium",
                                   })

        return alerts

    def _generate_trend_summary(
            self, trends: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable trend summary."""
        summary = {}

        for window_name, window_data in trends.items():
            changes = window_data["changes"]
            significant_changes = {
                k: v for k, v in changes.items() if abs(v) > 0.1  # 10% change threshold
            }

            if significant_changes:
                summary[window_name] = self._format_trend_summary(
                    significant_changes)

        return summary

    def _format_trend_summary(self, changes: Dict[str, float]) -> str:
        """Format trend changes into readable summary."""
        parts = []

        for metric, change in changes.items():
            direction = "increased" if change > 0 else "decreased"
            percentage = abs(round(change * 100, 1))
            parts.append(f"{metric} {direction} by {percentage}%")

        return "; ".join(parts)


class EnhancedAnalytics:
    """Enhanced analytics service with advanced features."""

    def __init__(self):
        """Initialize enhanced analytics."""
        self.reputation_scorer = ReputationScore()
        self.trend_analyzer = TrendAnalyzer()
        self.competitor_analyzer = CompetitorAnalysis()
        self.anomaly_detector = AnomalyDetector()
        self.realtime_analyzer = RealtimeTrendAnalyzer()

    @cache_warmer(["reputation_score", "trends", "competitor_analysis"])
    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def analyze_reputation(
        self,
        metrics: Dict[str, float],
        historical_data: List[Dict[str, Any]],
        competitor_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive reputation analysis."""
        start_time = datetime.now(timezone.utc)

        # Convert historical data to DataFrame
        df = pd.DataFrame(historical_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Calculate current reputation score
        score = await self.reputation_scorer.calculate_score(metrics)

        # Analyze trends and predictions
        trends = await self.trend_analyzer.analyze_trends(historical_data)

        # Detect anomalies
        anomalies = await self.anomaly_detector.detect_anomalies(df)

        # Analyze real-time trends
        realtime_trends = await self.realtime_analyzer.analyze_realtime_trends(df)

        # Analyze competitors if data available
        competitor_analysis = None
        if competitor_data:
            competitor_analysis = await self.competitor_analyzer.analyze_competitors(
                competitor_data
            )

        # Record latency
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        ANALYTICS_LATENCY.observe(duration)

        return {
            "reputation_score": score,
            "trends": trends,
            "anomalies": anomalies,
            "realtime_trends": realtime_trends,
            "competitor_analysis": competitor_analysis,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
