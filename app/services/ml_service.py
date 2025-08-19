"""
Machine learning service for advanced analytics and predictions.
Provides ML-powered insights and predictive capabilities.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

from app.core.config import get_settings
from app.core.metrics import ML_PREDICTION_LATENCY
from app.core.optimizations import CircuitBreaker

logger = logging.getLogger(__name__)
settings = get_settings()


class TrendPredictor:
    """Advanced trend prediction system."""

    def __init__(self):
        """Initialize trend predictor."""
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()

    async def predict_trends(
        self,
        historical_data: List[Dict[str, Any]],
        prediction_window: int = 30,
    ) -> Dict[str, Any]:
        """Predict future trends based on historical data."""
        try:
            # Prepare data
            df = pd.DataFrame(historical_data)
            features = self._prepare_features(df)

            if len(features) < prediction_window:
                return {"status": "insufficient_data"}

            # Train model
            X = features[:-prediction_window]
            y = df["value"].values[:-prediction_window]
            self.model.fit(X, y)

            # Generate predictions
            future_features = self._prepare_future_features(
                df, prediction_window
            )
            predictions = self.model.predict(future_features)

            return {
                "status": "success",
                "predictions": [
                    {
                        "date": (
                            datetime.now(timezone.utc) + timedelta(days=i)
                        ).isoformat(),
                        "value": float(pred),
                    }
                    for i, pred in enumerate(predictions)
                ],
                "confidence_scores": self._calculate_confidence_scores(
                    predictions
                ),
            }

        except Exception as e:
            logger.error("Error predicting trends: %s", e)
            return {"status": "error", "message": str(e)}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        df["date"] = pd.to_datetime(df["date"])

        features = []
        for i in range(len(df)):
            row_features = [
                df["value"].values[i],
                df["date"].dt.hour.values[i],
                df["date"].dt.dayofweek.values[i],
                df["date"].dt.month.values[i],
            ]
            features.append(row_features)

        return np.array(features)

    def _prepare_future_features(
        self, df: pd.DataFrame, window: int
    ) -> np.ndarray:
        """Prepare features for future predictions."""
        last_date = pd.to_datetime(df["date"].max())
        future_dates = [
            last_date + timedelta(days=i) for i in range(1, window + 1)
        ]

        features = []
        for date in future_dates:
            features.append(
                [
                    df["value"].mean(),  # Use mean as baseline
                    date.hour,
                    date.dayofweek,
                    date.month,
                ]
            )

        return np.array(features)

    def _calculate_confidence_scores(
        self, predictions: np.ndarray
    ) -> List[float]:
        """Calculate confidence scores for predictions."""
        # Implement confidence calculation logic
        return [0.8] * len(predictions)  # Placeholder


class AnomalyDetector:
    """Advanced anomaly detection system."""

    def __init__(self):
        """Initialize anomaly detector."""
        self.model = IsolationForest(contamination=0.1, random_state=42)

    async def detect_anomalies(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect anomalies in data."""
        try:
            # Prepare data
            df = pd.DataFrame(data)
            features = self._prepare_features(df)

            if len(features) < 10:
                return {"status": "insufficient_data"}

            # Detect anomalies
            predictions = self.model.fit_predict(features)
            anomaly_indices = np.where(predictions == -1)[0]

            return {
                "status": "success",
                "anomalies": [
                    {
                        "index": int(idx),
                        "data_point": data[idx],
                        "severity": self._calculate_severity(
                            features[idx], features
                        ),
                    }
                    for idx in anomaly_indices
                ],
                "total_anomalies": len(anomaly_indices),
            }

        except Exception as e:
            logger.error("Error detecting anomalies: %s", e)
            return {"status": "error", "message": str(e)}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        return df[numeric_columns].values

    def _calculate_severity(
        self, point: np.ndarray, all_points: np.ndarray
    ) -> str:
        """Calculate anomaly severity."""
        distance = np.linalg.norm(point - np.mean(all_points, axis=0))
        std_distance = np.std(
            [
                np.linalg.norm(p - np.mean(all_points, axis=0))
                for p in all_points
            ]
        )

        if distance > 3 * std_distance:
            return "critical"
        elif distance > 2 * std_distance:
            return "severe"
        return "moderate"


class PatternAnalyzer:
    """Pattern analysis system."""

    def __init__(self):
        """Initialize pattern analyzer."""
        self.cluster_model = KMeans(n_clusters=3)

    async def analyze_patterns(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns in data."""
        try:
            # Prepare data
            df = pd.DataFrame(data)
            features = self._prepare_features(df)

            if len(features) < 10:
                return {"status": "insufficient_data"}

            # Perform clustering
            clusters = self.cluster_model.fit_predict(features)

            # Analyze patterns
            patterns = self._analyze_clusters(df, clusters)

            return {
                "status": "success",
                "patterns": patterns,
                "cluster_stats": self._get_cluster_stats(df, clusters),
            }

        except Exception as e:
            logger.error("Error analyzing patterns: %s", e)
            return {"status": "error", "message": str(e)}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for pattern analysis."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        return self.scaler.fit_transform(df[numeric_columns])

    def _analyze_clusters(
        self, df: pd.DataFrame, clusters: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Analyze characteristics of each cluster."""
        patterns = []

        for cluster_id in np.unique(clusters):
            cluster_data = df[clusters == cluster_id]

            patterns.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": len(cluster_data),
                    "characteristics": self._get_cluster_characteristics(
                        cluster_data
                    ),
                }
            )

        return patterns

    def _get_cluster_characteristics(
        self, cluster_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get characteristics of cluster."""
        return {
            "mean_values": cluster_data.mean().to_dict(),
            "std_values": cluster_data.std().to_dict(),
            "dominant_categories": self._get_dominant_categories(cluster_data),
        }

    def _get_cluster_stats(
        self, df: pd.DataFrame, clusters: np.ndarray
    ) -> Dict[str, Any]:
        """Get statistical information about clusters."""
        return {
            "cluster_sizes": pd.Series(clusters).value_counts().to_dict(),
            "cluster_proportions": (
                pd.Series(clusters).value_counts(normalize=True).to_dict()
            ),
        }

    def _get_dominant_categories(
        self, df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Get dominant categories in categorical columns."""
        categorical_columns = df.select_dtypes(include=["object"]).columns

        dominants = {}
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            dominants[col] = value_counts.index.tolist()[:3]

        return dominants


class MLService:
    """Machine learning service."""

    def __init__(self):
        """Initialize ML service."""
        self.trend_predictor = TrendPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def analyze_data(
        self, data: List[Dict[str, Any]], analysis_types: List[str]
    ) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        start_time = datetime.now(timezone.utc)
        results = {}

        try:
            # Perform requested analyses
            if "trends" in analysis_types:
                results["trends"] = await self.trend_predictor.predict_trends(
                    data
                )

            if "anomalies" in analysis_types:
                results[
                    "anomalies"
                ] = await self.anomaly_detector.detect_anomalies(data)

            if "patterns" in analysis_types:
                results[
                    "patterns"
                ] = await self.pattern_analyzer.analyze_patterns(data)

            if "sentiment" in analysis_types:
                results["sentiment"] = await self._analyze_sentiment(data)

            # Record latency
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            ML_PREDICTION_LATENCY.observe(duration)

            return {
                "status": "success",
                "results": results,
                "analysis_time": duration,
            }

        except Exception as e:
            logger.error("Error performing analysis: %s", e)
            return {"status": "error", "message": str(e)}

    async def _analyze_sentiment(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment in text data."""
        try:
            texts = [item.get("text", "") for item in data if "text" in item]

            if not texts:
                return {"status": "no_text_data"}

            # Analyze sentiment
            sentiments = self.sentiment_analyzer(texts)

            return {
                "status": "success",
                "sentiments": sentiments,
                "summary": self._summarize_sentiments(sentiments),
            }

        except Exception as e:
            logger.error("Error analyzing sentiment: %s", e)
            return {"status": "error", "message": str(e)}

    def _summarize_sentiments(
        self, sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize sentiment analysis results."""
        labels = [s["label"] for s in sentiments]
        scores = [s["score"] for s in sentiments]

        return {
            "positive_ratio": labels.count("POSITIVE") / len(labels),
            "negative_ratio": labels.count("NEGATIVE") / len(labels),
            "average_score": sum(scores) / len(scores),
        }
