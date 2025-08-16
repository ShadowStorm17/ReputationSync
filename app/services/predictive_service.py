"""
Advanced predictive analytics service.
Provides ML-based forecasting and trend analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prometheus_client import Counter
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tenacity import retry, stop_after_attempt, wait_exponential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from app.core.config import get_settings
from app.core.metrics import PREDICTION_LATENCY, PREDICTION_ERRORS
from app.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)
settings = get_settings()


class PredictiveService:
    """Advanced predictive analytics service with ML-based forecasting."""

    def __init__(self):
        """Initialize advanced predictive models and tools."""
        self.sentiment_service = SentimentService()
        self.cache_ttl = settings.cache["predictions"] if hasattr(settings.cache, '__getitem__') and "predictions" in settings.cache else 3600

        # Initialize ML models
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1, random_state=42
        )

        # Initialize deep learning model
        self.lstm_model = self._build_lstm_model()

        # Performance monitoring
        self.prediction_counter = None  # Use shared metrics only
        self.error_counter = PREDICTION_ERRORS

        # Model performance tracking
        self.model_metrics = {
            "rf": {"mse": [], "r2": []},
            "lstm": {"mse": [], "r2": []},
            "prophet": {"mse": [], "r2": []},
        }

        # Prediction thresholds
        self.thresholds = {
            "confidence": 0.8,
            "anomaly_score": -0.5,
            "trend_significance": 0.1,
        }

    def _build_lstm_model(self) -> Sequential:
        """Build LSTM model for time series prediction."""
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(None, 5)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
        )

        return model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def predict_reputation_trend(
        self, platform_data: Dict, timeframe: str = "7d"
    ) -> Dict:
        """Predict reputation trends using ensemble of models."""
        try:
            start_time = datetime.now(timezone.utc)
            self.prediction_counter.inc()

            # Extract and prepare data
            metrics = self._extract_time_series(platform_data)
            features = self._prepare_features(metrics)

            # Make predictions using multiple models
            predictions = await self._ensemble_forecast(features, metrics)

            # Calculate risk scores
            risk_scores = self._calculate_risk_scores(predictions, metrics)

            # Generate insights
            insights = self._generate_predictive_insights(
                predictions, risk_scores, metrics
            )

            # Prepare response
            response = {
                "predictions": predictions,
                "risk_scores": risk_scores,
                "insights": insights,
                "confidence": self._calculate_prediction_confidence(
                    predictions
                ),
                "metadata": {
                    "timeframe": timeframe,
                    "models_used": ["rf", "lstm", "prophet"],
                    "features_used": list(features.columns),
                    "predicted_at": datetime.now(timezone.utc).isoformat(),
                    "processing_time": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                },
            }

            # Record latency
            PREDICTION_LATENCY.observe(
                (datetime.now(timezone.utc) - start_time).total_seconds()
            )

            return response

        except Exception as e:
            self.error_counter.inc()
            logger.error(f"Error predicting reputation trend: {str(e)}")
            raise

    async def _ensemble_forecast(
        self, features: pd.DataFrame, metrics: pd.DataFrame
    ) -> Dict:
        """Generate forecasts using multiple models."""
        try:
            # Split data for training
            train_size = int(len(features) * 0.8)
            train_features = features[:train_size]
            train_target = metrics["reputation_score"][:train_size]

            # Train models
            rf_predictions = self._rf_forecast(
                train_features, train_target, features
            )
            lstm_predictions = await self._lstm_forecast(
                train_features, train_target, features
            )
            prophet_predictions = await self._prophet_forecast(metrics)

            # Combine predictions
            ensemble_predictions = self._combine_predictions(
                rf_predictions, lstm_predictions, prophet_predictions
            )

            return {
                "ensemble": ensemble_predictions,
                "models": {
                    "rf": rf_predictions,
                    "lstm": lstm_predictions,
                    "prophet": prophet_predictions,
                },
            }

        except Exception as e:
            logger.error(f"Error in ensemble forecast: {str(e)}")
            return {}

    def _rf_forecast(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        features: pd.DataFrame,
    ) -> Dict:
        """Generate forecasts using Random Forest."""
        try:
            # Train model
            self.rf_model.fit(train_features, train_target)

            # Make predictions
            predictions = self.rf_model.predict(features)

            # Calculate performance metrics
            mse = mean_squared_error(
                train_target, self.rf_model.predict(train_features)
            )
            r2 = r2_score(train_target, self.rf_model.predict(train_features))

            # Update metrics
            self.model_metrics["rf"]["mse"].append(mse)
            self.model_metrics["rf"]["r2"].append(r2)

            return {
                "values": predictions.tolist(),
                "metrics": {"mse": mse, "r2": r2},
            }

        except Exception as e:
            logger.error(f"Error in RF forecast: {str(e)}")
            return {}

    async def _lstm_forecast(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        features: pd.DataFrame,
    ) -> Dict:
        """Generate forecasts using LSTM."""
        try:
            # Prepare sequences
            X, y = self._prepare_sequences(train_features, train_target)

            # Train model
            self.lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

            # Make predictions
            test_sequences = self._prepare_test_sequences(features)
            predictions = self.lstm_model.predict(test_sequences)

            # Calculate performance metrics
            train_pred = self.lstm_model.predict(X)
            mse = mean_squared_error(y, train_pred)
            r2 = r2_score(y, train_pred)

            # Update metrics
            self.model_metrics["lstm"]["mse"].append(mse)
            self.model_metrics["lstm"]["r2"].append(r2)

            return {
                "values": predictions.flatten().tolist(),
                "metrics": {"mse": mse, "r2": r2},
            }

        except Exception as e:
            logger.error(f"Error in LSTM forecast: {str(e)}")
            return {}

    async def _prophet_forecast(self, metrics: pd.DataFrame) -> Dict:
        """Generate forecasts using Prophet."""
        try:
            # Prepare data for Prophet
            df = pd.DataFrame(
                {"ds": metrics.index, "y": metrics["reputation_score"]}
            )

            # Initialize and train model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode="multiplicative",
            )
            model.fit(df)

            # Make predictions
            future = model.make_future_dataframe(periods=7, freq="D")
            forecast = model.predict(future)

            # Calculate performance metrics
            mse = mean_squared_error(df["y"], forecast["yhat"][: len(df)])
            r2 = r2_score(df["y"], forecast["yhat"][: len(df)])

            # Update metrics
            self.model_metrics["prophet"]["mse"].append(mse)
            self.model_metrics["prophet"]["r2"].append(r2)

            return {
                "values": forecast["yhat"].tolist(),
                "bounds": {
                    "lower": forecast["yhat_lower"].tolist(),
                    "upper": forecast["yhat_upper"].tolist(),
                },
                "metrics": {"mse": mse, "r2": r2},
            }

        except Exception as e:
            logger.error(f"Error in Prophet forecast: {str(e)}")
            return {}

    def _combine_predictions(
        self,
        rf_predictions: Dict,
        lstm_predictions: Dict,
        prophet_predictions: Dict,
    ) -> Dict:
        """Combine predictions from multiple models."""
        try:
            # Get predictions
            rf_values = np.array(rf_predictions.get("values", []))
            lstm_values = np.array(lstm_predictions.get("values", []))
            prophet_values = np.array(prophet_predictions.get("values", []))

            # Calculate weights based on model performance
            weights = self._calculate_model_weights()

            # Compute weighted average
            ensemble_predictions = (
                weights["rf"] * rf_values
                + weights["lstm"] * lstm_values
                + weights["prophet"] * prophet_values
            )

            return {
                "values": ensemble_predictions.tolist(),
                "weights": weights,
            }

        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return {}

    def _calculate_model_weights(self) -> Dict[str, float]:
        """Calculate weights for each model based on performance."""
        try:
            weights = {}
            total_score = 0

            for model in ["rf", "lstm", "prophet"]:
                # Calculate score based on recent performance
                mse = np.mean(self.model_metrics[model]["mse"][-5:])
                r2 = np.mean(self.model_metrics[model]["r2"][-5:])

                # Combine metrics into a single score
                score = (1 / mse) * r2
                weights[model] = score
                total_score += score

            # Normalize weights
            if total_score > 0:
                for model in weights:
                    weights[model] /= total_score
            else:
                # Equal weights if no performance history
                weights = {"rf": 1 / 3, "lstm": 1 / 3, "prophet": 1 / 3}

            return weights

        except Exception:
            # Return equal weights on error
            return {"rf": 1 / 3, "lstm": 1 / 3, "prophet": 1 / 3}

    def _calculate_prediction_confidence(self, predictions: Dict) -> float:
        """Calculate confidence in predictions."""
        try:
            # Get model performances
            performances = []
            for model in ["rf", "lstm", "prophet"]:
                if predictions["models"][model]:
                    performances.append(
                        predictions["models"][model]["metrics"]["r2"]
                    )

            # Calculate weighted confidence
            weights = predictions["ensemble"]["weights"]
            confidence = sum(
                p * w for p, w in zip(performances, weights.values())
            )

            return min(1.0, max(0.0, confidence))

        except Exception:
            return 0.0

    def _calculate_risk_scores(
        self, predictions: Dict, metrics: pd.DataFrame
    ) -> Dict:
        """Calculate comprehensive risk scores."""
        try:
            # Calculate trend risk
            trend_risk = self._calculate_trend_risk(
                predictions["ensemble"]["values"]
            )

            # Calculate volatility risk
            volatility_risk = self._calculate_volatility_risk(
                metrics["reputation_score"]
            )

            # Calculate anomaly risk
            anomaly_risk = self._calculate_anomaly_risk(
                predictions["ensemble"]["values"]
            )

            # Combine risks
            overall_risk = (
                trend_risk * 0.4 + volatility_risk * 0.3 + anomaly_risk * 0.3
            )

            return {
                "overall": overall_risk,
                "components": {
                    "trend": trend_risk,
                    "volatility": volatility_risk,
                    "anomaly": anomaly_risk,
                },
            }

        except Exception as e:
            logger.error(f"Error calculating risk scores: {str(e)}")
            return {}

    def _calculate_trend_risk(self, predictions: List[float]) -> float:
        """Calculate risk based on predicted trend."""
        try:
            # Calculate trend
            trend = np.polyfit(range(len(predictions)), predictions, 1)[0]

            # Convert trend to risk score
            return 1 / (1 + np.exp(-10 * -trend))

        except Exception:
            return 0.5

    def _calculate_volatility_risk(self, values: pd.Series) -> float:
        """Calculate risk based on volatility."""
        try:
            # Calculate rolling volatility
            volatility = values.rolling(window=7).std().mean()

            # Convert volatility to risk score
            return 1 / (1 + np.exp(-5 * volatility))

        except Exception:
            return 0.5

    def _calculate_anomaly_risk(self, predictions: List[float]) -> float:
        """Calculate risk based on potential anomalies."""
        try:
            # Reshape for anomaly detection
            X = np.array(predictions).reshape(-1, 1)

            # Get anomaly scores
            scores = self.anomaly_detector.score_samples(X)

            # Convert to risk score
            return 1 / (1 + np.exp(np.mean(scores)))

        except Exception:
            return 0.5

    def _generate_predictive_insights(
        self, predictions: Dict, risk_scores: Dict, metrics: pd.DataFrame
    ) -> List[Dict]:
        """Generate actionable insights from predictions."""
        try:
            insights = []

            # Analyze trend
            trend_insight = self._analyze_trend(
                predictions["ensemble"]["values"]
            )
            if trend_insight:
                insights.append(trend_insight)

            # Analyze risks
            risk_insight = self._analyze_risks(risk_scores)
            if risk_insight:
                insights.append(risk_insight)

            # Analyze patterns
            pattern_insight = self._analyze_patterns(metrics)
            if pattern_insight:
                insights.append(pattern_insight)

            return insights

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []

    def _analyze_trend(self, predictions: List[float]) -> Optional[Dict]:
        """Analyze predicted trend for insights."""
        try:
            # Calculate trend
            trend = np.polyfit(range(len(predictions)), predictions, 1)[0]

            if abs(trend) > self.thresholds["trend_significance"]:
                return {
                    "type": "trend",
                    "direction": "increasing" if trend > 0 else "decreasing",
                    "magnitude": abs(trend),
                    "confidence": self._calculate_trend_confidence(
                        predictions
                    ),
                }

            return None

        except Exception:
            return None

    def _analyze_risks(self, risk_scores: Dict) -> Optional[Dict]:
        """Analyze risk scores for insights."""
        try:
            if risk_scores["overall"] > 0.7:
                return {
                    "type": "risk_alert",
                    "level": "high",
                    "score": risk_scores["overall"],
                    "components": risk_scores["components"],
                }

            return None

        except Exception:
            return None

    def _analyze_patterns(self, metrics: pd.DataFrame) -> Optional[Dict]:
        """Analyze historical patterns for insights."""
        try:
            # Implementation would analyze patterns
            return None
        except Exception:
            return None

    def _calculate_trend_confidence(self, values: List[float]) -> float:
        """Calculate confidence in trend analysis."""
        try:
            # Implementation would calculate trend confidence
            return 0.8
        except Exception:
            return 0.0

    def _extract_time_series(self, platform_data: Dict) -> pd.DataFrame:
        """Extract time series data from platform data."""
        metrics = []

        for item in platform_data.get("historical_data", []):
            metrics.append(
                {
                    "timestamp": datetime.fromisoformat(item["timestamp"]),
                    "engagement_rate": item.get("engagement_rate", 0),
                    "sentiment_score": item.get("sentiment_score", 0),
                    "mention_count": item.get("mention_count", 0),
                    "follower_growth": item.get("follower_growth", 0),
                }
            )

        return pd.DataFrame(metrics)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        features = df[
            [
                "engagement_rate",
                "sentiment_score",
                "mention_count",
                "follower_growth",
            ]
        ].values

        return self.scaler.fit_transform(features)

    def _prepare_sequences(
        self, train_features: pd.DataFrame, train_target: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        X = []
        y = []
        for i in range(len(train_features) - 5):
            X.append(train_features[i : i + 5].values)
            y.append(train_target[i + 5])
        return np.array(X), np.array(y)

    def _prepare_test_sequences(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare test sequences for LSTM prediction."""
        X = []
        for i in range(len(features) - 5):
            X.append(features[i : i + 5].values)
        return np.array(X)

    async def detect_emerging_issues(
        self, platform_data: Dict, threshold: float = 0.7
    ) -> List[Dict]:
        """Detect potential emerging reputation issues."""
        try:
            # Analyze recent mentions and comments
            mentions = platform_data.get("mentions", [])
            comments = platform_data.get("comments", [])

            # Perform sentiment analysis
            mention_sentiments = (
                await self.sentiment_service.analyze_bulk_sentiment(
                    [m["text"] for m in mentions]
                )
            )
            comment_sentiments = (
                await self.sentiment_service.analyze_bulk_sentiment(
                    [c["text"] for c in comments]
                )
            )

            # Identify negative clusters
            issues = self._cluster_negative_content(
                mentions, mention_sentiments, comments, comment_sentiments
            )

            # Filter significant issues
            significant_issues = [
                issue for issue in issues if issue["severity"] >= threshold
            ]

            return {
                "issues": significant_issues,
                "total_analyzed": len(mentions) + len(comments),
                "threshold": threshold,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error detecting emerging issues: {str(e)}")
            raise

    async def analyze_competitor_benchmarks(
        self, platform: str, target_id: str, competitor_ids: List[str]
    ) -> Dict:
        """Analyze competitive benchmarks and positioning."""
        try:
            # Get target metrics
            target_metrics = await self._get_platform_metrics(
                platform, target_id
            )

            # Get competitor metrics
            competitor_metrics = []
            for comp_id in competitor_ids:
                metrics = await self._get_platform_metrics(platform, comp_id)
                competitor_metrics.append(metrics)

            # Calculate benchmarks
            benchmarks = self._calculate_benchmarks(
                target_metrics, competitor_metrics
            )

            # Generate insights
            insights = self._generate_competitive_insights(
                benchmarks, target_metrics, competitor_metrics
            )

            return {
                "benchmarks": benchmarks,
                "market_position": insights["position"],
                "strengths": insights["strengths"],
                "weaknesses": insights["weaknesses"],
                "opportunities": insights["opportunities"],
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing competitor benchmarks: {str(e)}")
            raise

    async def _get_platform_metrics(
        self, platform: str, entity_id: str
    ) -> Dict:
        """Get platform-specific metrics."""
        # This would be implemented to call appropriate platform service
        return {}
