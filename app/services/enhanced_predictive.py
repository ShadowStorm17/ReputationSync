"""
Enhanced predictive service.
Provides advanced predictive analytics with deep learning and ensemble methods.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from prometheus_client import Counter
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
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


class EnhancedPredictive:
    """Enhanced predictive analytics service with deep learning capabilities."""

    def __init__(self):
        """Initialize enhanced predictive models and tools."""
        # Initialize services
        self.sentiment_service = SentimentService()

        # Initialize ML models
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1, random_state=42
        )

        # Initialize LSTM model
        self.lstm_model = self._build_lstm_model()

        # Initialize Prophet model
        self.prophet = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode="multiplicative",
        )

        # Performance monitoring
        self.prediction_counter = None  # Use shared metrics only
        self.error_counter = PREDICTION_ERRORS

        # Cache configuration
        self.cache_ttl = settings.cache["predictions"] if hasattr(settings.cache, '__getitem__') and "predictions" in settings.cache else 3600

        # Model performance tracking
        self.model_metrics = {
            "rf": {"mse": [], "r2": []},
            "lstm": {"mse": [], "r2": []},
            "prophet": {"mse": [], "r2": []},
        }

    def _build_lstm_model(self) -> Sequential:
        """Build LSTM model for time series prediction."""
        model = Sequential(
            [
                LSTM(64, input_shape=(30, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        return model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def predict_reputation(
        self, data: Dict[str, Any], horizon: int = 30
    ) -> Dict[str, Any]:
        """Generate reputation predictions using ensemble methods."""
        try:
            start_time = datetime.now(timezone.utc)
            self.prediction_counter.inc()

            # Prepare data
            processed_data = await self._preprocess_data(data)

            # Generate predictions from each model
            rf_pred = await self._random_forest_predict(
                processed_data, horizon
            )
            lstm_pred = await self._lstm_predict(processed_data, horizon)
            prophet_pred = await self._prophet_predict(processed_data, horizon)

            # Ensemble predictions
            ensemble_pred = self._ensemble_predictions(
                [rf_pred, lstm_pred, prophet_pred]
            )

            # Detect anomalies
            anomalies = await self._detect_anomalies(processed_data)

            # Record latency
            PREDICTION_LATENCY.observe(
                (datetime.now(timezone.utc) - start_time).total_seconds()
            )

            return {
                "predictions": ensemble_pred,
                "anomalies": anomalies,
                "model_metrics": self.model_metrics,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.error_counter.inc()
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    async def _preprocess_data(
        self, data: Dict[str, Any]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Preprocess data for prediction."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data["historical_data"])

            # Handle missing values
            df = df.fillna(method="ffill").fillna(method="bfill")

            # Feature engineering
            df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

            # Scale features
            features = df[
                ["sentiment_score", "engagement_rate", "mention_count"]
            ].values
            scaled_features = self.scaler.fit_transform(features)

            return scaled_features, df

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    async def _random_forest_predict(
        self, data: np.ndarray, horizon: int
    ) -> np.ndarray:
        """Generate predictions using Random Forest."""
        try:
            # Prepare sequences
            X, y = self._prepare_sequences(data)

            # Train model
            self.rf_model.fit(X, y)

            # Generate predictions
            predictions = []
            last_sequence = X[-1]

            for _ in range(horizon):
                pred = self.rf_model.predict([last_sequence])[0]
                predictions.append(pred)
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = pred

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {str(e)}")
            raise

    async def _lstm_predict(
        self, data: np.ndarray, horizon: int
    ) -> np.ndarray:
        """Generate predictions using LSTM."""
        try:
            # Prepare sequences
            X, y = self._prepare_sequences(data)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Train model
            self.lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

            # Generate predictions
            predictions = []
            last_sequence = X[-1]

            for _ in range(horizon):
                pred = self.lstm_model.predict(
                    last_sequence.reshape(1, 30, 1), verbose=0
                )[0][0]
                predictions.append(pred)
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = pred

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Error in LSTM prediction: {str(e)}")
            raise

    async def _prophet_predict(
        self, data: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        """Generate predictions using Prophet."""
        try:
            # Prepare data
            prophet_df = pd.DataFrame(
                {
                    "ds": pd.to_datetime(data["timestamp"]),
                    "y": data["sentiment_score"],
                }
            )

            # Fit model
            self.prophet.fit(prophet_df)

            # Generate predictions
            future = self.prophet.make_future_dataframe(
                periods=horizon, freq="D"
            )
            forecast = self.prophet.predict(future)

            return forecast.tail(horizon)["yhat"].values

        except Exception as e:
            logger.error(f"Error in Prophet prediction: {str(e)}")
            raise

    def _ensemble_predictions(
        self, predictions: List[np.ndarray]
    ) -> np.ndarray:
        """Combine predictions using weighted average."""
        try:
            # Calculate weights based on model performance
            weights = self._calculate_model_weights()

            # Combine predictions
            weighted_preds = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, weights):
                weighted_preds += pred * weight

            return weighted_preds

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise

    async def _detect_anomalies(
        self, data: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in the data."""
        try:
            # Fit anomaly detector
            self.anomaly_detector.fit(data)

            # Get anomaly scores
            scores = self.anomaly_detector.score_samples(data)
            predictions = self.anomaly_detector.predict(data)

            # Identify anomalies
            anomalies = []
            for i, (score, pred) in enumerate(zip(scores, predictions)):
                if pred == -1:  # Anomaly
                    anomalies.append(
                        {
                            "index": i,
                            "score": float(score),
                            "severity": "high" if score < -0.5 else "medium",
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []

    def _prepare_sequences(
        self, data: np.ndarray, sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series prediction."""
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            sequences.append(data[i : i + sequence_length])
            targets.append(data[i + sequence_length])

        return np.array(sequences), np.array(targets)

    def _calculate_model_weights(self) -> List[float]:
        """Calculate model weights based on performance metrics."""
        weights = [0.4, 0.3, 0.3]  # Default weights

        try:
            # Calculate weights based on R² scores
            total_r2 = sum(
                [
                    np.mean(self.model_metrics["rf"]["r2"]),
                    np.mean(self.model_metrics["lstm"]["r2"]),
                    np.mean(self.model_metrics["prophet"]["r2"]),
                ]
            )

            if total_r2 > 0:
                weights = [
                    np.mean(self.model_metrics["rf"]["r2"]) / total_r2,
                    np.mean(self.model_metrics["lstm"]["r2"]) / total_r2,
                    np.mean(self.model_metrics["prophet"]["r2"]) / total_r2,
                ]

            return weights

        except Exception:
            return weights
