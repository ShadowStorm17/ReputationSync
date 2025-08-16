import logging
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from scipy.stats import norm

from app.core.config import get_settings
from app.services.predictive_service import PredictiveService
from app.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)
settings = get_settings()


class AnalyticsEngine:
    """Advanced analytics engine with deep learning capabilities."""

    def __init__(self):
        self.sentiment_service = SentimentService()
        self.predictive_service = PredictiveService()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)

        # Initialize ML models
        self.rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42
        )

        # Initialize deep learning model for sequence analysis
        self.sequence_model = self._build_sequence_model()

        # Analysis configurations
        self.analysis_config = {
            "trend_window": 30,  # days
            "seasonality_periods": [7, 30, 90],  # days
            "correlation_threshold": 0.7,
            "anomaly_sensitivity": 0.95,
            "growth_baseline": 0.05,  # 5% baseline for growth metrics
        }

        # Market analysis parameters
        self.market_params = {
            "position_dimensions": [
                "sentiment",
                "engagement",
                "growth",
                "influence",
            ],
            "competitive_factors": [
                "market_share",
                "growth_rate",
                "sentiment_advantage",
                "engagement_efficiency",
            ],
        }

    def _build_sequence_model(self) -> Sequential:
        """Build LSTM model for sequence analysis."""
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(None, 5)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
        )

        return model

    async def analyze_reputation_metrics(
        self, data: Dict, timeframe: str = "30d"
    ) -> Dict:
        """Perform comprehensive reputation analysis with deep learning insights."""
        try:
            # Extract and preprocess metrics
            metrics_df = self._prepare_metrics_data(data)

            # Perform deep learning analysis
            sequence_predictions = self._analyze_sequences(metrics_df)

            # Calculate advanced metrics
            sentiment_metrics = self._analyze_sentiment_metrics(data)
            engagement_metrics = self._analyze_engagement_metrics(data)
            influence_metrics = self._analyze_influence_metrics(data)
            growth_metrics = self._analyze_growth_metrics(data)

            # Detect patterns and anomalies
            patterns = self._detect_advanced_patterns(metrics_df)
            anomalies = self._detect_anomalies(metrics_df)

            # Generate insights
            insights = self._generate_advanced_insights(
                metrics_df, patterns, anomalies
            )

            # Combine all analyses
            combined_metrics = {
                "sentiment": sentiment_metrics,
                "engagement": engagement_metrics,
                "influence": influence_metrics,
                "growth": growth_metrics,
                "patterns": patterns,
                "anomalies": anomalies,
                "insights": insights,
                "sequence_predictions": sequence_predictions,
                "composite_score": self._calculate_composite_score(
                    [
                        sentiment_metrics,
                        engagement_metrics,
                        influence_metrics,
                        growth_metrics,
                    ]
                ),
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }

            return combined_metrics

        except Exception as e:
            logger.error(f"Error analyzing reputation metrics: {str(e)}")
            raise

    def _prepare_metrics_data(self, data: Dict) -> pd.DataFrame:
        """Prepare metrics data for analysis."""
        try:
            # Extract time series data
            metrics = []
            for item in data.get("historical_data", []):
                metrics.append(
                    {
                        "timestamp": datetime.fromisoformat(item["timestamp"]),
                        "sentiment": item.get("sentiment", 0),
                        "engagement": item.get("engagement", 0),
                        "influence": item.get("influence", 0),
                        "growth": item.get("growth", 0),
                        "mentions": item.get("mentions", 0),
                    }
                )

            # Convert to DataFrame
            df = pd.DataFrame(metrics)
            df.set_index("timestamp", inplace=True)

            # Handle missing values
            df.fillna(method="ffill", inplace=True)
            df.fillna(0, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error preparing metrics data: {str(e)}")
            return pd.DataFrame()

    def _analyze_sequences(self, df: pd.DataFrame) -> Dict:
        """Analyze sequences using LSTM model."""
        try:
            # Prepare sequences
            sequence_length = 10
            sequences = []

            for i in range(len(df) - sequence_length):
                sequences.append(df.iloc[i : i + sequence_length].values)

            if not sequences:
                return {}

            # Convert to numpy array and reshape for LSTM
            X = np.array(sequences)

            # Make predictions
            predictions = self.sequence_model.predict(X)

            return {
                "predictions": predictions.tolist(),
                "confidence": self._calculate_prediction_confidence(
                    predictions
                ),
                "trend_direction": self._get_sequence_trend(predictions),
            }

        except Exception as e:
            logger.error(f"Error analyzing sequences: {str(e)}")
            return {}

    def _detect_advanced_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect advanced patterns in metrics."""
        try:
            patterns = {}

            # Detect seasonality
            for period in self.analysis_config["seasonality_periods"]:
                seasonal_patterns = self._analyze_seasonality(df, period)
                patterns[f"seasonality_{period}d"] = seasonal_patterns

            # Detect trends
            trends = self._analyze_trends(df)
            patterns["trends"] = trends

            # Detect cycles
            cycles = self._detect_cycles(df)
            patterns["cycles"] = cycles

            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {}

    def _analyze_seasonality(self, df: pd.DataFrame, period: int) -> Dict:
        """Analyze seasonality patterns."""
        try:
            seasonal_patterns = {}

            for column in df.columns:
                if len(df[column]) >= period * 2:
                    # Calculate seasonal decomposition
                    seasonal = self._decompose_seasonal(df[column], period)

                    seasonal_patterns[column] = {
                        "strength": self._calculate_seasonality_strength(
                            seasonal
                        ),
                        "peaks": self._find_seasonal_peaks(seasonal),
                        "pattern": seasonal.tolist(),
                    }

            return seasonal_patterns

        except Exception:
            return {}

    def _decompose_seasonal(
        self, series: pd.Series, period: int
    ) -> np.ndarray:
        """Decompose time series into seasonal components."""
        try:
            # Simple moving average method
            ma = series.rolling(window=period, center=True).mean()
            seasonal = series - ma
            return seasonal.fillna(0).values
        except Exception:
            return np.zeros(len(series))

    def _detect_cycles(self, df: pd.DataFrame) -> Dict:
        """Detect cyclical patterns in metrics."""
        try:
            cycles = {}

            for column in df.columns:
                if len(df[column]) >= 30:  # Minimum length for cycle detection
                    # Perform autocorrelation analysis
                    autocorr = self._calculate_autocorrelation(df[column])

                    # Find cycle lengths
                    cycle_lengths = self._find_cycle_lengths(autocorr)

                    cycles[column] = {
                        "lengths": cycle_lengths,
                        "strength": self._calculate_cycle_strength(autocorr),
                        "confidence": self._calculate_cycle_confidence(
                            df[column], cycle_lengths
                        ),
                    }

            return cycles

        except Exception:
            return {}

    def _calculate_autocorrelation(self, series: pd.Series) -> np.ndarray:
        """Calculate autocorrelation for cycle detection."""
        try:
            # Remove trend
            detrended = series - series.rolling(window=7, min_periods=1).mean()

            # Calculate autocorrelation
            autocorr = np.correlate(detrended, detrended, mode="full")
            return autocorr[len(autocorr) // 2 :]
        except Exception:
            return np.array([])

    def _find_cycle_lengths(self, autocorr: np.ndarray) -> List[int]:
        """Find potential cycle lengths from autocorrelation."""
        try:
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if (
                    autocorr[i] > autocorr[i - 1]
                    and autocorr[i] > autocorr[i + 1]
                ):
                    peaks.append(i)

            return peaks[:3]  # Return top 3 cycle lengths
        except Exception:
            return []

    def _generate_advanced_insights(
        self, df: pd.DataFrame, patterns: Dict, anomalies: Dict
    ) -> List[Dict]:
        """Generate advanced insights from analysis."""
        try:
            insights = []

            # Add pattern-based insights
            for metric, pattern in patterns.get("trends", {}).items():
                if pattern["strength"] > 0.7:
                    insights.append(
                        {
                            "type": "trend",
                            "metric": metric,
                            "description": f"Strong {pattern['trend']} trend detected",
                            "confidence": pattern["confidence"],
                            "impact": "high"
                            if pattern["strength"] > 0.9
                            else "medium",
                        }
                    )

            # Add seasonality insights
            for period, seasonal in patterns.items():
                if period.startswith("seasonality_"):
                    for metric, data in seasonal.items():
                        if data["strength"] > 0.6:
                            insights.append(
                                {
                                    "type": "seasonality",
                                    "metric": metric,
                                    "period": period.split("_")[1],
                                    "description": "Strong seasonal pattern detected",
                                    "peaks": data["peaks"],
                                    "confidence": data["strength"],
                                }
                            )

            # Add anomaly insights
            for anomaly in anomalies:
                insights.append(
                    {
                        "type": "anomaly",
                        "metrics": anomaly["metrics"],
                        "severity": anomaly["severity"],
                        "description": anomaly["description"],
                        "recommendations": self._generate_anomaly_recommendations(
                            anomaly
                        ),
                    }
                )

            return insights

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []

    def _generate_anomaly_recommendations(self, anomaly: Dict) -> List[Dict]:
        """Generate recommendations for handling anomalies."""
        try:
            recommendations = []

            if anomaly["severity"] == "high":
                recommendations.extend(
                    [
                        {
                            "action": "immediate_investigation",
                            "priority": "high",
                            "description": "Investigate root cause immediately",
                        },
                        {
                            "action": "stakeholder_notification",
                            "priority": "high",
                            "description": "Notify key stakeholders",
                        },
                    ]
                )
            elif anomaly["severity"] == "medium":
                recommendations.append(
                    {
                        "action": "monitoring_increase",
                        "priority": "medium",
                        "description": "Increase monitoring frequency",
                    }
                )

            return recommendations

        except Exception:
            return []

    def _analyze_sentiment_metrics(self, data: Dict) -> Dict:
        """Analyze detailed sentiment metrics."""
        try:
            sentiment_data = data.get("sentiment", {})

            # Calculate advanced metrics
            sentiment_stats = {
                "average": np.mean(sentiment_data.get("values", [0])),
                "std_dev": np.std(sentiment_data.get("values", [0])),
                "volatility": self._calculate_volatility(
                    sentiment_data.get("values", [])
                ),
                "trend": self._calculate_trend(
                    sentiment_data.get("values", [])
                ),
            }

            # Add sentiment distribution
            distribution = self._calculate_sentiment_distribution(
                sentiment_data.get("values", [])
            )

            return {
                "stats": sentiment_stats,
                "distribution": distribution,
                "risk_level": self._assess_sentiment_risk(sentiment_stats),
            }

        except Exception:
            return {
                "stats": {"average": 0, "std_dev": 0, "volatility": 0},
                "distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "risk_level": "unknown",
            }

    def _analyze_engagement_metrics(self, data: Dict) -> Dict:
        """Analyze detailed engagement metrics."""
        try:
            engagement_data = data.get("engagement", {})

            # Calculate engagement metrics
            metrics = {
                "total_engagement": sum(engagement_data.get("values", [0])),
                "average_engagement": np.mean(
                    engagement_data.get("values", [0])
                ),
                "engagement_rate": self._calculate_engagement_rate(
                    engagement_data
                ),
                "growth_rate": self._calculate_growth_rate(
                    engagement_data.get("values", [])
                ),
            }

            # Add engagement patterns
            patterns = self._detect_engagement_patterns(
                engagement_data.get("values", [])
            )

            return {
                "metrics": metrics,
                "patterns": patterns,
                "quality_score": self._calculate_engagement_quality(
                    engagement_data
                ),
            }

        except Exception:
            return {
                "metrics": {
                    "total_engagement": 0,
                    "average_engagement": 0,
                    "engagement_rate": 0,
                    "growth_rate": 0,
                },
                "patterns": [],
                "quality_score": 0,
            }

    def _analyze_influence_metrics(self, data: Dict) -> Dict:
        """Analyze influence and reach metrics."""
        try:
            influence_data = data.get("influence", {})

            # Calculate influence metrics
            metrics = {
                "reach": influence_data.get("reach", 0),
                "amplification": self._calculate_amplification(influence_data),
                "impact_score": self._calculate_impact_score(influence_data),
            }

            # Add influence distribution
            distribution = self._analyze_influence_distribution(influence_data)

            return {
                "metrics": metrics,
                "distribution": distribution,
                "key_influencers": self._identify_key_influencers(
                    influence_data
                ),
            }

        except Exception:
            return {
                "metrics": {"reach": 0, "amplification": 0, "impact_score": 0},
                "distribution": {},
                "key_influencers": [],
            }

    def _analyze_growth_metrics(self, data: Dict) -> Dict:
        """Analyze growth and trend metrics."""
        try:
            growth_data = data.get("growth", {})

            # Calculate growth metrics
            metrics = {
                "growth_rate": self._calculate_growth_rate(
                    growth_data.get("values", [])
                ),
                "acceleration": self._calculate_growth_acceleration(
                    growth_data.get("values", [])
                ),
                "momentum": self._calculate_momentum(
                    growth_data.get("values", [])
                ),
            }

            # Add trend analysis
            trends = self._analyze_growth_trends(growth_data)

            return {
                "metrics": metrics,
                "trends": trends,
                "forecast": self._forecast_growth(growth_data),
            }

        except Exception:
            return {
                "metrics": {
                    "growth_rate": 0,
                    "acceleration": 0,
                    "momentum": 0,
                },
                "trends": [],
                "forecast": {},
            }

    def _calculate_composite_score(self, metrics: List[Dict]) -> float:
        """Calculate composite reputation score."""
        try:
            weights = {
                "sentiment": 0.3,
                "engagement": 0.25,
                "influence": 0.25,
                "growth": 0.2,
            }

            scores = []
            for metric, weight in weights.items():
                if metric == "sentiment":
                    score = metrics[0]["stats"]["average"] * weight
                elif metric == "engagement":
                    score = metrics[1]["metrics"]["engagement_rate"] * weight
                elif metric == "influence":
                    score = metrics[2]["metrics"]["impact_score"] * weight
                else:  # growth
                    score = metrics[3]["metrics"]["momentum"] * weight
                scores.append(score)

            return sum(scores)

        except Exception:
            return 0.0

    def _calculate_market_metrics(self, market_data: List[Dict]) -> Dict:
        """Calculate market-wide metrics."""
        try:
            # Extract metrics
            sentiment_values = [
                d.get("sentiment", {}).get("average", 0) for d in market_data
            ]
            engagement_values = [
                d.get("engagement", {}).get("total", 0) for d in market_data
            ]

            return {
                "sentiment": {
                    "average": np.mean(sentiment_values),
                    "std_dev": np.std(sentiment_values),
                    "range": (min(sentiment_values), max(sentiment_values)),
                },
                "engagement": {
                    "average": np.mean(engagement_values),
                    "std_dev": np.std(engagement_values),
                    "range": (min(engagement_values), max(engagement_values)),
                },
            }

        except Exception:
            return {
                "sentiment": {"average": 0, "std_dev": 0, "range": (0, 0)},
                "engagement": {"average": 0, "std_dev": 0, "range": (0, 0)},
            }

    def _analyze_market_positioning(
        self, entity_data: Dict, market_metrics: Dict
    ) -> Dict:
        """Analyze market positioning."""
        try:
            sentiment_position = (
                entity_data.get("sentiment", {}).get("average", 0)
                - market_metrics["sentiment"]["average"]
            ) / market_metrics["sentiment"]["std_dev"]

            engagement_position = (
                entity_data.get("engagement", {}).get("total", 0)
                - market_metrics["engagement"]["average"]
            ) / market_metrics["engagement"]["std_dev"]

            return {
                "sentiment_percentile": self._calculate_percentile(
                    sentiment_position
                ),
                "engagement_percentile": self._calculate_percentile(
                    engagement_position
                ),
                "overall_position": self._calculate_position_label(
                    (sentiment_position + engagement_position) / 2
                ),
            }

        except Exception:
            return {
                "sentiment_percentile": 50,
                "engagement_percentile": 50,
                "overall_position": "average",
            }

    def _identify_competitive_advantages(
        self, entity_data: Dict, market_metrics: Dict
    ) -> List[Dict]:
        """Identify competitive advantages."""
        try:
            advantages = []

            # Check sentiment advantage
            if (
                entity_data.get("sentiment", {}).get("average", 0)
                > market_metrics["sentiment"]["average"]
                + market_metrics["sentiment"]["std_dev"]
            ):
                advantages.append(
                    {
                        "type": "sentiment",
                        "strength": "high",
                        "description": "Above-market sentiment performance",
                    }
                )

            # Check engagement advantage
            if (
                entity_data.get("engagement", {}).get("total", 0)
                > market_metrics["engagement"]["average"]
                + market_metrics["engagement"]["std_dev"]
            ):
                advantages.append(
                    {
                        "type": "engagement",
                        "strength": "high",
                        "description": "Superior engagement metrics",
                    }
                )

            return advantages

        except Exception:
            return []

    def _generate_competitive_recommendations(
        self, positioning: Dict, advantages: List[Dict]
    ) -> List[Dict]:
        """Generate competitive strategy recommendations."""
        try:
            recommendations = []

            # Add positioning-based recommendations
            if positioning["overall_position"] == "leader":
                recommendations.append(
                    {
                        "type": "strategy",
                        "priority": "high",
                        "action": "maintain_leadership",
                        "description": "Maintain market leadership position",
                        "steps": [
                            "Monitor competitor activities closely",
                            "Invest in innovation",
                            "Strengthen brand advocacy",
                        ],
                    }
                )
            elif positioning["overall_position"] == "challenger":
                recommendations.append(
                    {
                        "type": "strategy",
                        "priority": "high",
                        "action": "close_gap",
                        "description": "Close gap with market leaders",
                        "steps": [
                            "Identify key differentiators",
                            "Focus on engagement quality",
                            "Develop unique value proposition",
                        ],
                    }
                )

            return recommendations

        except Exception:
            return []

    def _analyze_metric_patterns(self, values: np.ndarray) -> Dict:
        """Analyze patterns in metric values."""
        try:
            # Calculate basic statistics
            stats = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

            # Detect trends
            trend = self._calculate_trend(values)

            # Detect cycles
            cycles = self._detect_cycles(values)

            return {
                "statistics": stats,
                "trend": trend,
                "cycles": cycles,
                "volatility": self._calculate_volatility(values),
            }

        except Exception:
            return {
                "statistics": {"mean": 0, "std": 0, "min": 0, "max": 0},
                "trend": "stable",
                "cycles": [],
                "volatility": 0,
            }

    def _analyze_metric_correlations(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between metrics."""
        try:
            correlation_matrix = df.corr()

            correlations = {}
            for col1 in df.columns:
                correlations[col1] = {}
                for col2 in df.columns:
                    if col1 != col2:
                        correlations[col1][col2] = correlation_matrix.loc[
                            col1, col2
                        ]

            return correlations

        except Exception:
            return {}

    def _detect_seasonality(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect seasonality in metrics."""
        try:
            seasonality = {}

            for column in df.columns:
                values = df[column].values

                # Detect daily patterns
                daily_pattern = self._detect_daily_pattern(values)

                # Detect weekly patterns
                weekly_pattern = self._detect_weekly_pattern(values)

                seasonality[column] = {
                    "daily_pattern": daily_pattern,
                    "weekly_pattern": weekly_pattern,
                    "strength": self._calculate_seasonality_strength(values),
                }

            return seasonality

        except Exception:
            return {}

    def _generate_trend_insights(
        self, patterns: Dict, correlations: Dict, seasonality: Dict
    ) -> List[Dict]:
        """Generate insights from trend analysis."""
        try:
            insights = []

            # Add pattern-based insights
            for metric, pattern in patterns.items():
                if pattern["trend"] != "stable":
                    insights.append(
                        {
                            "type": "trend",
                            "metric": metric,
                            "description": f"Strong {pattern['trend']} trend detected",
                            "confidence": self._calculate_trend_confidence(
                                pattern
                            ),
                        }
                    )

            # Add correlation insights
            strong_correlations = self._find_strong_correlations(correlations)
            for corr in strong_correlations:
                insights.append(
                    {
                        "type": "correlation",
                        "metrics": [corr["metric1"], corr["metric2"]],
                        "description": "Strong correlation detected",
                        "correlation": corr["value"],
                    }
                )

            # Add seasonality insights
            for metric, season in seasonality.items():
                if season["strength"] > 0.7:
                    insights.append(
                        {
                            "type": "seasonality",
                            "metric": metric,
                            "description": "Strong seasonal pattern detected",
                            "pattern": season,
                        }
                    )

            return insights

        except Exception:
            return []

    def _calculate_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile."""
        try:
            return float(norm.cdf(z_score) * 100)
        except Exception:
            return 50.0

    def _calculate_position_label(self, score: float) -> str:
        """Convert position score to label."""
        if score > 1.5:
            return "leader"
        elif score > 0.5:
            return "challenger"
        elif score > -0.5:
            return "average"
        else:
            return "laggard"

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate metric volatility."""
        try:
            return float(np.std(values))
        except Exception:
            return 0.0
