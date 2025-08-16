import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd

from app.core.config import get_settings
from app.services.platform_service import PlatformService
from app.services.predictive_service import PredictiveService
from app.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)
settings = get_settings()


class ReportingService:
    """Service for generating comprehensive reports and analytics."""

    def __init__(self):
        self.platform_service = PlatformService()
        self.sentiment_service = SentimentService()
        self.predictive_service = PredictiveService()

    async def generate_reputation_report(
        self,
        platform: str,
        entity_id: str,
        timeframe: str = "30d"
    ) -> Dict:
        """Generate comprehensive reputation report."""
        try:
            # Get platform data
            platform_data = await self.platform_service.get_platform_data(
                platform,
                entity_id
            )

            # Get sentiment analysis
            sentiment_data = await self.sentiment_service.get_reputation_score(
                platform_data,
                timeframe
            )

            # Get predictions
            predictions = await self.predictive_service.predict_reputation_trend(
                platform_data,
                timeframe
            )

            # Generate report sections
            overview = self._generate_overview(platform_data, sentiment_data)
            trends = self._analyze_trends(platform_data, timeframe)
            insights = self._generate_insights(platform_data, predictions)

            return {
                "overview": overview,
                "trends": trends,
                "insights": insights,
                "predictions": predictions,
                "recommendations": self._generate_recommendations(
                    overview,
                    trends,
                    predictions
                ),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating reputation report: {str(e)}")
            raise

    async def generate_competitor_analysis(
        self,
        platform: str,
        entity_id: str,
        competitor_ids: List[str],
        timeframe: str = "30d"
    ) -> Dict:
        """Generate competitive analysis report."""
        try:
            # Get platform data for the main entity
            platform_data = await self.platform_service.get_platform_data(
                platform,
                entity_id
            )
            # Get competitor data
            competitor_data = []
            for comp_id in competitor_ids:
                data = await self.platform_service.get_platform_data(
                    platform,
                    comp_id
                )
                competitor_data.append(data)

            # Analyze competitive position
            benchmarks = await self.predictive_service.analyze_competitor_benchmarks(
                platform,
                entity_id,
                competitor_ids
            )

            # Generate report sections
            market_analysis = self._analyze_market_position(
                platform_data,
                competitor_data
            )
            competitive_gaps = self._identify_competitive_gaps(
                platform_data,
                competitor_data
            )

            return {
                "market_analysis": market_analysis,
                "competitive_gaps": competitive_gaps,
                "benchmarks": benchmarks,
                "recommendations": self._generate_competitive_recommendations(
                    market_analysis,
                    competitive_gaps
                ),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating competitor analysis: {str(e)}")
            raise

    async def generate_trend_report(
        self,
        platform: str,
        entity_id: str,
        metrics: List[str],
        timeframe: str = "90d"
    ) -> Dict:
        """Generate detailed trend analysis report."""
        try:
            # Get historical data
            historical_data = await self.platform_service.get_historical_data(
                platform,
                entity_id,
                timeframe
            )

            # Analyze trends for each metric
            trend_analysis = {}
            for metric in metrics:
                trend_analysis[metric] = self._analyze_metric_trend(
                    historical_data,
                    metric
                )

            # Generate correlations
            correlations = self._analyze_correlations(historical_data, metrics)

            return {
                "trends": trend_analysis,
                "correlations": correlations,
                "summary": self._generate_trend_summary(trend_analysis),
                "anomalies": self._detect_anomalies(historical_data, metrics),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating trend report: {str(e)}")
            raise

    def _generate_overview(
        self,
        platform_data: Dict,
        sentiment_data: Dict
    ) -> Dict:
        """Generate overview section of the report."""
        return {
            "reputation_score": sentiment_data["overall_score"],
            "sentiment_distribution": sentiment_data["sentiment_distribution"],
            "engagement_metrics": {
                "rate": platform_data.get(
                    "engagement_rate",
                    0),
                "total_interactions": platform_data.get(
                    "total_interactions",
                    0),
                "active_users": platform_data.get(
                    "active_users",
                    0)},
            "reach_metrics": {
                "followers": platform_data.get(
                    "follower_count",
                    0),
                "impressions": platform_data.get(
                    "impressions",
                    0),
                "reach": platform_data.get(
                    "reach",
                    0)}}

    def _analyze_trends(
        self,
        platform_data: Dict,
        timeframe: str
    ) -> Dict:
        """Analyze trends in platform data."""
        df = pd.DataFrame(platform_data.get("historical_data", []))
        if df.empty:
            return {}

        trends = {}
        metrics = ["engagement_rate", "sentiment_score", "mention_count"]

        for metric in metrics:
            if metric in df.columns:
                trend = self._calculate_trend(df[metric])
                trends[metric] = {
                    "direction": trend["direction"],
                    "change": trend["change"],
                    "volatility": trend["volatility"]
                }

        return trends

    def _calculate_trend(self, series: pd.Series) -> Dict:
        """Calculate trend metrics for a time series."""
        if len(series) < 2:
            return {
                "direction": "stable",
                "change": 0,
                "volatility": 0
            }

        # Calculate direction and change
        start_value = series.iloc[0]
        end_value = series.iloc[-1]
        change = ((end_value - start_value) / start_value) * 100

        if change > 5:
            direction = "increasing"
        elif change < -5:
            direction = "decreasing"
        else:
            direction = "stable"

        # Calculate volatility
        volatility = series.std() / series.mean() if series.mean() != 0 else 0

        return {
            "direction": direction,
            "change": round(change, 2),
            "volatility": round(volatility, 2)
        }

    def _generate_insights(
        self,
        platform_data: Dict,
        predictions: Dict
    ) -> List[Dict]:
        """Generate insights from data and predictions."""
        insights = []

        # Analyze engagement patterns
        engagement_insight = self._analyze_engagement_patterns(platform_data)
        if engagement_insight:
            insights.append(engagement_insight)

        # Analyze content performance
        content_insight = self._analyze_content_performance(platform_data)
        if content_insight:
            insights.append(content_insight)

        # Analyze audience behavior
        audience_insight = self._analyze_audience_behavior(platform_data)
        if audience_insight:
            insights.append(audience_insight)

        # Add predictive insights
        if predictions.get("risk_score", 0) > 0.5:
            insights.append({
                "type": "risk_alert",
                "title": "High Risk Alert",
                "description": "Elevated risk of reputation damage detected",
                "priority": "high"
            })

        return insights

    def _analyze_engagement_patterns(self, data: Dict) -> Optional[Dict]:
        """Analyze patterns in engagement data."""
        try:
            engagement_data = pd.DataFrame(data.get("engagement_history", []))
            if engagement_data.empty:
                return None

            # Find peak engagement times
            engagement_data["hour"] = pd.to_datetime(
                engagement_data["timestamp"]
            ).dt.hour
            peak_hours = engagement_data.groupby(
                "hour")["engagement_rate"].mean()
            best_hours = peak_hours.nlargest(3)

            return {
                "type": "engagement_timing",
                "title": "Optimal Engagement Times",
                "description": f"Best posting times are at {', '.join(str(h) for h in best_hours.index)}:00",
                "priority": "medium"}

        except Exception:
            return None

    def _analyze_content_performance(self, data: Dict) -> Optional[Dict]:
        """Analyze content performance patterns."""
        try:
            content_data = pd.DataFrame(data.get("content", []))
            if content_data.empty:
                return None

            # Find best performing content types
            performance = content_data.groupby(
                "type")["engagement_rate"].mean()
            best_type = performance.idxmax()

            return {
                "type": "content_strategy",
                "title": "Content Performance",
                "description": f"{best_type} content shows highest engagement",
                "priority": "medium"
            }

        except Exception:
            return None

    def _analyze_audience_behavior(self, data: Dict) -> Optional[Dict]:
        """Analyze audience behavior patterns."""
        try:
            audience_data = data.get("audience_metrics", {})
            if not audience_data:
                return None

            growth_rate = audience_data.get("growth_rate", 0)
            churn_rate = audience_data.get("churn_rate", 0)

            if growth_rate < churn_rate:
                return {
                    "type": "audience_retention",
                    "title": "Audience Retention Alert",
                    "description": "Audience churn rate exceeds growth rate",
                    "priority": "high"
                }

            return None

        except Exception:
            return None

    def _generate_recommendations(
        self,
        overview: Dict,
        trends: Dict,
        predictions: Dict
    ) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check engagement trends
        if trends.get("engagement_rate", {}).get("direction") == "decreasing":
            recommendations.append({
                "type": "engagement",
                "priority": "high",
                "action": "Increase posting frequency and interactive content",
                "impact": "Improve engagement rates"
            })

        # Check sentiment trends
        if trends.get("sentiment_score", {}).get("direction") == "decreasing":
            recommendations.append({
                "type": "sentiment",
                "priority": "high",
                "action": "Address negative sentiment in recent content",
                "impact": "Improve brand perception"
            })

        # Check prediction-based recommendations
        if predictions.get("risk_score", 0) > 0.7:
            recommendations.append({
                "type": "risk_mitigation",
                "priority": "critical",
                "action": "Implement reputation protection measures",
                "impact": "Prevent potential reputation damage"
            })

        return recommendations
