"""
Analytics router.
Provides endpoints for advanced analytics and visualization.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_analytics_service
from app.core.error_handling import ReputationError
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/sentiment-heatmap")
async def generate_sentiment_heatmap(
    data: List[Dict[str, Any]],
    region_field: str = "region",
    platform_field: str = "platform",
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """Generate sentiment heatmap by region and platform."""
    try:
        return await analytics_service.generate_sentiment_heatmap(
            data=data, region_field=region_field, platform_field=platform_field
        )
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating sentiment heatmap: {str(e)}"
        )


@router.post("/influence-graph")
async def generate_influence_graph(
    data: List[Dict[str, Any]],
    min_influence: float = 0.1,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """Generate influence graph based on mention reach."""
    try:
        return await analytics_service.generate_influence_graph(
            data=data, min_influence=min_influence
        )
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating influence graph: {str(e)}"
        )


@router.post("/forecast-trends")
async def forecast_trends(
    data: List[Dict[str, Any]],
    forecast_days: int = 30,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """Forecast sentiment trends."""
    try:
        return await analytics_service.forecast_trends(
            data=data, forecast_days=forecast_days
        )
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error forecasting trends: {str(e)}"
        )


@router.post("/reputation-score")
async def calculate_reputation_score(
    data: List[Dict[str, Any]],
    industry_average: Optional[Dict[str, float]] = None,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """Calculate cross-platform reputation score with benchmarking."""
    try:
        return await analytics_service.calculate_reputation_score(
            data=data, industry_average=industry_average
        )
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating reputation score: {str(e)}"
        )
