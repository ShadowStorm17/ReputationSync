from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.cache import cache_manager
from app.core.security import get_current_user
from app.models.user import User
from app.services.analytics import (
    get_reputation_metrics,
    get_sentiment_metrics,
)
from app.services.engagement import get_engagement_metrics

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request, current_user: User = Depends(get_current_user)
):
    """Render the dashboard page."""
    try:
        # Get cached dashboard data or fetch new data
        dashboard_data = await cache_manager.get("dashboard_data")
        if not dashboard_data:
            dashboard_data = await fetch_dashboard_data()
            # Cache for 5 minutes
            await cache_manager.set(
                "dashboard_data", dashboard_data, expire=300
            )

        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "user": current_user, **dashboard_data},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/refresh")
async def refresh_dashboard(current_user: User = Depends(get_current_user)):
    """Refresh dashboard data."""
    try:
        data = await fetch_dashboard_data()
        await cache_manager.set("dashboard_data", data, expire=300)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_dashboard_data() -> Dict[str, Any]:
    """Fetch all dashboard data."""
    # Get reputation metrics
    reputation_metrics = await get_reputation_metrics()

    # Get sentiment metrics
    sentiment_metrics = await get_sentiment_metrics()

    # Get engagement metrics
    engagement_metrics = await get_engagement_metrics()

    # Get recent activities
    recent_activities = get_recent_activities()

    # Calculate changes
    total_score_change = calculate_percentage_change(
        reputation_metrics["current_score"],
        reputation_metrics["previous_score"],
    )

    active_users_change = calculate_percentage_change(
        engagement_metrics["active_users"],
        engagement_metrics["previous_active_users"],
    )

    sentiment_change = calculate_percentage_change(
        sentiment_metrics["average_sentiment"],
        sentiment_metrics["previous_sentiment"],
    )

    engagement_change = calculate_percentage_change(
        engagement_metrics["engagement_rate"],
        engagement_metrics["previous_engagement_rate"],
    )

    return {
        "total_score": reputation_metrics["current_score"],
        "score_change": total_score_change,
        "active_users": engagement_metrics["active_users"],
        "user_change": active_users_change,
        "sentiment_score": sentiment_metrics["average_sentiment"],
        "sentiment_change": sentiment_change,
        "engagement_rate": engagement_metrics["engagement_rate"],
        "engagement_change": engagement_change,
        "dates": reputation_metrics["dates"],
        "scores": reputation_metrics["scores"],
        "sentiment_distribution": sentiment_metrics["distribution"],
        "recent_activities": recent_activities,
    }


def get_recent_activities() -> List[Dict[str, Any]]:
    """Get recent user activities."""
    # This would typically come from your database
    # For now, returning mock data
    return [
        {
            "user_name": "John Doe",
            "user_avatar": "/static/img/avatars/user1.jpg",
            "description": "Updated reputation score for @company",
            "status": "Completed",
            "status_class": "status-success",
        },
        {
            "user_name": "Jane Smith",
            "user_avatar": "/static/img/avatars/user2.jpg",
            "description": "Analyzed sentiment for new product launch",
            "status": "In Progress",
            "status_class": "status-warning",
        },
        {
            "user_name": "Mike Johnson",
            "user_avatar": "/static/img/avatars/user3.jpg",
            "description": "Generated monthly report",
            "status": "Completed",
            "status_class": "status-success",
        },
    ]


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100
