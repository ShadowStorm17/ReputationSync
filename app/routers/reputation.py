"""
Reputation router.
Handles reputation management endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from datetime import timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    status,
)

from app.core.cache import cache_response
from app.core.error_handling import ErrorCategory, ErrorSeverity, handle_errors
from app.core.metrics import track_performance
from app.core.rate_limiting import rate_limit
from app.core.security import User, get_current_active_user
from app.core.validation import (
    validate_date_range,
    validate_platform,
    validate_username,
)
from app.models.reputation import (
    AlertSeverity,
    AlertType,
    MonitoringConfig,
    ReputationAlert,
    ReputationMetrics,
    ReputationScore,
    MultiPlatformReputation,
    PlatformBreakdown,
)
from app.services.analytics_engine import AnalyticsEngine
from app.services.comment_service import CommentService
from app.services.notification_service import NotificationService
from app.services.predictive_service import PredictiveService
from app.services.reporting_service import ReportingService
from app.services.response_service import ResponseService
from app.services.sentiment_service import SentimentService

router = APIRouter(
    prefix="/reputation",
    tags=["reputation"],
    responses={
        404: {"description": "Not found"},
        400: {"description": "Bad request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Too many requests"},
        500: {"description": "Internal server error"},
    },
)

# Initialize services
sentiment_service = SentimentService()
predictive_service = PredictiveService()
response_service = ResponseService()
comment_service = CommentService()
analytics_engine = AnalyticsEngine()
reporting_service = ReportingService()
notification_service = NotificationService()


@router.get("/score", response_model=ReputationScore)
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=100, period=3600)  # 100 requests per hour
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_reputation_score(
    platform: str = Query(
        ..., description="Platform name (e.g., 'linkedin', 'twitter')"
    ),
    username: str = Query(..., description="Username to analyze"),
    timeframe: str = Query(
        "7d", description="Timeframe for analysis (e.g., '7d', '30d', '1y')"
    ),
    include_trends: bool = Query(True, description="Include trend analysis"),
    current_user: User = Depends(get_current_active_user),
) -> ReputationScore:
    """Get reputation score for a user on a platform."""
    try:
        # Validate inputs
        validate_platform(platform)
        validate_username(username)

        # Get sentiment analysis
        sentiment = await sentiment_service.analyze_user_sentiment(
            platform, username
        )

        # Get predictive insights
        predictions = await predictive_service.get_reputation_predictions(
            platform, username
        )

        # Get analytics
        analytics = await analytics_engine.get_user_analytics(
            platform, username
        )

        # Get trend analysis if requested
        trends = None
        if include_trends:
            trends = await analytics_engine.get_trend_analysis(
                platform, username, timeframe
            )

        # Combine into reputation score
        score = ReputationScore(
            platform=platform,
            username=username,
            sentiment_score=sentiment.score,
            engagement_score=analytics.engagement_score,
            influence_score=analytics.influence_score,
            growth_score=predictions.growth_potential,
            overall_score=analytics.calculate_overall_score(),
            timeframe=timeframe,
            calculated_at=datetime.now(timezone.utc),
            trend_direction=trends.direction if trends else "stable",
            trend_magnitude=trends.magnitude if trends else 0.0,
            confidence_score=analytics.confidence_score,
        )

        return score

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating reputation score",
        )


@router.get("/metrics", response_model=ReputationMetrics)
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=50, period=3600)  # 50 requests per hour
@cache_response(ttl=600)  # Cache for 10 minutes
async def get_reputation_metrics(
    platform: str = Query(..., description="Platform name"),
    username: str = Query(..., description="Username to analyze"),
    timeframe: str = Query("30d", description="Timeframe for analysis"),
    include_historical: bool = Query(
        False, description="Include historical data"
    ),
    current_user: User = Depends(get_current_active_user),
) -> ReputationMetrics:
    """Get detailed reputation metrics."""
    try:
        # Validate inputs
        validate_platform(platform)
        validate_username(username)

        # Get analytics
        analytics = await analytics_engine.get_user_analytics(
            platform, username
        )

        # Get sentiment trends
        sentiment_trends = await sentiment_service.get_sentiment_trends(
            platform, username
        )

        # Get engagement metrics
        engagement = await analytics_engine.get_engagement_metrics(
            platform, username
        )

        # Get predictions
        predictions = await predictive_service.get_reputation_predictions(
            platform, username
        )

        # Get historical data if requested
        historical_data = None
        if include_historical:
            historical_data = await analytics_engine.get_historical_metrics(
                platform, username, timeframe
            )

        # Combine into metrics
        metrics = ReputationMetrics(
            platform=platform,
            username=username,
            sentiment_trends=sentiment_trends,
            engagement_metrics=engagement,
            influence_metrics=analytics.influence_metrics,
            growth_predictions=predictions.metrics,
            risk_factors=analytics.risk_factors,
            timeframe=timeframe,
            calculated_at=datetime.now(timezone.utc),
            historical_data=historical_data,
        )

        return metrics

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating reputation metrics",
        )


@router.post("/analyze/comment")
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
@rate_limit(limit=200, period=3600)  # 200 requests per hour
async def analyze_comment(
    platform: str = Query(..., description="Platform name"),
    comment_id: str = Query(..., description="Comment ID to analyze"),
    include_suggestions: bool = Query(
        True, description="Include response suggestions"
    ),
    include_context: bool = Query(
        False, description="Include surrounding context"
    ),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Analyze a specific comment."""
    try:
        # Validate inputs
        validate_platform(platform)

        # Get comment analysis
        analysis = await comment_service.analyze_comment(platform, comment_id)

        # Get context if requested
        context = None
        if include_context:
            context = await comment_service.get_comment_context(
                platform, comment_id
            )

        result = {
            "analysis": analysis,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        if context:
            result["context"] = context

        # Get response suggestions if requested
        if include_suggestions:
            suggestions = await response_service.get_response_suggestions(
                platform, comment_id, analysis
            )
            result["suggestions"] = suggestions

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error analyzing comment",
        )


@router.get("/report")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=20, period=3600)  # 20 requests per hour
async def generate_report(
    platform: str = Query(..., description="Platform name"),
    username: str = Query(..., description="Username to analyze"),
    report_type: str = Query(
        ..., description="Report type (e.g., 'summary', 'detailed', 'trends')"
    ),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    include_recommendations: bool = Query(
        True, description="Include recommendations"
    ),
    include_competitors: bool = Query(
        False, description="Include competitor analysis"
    ),
    format: str = Query("json", description="Report format (json, pdf, csv)"),
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict[str, Any]:
    """Generate a reputation report."""
    try:
        # Validate inputs
        validate_platform(platform)
        validate_username(username)
        validate_date_range(start_date, end_date)

        # Generate report
        report = await reporting_service.generate_report(
            platform=platform,
            username=username,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            include_recommendations=include_recommendations,
            include_competitors=include_competitors,
            format=format,
        )

        # Schedule notification if report is large
        if report.get("size", 0) > 1024 * 1024:  # 1MB
            background_tasks.add_task(
                notification_service.send_report_notification,
                user_id=current_user.id,
                report_id=report["id"],
            )

        return {**report, "generated_at": datetime.now(timezone.utc).isoformat()}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating report",
        )


@router.post("/monitoring/config")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=10, period=3600)  # 10 requests per hour
async def configure_monitoring(
    config: MonitoringConfig,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Configure reputation monitoring."""
    try:
        # Validate configuration
        if not config.is_active and config.alert_channels:
            raise ValueError(
                "Cannot set alert channels for inactive monitoring"
            )

        # Save configuration
        result = await analytics_engine.save_monitoring_config(config)

        return {
            "status": "success",
            "message": "Monitoring configuration saved",
            "config": result,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error configuring monitoring",
        )


@router.get("/alerts")
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
@rate_limit(limit=100, period=3600)  # 100 requests per hour
async def get_alerts(
    platform: str = Query(..., description="Platform name"),
    username: str = Query(..., description="Username to analyze"),
    severity: Optional[AlertSeverity] = Query(
        None, description="Filter by alert severity"
    ),
    alert_type: Optional[AlertType] = Query(
        None, description="Filter by alert type"
    ),
    resolved: Optional[bool] = Query(
        None, description="Filter by resolution status"
    ),
    start_date: Optional[str] = Query(
        None, description="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_active_user),
) -> List[ReputationAlert]:
    """Get reputation alerts."""
    try:
        # Validate inputs
        validate_platform(platform)
        validate_username(username)
        if start_date and end_date:
            validate_date_range(start_date, end_date)

        # Get alerts
        alerts = await analytics_engine.get_alerts(
            platform=platform,
            username=username,
            severity=severity,
            alert_type=alert_type,
            resolved=resolved,
            start_date=start_date,
            end_date=end_date,
        )

        return alerts

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving alerts",
        )


@router.post("/alerts/{alert_id}/resolve")
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
@rate_limit(limit=50, period=3600)  # 50 requests per hour
async def resolve_alert(
    alert_id: str,
    resolution_notes: str = Query(
        ..., description="Notes about the resolution"
    ),
    action_taken: str = Query(
        ..., description="Action taken to resolve the alert"
    ),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Resolve a reputation alert."""
    try:
        # Resolve alert
        result = await analytics_engine.resolve_alert(
            alert_id=alert_id,
            resolution_notes=resolution_notes,
            action_taken=action_taken,
            resolved_by=current_user.id,
        )

        return {
            "status": "success",
            "message": "Alert resolved successfully",
            "alert": result,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error resolving alert",
        )


@router.get("/aggregate", response_model=MultiPlatformReputation)
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=20, period=3600)  # 20 requests per hour
@cache_response(ttl=300)  # Cache for 5 minutes
async def aggregate_reputation(
    person: str = Query(..., description="Person identifier (username/customer id)"),
    platforms: List[str] = Query(..., description="List of platforms"),
    usernames: List[str] = Query(..., description="List of usernames aligned with platforms"),
    timeframe: str = Query("30d", description="Timeframe for analysis"),
    weights: Optional[str] = Query(None, description="JSON map of platform->weight"),
    current_user: User = Depends(get_current_active_user),
) -> MultiPlatformReputation:
    """Aggregate multi-platform reputation for a person.

    platforms and usernames must be the same length and aligned by index.
    weights is an optional JSON string like: {"twitter": 0.6, "linkedin": 0.4}
    """
    try:
        if len(platforms) != len(usernames):
            raise ValueError("platforms and usernames must be the same length")

        # Validate inputs and prepare weights
        for p in platforms:
            validate_platform(p)
        for u in usernames:
            validate_username(u)

        weight_map: Dict[str, float] = {}
        if weights:
            try:
                import json as _json
                parsed = _json.loads(weights)
                if not isinstance(parsed, dict):
                    raise ValueError("weights must be a JSON object of platform->weight")
                weight_map = {str(k): float(v) for k, v in parsed.items()}
            except Exception as e:
                raise ValueError(f"Invalid weights: {e}")

        # Equal weights by default
        n = len(platforms)
        default_weight = 1.0 / n if n > 0 else 0.0

        breakdown: List[PlatformBreakdown] = []

        for p, u in zip(platforms, usernames):
            # Use existing services similar to get_reputation_score
            sentiment = await sentiment_service.analyze_user_sentiment(p, u)
            predictions = await predictive_service.get_reputation_predictions(p, u)
            analytics = await analytics_engine.get_user_analytics(p, u)

            # Compute overall score using existing analytics method
            platform_overall = analytics.calculate_overall_score()

            w = float(weight_map.get(p, default_weight))

            breakdown.append(
                PlatformBreakdown(
                    platform=p,
                    username=u,
                    overall_score=platform_overall,
                    weight=w,
                    sentiment_score=sentiment.score,
                    engagement_score=analytics.engagement_score,
                    influence_score=analytics.influence_score,
                    growth_score=predictions.growth_potential,
                )
            )

        # Normalize weights to sum to 1 (avoid division by zero)
        total_w = sum(b.weight for b in breakdown)
        if total_w <= 0:
            total_w = 1.0
        norm_breakdown = []
        for b in breakdown:
            norm_w = b.weight / total_w
            norm_breakdown.append(
                PlatformBreakdown(
                    platform=b.platform,
                    username=b.username,
                    overall_score=b.overall_score,
                    weight=norm_w,
                    sentiment_score=b.sentiment_score,
                    engagement_score=b.engagement_score,
                    influence_score=b.influence_score,
                    growth_score=b.growth_score,
                )
            )

        composite_score = sum(b.overall_score * b.weight for b in norm_breakdown)

        return MultiPlatformReputation(
            person=person,
            timeframe=timeframe,
            overall_score=composite_score,
            weights={b.platform: b.weight for b in norm_breakdown},
            breakdown=norm_breakdown,
            calculated_at=datetime.now(timezone.utc),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error aggregating reputation",
        )
