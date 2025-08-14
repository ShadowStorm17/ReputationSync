"""
Premium features router.
Handles additional revenue streams and premium features.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    status,
)

from app.core.error_handling import ErrorCategory, ErrorSeverity, handle_errors
from app.core.metrics import track_performance
from app.core.rate_limiting import rate_limit
from app.core.security import User, get_current_active_user
from app.models.premium_features import (
    CustomIntegration,
    CustomReport,
    DataExport,
    ExportFormat,
    HistoricalData,
    IntegrationType,
    PayPerUse,
    TrainingSession,
    TrainingType,
)
from app.services.notification_service import NotificationService
from app.services.premium_service import PremiumService

router = APIRouter(
    prefix="/premium",
    tags=["premium"],
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
premium_service = PremiumService()
notification_service = NotificationService()


@router.post("/pay-per-use")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=100, period=3600)  # 100 requests per hour
async def create_pay_per_use(
    current_user: User = Depends(get_current_active_user),
    api_calls: int = Query(..., description="Number of API calls to purchase"),
) -> PayPerUse:
    """Purchase additional API calls beyond plan limits."""
    try:
        result = await premium_service.create_pay_per_use(
            user_id=current_user.id, api_calls=api_calls
        )
        return result
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing pay-per-use request",
        )


@router.post("/reports")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=10, period=3600)  # 10 requests per hour
async def create_custom_report(
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    report_type: str = Query(..., description="Type of report"),
    format: ExportFormat = Query(..., description="Export format"),
    custom_metrics: List[str] = Query(
        ..., description="Custom metrics to include"
    ),
    include_visualizations: bool = Query(
        True, description="Include visualizations"
    ),
    include_recommendations: bool = Query(
        True, description="Include recommendations"
    ),
    delivery_method: str = Query(..., description="Delivery method"),
) -> CustomReport:
    """Request a custom report generation."""
    try:
        report = await premium_service.create_custom_report(
            user_id=current_user.id,
            report_type=report_type,
            format=format,
            custom_metrics=custom_metrics,
            include_visualizations=include_visualizations,
            include_recommendations=include_recommendations,
            delivery_method=delivery_method,
        )

        # Schedule report generation
        background_tasks.add_task(
            premium_service.generate_report, report_id=report.report_id
        )

        return report
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating custom report",
        )


@router.post("/export")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=20, period=3600)  # 20 requests per hour
async def create_data_export(
    current_user: User = Depends(get_current_active_user),
    format: ExportFormat = Query(..., description="Export format"),
    start_date: datetime = Query(..., description="Start date"),
    end_date: datetime = Query(..., description="End date"),
    include_metadata: bool = Query(True, description="Include metadata"),
    compression: bool = Query(False, description="Enable compression"),
    encryption: bool = Query(False, description="Enable encryption"),
) -> DataExport:
    """Request a data export in premium format."""
    try:
        export = await premium_service.create_data_export(
            user_id=current_user.id,
            format=format,
            start_date=start_date,
            end_date=end_date,
            include_metadata=include_metadata,
            compression=compression,
            encryption=encryption,
        )
        return export
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating data export",
        )


@router.post("/historical-data")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=5, period=3600)  # 5 requests per hour
async def request_historical_data(
    current_user: User = Depends(get_current_active_user),
    start_date: datetime = Query(..., description="Start date"),
    end_date: datetime = Query(..., description="End date"),
    data_types: List[str] = Query(..., description="Types of data to access"),
    retention_period: int = Query(..., description="Retention period in days"),
) -> HistoricalData:
    """Request access to historical data."""
    try:
        access = await premium_service.create_historical_data_access(
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            retention_period=retention_period,
        )
        return access
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error requesting historical data access",
        )


@router.post("/integrations")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=3, period=3600)  # 3 requests per hour
async def request_custom_integration(
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    integration_type: IntegrationType = Query(
        ..., description="Type of integration"
    ),
    platform: str = Query(..., description="Target platform"),
    requirements: Dict[str, Any] = Query(
        ..., description="Integration requirements"
    ),
) -> CustomIntegration:
    """Request a custom integration."""
    try:
        integration = await premium_service.create_custom_integration(
            user_id=current_user.id,
            integration_type=integration_type,
            platform=platform,
            requirements=requirements,
        )

        # Schedule integration setup
        background_tasks.add_task(
            premium_service.setup_integration,
            integration_id=integration.integration_id,
        )

        return integration
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error requesting custom integration",
        )


@router.post("/training")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=5, period=3600)  # 5 requests per hour
async def schedule_training(
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    session_type: TrainingType = Query(
        ..., description="Type of training session"
    ),
    duration: int = Query(..., description="Duration in hours"),
    topics: List[str] = Query(..., description="Topics to cover"),
    participants: int = Query(..., description="Number of participants"),
    scheduled_at: datetime = Query(..., description="Scheduled date and time"),
) -> TrainingSession:
    """Schedule a training or consulting session."""
    try:
        session = await premium_service.create_training_session(
            user_id=current_user.id,
            session_type=session_type,
            duration=duration,
            topics=topics,
            participants=participants,
            scheduled_at=scheduled_at,
        )

        # Schedule confirmation emails
        background_tasks.add_task(
            notification_service.send_training_confirmation,
            session_id=session.session_id,
        )

        return session
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error scheduling training session",
        )


@router.get("/pricing")
@track_performance
@handle_errors(ErrorSeverity.LOW, ErrorCategory.BUSINESS)
@rate_limit(limit=50, period=3600)  # 50 requests per hour
async def get_premium_pricing(
    current_user: Optional[User] = Depends(get_current_active_user),
    feature_type: Optional[str] = Query(
        None, description="Type of premium feature"
    ),
) -> Dict[str, Any]:
    """Get pricing information for premium features."""
    try:
        pricing = await premium_service.get_premium_pricing(feature_type)
        return pricing
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving premium pricing",
        )
