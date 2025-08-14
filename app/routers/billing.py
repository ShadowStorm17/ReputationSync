"""
Billing router.
Handles subscription management, payments, and usage tracking.
"""

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
from app.models.pricing import (
    Invoice,
    PlanType,
    SubscriptionPlan,
    UsageMetrics,
)
from app.services.billing_service import BillingService
from app.services.notification_service import NotificationService

router = APIRouter(
    prefix="/billing",
    tags=["billing"],
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
billing_service = BillingService()
notification_service = NotificationService()


@router.get("/plans", response_model=List[SubscriptionPlan])
@track_performance
@handle_errors(ErrorSeverity.LOW, ErrorCategory.BUSINESS)
@rate_limit(limit=100, period=3600)  # 100 requests per hour
async def get_subscription_plans(
    include_enterprise: bool = Query(
        False, description="Include enterprise plans"
    ),
    current_user: Optional[User] = Depends(get_current_active_user),
) -> List[SubscriptionPlan]:
    """Get available subscription plans."""
    try:
        plans = await billing_service.get_available_plans(include_enterprise)
        return plans
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving subscription plans",
        )


@router.post("/subscribe")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=10, period=3600)  # 10 requests per hour
async def subscribe_to_plan(
    plan_type: PlanType,
    billing_cycle: str = Query(
        ..., description="Billing cycle (monthly/annual)"
    ),
    payment_method: str = Query(..., description="Payment method ID"),
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict[str, Any]:
    """Subscribe to a plan."""
    try:
        # Create subscription
        subscription = await billing_service.create_subscription(
            user_id=current_user.id,
            plan_type=plan_type,
            billing_cycle=billing_cycle,
            payment_method=payment_method,
        )

        # Schedule welcome email
        background_tasks.add_task(
            notification_service.send_welcome_email,
            user_id=current_user.id,
            plan_type=plan_type,
        )

        return {
            "status": "success",
            "message": "Subscription created successfully",
            "subscription": subscription,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating subscription",
        )


@router.get("/usage", response_model=UsageMetrics)
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
@rate_limit(limit=50, period=3600)  # 50 requests per hour
async def get_usage_metrics(
    start_date: Optional[str] = Query(
        None, description="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_active_user),
) -> UsageMetrics:
    """Get usage metrics for the current subscription."""
    try:
        metrics = await billing_service.get_usage_metrics(
            user_id=current_user.id, start_date=start_date, end_date=end_date
        )
        return metrics
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving usage metrics",
        )


@router.get("/invoices", response_model=List[Invoice])
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
@rate_limit(limit=50, period=3600)  # 50 requests per hour
async def get_invoices(
    status: Optional[str] = Query(
        None, description="Filter by invoice status"
    ),
    start_date: Optional[str] = Query(
        None, description="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_active_user),
) -> List[Invoice]:
    """Get invoice history."""
    try:
        invoices = await billing_service.get_invoices(
            user_id=current_user.id,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )
        return invoices
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving invoices",
        )


@router.post("/upgrade")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=5, period=3600)  # 5 requests per hour
async def upgrade_plan(
    new_plan_type: PlanType,
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict[str, Any]:
    """Upgrade to a higher tier plan."""
    try:
        # Upgrade subscription
        result = await billing_service.upgrade_plan(
            user_id=current_user.id, new_plan_type=new_plan_type
        )

        # Schedule upgrade notification
        background_tasks.add_task(
            notification_service.send_upgrade_notification,
            user_id=current_user.id,
            new_plan_type=new_plan_type,
        )

        return {
            "status": "success",
            "message": "Plan upgraded successfully",
            "subscription": result,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error upgrading plan",
        )


@router.post("/cancel")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=3, period=3600)  # 3 requests per hour
async def cancel_subscription(
    reason: str = Query(..., description="Reason for cancellation"),
    feedback: Optional[str] = Query(None, description="Additional feedback"),
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict[str, Any]:
    """Cancel the current subscription."""
    try:
        # Cancel subscription
        result = await billing_service.cancel_subscription(
            user_id=current_user.id, reason=reason, feedback=feedback
        )

        # Schedule cancellation notification
        background_tasks.add_task(
            notification_service.send_cancellation_notification,
            user_id=current_user.id,
            reason=reason,
        )

        return {
            "status": "success",
            "message": "Subscription cancelled successfully",
            "details": result,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error cancelling subscription",
        )


@router.post("/payment-method")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
@rate_limit(limit=10, period=3600)  # 10 requests per hour
async def update_payment_method(
    payment_method_id: str = Query(..., description="New payment method ID"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Update payment method."""
    try:
        result = await billing_service.update_payment_method(
            user_id=current_user.id, payment_method_id=payment_method_id
        )

        return {
            "status": "success",
            "message": "Payment method updated successfully",
            "details": result,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating payment method",
        )
