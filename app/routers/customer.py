"""
Customer router.
Handles customer profile operations.
"""

from datetime import datetime
from typing import Dict, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.error_handling import ErrorCategory, ErrorSeverity, handle_errors
from app.core.metrics import track_performance
from app.core.security import User, get_current_active_user
from app.models.customer import (
    CustomerProfileCreate,
    CustomerProfileResponse,
    CustomerProfileUpdate,
    PlatformProfile,
    SocialPlatform,
    SubscriptionPlan,
)
from app.services.customer_service import CustomerService
from app.services.reputation_service import ReputationService

router = APIRouter(
    prefix="/customers",
    tags=["customers"],
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
customer_service = CustomerService()
reputation_service = ReputationService()


@router.post("", response_model=CustomerProfileResponse)
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
async def create_customer_profile(
    profile: CustomerProfileCreate,
    current_user: User = Depends(get_current_active_user),
) -> CustomerProfileResponse:
    """Create a new customer profile."""
    return await customer_service.create_profile(profile)


@router.get("/{customer_id}", response_model=CustomerProfileResponse)
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
async def get_customer_profile(
    customer_id: str, current_user: User = Depends(get_current_active_user)
) -> CustomerProfileResponse:
    """Get customer profile by ID."""
    return await customer_service.get_profile(customer_id)


@router.put("/{customer_id}", response_model=CustomerProfileResponse)
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
async def update_customer_profile(
    customer_id: str,
    profile: CustomerProfileUpdate,
    current_user: User = Depends(get_current_active_user),
) -> CustomerProfileResponse:
    """Update customer profile."""
    return await customer_service.update_profile(customer_id, profile)


@router.post("/{customer_id}/platforms/{platform}")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
async def add_platform_profile(
    customer_id: str,
    platform: SocialPlatform,
    profile: PlatformProfile,
    current_user: User = Depends(get_current_active_user),
) -> CustomerProfileResponse:
    """Add a social media platform profile."""
    return await customer_service.add_platform_profile(
        customer_id, platform, profile
    )


@router.delete("/{customer_id}/platforms/{platform}")
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
async def remove_platform_profile(
    customer_id: str,
    platform: SocialPlatform,
    current_user: User = Depends(get_current_active_user),
) -> CustomerProfileResponse:
    """Remove a social media platform profile."""
    return await customer_service.remove_platform_profile(
        customer_id, platform
    )


@router.get("/{customer_id}/reputation")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
async def get_customer_reputation(
    customer_id: str,
    platform: Optional[SocialPlatform] = None,
    timeframe: str = Query("7d", description="Timeframe for analysis"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get customer's reputation across platforms."""
    customer = await customer_service.get_profile(customer_id)

    # Check subscription features
    if not customer.subscription.features.real_time_monitoring:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Real-time monitoring not available in current plan",
        )

    # Get reputation for specified platform or all platforms
    platforms = [platform] if platform else customer.platform_profiles.keys()

    reputation_data = {}
    for p in platforms:
        if p in customer.platform_profiles:
            profile = customer.platform_profiles[p]
            reputation = await reputation_service.get_reputation(
                platform=p, username=profile.username, timeframe=timeframe
            )
            reputation_data[p.value] = reputation

    return {
        "customer_id": customer_id,
        "reputation_data": reputation_data,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.post("/{customer_id}/reputation/fix")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
async def fix_customer_reputation(
    customer_id: str,
    platform: Optional[SocialPlatform] = None,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Fix customer's reputation across platforms."""
    customer = await customer_service.get_profile(customer_id)

    # Check subscription features
    if not customer.subscription.features.reputation_fix:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reputation fix not available in current plan",
        )

    # Fix reputation for specified platform or all platforms
    platforms = [platform] if platform else customer.platform_profiles.keys()

    fix_results = {}
    for p in platforms:
        if p in customer.platform_profiles:
            profile = customer.platform_profiles[p]
            result = await reputation_service.fix_reputation(
                platform=p, username=profile.username
            )
            fix_results[p.value] = result

    return {
        "customer_id": customer_id,
        "fix_results": fix_results,
        "completed_at": datetime.utcnow().isoformat(),
    }


@router.get("/{customer_id}/subscription")
@track_performance
@handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS)
async def get_subscription_details(
    customer_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get customer's subscription details."""
    customer = await customer_service.get_profile(customer_id)

    return {
        "customer_id": customer_id,
        "subscription": customer.subscription,
        "platform_count": len(customer.platform_profiles),
        "max_platforms": customer.subscription.features.platform_limit,
    }


@router.post("/{customer_id}/subscription/upgrade")
@track_performance
@handle_errors(ErrorSeverity.HIGH, ErrorCategory.BUSINESS)
async def upgrade_subscription(
    customer_id: str,
    new_plan: SubscriptionPlan,
    current_user: User = Depends(get_current_active_user),
) -> CustomerProfileResponse:
    """Upgrade customer's subscription plan."""
    return await customer_service.upgrade_subscription(customer_id, new_plan)
