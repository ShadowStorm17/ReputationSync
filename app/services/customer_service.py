"""
Customer service.
Handles customer profile operations and business logic.
"""

import logging
from datetime import datetime
from typing import Dict

from app.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    ReputationError,
)
from app.core.metrics import metrics_manager
from app.models.customer import (
    CustomerProfile,
    CustomerProfileCreate,
    CustomerProfileResponse,
    CustomerProfileUpdate,
    PlatformProfile,
    SocialPlatform,
    SubscriptionFeatures,
    SubscriptionPlan,
)

logger = logging.getLogger(__name__)


class CustomerService:
    """Service for managing customer profiles."""

    def __init__(self):
        """Initialize customer service."""
        self._profiles: Dict[str, CustomerProfile] = {}
        self._initialize_subscription_features()

    def _initialize_subscription_features(self):
        """Initialize subscription plan features."""
        self._subscription_features = {
            SubscriptionPlan.FREE: SubscriptionFeatures(
                real_time_monitoring=False,
                one_time_analysis=True,
                platform_limit=1,
                analysis_frequency="monthly",
                reputation_fix=False,
                custom_reports=False,
                api_access=False,
                priority_support=False,
            ),
            SubscriptionPlan.BASIC: SubscriptionFeatures(
                real_time_monitoring=True,
                one_time_analysis=True,
                platform_limit=3,
                analysis_frequency="weekly",
                reputation_fix=False,
                custom_reports=False,
                api_access=False,
                priority_support=False,
            ),
            SubscriptionPlan.PROFESSIONAL: SubscriptionFeatures(
                real_time_monitoring=True,
                one_time_analysis=True,
                platform_limit=5,
                analysis_frequency="daily",
                reputation_fix=True,
                custom_reports=True,
                api_access=True,
                priority_support=True,
            ),
            SubscriptionPlan.ENTERPRISE: SubscriptionFeatures(
                real_time_monitoring=True,
                one_time_analysis=True,
                platform_limit=10,
                analysis_frequency="hourly",
                reputation_fix=True,
                custom_reports=True,
                api_access=True,
                priority_support=True,
            ),
        }

    async def create_profile(
        self, profile_data: CustomerProfileCreate
    ) -> CustomerProfileResponse:
        """Create a new customer profile."""
        try:
            # Generate unique ID
            profile_id = f"cust_{datetime.utcnow().timestamp()}"

            # Create profile
            profile = CustomerProfile(
                id=profile_id,
                email=profile_data.email,
                name=profile_data.name,
                company=profile_data.company,
                subscription=profile_data.subscription_plan,
                platform_profiles=profile_data.platform_profiles or {},
            )

            # Set subscription features
            profile.subscription.features = self._subscription_features[
                profile.subscription.plan
            ]

            # Store profile
            self._profiles[profile_id] = profile

            # Record metrics
            await metrics_manager.record_customer_metric(
                metric_type="profile_created",
                value=1,
                labels={"plan": profile.subscription.plan.value},
            )

            return CustomerProfileResponse(**profile.dict())

        except Exception as e:
            logger.error(f"Error creating profile: {str(e)}")
            raise ReputationError(
                message="Failed to create customer profile",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
                details={"error": str(e)},
            )

    async def get_profile(self, customer_id: str) -> CustomerProfileResponse:
        """Get customer profile by ID."""
        try:
            profile = self._profiles.get(customer_id)
            if not profile:
                raise ReputationError(
                    message=f"Customer profile not found: {customer_id}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.BUSINESS,
                )

            return CustomerProfileResponse(**profile.dict())

        except Exception as e:
            logger.error(f"Error getting profile: {str(e)}")
            raise ReputationError(
                message="Failed to get customer profile",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.BUSINESS,
                details={"error": str(e)},
            )

    async def update_profile(
        self, customer_id: str, profile_data: CustomerProfileUpdate
    ) -> CustomerProfileResponse:
        """Update customer profile."""
        try:
            profile = self._profiles.get(customer_id)
            if not profile:
                raise ReputationError(
                    message=f"Customer profile not found: {customer_id}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.BUSINESS,
                )

            # Update profile fields
            update_data = profile_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(profile, field, value)

            # Update timestamp
            profile.updated_at = datetime.utcnow()

            # Store updated profile
            self._profiles[customer_id] = profile

            return CustomerProfileResponse(**profile.dict())

        except Exception as e:
            logger.error(f"Error updating profile: {str(e)}")
            raise ReputationError(
                message="Failed to update customer profile",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
                details={"error": str(e)},
            )

    async def add_platform_profile(
        self,
        customer_id: str,
        platform: SocialPlatform,
        profile: PlatformProfile,
    ) -> CustomerProfileResponse:
        """Add a social media platform profile."""
        try:
            customer = await self.get_profile(customer_id)

            # Check platform limit
            if (
                len(customer.platform_profiles)
                >= customer.subscription.features.platform_limit
            ):
                raise ReputationError(
                    message="Platform limit reached for current subscription",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.BUSINESS,
                )

            # Add platform profile
            customer.platform_profiles[platform] = profile

            # Update profile
            return await self.update_profile(
                customer_id,
                CustomerProfileUpdate(
                    platform_profiles=customer.platform_profiles
                ),
            )

        except Exception as e:
            logger.error(f"Error adding platform profile: {str(e)}")
            raise ReputationError(
                message="Failed to add platform profile",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
                details={"error": str(e)},
            )

    async def remove_platform_profile(
        self, customer_id: str, platform: SocialPlatform
    ) -> CustomerProfileResponse:
        """Remove a social media platform profile."""
        try:
            customer = await self.get_profile(customer_id)

            # Remove platform profile
            if platform in customer.platform_profiles:
                del customer.platform_profiles[platform]

            # Update profile
            return await self.update_profile(
                customer_id,
                CustomerProfileUpdate(
                    platform_profiles=customer.platform_profiles
                ),
            )

        except Exception as e:
            logger.error(f"Error removing platform profile: {str(e)}")
            raise ReputationError(
                message="Failed to remove platform profile",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.BUSINESS,
                details={"error": str(e)},
            )

    async def upgrade_subscription(
        self, customer_id: str, new_plan: SubscriptionPlan
    ) -> CustomerProfileResponse:
        """Upgrade customer's subscription plan."""
        try:
            customer = await self.get_profile(customer_id)

            # Update subscription
            customer.subscription.plan = new_plan
            customer.subscription.features = self._subscription_features[
                new_plan
            ]
            customer.subscription.updated_at = datetime.utcnow()

            # Update profile
            return await self.update_profile(
                customer_id,
                CustomerProfileUpdate(subscription=customer.subscription),
            )

        except Exception as e:
            logger.error(f"Error upgrading subscription: {str(e)}")
            raise ReputationError(
                message="Failed to upgrade subscription",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
                details={"error": str(e)},
            )
