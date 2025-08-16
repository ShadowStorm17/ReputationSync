"""
API key service.
Handles API key generation, validation, and tracking.
"""

import logging
import secrets
import string
from datetime import datetime, timezone
from typing import Dict, List

from app.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    ReputationError,
)
from app.core.metrics import metrics_manager
from app.core.notifications import NotificationService
from app.models.api_key import (
    APIKey,
    APIKeyCreate,
    APIKeyPermissions,
    APIKeyResponse,
    APIKeyStats,
    APIKeyStatus,
    APIKeyUsage,
)
from app.models.customer import SubscriptionPlan

logger = logging.getLogger(__name__)


class APIKeyService:
    """Service for managing API keys."""

    def __init__(self):
        """Initialize API key service."""
        self._keys: Dict[str, APIKey] = {}
        self._usage_history: List[APIKeyUsage] = []
        self._notification_service = NotificationService()
        self._initialize_permissions()

    def _initialize_permissions(self):
        """Initialize API key permissions by subscription plan."""
        self._permissions = {
            SubscriptionPlan.FREE: APIKeyPermissions(
                rate_limit=10,
                max_requests=1000,
                allowed_endpoints=["/api/v1/reputation/score"],
                webhook_access=False,
                real_time_access=False,
                custom_reports=False,
            ),
            SubscriptionPlan.BASIC: APIKeyPermissions(
                rate_limit=30,
                max_requests=5000,
                allowed_endpoints=[
                    "/api/v1/reputation/score",
                    "/api/v1/reputation/metrics",
                    "/api/v1/reputation/analyze/comment",
                ],
                webhook_access=True,
                real_time_access=False,
                custom_reports=False,
            ),
            SubscriptionPlan.PROFESSIONAL: APIKeyPermissions(
                rate_limit=60,
                max_requests=20000,
                allowed_endpoints=[
                    "/api/v1/reputation/score",
                    "/api/v1/reputation/metrics",
                    "/api/v1/reputation/analyze/comment",
                    "/api/v1/reputation/report",
                    "/api/v1/customers/reputation",
                ],
                webhook_access=True,
                real_time_access=True,
                custom_reports=True,
            ),
            SubscriptionPlan.ENTERPRISE: APIKeyPermissions(
                rate_limit=120,
                max_requests=100000,
                allowed_endpoints=["*"],  # All endpoints
                webhook_access=True,
                real_time_access=True,
                custom_reports=True,
            ),
        }

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(32))

    async def create_api_key(self, key_data: APIKeyCreate) -> APIKeyResponse:
        """Create a new API key."""
        try:
            # Generate unique ID and key
            key_id = f"key_{datetime.now(timezone.utc).timestamp()}"
            api_key = self._generate_api_key()

            # Get permissions based on subscription plan
            permissions = self._permissions[key_data.subscription_plan]

            # Create API key
            key = APIKey(
                id=key_id,
                key=api_key,
                customer_id=key_data.customer_id,
                name=key_data.name,
                status=APIKeyStatus.ACTIVE,
                expires_at=key_data.expires_at,
                permissions=permissions,
            )

            # Store key
            self._keys[key_id] = key

            # Record metrics
            await metrics_manager.record_api_key_metric(
                metric_type="key_created",
                value=1,
                labels={
                    "plan": key_data.subscription_plan.value,
                    "status": key.status.value,
                },
            )

            # Send notification
            await self._notification_service.send_notification(
                type="api_key_created",
                title="New API Key Generated",
                message="API key '%s' has been generated for your account." % key.name,
                data={
                    "key_id": key_id,
                    "customer_id": key_data.customer_id,
                    "plan": key_data.subscription_plan.value,
                },
            )

            return APIKeyResponse(**key.dict())

        except Exception as e:
            logger.error("Error creating API key: %s", str(e))
            raise ReputationError(
                message="Failed to create API key",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SECURITY,
                details={"error": str(e)},
            )

    async def revoke_api_key(self, key_id: str) -> APIKeyResponse:
        """Revoke an API key."""
        try:
            key = self._keys.get(key_id)
            if not key:
                raise ReputationError(
                    message="API key not found: %s" % key_id,
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.SECURITY,
                )

            # Update key status
            key.status = APIKeyStatus.REVOKED
            key.is_active = False
            key.metadata["revoked_at"] = datetime.now(timezone.utc).isoformat()

            # Store updated key
            self._keys[key_id] = key

            # Record metrics
            await metrics_manager.record_api_key_metric(
                metric_type="key_revoked",
                value=1,
                labels={
                    "plan": key.permissions.rate_limit,
                    "status": key.status.value,
                },
            )

            # Send notification
            await self._notification_service.send_notification(
                type="api_key_revoked",
                title="API Key Revoked",
                message="API key '%s' has been revoked." % key.name,
                data={
                    "key_id": key_id,
                    "customer_id": key.customer_id,
                    "reason": "Manual revocation",
                },
            )

            return APIKeyResponse(**key.dict())

        except Exception as e:
            logger.error("Error revoking API key: %s", str(e))
            raise ReputationError(
                message="Failed to revoke API key",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SECURITY,
                details={"error": str(e)},
            )

    async def validate_api_key(
        self, api_key: str, endpoint: str, method: str
    ) -> APIKey:
        """Validate API key and check permissions."""
        try:
            # Find key
            key = next(
                (k for k in self._keys.values() if k.key == api_key), None
            )

            if not key:
                raise ReputationError(
                    message="Invalid API key",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY,
                )

            # Check if key is active
            if not key.is_active:
                raise ReputationError(
                    message="API key is not active",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY,
                )

            # Check if key has expired
            if key.expires_at and datetime.now(timezone.utc) > key.expires_at:
                key.status = APIKeyStatus.EXPIRED
                key.is_active = False
                self._keys[key.id] = key
                raise ReputationError(
                    message="API key has expired",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY,
                )

            # Check endpoint access
            if (
                "*" not in key.permissions.allowed_endpoints
                and endpoint not in key.permissions.allowed_endpoints
            ):
                raise ReputationError(
                    message="Endpoint not allowed for this API key",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY,
                )

            # Update usage
            key.last_used_at = datetime.now(timezone.utc)
            key.usage_count += 1
            self._keys[key.id] = key

            # Record usage
            usage = APIKeyUsage(
                key_id=key.id,
                endpoint=endpoint,
                method=method,
                response_time=0.0,  # Will be updated after request
                status_code=200,
            )
            self._usage_history.append(usage)

            return key

        except Exception as e:
            logger.error("Error validating API key: %s", str(e))
            raise ReputationError(
                message="Failed to validate API key",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SECURITY,
                details={"error": str(e)},
            )

    async def get_api_key_stats(self) -> APIKeyStats:
        """Get API key statistics."""
        try:
            # Calculate statistics
            total_keys = len(self._keys)
            active_keys = sum(1 for k in self._keys.values() if k.is_active)
            expired_keys = sum(
                1
                for k in self._keys.values()
                if k.status == APIKeyStatus.EXPIRED
            )
            revoked_keys = sum(
                1
                for k in self._keys.values()
                if k.status == APIKeyStatus.REVOKED
            )

            # Calculate usage by plan
            usage_by_plan = {}
            for key in self._keys.values():
                plan = next(
                    p
                    for p, perms in self._permissions.items()
                    if perms.rate_limit == key.permissions.rate_limit
                )
                usage_by_plan[plan.value] = (
                    usage_by_plan.get(plan.value, 0) + key.usage_count
                )

            # Calculate usage by status
            usage_by_status = {}
            for key in self._keys.values():
                usage_by_status[key.status.value] = (
                    usage_by_status.get(key.status.value, 0) + key.usage_count
                )

            # Get recent activity
            recent_activity = sorted(
                self._usage_history, key=lambda x: x.timestamp, reverse=True
            )[
                :100
            ]  # Last 100 activities

            return APIKeyStats(
                total_keys=total_keys,
                active_keys=active_keys,
                expired_keys=expired_keys,
                revoked_keys=revoked_keys,
                total_usage=sum(k.usage_count for k in self._keys.values()),
                usage_by_plan=usage_by_plan,
                usage_by_status=usage_by_status,
                recent_activity=recent_activity,
            )

        except Exception as e:
            logger.error("Error getting API key stats: %s", str(e))
            raise ReputationError(
                message="Failed to get API key statistics",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM,
                details={"error": str(e)},
            )

    async def cleanup_expired_keys(self):
        """Clean up expired API keys."""
        try:
            current_time = datetime.now(timezone.utc)
            expired_keys = [
                key
                for key in self._keys.values()
                if key.expires_at and key.expires_at < current_time
            ]

            for key in expired_keys:
                key.status = APIKeyStatus.EXPIRED
                key.is_active = False
                self._keys[key.id] = key

                # Send notification
                await self._notification_service.send_notification(
                    type="api_key_expired",
                    title="API Key Expired",
                    message="API key '%s' has expired." % key.name,
                    data={
                        "key_id": key.id,
                        "customer_id": key.customer_id,
                        "expired_at": key.expires_at.isoformat(),
                    },
                )

            return len(expired_keys)

        except Exception as e:
            logger.error("Error cleaning up expired keys: %s", str(e))
            raise ReputationError(
                message="Failed to clean up expired API keys",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM,
                details={"error": str(e)},
            )
