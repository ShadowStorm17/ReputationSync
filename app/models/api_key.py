"""
API key models.
Defines API key management and tracking.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.customer import SubscriptionPlan


class APIKeyStatus(str, Enum):
    """API key status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class APIKeyPermissions(BaseModel):
    """API key permissions based on subscription plan."""
    rate_limit: int  # Requests per minute
    max_requests: int  # Total requests allowed
    allowed_endpoints: List[str]
    webhook_access: bool
    real_time_access: bool
    custom_reports: bool


class APIKey(BaseModel):
    """API key model."""
    id: str
    key: str
    customer_id: str
    name: str
    status: APIKeyStatus = APIKeyStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    permissions: APIKeyPermissions
    usage_count: int = 0
    is_active: bool = True
    metadata: dict = Field(default_factory=dict)

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyCreate(BaseModel):
    """API key creation model."""
    customer_id: str
    name: str
    subscription_plan: SubscriptionPlan
    expires_at: Optional[datetime] = None


class APIKeyResponse(BaseModel):
    """API key response model."""
    id: str
    key: str
    name: str
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    permissions: APIKeyPermissions
    usage_count: int
    is_active: bool

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyUsage(BaseModel):
    """API key usage tracking."""
    key_id: str
    endpoint: str
    method: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_time: float
    status_code: int
    error: Optional[str] = None


class APIKeyStats(BaseModel):
    """API key statistics."""
    total_keys: int
    active_keys: int
    expired_keys: int
    revoked_keys: int
    total_usage: int
    usage_by_plan: dict
    usage_by_status: dict
    recent_activity: List[APIKeyUsage]
