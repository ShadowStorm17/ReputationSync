"""
Customer profile models.
Defines customer profiles and subscription plans.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any

from pydantic import BaseModel, EmailStr, Field


class SubscriptionPlan(str, Enum):
    """Subscription plan types."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class PlatformAccess(str, Enum):
    """Platform access levels."""
    READ_ONLY = "read_only"
    BASIC = "basic"
    ADVANCED = "advanced"
    FULL = "full"


class SocialPlatform(str, Enum):
    """Supported social media platforms."""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    GITHUB = "github"


class PlatformProfile(BaseModel):
    """Social media platform profile."""
    platform: SocialPlatform
    username: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    is_active: bool = True
    access_level: PlatformAccess = PlatformAccess.BASIC


class SubscriptionFeatures(BaseModel):
    """Subscription plan features."""
    real_time_monitoring: bool = False
    one_time_analysis: bool = True
    platform_limit: int = 1
    analysis_frequency: str = "daily"  # daily, weekly, monthly
    reputation_fix: bool = False
    custom_reports: bool = False
    api_access: bool = False
    priority_support: bool = False


class Subscription(BaseModel):
    """Customer subscription details."""
    plan: SubscriptionPlan = SubscriptionPlan.FREE
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    is_active: bool = True
    features: SubscriptionFeatures = Field(
        default_factory=SubscriptionFeatures)
    auto_renew: bool = False
    payment_method: Optional[str] = None


class CustomerProfile(BaseModel):
    """Customer profile model."""
    id: str
    email: EmailStr
    name: str
    company: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    subscription: Subscription = Field(default_factory=Subscription)
    platform_profiles: Dict[SocialPlatform,
                            PlatformProfile] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CustomerProfileCreate(BaseModel):
    """Customer profile creation model."""
    email: EmailStr
    name: str
    company: Optional[str] = None
    subscription_plan: SubscriptionPlan = SubscriptionPlan.FREE
    platform_profiles: Optional[Dict[SocialPlatform, PlatformProfile]] = None


class CustomerProfileUpdate(BaseModel):
    """Customer profile update model."""
    name: Optional[str] = None
    company: Optional[str] = None
    subscription_plan: Optional[SubscriptionPlan] = None
    platform_profiles: Optional[Dict[SocialPlatform, PlatformProfile]] = None
    preferences: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class CustomerProfileResponse(BaseModel):
    """Customer profile response model."""
    id: str
    email: EmailStr
    name: str
    company: Optional[str] = None
    subscription: Subscription
    platform_profiles: Dict[SocialPlatform, PlatformProfile]
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
