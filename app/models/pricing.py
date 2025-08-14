from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class PlanType(str, Enum):
    """Enum for subscription plan types."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class FeatureType(str, Enum):
    """Enum for feature types."""
    API_CALLS = "api_calls"
    MONITORING = "monitoring"
    ALERTS = "alerts"
    REPORTS = "reports"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    CUSTOM_METRICS = "custom_metrics"
    WHITE_LABEL = "white_label"
    DEDICATED_SUPPORT = "dedicated_support"


class PlanFeature(BaseModel):
    """Model for plan features."""
    feature_type: FeatureType
    limit: Optional[int] = None
    enabled: bool = True
    additional_cost: Optional[float] = None


class SubscriptionPlan(BaseModel):
    """Model for subscription plans."""
    plan_type: PlanType
    name: str
    description: str
    monthly_price: float
    annual_price: float
    features: List[PlanFeature]
    max_users: int
    max_platforms: int
    retention_period: int  # in days
    support_level: str
    custom_domain: bool = False
    ssl_enabled: bool = True
    api_rate_limit: int  # requests per hour
    webhook_limit: Optional[int] = None
    export_formats: List[str] = ["json"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UsageMetrics(BaseModel):
    """Model for tracking API usage."""
    plan_type: PlanType
    api_calls_used: int
    api_calls_limit: int
    monitoring_used: int
    monitoring_limit: int
    alerts_used: int
    alerts_limit: int
    reports_generated: int
    reports_limit: int
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class BillingInfo(BaseModel):
    """Model for billing information."""
    customer_id: str
    plan_type: PlanType
    billing_cycle: str  # monthly/annual
    next_billing_date: datetime
    payment_method: str
    billing_address: Dict[str, str]
    tax_id: Optional[str] = None
    vat_number: Optional[str] = None
    currency: str = "USD"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Invoice(BaseModel):
    """Model for invoices."""
    invoice_id: str
    customer_id: str
    amount: float
    currency: str
    status: str
    items: List[Dict[str, Any]]
    due_date: datetime
    paid_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
