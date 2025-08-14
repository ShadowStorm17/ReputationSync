from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import deque

from pydantic import BaseModel, Field, HttpUrl, validator

from .base import TimestampedModel


class AlertSeverity(str, Enum):
    """Enum for alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Enum for alert types."""
    SENTIMENT_DROP = "sentiment_drop"
    ENGAGEMENT_SPIKE = "engagement_spike"
    MENTION_SPIKE = "mention_spike"
    CRISIS = "crisis"
    GROWTH_OPPORTUNITY = "growth_opportunity"


class ReputationMetrics(BaseModel):
    """Model for reputation metrics."""
    sentiment_score: float = Field(...,
                                   ge=-1.0,
                                   le=1.0,
                                   description="Overall sentiment score (-1 to 1)")
    engagement_rate: float = Field(..., ge=0.0,
                                   description="Engagement rate as percentage")
    follower_growth_rate: float = Field(...,
                                        description="Follower growth rate as percentage")
    mention_count: int = Field(..., ge=0,
                               description="Total number of mentions")
    positive_mentions: int = Field(..., ge=0,
                                   description="Number of positive mentions")
    negative_mentions: int = Field(..., ge=0,
                                   description="Number of negative mentions")
    neutral_mentions: int = Field(..., ge=0,
                                  description="Number of neutral mentions")
    reach_score: float = Field(..., ge=0.0, le=100.0,
                               description="Content reach score")
    influence_score: float = Field(...,
                                   ge=0.0,
                                   le=100.0,
                                   description="Influence score")
    response_rate: float = Field(...,
                                 ge=0.0,
                                 le=100.0,
                                 description="Response rate to mentions")
    average_response_time: float = Field(...,
                                         ge=0.0,
                                         description="Average response time in minutes")

    @validator('sentiment_score')
    def validate_sentiment_score(cls, v):
        """Validate sentiment score is within bounds."""
        if not -1.0 <= v <= 1.0:
            raise ValueError('Sentiment score must be between -1.0 and 1.0')
        return v

    @validator('engagement_rate', 'follower_growth_rate')
    def validate_percentage(cls, v):
        """Validate percentage values."""
        if v < 0:
            raise ValueError('Percentage values cannot be negative')
        return v


class ReputationAlert(TimestampedModel):
    """Model for reputation alerts."""
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    message: str = Field(..., min_length=10, description="Alert message")
    metrics: Dict[str,
                  float] = Field(...,
                                 description="Relevant metrics at time of alert")
    source_url: Optional[HttpUrl] = None
    resolved: bool = Field(
        False, description="Whether the alert has been resolved")
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    action_taken: Optional[str] = None
    impact_score: float = Field(..., ge=0.0, le=100.0,
                                description="Impact score of the alert")

    @validator('message')
    def validate_message_length(cls, v):
        """Validate message length."""
        if len(v) < 10:
            raise ValueError('Message must be at least 10 characters long')
        return v


class ReputationScore(TimestampedModel):
    """Model for overall reputation score."""
    platform: str = Field(..., description="Social media platform")
    username: str = Field(..., description="Username or handle")
    score: float = Field(..., ge=0.0, le=100.0,
                         description="Overall reputation score")
    metrics: ReputationMetrics = Field(..., description="Detailed metrics")
    alerts: List[ReputationAlert] = Field(
        default_factory=list, description="Active alerts")
    timeframe: str = Field(..., description="Analysis timeframe")
    trend_direction: str = Field(...,
                                 description="Score trend direction (up/down/stable)")
    trend_magnitude: float = Field(..., ge=0.0, description="Trend magnitude")
    confidence_score: float = Field(...,
                                    ge=0.0,
                                    le=100.0,
                                    description="Confidence in the score")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MonitoringConfig(BaseModel):
    """Model for reputation monitoring configuration."""
    platform: str = Field(..., description="Social media platform to monitor")
    username: str = Field(..., description="Username or handle to monitor")
    alert_thresholds: Dict[str,
                           float] = Field(...,
                                          description="Thresholds for different metrics")
    monitoring_interval: int = Field(
        300, ge=60, description="Monitoring interval in seconds")
    alert_channels: List[str] = Field(
        ["email"], description="Channels to send alerts")
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords to monitor")
    is_active: bool = Field(True, description="Whether monitoring is active")
    custom_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Custom metrics to track")
    notification_preferences: Dict[str, bool] = Field(
        default_factory=lambda: {
            "email": True,
            "sms": False,
            "webhook": False
        },
        description="Notification preferences"
    )
    auto_response_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rules for automatic responses"
    )

    @validator('monitoring_interval')
    def validate_interval(cls, v):
        """Validate monitoring interval."""
        if v < 60:
            raise ValueError('Monitoring interval must be at least 60 seconds')
        return v

    @validator('alert_channels')
    def validate_channels(cls, v):
        """Validate alert channels."""
        valid_channels = {"email", "sms", "webhook", "slack"}
        if not all(channel in valid_channels for channel in v):
            raise ValueError(
                f'Invalid alert channel. Must be one of: {valid_channels}')
        return v


class AlertConfig(BaseModel):
    name: str
    metric: str
    threshold: float
    condition: str  # e.g., 'above' or 'below'
    window: timedelta
    cooldown: timedelta


class AlertEvent(BaseModel):
    name: str
    timestamp: datetime
    severity: str
    message: str
    details: Optional[dict] = None


class MetricsWindow:
    def __init__(self, duration: timedelta):
        self.duration = duration
        self.values = deque()
        self.timestamps = deque()
