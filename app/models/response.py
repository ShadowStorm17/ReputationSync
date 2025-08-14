from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .base import TimestampedModel


class ResponseTemplate(TimestampedModel):
    """Model for response templates."""
    name: str
    content: str
    category: str
    sentiment: str = Field(...,
                           description="Intended sentiment (positive, neutral, negative)")
    variables: List[str] = []
    is_active: bool = True


class ResponseRule(BaseModel):
    """Model for response rules."""
    trigger_type: str = Field(...,
                              description="Type of trigger (keyword, sentiment, etc.)")
    trigger_value: str
    template_id: int
    priority: int = Field(1, ge=1, le=5)
    conditions: Dict[str, str] = {}


class AutoResponse(TimestampedModel):
    """Model for automated responses."""
    platform: str
    comment_id: str
    user_id: str
    original_text: str
    response_text: str
    template_id: Optional[int] = None
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    status: str = "pending"
    response_time: Optional[float] = None  # in seconds


class ResponseMetrics(BaseModel):
    """Model for response performance metrics."""
    total_responses: int = 0
    average_response_time: float = 0.0
    sentiment_distribution: Dict[str, int] = {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    }
    template_usage: Dict[int, int] = {}
    success_rate: float = 0.0
