from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class InstagramUserBase(BaseModel):
    """Base model for Instagram user data."""
    username: str
    full_name: Optional[str] = None
    biography: Optional[str] = None
    website: Optional[HttpUrl] = None
    is_private: bool = False
    is_verified: bool = False
    media_count: Optional[int] = None
    follower_count: Optional[int] = None
    following_count: Optional[int] = None


class InstagramUserResponse(InstagramUserBase):
    """Response model for Instagram user data."""
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    engagement_rate: Optional[float] = None
    sentiment_score: Optional[float] = None
    error: Optional[str] = None


class BulkUserRequest(BaseModel):
    """Request model for bulk user lookup."""
    usernames: List[str] = Field(..., max_items=100)


class InstagramPost(BaseModel):
    """Model for Instagram post data."""
    id: str
    caption: Optional[str] = None
    media_type: str
    media_url: Optional[HttpUrl] = None
    permalink: HttpUrl
    timestamp: datetime
    like_count: Optional[int] = None
    comments_count: Optional[int] = None
    engagement_rate: Optional[float] = None
    sentiment_score: Optional[float] = None


class InstagramComment(BaseModel):
    """Model for Instagram comment data."""
    id: str
    text: str
    username: str
    timestamp: datetime
    like_count: Optional[int] = None
    sentiment_score: Optional[float] = None
    is_reply: bool = False
    parent_comment_id: Optional[str] = None


class InstagramMetrics(BaseModel):
    """Model for Instagram metrics data."""
    follower_growth: Dict[str, int]  # Date -> Count
    engagement_rates: Dict[str, float]  # Date -> Rate
    sentiment_scores: Dict[str, float]  # Date -> Score
    post_frequency: Dict[str, int]  # Date -> Count
    best_posting_times: List[Dict[str, float]]  # Hour -> Engagement Rate
