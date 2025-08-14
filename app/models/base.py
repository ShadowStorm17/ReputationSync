from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TimestampedModel(BaseModel):
    """Base model with timestamps."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str
    status_code: int = 500
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None
    method: Optional[str] = None
