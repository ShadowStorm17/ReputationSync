from fastapi import APIRouter, Depends, HTTPException, status
from app.core.auth import get_current_active_user
from app.services.sentiment_service import SentimentService
from typing import Dict

router = APIRouter(
    prefix="/analyze",
    tags=["analysis"],
)

sentiment_service = SentimentService()

@router.post("/sentiment")
async def analyze_sentiment(
    payload: Dict,
    current_user=Depends(get_current_active_user)
):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    result = await sentiment_service.analyze_text(text)
    return result 