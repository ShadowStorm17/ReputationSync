from fastapi import APIRouter, Depends, HTTPException, status
from app.core.auth import get_current_active_user
from typing import Dict

router = APIRouter(
    prefix="/crisis",
    tags=["crisis"],
)

@router.post("/check")
async def check_crisis(
    metrics: Dict,
    current_user=Depends(get_current_active_user)
):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Crisis check not implemented yet")