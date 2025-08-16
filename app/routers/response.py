from fastapi import APIRouter, Depends, HTTPException, status, Body
from app.core.auth import get_current_active_user
from typing import Dict

router = APIRouter(
    prefix="/response",
    tags=["response"],
)

@router.post("/generate")
async def generate_response(
    payload: Dict = Body(...),
    current_user=Depends(get_current_active_user)
):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Response generation not implemented yet")