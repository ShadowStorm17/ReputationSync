from fastapi import APIRouter, Depends, HTTPException, status
from app.core.auth import get_current_active_user
from app.core.metrics import metrics_manager

router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
)

@router.get("/")
async def get_dashboard(current_user=Depends(get_current_active_user)):
    try:
        data = await metrics_manager.get_metrics_summary()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/refresh")
async def refresh_dashboard(current_user=Depends(get_current_active_user)):
    try:
        data = await metrics_manager.get_metrics_summary()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 