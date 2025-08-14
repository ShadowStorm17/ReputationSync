"""
Monitoring router.
Handles monitoring and analytics endpoints.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.metrics import track_performance
from app.core.security import User, get_current_admin_user
from app.core.auth import get_current_active_user
from app.services.analytics_engine import AnalyticsEngine
from app.services.enhanced_monitoring import EnhancedMonitoring
from app.services.monitoring_orchestrator import MonitoringOrchestrator
from app.services.monitoring_service import MonitoringService

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    responses={404: {"description": "Not found"}},
)

# Initialize services
monitoring_service = MonitoringService()
enhanced_monitoring = EnhancedMonitoring()
monitoring_orchestrator = MonitoringOrchestrator()
analytics_engine = AnalyticsEngine()


@router.get("/status")
@track_performance
async def get_monitoring_status(
    current_user: User = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Get monitoring system status."""
    try:
        return await monitoring_service.get_system_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/metrics")
@track_performance
async def get_monitoring_metrics(
    current_user: User = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Get monitoring metrics."""
    try:
        return await monitoring_service.get_metrics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/alerts")
@track_performance
async def get_active_alerts(
    current_user: User = Depends(get_current_admin_user),
) -> List[Dict[str, Any]]:
    """Get active alerts."""
    try:
        return await monitoring_service.get_active_alerts()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/analytics/summary")
@track_performance
async def get_analytics_summary(
    current_user: User = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Get analytics summary."""
    try:
        return await analytics_engine.get_system_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/analytics/trends")
@track_performance
async def get_analytics_trends(
    metric: str,
    timeframe: str,
    current_user: User = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Get analytics trends."""
    try:
        return await analytics_engine.get_metric_trends(metric, timeframe)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/alerts/configure")
@track_performance
async def configure_alerts(
    config: Dict[str, Any],
    current_user: User = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Configure monitoring alerts."""
    try:
        return await monitoring_service.configure_alerts(config)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/start")
async def start_monitoring(
    config: Dict[str, Any],
    current_user=Depends(get_current_active_user)
):
    return {"message": "Monitoring started.", "config": config}

@router.post("/stop")
async def stop_monitoring(
    current_user=Depends(get_current_active_user)
):
    return {"message": "Monitoring stopped."}


@router.get("/health/check")
@track_performance
async def run_health_check(
    component: str, current_user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Run health check on a specific component."""
    try:
        return await monitoring_orchestrator.run_health_check(component)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
