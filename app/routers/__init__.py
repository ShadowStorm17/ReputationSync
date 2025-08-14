"""
Routers package.
Contains all API routers.
"""

from app.routers.auth import router as auth_router
from app.routers.monitoring import router as monitoring_router
from app.routers.platforms import router as platforms_router
from app.routers.reputation import router as reputation_router
from app.routers.dashboard import router as dashboard_router
from app.routers.analysis import router as analysis_router
from app.routers.crisis import router as crisis_router
from app.routers.websocket import router as websocket_router
from app.routers.response import router as response_router

__all__ = [
    "auth_router",
    "reputation_router",
    "platforms_router",
    "monitoring_router",
    "dashboard_router",
    "analysis_router",
    "crisis_router",
    "websocket_router",
    "response_router",
]
