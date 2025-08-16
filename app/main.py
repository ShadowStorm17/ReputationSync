"""
Main application module.
Initializes and configures the FastAPI application.
"""

# app/main.py

import logging
from datetime import datetime, timezone
import asyncio

from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings

from app.core.config import Settings, get_settings
from app.core.middleware.auth import AuthMiddleware
from app.core.middleware.rate_limit import RateLimitMiddleware
from app.core.middleware.headers import HeadersMiddleware
from app.core.middleware.request_logging import RequestLoggingMiddleware
from app.core.middleware.error_handling import ErrorHandlingMiddleware
from app.core.middleware.caching import CachingMiddleware
from app.core.middleware.transformation import TransformationMiddleware
from app.core.metrics import metrics_manager
from app.core.lifecycle import on_startup, on_shutdown
from app.core.errors import ReputationError, ErrorResponseModel
from app.core.security import get_current_admin_user, User

from app.routers import (
    reputation,
    customer,
    api_key,
    content_analysis,
    analytics,
    platforms,
    dashboard,
    analysis,
    crisis,
    websocket,
    response,
    monitoring,
)

# Load settings via Pydantic BaseSettings
settings: Settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("reputation_sync")

app = FastAPI(
    title="Reputation Management API",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url=None,
    openapi_url="/api/v1/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware order: Logging → Auth → Rate Limit → Headers → Caching → Transform → Error Handling
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(HeadersMiddleware)
app.add_middleware(CachingMiddleware)
app.add_middleware(TransformationMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Lifecycle events
app.add_event_handler("startup", on_startup)
app.add_event_handler("shutdown", on_shutdown)

# Exception handlers
@app.exception_handler(ReputationError)
async def handle_reputation_error(request: Request, exc: ReputationError):
    # Internal logging and metrics
    await metrics_manager.record_error(exc)
    logger.warning(f"{exc.category} - {exc.message}")
    return ErrorResponseModel(
        code=exc.code,
        message=exc.message,
        details=exc.details,
    ).dict(), exc.status_code

@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    await metrics_manager.record_error(exc, severity="HIGH", category="SYSTEM")
    logger.error("Unexpected error", exc_info=exc)
    # Return generic message
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
    )

# Health checks (public)
@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"status": "ok", "version": settings.APP_VERSION}

@app.get("/health", tags=["health"])
def health() -> dict:
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# API v1 router group
api_v1 = FastAPI(
    openapi_prefix="/api/v1",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

api_v1.include_router(reputation.router, prefix="/reputation", tags=["Reputation"])
api_v1.include_router(customer.router, prefix="/customers", tags=["Customers"])
api_v1.include_router(content_analysis.router, prefix="/content-analysis", tags=["Content Analysis"])
api_v1.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_v1.include_router(platforms.router, prefix="/platforms", tags=["Platforms"])
api_v1.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
api_v1.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_v1.include_router(crisis.router, prefix="/crisis", tags=["Crisis"])
api_v1.include_router(response.router, prefix="/response", tags=["Response"])
api_v1.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])

# Secure API-key management
@api_v1.post("/api-keys", status_code=status.HTTP_201_CREATED, tags=["API Keys"])
async def create_api_key(current_user: User = Depends(get_current_admin_user)):
    key = await api_key.create_key(owner=current_user.id)
    return {"key": key.plaintext, "created_at": key.created_at}

@api_v1.delete("/api-keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(key_id: str, current_user: User = Depends(get_current_admin_user)):
    await api_key.revoke_key(key_id, revoked_by=current_user.id)
    return {"message": "API key revoked successfully", "key_id": key_id}

app.mount("/api/v1", api_v1)
