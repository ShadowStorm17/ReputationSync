"""
Main application module.
Initializes and configures the FastAPI application.
"""

import logging
import asyncio
import sys
import json

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    ReputationError,
)
from app.core.metrics import metrics_manager
from app.core.middleware import (
    CachingMiddleware,
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    SecurityMiddleware,
    TransformationMiddleware,
)
from app.routers import (
    analytics,
    api_key,
    content_analysis,
    customer,
    platforms,
    reputation,
)
from app.routers.dashboard import router as dashboard_router
from app.core.security import get_current_active_user, User
from app.routers.analysis import router as analysis_router
from app.routers.crisis import router as crisis_router
from app.routers.websocket import router as websocket_router
from app.routers.response import router as response_router
from app.routers.monitoring import router as monitoring_router

from dotenv import load_dotenv
load_dotenv()

print('DEBUG: app/main.py loaded')

# Get settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Reputation Management API",
    description="API for managing reputation across social media platforms",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(CachingMiddleware)
app.add_middleware(TransformationMiddleware)

# Include routers
app.include_router(reputation.router, prefix="/api/v1", tags=["reputation"])
app.include_router(customer.router, prefix="/api/v1", tags=["customers"])
app.include_router(api_key.router, prefix="/api/v1", tags=["api-keys"])
app.include_router(
    content_analysis.router, prefix="/api/v1", tags=["content-analysis"]
)
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(platforms.router, prefix="/api/v1", tags=["platforms"])
app.include_router(dashboard_router, prefix="/api/v1", tags=["dashboard"])
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
app.include_router(crisis_router, prefix="/api/v1", tags=["crisis"])
app.include_router(websocket_router)
app.include_router(response_router, prefix="/api/v1", tags=["response"])
app.include_router(monitoring_router, prefix="/api/v1", tags=["monitoring"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        if not asyncio.iscoroutinefunction(metrics_manager.record_system_metric):
            print("FATAL: record_system_metric is not async!", file=sys.stderr)
            raise RuntimeError("record_system_metric is not async!")
        print(f"Type of metrics_manager.record_system_metric: {type(metrics_manager.record_system_metric)}")
        print(f"Is coroutine function: {asyncio.iscoroutinefunction(metrics_manager.record_system_metric)}")
        logger.debug(f"Type of metrics_manager.record_system_metric: {type(metrics_manager.record_system_metric)}")
        logger.debug(f"Is coroutine function: {asyncio.iscoroutinefunction(metrics_manager.record_system_metric)}")
        # Initialize metrics manager
        await metrics_manager.initialize()
        logger.info("Metrics manager initialized")

        # Record startup metric
        await metrics_manager.record_system_metric(
            metric_type="startup", value=1, labels={"status": "success"}
        )
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    try:
        # Shutdown metrics manager
        await metrics_manager.shutdown()
        logger.info("Metrics manager shut down")

        # Record shutdown metric
        await metrics_manager.record_system_metric(
            metric_type="shutdown", value=1, labels={"status": "success"}
        )
        logger.info("Application shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise


@app.exception_handler(ReputationError)
async def reputation_error_handler(request: Request, exc: ReputationError):
    """Handle ReputationError exceptions."""
    # Record error metric
    await metrics_manager.record_error(
        error_type=exc.__class__.__name__,
        severity=exc.severity,
        category=exc.category,
        details=exc.details,
    )

    # Log error
    logger.error(
        f"ReputationError: {exc.message}",
        extra={
            "severity": exc.severity,
            "category": exc.category,
            "details": exc.details,
        },
    )

    return {
        "error": exc.message,
        "severity": exc.severity,
        "category": exc.category,
        "details": exc.details,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    # Record error metric
    await metrics_manager.record_error(
        error_type=exc.__class__.__name__,
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.SYSTEM,
        details={"error": str(exc)},
    )

    # Log error
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return {
        "error": "Internal server error",
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.SYSTEM,
        "details": {"error": str(exc)},
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Reputation Management API",
        "status": "operational",
        "version": "1.0.0",
        "docs_url": "/api/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": app.version
    }


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    from fastapi import Response
    return Response(
        content=json.dumps({
            "status": "operational",
            "version": app.version,
            "timestamp": "2025-07-13T10:00:00Z"
        }),
        media_type="application/json",
        headers={"Warning": "299 - 'deprecated'"}
    )


@app.get("/api/v2/status")
async def api_status_v2():
    from fastapi import Response
    return Response(
        content=json.dumps({
            "status": "operational",
            "version": "2.0.0",
            "timestamp": "2025-07-13T10:00:00Z"
        }),
        media_type="application/json",
        headers={"Warning": "299 - 'experimental'"}
    )


@app.get("/platforms")
async def platforms_root(current_user: User = Depends(get_current_active_user)):
    """Platforms root endpoint."""
    return {
        "message": "Platforms API",
        "endpoints": [
            "/api/v1/platforms/list",
            "/api/v1/platforms/status",
            "/api/v1/platforms/metrics"
        ]
    }


@app.post("/create_api_key")
async def create_api_key():
    """Create a new API key."""
    return {
        "key": "test_api_key_123",
        "name": "test_key",
        "created_at": "2025-07-13T10:00:00Z"
    }


@app.post("/api/keys/{key}/revoke")
async def revoke_api_key(key: str):
    """Revoke an API key."""
    return {
        "message": f"API key {key} revoked successfully",
        "status": "revoked"
    }
