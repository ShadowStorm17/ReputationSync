from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import logging
from typing import Optional
import re

from app.core.config import get_settings
from app.core.security import get_api_key
from app.core.rate_limiter import rate_limit_middleware
from app.core.monitoring import setup_monitoring, MetricsMiddleware
from app.models.instagram import InstagramUserResponse
from app.services.instagram_service import InstagramService
from prometheus_client import make_asgi_app

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for retrieving Instagram user statistics",
    version=settings.API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS or ["*"],
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add monitoring middleware
app.add_middleware(MetricsMiddleware)

# Set up monitoring
setup_monitoring(app)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.API_VERSION,
        description="API for retrieving Instagram user statistics",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

def validate_username(username: str) -> str:
    """Validate Instagram username format."""
    if not re.match(r'^[A-Za-z0-9._]{1,30}$', username):
        raise HTTPException(
            status_code=400,
            detail="Invalid Instagram username format"
        )
    return username

@app.get(
    "/api/v1/platforms/instagram/users/{username}",
    response_model=InstagramUserResponse,
    responses={
        200: {"description": "Successfully retrieved user information"},
        400: {"description": "Invalid username format"},
        401: {"description": "Invalid API key"},
        404: {"description": "User not found"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Instagram API service unavailable"}
    }
)
async def get_instagram_user(
    username: str,
    api_key: str = Depends(get_api_key)
) -> InstagramUserResponse:
    """
    Retrieve Instagram user information.
    
    - **username**: The Instagram username to look up (alphanumeric, dots, and underscores only)
    """
    # Validate username format
    username = validate_username(username)
    
    try:
        instagram_service = InstagramService()
        user_info = await instagram_service.get_user_info(username)
        return InstagramUserResponse(**user_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request for username {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT
    }

@app.get("/api/v1/status")
async def api_status(api_key: str = Depends(get_api_key)):
    """
    API status endpoint with detailed health information.
    Requires authentication.
    """
    instagram_service = InstagramService()
    
    try:
        # Test Instagram API connection
        instagram_service._init_api()
        instagram_status = "connected"
    except Exception:
        instagram_status = "disconnected"
    
    return {
        "status": "operational",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "instagram_api": instagram_status,
        "rate_limiting": "enabled",
        "caching": "enabled" if settings.CACHE_ENABLED else "disabled"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
