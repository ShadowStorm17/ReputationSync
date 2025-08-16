from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.core.config import get_settings

# Create FastAPI app instance
app = FastAPI()
logger = logging.getLogger(__name__)
settings = get_settings()

# CORS middleware (avoid wildcard in non-debug environments)
configured_origins = list(settings.CORS_ORIGINS or [])
filtered_origins = [o for o in configured_origins if o != "*"]

if not settings.DEBUG and (not filtered_origins):
    # In non-debug, do not allow wildcard. Expect explicit origins via CORS_ORIGINS.
    logger.warning("CORS is configured without explicit origins in non-debug environment. Set CORS_ORIGINS to allowed origins.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=filtered_origins if not settings.DEBUG else configured_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the application is working"""
    return {"status": "ok", "message": "Application is running"}


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    try:
        # Close Instagram API HTTP client if global instance exists
        from app.services.instagram_service import instagram_api  # type: ignore
        if instagram_api:
            await instagram_api.aclose()
            logger.info("Closed InstagramAPI HTTP client")

        # Close service instances created in routers
        try:
            from app.routers.platforms import (
                instagram_service as router_instagram_service,
                twitter_service as router_twitter_service,
                youtube_service as router_youtube_service,
            )

            if router_instagram_service:
                await router_instagram_service.aclose()
                logger.info("Closed router Instagram service client")
            if router_twitter_service:
                await router_twitter_service.aclose()
                logger.info("Closed router Twitter service client")
            if router_youtube_service:
                await router_youtube_service.aclose()
                logger.info("Closed router YouTube service client")
        except Exception as e:
            logger.warning("Error closing router service clients: %s", str(e))
    except Exception as e:
        # Log and continue shutdown
        logger.warning("Error during shutdown cleanup: %s", str(e))
