from fastapi import HTTPException, Request
from ratelimit import RateLimitException, limits, RateLimitDecorator
from app.core.config import get_settings
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

def create_rate_limiter(calls: int = None, period: int = None) -> RateLimitDecorator:
    """Create a rate limiter with configurable calls and period."""
    return limits(
        calls=calls or settings.RATE_LIMIT_CALLS,
        period=period or settings.RATE_LIMIT_PERIOD
    )

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to apply rate limiting based on client IP."""
    client_ip = request.client.host
    
    @create_rate_limiter()
    async def _rate_limited():
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error in rate limited request: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    try:
        return await _rate_limited()
    except RateLimitException:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        ) 