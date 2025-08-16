"""
Advanced rate limiting system.
Provides intelligent rate limiting with adaptive thresholds and burst handling.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import Counter, Gauge
from app.core.metrics import REQUEST_COUNT

from app.core.config import get_settings
from app.core.error_handling import ErrorCategory, ErrorSeverity, handle_errors

from fastapi import Request, HTTPException, status
from functools import wraps
import inspect

# Rate limiting metrics
RATE_LIMIT_EXCEEDED = Counter(
    "rate_limit_exceeded_total",
    "Rate limit exceeded count",
    ["endpoint", "client"],
)
BURST_ALLOWED = Counter(
    "burst_allowed_total", "Burst requests allowed", ["endpoint", "client"]
)
ADAPTIVE_LIMIT = Gauge(
    "adaptive_rate_limit", "Adaptive rate limit", ["endpoint", "client"]
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RateLimitState:
    """Rate limit state tracking."""

    requests: List[datetime] = field(default_factory=list)
    burst_tokens: int = field(
        default_factory=lambda: settings.rate_limit.DEFAULT_LIMIT
    )
    last_refill: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    adaptive_limit: int = field(
        default_factory=lambda: settings.rate_limit.DEFAULT_LIMIT
    )
    error_count: int = 0
    total_requests: int = 0


class RateLimiter:
    """Advanced rate limiting system."""

    def __init__(self):
        """Initialize rate limiter."""
        self._lock = threading.Lock()
        self._states: Dict[str, Dict[str, RateLimitState]] = {}
        self._background_tasks = []

    async def initialize(self):
        """Initialize rate limiter and start background tasks."""
        self._background_tasks = [
            asyncio.create_task(self._cleanup_old_requests()),
            asyncio.create_task(self._refill_burst_tokens()),
            asyncio.create_task(self._adjust_adaptive_limits()),
        ]

    async def shutdown(self):
        """Shutdown rate limiter and cleanup tasks."""
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()

    async def check_rate_limit(
        self, client_id: str, endpoint: str = "*", cost: int = 1
    ) -> Tuple[bool, Optional[int]]:
        """Check if a request is allowed."""
        try:
            # Get or create state
            state = self._get_state(client_id, endpoint)

            current_time = datetime.now(timezone.utc)
            window_start = current_time - timedelta(
                seconds=settings.rate_limit.DEFAULT_WINDOW
            )

            with self._lock:
                # Clean old requests
                state.requests = [
                    req for req in state.requests if req > window_start
                ]

                # Check regular rate limit
                if len(state.requests) >= state.adaptive_limit:
                    # Try burst
                    if (
                        settings.rate_limit.BURST_MULTIPLIER > 1.0
                        and state.burst_tokens >= cost
                    ):
                        # Allow burst
                        state.burst_tokens -= cost
                        state.requests.append(current_time)
                        state.total_requests += 1

                        BURST_ALLOWED.labels(
                            endpoint=endpoint, client=client_id
                        ).inc()

                        return True, None

                    # Rate limit exceeded
                    RATE_LIMIT_EXCEEDED.labels(
                        endpoint=endpoint, client=client_id
                    ).inc()

                    # Calculate reset time
                    oldest_request = state.requests[0]
                    reset_time = int(
                        (
                            oldest_request
                            + timedelta(
                                seconds=settings.rate_limit.DEFAULT_WINDOW
                            )
                            - current_time
                        ).total_seconds()
                    )

                    return False, reset_time

                # Allow request
                state.requests.append(current_time)
                state.total_requests += 1

                REQUEST_COUNT.labels(endpoint=endpoint, client=client_id).inc()

                return True, None

        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            # Fail open to prevent blocking legitimate traffic
            return True, None

    def _get_state(self, client_id: str, endpoint: str) -> RateLimitState:
        """Get or create rate limit state."""
        with self._lock:
            if endpoint not in self._states:
                self._states[endpoint] = {}

            if client_id not in self._states[endpoint]:
                self._states[endpoint][client_id] = RateLimitState()

            return self._states[endpoint][client_id]

    async def _cleanup_old_requests(self):
        """Clean up old requests periodically."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                window_start = current_time - timedelta(
                    seconds=settings.rate_limit.DEFAULT_WINDOW
                )

                with self._lock:
                    for endpoint in self._states:
                        for client_id in list(self._states[endpoint].keys()):
                            state = self._states[endpoint][client_id]

                            # Clean old requests
                            state.requests = [
                                req
                                for req in state.requests
                                if req > window_start
                            ]

                            # Remove empty states
                            if not state.requests:
                                del self._states[endpoint][client_id]

                        # Remove empty endpoints
                        if not self._states[endpoint]:
                            del self._states[endpoint]

                await asyncio.sleep(60)  # Clean up every minute

            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                await asyncio.sleep(60)

    async def _refill_burst_tokens(self):
        """Refill burst tokens periodically."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)

                with self._lock:
                    for endpoint in self._states.values():
                        for state in endpoint.values():
                            # Calculate tokens to add
                            time_passed = (
                                current_time - state.last_refill
                            ).total_seconds()
                            # Refill over 1 hour
                            tokens_to_add = int(
                                time_passed
                                * (settings.rate_limit.DEFAULT_LIMIT / 3600)
                            )

                            # Add tokens up to limit
                            state.burst_tokens = min(
                                settings.rate_limit.DEFAULT_LIMIT,
                                state.burst_tokens + tokens_to_add,
                            )
                            state.last_refill = current_time

                await asyncio.sleep(60)  # Refill every minute

            except Exception as e:
                logger.error(f"Burst token refill error: {str(e)}")
                await asyncio.sleep(60)

    async def _adjust_adaptive_limits(self):
        """Adjust rate limits based on usage patterns."""
        while True:
            try:
                with self._lock:
                    for endpoint in self._states.values():
                        for client_id, state in endpoint.items():
                            # Calculate error rate
                            error_rate = state.error_count / max(
                                1, state.total_requests
                            )

                            # Adjust limit based on error rate
                            if (
                                error_rate
                                > settings.rate_limit.ERROR_THRESHOLD
                            ):
                                # Reduce limit
                                state.adaptive_limit = max(
                                    settings.rate_limit.MIN_LIMIT,
                                    int(state.adaptive_limit * 0.8),
                                )
                            elif (
                                error_rate
                                < settings.rate_limit.ERROR_THRESHOLD / 2
                            ):
                                # Increase limit
                                state.adaptive_limit = min(
                                    settings.rate_limit.MAX_LIMIT,
                                    int(state.adaptive_limit * 1.2),
                                )

                            # Update metric
                            ADAPTIVE_LIMIT.labels(
                                endpoint="*", client=client_id
                            ).set(state.adaptive_limit)

                            # Reset counters
                            state.error_count = 0
                            state.total_requests = 0

                await asyncio.sleep(300)  # Adjust every 5 minutes

            except Exception as e:
                logger.error(f"Adaptive limit adjustment error: {str(e)}")
                await asyncio.sleep(60)

    @handle_errors(ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
    async def record_error(self, client_id: str, endpoint: str = "*"):
        """Record an error for adaptive rate limiting."""
        try:
            state = self._get_state(client_id, endpoint)
            with self._lock:
                state.error_count += 1
        except Exception as e:
            logger.error(f"Error recording rate limit error: {str(e)}")

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def get_limit_status(
        self, client_id: str, endpoint: str = "*"
    ) -> Dict[str, Any]:
        """Get current rate limit status."""
        try:
            state = self._get_state(client_id, endpoint)
            current_time = datetime.now(timezone.utc)
            window_start = current_time - timedelta(
                seconds=settings.rate_limit.DEFAULT_WINDOW
            )

            with self._lock:
                return {
                    "limit": state.adaptive_limit,
                    "remaining": max(
                        0, state.adaptive_limit - len(state.requests)
                    ),
                    "burst_tokens": state.burst_tokens,
                    "window_size": settings.rate_limit.DEFAULT_WINDOW,
                    "requests_in_window": len(
                        [r for r in state.requests if r > window_start]
                    ),
                }
        except Exception as e:
            logger.error(f"Error getting rate limit status: {str(e)}")
            return {}


# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(limit: int = 100, period: int = 60):
    """
    Decorator to apply rate limiting to FastAPI endpoints.
    - limit: max requests per period
    - period: period in seconds
    """
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Find request object
                request: Request = kwargs.get("request")
                if not request:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break
                if not request:
                    raise RuntimeError("Request object not found for rate_limit")

                # Identify client (user id, API key, or IP)
                client_id = None
                user = kwargs.get("current_user") or getattr(request.state, "user", None)
                if user and hasattr(user, "id"):
                    client_id = str(user.id)
                elif "x-api-key" in request.headers:
                    client_id = request.headers["x-api-key"]
                else:
                    client_id = request.client.host if request.client else "anonymous"

                endpoint = request.url.path

                allowed, _ = await rate_limiter.check_rate_limit(client_id, endpoint)
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded: {limit} requests per {period} seconds."
                    )
                return await func(*args, **kwargs)
            return wrapper
        else:
            raise NotImplementedError("rate_limit only supports async endpoints.")
    return decorator
