"""
Circuit breaker service for fault tolerance.
Provides circuit breaker pattern implementation.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CircuitState:
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service unavailable
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_timeout: int = 30,
        error_types: Optional[List[type]] = None,
    ):
        """Initialize circuit breaker."""
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_timeout = half_open_timeout
        self.error_types = error_types or [Exception]

        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )

    async def get_state(self) -> str:
        """Get current circuit state."""
        try:
            state_data = await self.redis.get(f"circuit:{self.name}")
            if not state_data:
                return CircuitState.CLOSED

            state = json.loads(state_data)

            # Check if recovery timeout has passed
            if state["state"] == CircuitState.OPEN:
                opened_at = datetime.fromisoformat(state["opened_at"])
                if datetime.now(timezone.utc) - opened_at > timedelta(
                    seconds=self.recovery_timeout
                ):
                    await self.transition_to_half_open()
                    return CircuitState.HALF_OPEN

            return state["state"]

        except Exception as e:
            logger.error(f"Get state error: {str(e)}")
            return CircuitState.CLOSED

    async def record_success(self):
        """Record successful operation."""
        try:
            state = await self.get_state()

            if state == CircuitState.HALF_OPEN:
                await self.close_circuit()

            # Reset failure count
            await self.redis.set(f"circuit:{self.name}:failures", 0)

        except Exception as e:
            logger.error(f"Record success error: {str(e)}")

    async def record_failure(self):
        """Record failed operation."""
        try:
            state = await self.get_state()

            if state == CircuitState.HALF_OPEN:
                await self.open_circuit()
                return

            # Increment failure count
            failures = await self.redis.incr(f"circuit:{self.name}:failures")

            if failures >= self.failure_threshold:
                await self.open_circuit()

        except Exception as e:
            logger.error(f"Record failure error: {str(e)}")

    async def open_circuit(self):
        """Open the circuit."""
        try:
            state_data = {
                "state": CircuitState.OPEN,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }

            await self.redis.set(
                f"circuit:{self.name}", json.dumps(state_data)
            )

        except Exception as e:
            logger.error(f"Open circuit error: {str(e)}")

    async def close_circuit(self):
        """Close the circuit."""
        try:
            state_data = {
                "state": CircuitState.CLOSED,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }

            await self.redis.set(
                f"circuit:{self.name}", json.dumps(state_data)
            )

            # Reset failure count
            await self.redis.set(f"circuit:{self.name}:failures", 0)

        except Exception as e:
            logger.error(f"Close circuit error: {str(e)}")

    async def transition_to_half_open(self):
        """Transition to half-open state."""
        try:
            state_data = {
                "state": CircuitState.HALF_OPEN,
                "transitioned_at": datetime.now(timezone.utc).isoformat(),
            }

            await self.redis.set(
                f"circuit:{self.name}", json.dumps(state_data)
            )

        except Exception as e:
            logger.error(f"Half-open transition error: {str(e)}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get circuit metrics."""
        try:
            state_data = await self.redis.get(f"circuit:{self.name}")
            failures = await self.redis.get(f"circuit:{self.name}:failures")

            metrics = {
                "name": self.name,
                "state": CircuitState.CLOSED,
                "failures": int(failures) if failures else 0,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }

            if state_data:
                state = json.loads(state_data)
                metrics.update(state)

            return {"status": "success", "metrics": metrics}

        except Exception as e:
            logger.error(f"Get metrics error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def reset(self):
        """Reset circuit state."""
        try:
            # Delete all circuit data
            keys = await self.redis.keys(f"circuit:{self.name}*")
            if keys:
                await self.redis.delete(*keys)

            return {"status": "success", "message": "Circuit reset"}

        except Exception as e:
            logger.error(f"Reset error: {str(e)}")
            return {"status": "error", "message": str(e)}


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    half_open_timeout: int = 30,
    error_types: Optional[List[type]] = None,
):
    """Circuit breaker decorator."""

    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(
            name,
            failure_threshold,
            recovery_timeout,
            half_open_timeout,
            error_types,
        )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            state = await breaker.get_state()

            if state == CircuitState.OPEN:
                raise Exception(f"Circuit {name} is OPEN")

            try:
                result = await func(*args, **kwargs)
                await breaker.record_success()
                return result

            except Exception as e:
                if any(
                    isinstance(e, error_type)
                    for error_type in breaker.error_types
                ):
                    await breaker.record_failure()
                raise

        return wrapper

    return decorator


class CircuitBreakerService:
    """Circuit breaker management service."""

    def __init__(self):
        """Initialize circuit breaker service."""
        self.breakers: Dict[str, CircuitBreaker] = {}

    def create_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_timeout: int = 30,
        error_types: Optional[List[type]] = None,
    ) -> CircuitBreaker:
        """Create new circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(
                name,
                failure_threshold,
                recovery_timeout,
                half_open_timeout,
                error_types,
            )
        return self.breakers[name]

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker."""
        return self.breakers.get(name)

    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers."""
        try:
            metrics = {}
            for name, breaker in self.breakers.items():
                result = await breaker.get_metrics()
                if result["status"] == "success":
                    metrics[name] = result["metrics"]

            return {"status": "success", "metrics": metrics}

        except Exception as e:
            logger.error(f"Get all metrics error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def reset_all(self) -> Dict[str, Any]:
        """Reset all circuit breakers."""
        try:
            results = {}
            for name, breaker in self.breakers.items():
                result = await breaker.reset()
                results[name] = result

            return {"status": "success", "results": results}

        except Exception as e:
            logger.error(f"Reset all error: {str(e)}")
            return {"status": "error", "message": str(e)}
