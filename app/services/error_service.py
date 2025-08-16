"""
Error handling service for comprehensive error management.
Provides error tracking, logging, and recovery capabilities.
"""

import json
import logging
import traceback
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ErrorTracker:
    """Error tracking system."""

    def __init__(self):
        """Initialize error tracker."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        self.retention_period = 86400  # 24 hours

    async def track_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track error occurrence."""
        try:
            error_id = f"error:{datetime.now(timezone.utc).isoformat()}"
            error_data = {
                'type': error.__class__.__name__,
                'message': str(error),
                'traceback': traceback.format_exc(),
                'context': context,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            # Store error
            await self.redis.set(
                error_id,
                json.dumps(error_data),
                ex=self.retention_period
            )

            # Update error stats
            await self._update_error_stats(error.__class__.__name__)

            return {
                'status': 'success',
                'error_id': error_id,
                'error_data': error_data
            }

        except Exception as e:
            logger.error(f"Error tracking error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def get_error(
        self,
        error_id: str
    ) -> Dict[str, Any]:
        """Get error details."""
        try:
            error_data = await self.redis.get(error_id)
            if not error_data:
                return {
                    'status': 'error',
                    'message': 'Error not found'
                }

            return {
                'status': 'success',
                'error': json.loads(error_data)
            }

        except Exception as e:
            logger.error(f"Error getting error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def _update_error_stats(
        self,
        error_type: str
    ):
        """Update error statistics."""
        try:
            # Update error count
            await self.redis.hincrby('error_stats', error_type, 1)

            # Update error timeline
            timeline_key = f"error_timeline:{
                datetime.now(timezone.utc).strftime('%Y%m%d')}"
            await self.redis.hincrby(timeline_key, error_type, 1)
            await self.redis.expire(timeline_key, self.retention_period)

        except Exception as e:
            logger.error(f"Error updating error stats: {str(e)}")


class ErrorRecovery:
    """Error recovery system."""

    def __init__(self):
        """Initialize error recovery."""
        self.recovery_strategies = {}

    def register_strategy(
        self,
        error_type: Type[Exception],
        strategy: Callable
    ):
        """Register recovery strategy for error type."""
        self.recovery_strategies[error_type] = strategy

    async def attempt_recovery(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to recover from error."""
        try:
            error_type = type(error)

            # Check for registered strategy
            if error_type in self.recovery_strategies:
                strategy = self.recovery_strategies[error_type]
                result = await strategy(error, context)

                return {
                    'status': 'success',
                    'recovered': True,
                    'result': result
                }

            return {
                'status': 'success',
                'recovered': False,
                'message': 'No recovery strategy found'
            }

        except Exception as e:
            logger.error(f"Error attempting recovery: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }


def handle_errors(
    error_types: Optional[List[Type[Exception]]] = None,
    track: bool = True,
    recover: bool = True
):
    """Error handling decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            error_service = ErrorService()

            try:
                return await func(*args, **kwargs)

            except Exception as e:
                if error_types and not isinstance(e, tuple(error_types)):
                    raise

                context = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }

                # Track error
                if track:
                    await error_service.track_error(e, context)

                # Attempt recovery
                if recover:
                    recovery = await error_service.attempt_recovery(
                        e,
                        context
                    )
                    if recovery['recovered']:
                        return recovery['result']

                raise

        return wrapper
    return decorator


class ErrorService:
    """Comprehensive error handling service."""

    def __init__(self):
        """Initialize error service."""
        self.tracker = ErrorTracker()
        self.recovery = ErrorRecovery()

    async def track_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track error occurrence."""
        return await self.tracker.track_error(error, context)

    async def get_error(
        self,
        error_id: str
    ) -> Dict[str, Any]:
        """Get error details."""
        return await self.tracker.get_error(error_id)

    async def attempt_recovery(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to recover from error."""
        return await self.recovery.attempt_recovery(error, context)

    def register_recovery_strategy(
        self,
        error_type: Type[Exception],
        strategy: Callable
    ):
        """Register error recovery strategy."""
        self.recovery.register_strategy(error_type, strategy)

    async def get_error_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get error statistics."""
        try:
            if not start_date:
                start_date = datetime.now(timezone.utc)
            if not end_date:
                end_date = datetime.now(timezone.utc)

            # Get overall stats
            overall_stats = await self.tracker.redis.hgetall('error_stats')

            # Get timeline stats
            timeline_stats = {}
            current_date = start_date
            while current_date <= end_date:
                date_key = current_date.strftime('%Y%m%d')
                timeline_key = f"error_timeline:{date_key}"

                day_stats = await self.tracker.redis.hgetall(timeline_key)
                if day_stats:
                    timeline_stats[date_key] = day_stats

                current_date = current_date + timedelta(days=1)

            return {
                'status': 'success',
                'overall_stats': overall_stats,
                'timeline_stats': timeline_stats
            }

        except Exception as e:
            logger.error(f"Error getting error stats: {str(e)}")
            return {'status': 'error', 'message': str(e)}
