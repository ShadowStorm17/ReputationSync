"""
Error handling module.
Defines error types and handling for the application.
"""

import logging
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

from fastapi import HTTPException, status

from app.core.metrics import metrics_manager
from app.core.errors import ErrorCategory
from app.core.error_utils import ErrorSeverity, handle_errors

logger = logging.getLogger(__name__)


class ReputationError(Exception):
    """Base error class for reputation monitoring."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.BUSINESS,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize error."""
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "message": self.message,
            "severity": self.severity,
            "category": self.category,
            "details": self.details,
            "context": self.context,
        }

    def __str__(self) -> str:
        """Get string representation."""
        return f"{self.category} error ({self.severity}): {self.message}"


class ValidationError(ReputationError):
    """Error for validation failures."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize validation error."""
        super().__init__(
            message=message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            details=details,
        )


class IntegrationError(ReputationError):
    """Error for integration failures."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize integration error."""
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INTEGRATION,
            details=details,
        )


class SecurityError(ReputationError):
    """Error for security violations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize security error."""
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            details=details,
        )


class SystemError(ReputationError):
    """Error for system failures."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize system error."""
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            details=details,
        )


class BusinessError(ReputationError):
    """Error for business rule violations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize business error."""
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS,
            details=details,
        )


def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.BUSINESS,
    reraise: bool = True,
) -> Callable:
    """Error handling decorator."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except ReputationError as e:
                # Log the error
                logger.error(
                    "Reputation error: %s",
                    e,
                    extra={
                        "severity": e.severity,
                        "category": e.category,
                        "details": e.details,
                    },
                )

                # Record metrics
                await metrics_manager.record_error(
                    severity=e.severity, category=e.category, details=e.details
                )

                # Convert to HTTP exception
                status_code = _get_status_code(e)
                raise HTTPException(
                    status_code=status_code,
                    detail={
                        "message": str(e),
                        "category": e.category,
                        "details": e.details,
                    },
                )
            except Exception as e:
                # Log unexpected errors
                logger.error(
                    "Unexpected error: %s",
                    e,
                    exc_info=True,
                    extra={"severity": severity, "category": category},
                )

                # Record metrics
                await metrics_manager.record_error(
                    severity=severity,
                    category=category,
                    details={"error": str(e)},
                )

                if reraise:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Internal server error",
                    )
                return None

        return wrapper

    return decorator


def _get_status_code(error: ReputationError) -> int:
    """Get HTTP status code for error."""
    if isinstance(error, ValidationError):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(error, SecurityError):
        return status.HTTP_403_FORBIDDEN
    elif isinstance(error, IntegrationError):
        return status.HTTP_502_BAD_GATEWAY
    else:
        return status.HTTP_500_INTERNAL_SERVER_ERROR


def handle_validation_error(
    func: Callable, error_type: Type[Exception] = ValueError
) -> Callable:
    """Validation error handling decorator."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except error_type as e:
            raise ValidationError(str(e))

    return wrapper


def handle_integration_error(
    func: Callable, error_type: Type[Exception] = Exception
) -> Callable:
    """Integration error handling decorator."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except error_type as e:
            raise IntegrationError(str(e))

    return wrapper


def handle_security_error(
    func: Callable, error_type: Type[Exception] = Exception
) -> Callable:
    """Security error handling decorator."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except error_type as e:
            raise SecurityError(str(e))

    return wrapper
