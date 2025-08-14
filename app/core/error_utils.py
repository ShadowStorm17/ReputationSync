import logging
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ErrorCategory should still be imported from app.core.errors

def handle_errors(
    severity: 'ErrorSeverity' = ErrorSeverity.MEDIUM,
    category: 'ErrorCategory' = None,
    reraise: bool = True,
) -> Callable:
    """Error handling decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Unexpected error: {str(e)}",
                    exc_info=True,
                    extra={"severity": severity, "category": category},
                )
                if reraise:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Internal server error",
                    )
                return None
        return wrapper
    return decorator 