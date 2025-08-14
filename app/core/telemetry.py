import logging
import time
import asyncio
from functools import wraps
from typing import Callable, Any, Optional

def log_error(error: Exception, context: str = ""):
    logging.error(f"Telemetry error: {error} | Context: {context}")

def monitor_execution_time(metric: Optional[Any] = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if metric is not None:
                    try:
                        metric.labels(operation=func.__name__, status="success").observe(duration)
                    except Exception:
                        pass
                else:
                    logging.info(f"Execution time for {func.__name__}: {duration:.4f}s")

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if metric is not None:
                    try:
                        metric.labels(operation=func.__name__, status="success").observe(duration)
                    except Exception:
                        pass
                else:
                    logging.info(f"Execution time for {func.__name__}: {duration:.4f}s")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    # If used as @monitor_execution_time without parentheses
    if callable(metric):
        return decorator(metric)
    return decorator 