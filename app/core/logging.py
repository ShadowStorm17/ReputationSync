"""
Advanced logging and monitoring system for the reputation management API.
Provides comprehensive logging, monitoring, and observability features.
"""

import json
import logging
import queue
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

from app.core.config import get_settings
from app.core.constants import APP_LOG_FILE
settings = get_settings()

# Enhanced logging metrics
LOG_ENTRIES = Counter(
    "log_entries_total", "Log entry count", ["level", "module"]
)
ERROR_ENTRIES = Counter("error_entries_total", "Error entry count", ["type"])
LOG_PROCESSING_TIME = Histogram(
    "log_processing_seconds", "Log processing time"
)
LOG_QUEUE_SIZE = Gauge("log_queue_size", "Current log queue size")


class LogLevel:
    """Extended log levels with custom severity."""

    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogContext:
    """Thread-local storage for log context."""

    _context = threading.local()

    @classmethod
    def get_context(cls) -> Dict:
        """Get current context."""
        if not hasattr(cls._context, "data"):
            cls._context.data = {}
        return cls._context.data

    @classmethod
    def set_context(cls, **kwargs):
        """Set context values."""
        cls.get_context().update(kwargs)

    @classmethod
    def clear_context(cls):
        """Clear current context."""
        if hasattr(cls._context, "data"):
            cls._context.data.clear()


class StructuredLogRecord(logging.LogRecord):
    """Enhanced log record with structured data."""

    def __init__(self, *args, **kwargs):
        """Initialize structured log record."""
        super().__init__(*args, **kwargs)
        self.context = LogContext.get_context().copy()
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.trace_id = self.context.get("trace_id")
        self.correlation_id = self.context.get("correlation_id")


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler with queue."""

    def __init__(self, capacity: int = 10000):
        """Initialize async handler."""
        super().__init__()
        self.queue = queue.Queue(maxsize=capacity)
        self.worker = threading.Thread(target=self._process_logs, daemon=True)
        self.worker.start()
        self.handlers: List[logging.Handler] = []

    def emit(self, record: logging.LogRecord):
        """Queue log record for processing."""
        try:
            self.queue.put_nowait(record)
            LOG_QUEUE_SIZE.set(self.queue.qsize())
        except queue.Full:
            sys.stderr.write(f"Log queue full, dropping message: {record}\n")

    def _process_logs(self):
        """Process queued log records."""
        while True:
            try:
                record = self.queue.get()
                start_time = time.time()

                for handler in self.handlers:
                    try:
                        if record.levelno >= handler.level:
                            handler.emit(record)
                    except Exception as e:
                        sys.stderr.write(f"Error in log handler: {e}\n")

                LOG_PROCESSING_TIME.observe(time.time() - start_time)
                LOG_QUEUE_SIZE.set(self.queue.qsize())

            except Exception as e:
                sys.stderr.write(f"Error processing log: {e}\n")

    def add_handler(self, handler: logging.Handler):
        """Add a handler for log processing."""
        self.handlers.append(handler)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, **kwargs):
        """Initialize structured formatter."""
        super().__init__()
        self.default_fields = kwargs

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        message = {
            "timestamp": getattr(
                record, "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
            "logger": record.name,
        }

        # Add context if available
        if hasattr(record, "context"):
            message["context"] = record.context

        # Add trace IDs if available
        if hasattr(record, "trace_id"):
            message["trace_id"] = record.trace_id
        if hasattr(record, "correlation_id"):
            message["correlation_id"] = record.correlation_id

        # Add exception info if present
        if record.exc_info:
            message["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add default fields
        message.update(self.default_fields)

        return json.dumps(message)


class EnhancedLogger:
    """Advanced logger with structured logging and monitoring."""

    def __init__(self, name: str):
        """Initialize enhanced logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(settings.LOG_LEVEL)

        # Set custom log record factory
        logging.setLogRecordFactory(StructuredLogRecord)

        # Initialize async handler
        self.async_handler = AsyncLogHandler()

        # Initialize handlers
        self._initialize_handlers()

        # Track logger metrics
        self.metrics_enabled = settings.ENABLE_METRICS

    def _initialize_handlers(self):
        """Initialize log handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            StructuredFormatter(
                app_name=settings.APP_NAME,
                app_version=settings.APP_VERSION,
                environment=settings.ENVIRONMENT,
            )
        )
        self.async_handler.add_handler(console_handler)

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / APP_LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setFormatter(
            StructuredFormatter(
                app_name=settings.APP_NAME,
                app_version=settings.APP_VERSION,
                environment=settings.ENVIRONMENT,
            )
        )
        self.async_handler.add_handler(file_handler)

        # Add async handler to logger
        self.logger.addHandler(self.async_handler)

    def _log(
        self,
        level: int,
        msg: str,
        *args,
        exc_info: Optional[Exception] = None,
        extra: Optional[Dict] = None,
        **kwargs,
    ):
        """Internal logging method with metrics."""
        if self.metrics_enabled:
            LOG_ENTRIES.labels(
                level=logging.getLevelName(level), module=self.logger.name
            ).inc()

            if level >= logging.ERROR:
                ERROR_ENTRIES.labels(
                    type=exc_info.__class__.__name__ if exc_info else "unknown"
                ).inc()

        # Add kwargs to context
        context = LogContext.get_context()
        if kwargs:
            context.update(kwargs)

        # Log message
        self.logger.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            extra={"log_context": context, **kwargs}
            if extra is None
            else {**extra, "log_context": context, **kwargs},
        )

    def trace(self, msg: str, *args, **kwargs):
        """Log trace message."""
        self._log(LogLevel.TRACE, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, msg, *args, **kwargs)

    def error(
        self, msg: str, *args, exc_info: Optional[Exception] = None, **kwargs
    ):
        """Log error message."""
        self._log(LogLevel.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(
        self, msg: str, *args, exc_info: Optional[Exception] = None, **kwargs
    ):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)

    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding context to logs."""
        previous = LogContext.get_context().copy()
        LogContext.set_context(**kwargs)
        try:
            yield
        finally:
            LogContext.get_context().clear()
            LogContext.get_context().update(previous)

    def with_context(self, **context):
        """Decorator for adding context to logs."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.context(**context):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


# Global logger instance
logger = EnhancedLogger(__name__)
