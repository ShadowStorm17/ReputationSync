"""
Middleware module.
Handles request/response processing and common middleware functionality.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import xmltodict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from .config import get_settings
from .constants import (
    METRICS_ENDPOINT, HEALTH_ENDPOINT, DOCS_ENDPOINT, REDOC_ENDPOINT,
    OPENAPI_ENDPOINT, APP_LOG_FILE, CONTENT_TYPE_JSON, SECURITY_HEADERS,
    MAX_REQUEST_SIZE, MAX_BODY_SIZE
)
from .error_handling import ErrorCategory, ErrorSeverity, ReputationError
from .metrics import metrics_manager
from .rate_limiter import rate_limiter
from .security_headers import security_headers

logger = logging.getLogger(__name__)
settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging request/response details."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware."""
        super().__init__(app)
        self._exclude_paths: Set[str] = {
            METRICS_ENDPOINT,
            HEALTH_ENDPOINT,
            DOCS_ENDPOINT,
            REDOC_ENDPOINT,
            OPENAPI_ENDPOINT
        }
        self._max_body_size: int = MAX_BODY_SIZE
        self._log_file_size_limit: int = 100 * 1024 * 1024  # 100MB
        self._log_rotation_interval: int = 24 * 60 * 60  # 24 hours
        self._last_rotation: float = time.time()
        self._log_lock = asyncio.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and response."""
        try:
            # Skip logging for excluded paths
            if request.url.path in self._exclude_paths:
                return await call_next(request)

            # Check log rotation
            await self._check_log_rotation()

            # Log request
            await self._log_request(request)

            # Process request
            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time

            # Log response
            await self._log_response(request, response, duration)

            # Record metrics
            await self._record_metrics(request, response, duration)

            return response

        except Exception as e:
            logger.error(f"Middleware error: {str(e)}", exc_info=True)
            raise ReputationError(
                message=f"Middleware processing failed: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM
            )

    async def _check_log_rotation(self) -> None:
        """Check if log rotation is needed."""
        try:
            async with self._log_lock:
                current_time = time.time()
                if current_time - self._last_rotation >= self._log_rotation_interval:
                    # Check log file size
                    if os.path.exists(APP_LOG_FILE):
                        size = os.path.getsize(APP_LOG_FILE)
                        if size >= self._log_file_size_limit:
                            # Rotate log file
                            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                            os.rename(APP_LOG_FILE, f"app_{timestamp}.log")
                            logger.info("Log file rotated")

                    self._last_rotation = current_time

        except Exception as e:
            logger.error(f"Log rotation error: {str(e)}", exc_info=True)

    async def _log_request(self, request: Request) -> None:
        """Log request details."""
        try:
            # Get request body
            body = await self._get_request_body(request)

            # Log request
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"Headers: {dict(request.headers)} "
                f"Body: {body}"
            )

        except Exception as e:
            logger.error(f"Error logging request: {str(e)}", exc_info=True)

    async def _log_response(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Log response details."""
        try:
            # Get response body
            body = await self._get_response_body(response)

            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {duration:.2f}s "
                f"Body: {body}"
            )

        except Exception as e:
            logger.error(f"Error logging response: {str(e)}", exc_info=True)

    async def _record_metrics(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Record request/response metrics."""
        try:
            try:
                req_body = await request.body()
                req_size = len(req_body) if req_body else 0
            except Exception:
                req_size = 0
            try:
                resp_body = getattr(response, 'body', None)
                resp_size = len(resp_body) if resp_body else 0
            except Exception:
                resp_size = 0
            await metrics_manager.record_request(request, response, req_size, resp_size)
            
        except Exception as e:
            logger.error(f"Error recording metrics: {str(e)}", exc_info=True)

    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Get request body as string."""
        try:
            body = await request.body()
            if not body:
                return None

            # Check body size
            if len(body) > self._max_body_size:
                return f"[Body too large: {len(body)} bytes]"

            try:
                return body.decode()
            except UnicodeDecodeError:
                return "[Binary body]"

        except Exception as e:
            logger.error(
                f"Error getting request body: {str(e)}", exc_info=True
            )
            return None

    async def _get_response_body(self, response: Response) -> Optional[str]:
        """Get response body as string."""
        try:
            if not hasattr(response, "body"):
                return None

            body = response.body
            if not body:
                return None

            # Check body size
            if len(body) > self._max_body_size:
                return f"[Body too large: {len(body)} bytes]"

            try:
                return body.decode()
            except UnicodeDecodeError:
                return "[Binary body]"

        except Exception as e:
            logger.error(
                f"Error getting response body: {str(e)}", exc_info=True
            )
            return None


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware."""
        super().__init__(app)
        self._error_context: Dict[str, Any] = {}
        self._error_lock = asyncio.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and handle errors."""
        try:
            # Generate correlation ID
            correlation_id = request.headers.get(
                "X-Correlation-ID", str(uuid.uuid4()))
            self._error_context["correlation_id"] = correlation_id
            self._error_context["request_id"] = request.headers.get(
                "X-Request-ID")
            self._error_context["client_ip"] = request.client.host if request.client else None
            self._error_context["user_agent"] = request.headers.get(
                "User-Agent")

            return await call_next(request)

        except ReputationError as e:
            # Handle known errors
            return await self._handle_reputation_error(e)

        except Exception as e:
            # Handle unknown errors
            return await self._handle_unknown_error(e)

    async def _handle_reputation_error(
            self, error: ReputationError) -> Response:
        """Handle ReputationError."""
        try:
            async with self._error_lock:
                # Add context to error
                error.context.update(self._error_context)

                # Log error with context
                logger.error(
                    f"ReputationError: {error.message} "
                    f"Severity: {error.severity} "
                    f"Category: {error.category} "
                    f"Context: {error.context}",
                    exc_info=True
                )

                # Record error metric
                metrics_manager.record_error(
                    error_type="reputation_error",
                    message=error.message,
                    severity=error.severity,
                    category=error.category
                )

                # Determine status code
                if "Rate limit exceeded" in error.message:
                    status_code = 429
                else:
                    status_code = self._get_status_code(error.severity)

                # Create error response
                response = Response(
                    content=json.dumps({
                        "error": error.message,
                        "severity": error.severity.value,
                        "category": error.category.value,
                        "correlation_id": error.context.get("correlation_id"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "request_id": error.context.get("request_id"),
                        "context": error.context
                    }),
                    status_code=status_code,
                    media_type="application/json",
                    headers={
                        "X-Correlation-ID": error.context.get("correlation_id", ""),
                        "X-Error-Type": "reputation_error"
                    }
                )

                # Add security headers
                security_headers.add_headers(response)

            return response
            
        except Exception as e:
            logger.error(
                f"Error handling ReputationError: {str(e)}", exc_info=True
            )
            return await self._handle_unknown_error(e)

    async def _handle_unknown_error(self, error: Exception) -> Response:
        """Handle unknown errors."""
        try:
            async with self._error_lock:
                # Create error context
                error_context = {
                    **self._error_context,
                    "error_type": type(error).__name__,
                    "error_message": str(error)
                }

                # Log error with context
                logger.error(
                    f"Unknown error: {str(error)} "
                    f"Context: {error_context}",
                    exc_info=True
                )

                # Record error metric
                metrics_manager.record_error(
                    error_type="unknown_error",
                    message=str(error),
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SYSTEM
                )

                # Create error response
                response = Response(
                    content=json.dumps({
                        "error": "Internal server error",
                        "severity": ErrorSeverity.HIGH.value,
                        "category": ErrorCategory.SYSTEM.value,
                        "correlation_id": error_context.get("correlation_id"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "request_id": error_context.get("request_id"),
                        "context": error_context
                    }),
                    status_code=500,
                    media_type="application/json",
                    headers={
                        "X-Correlation-ID": error_context.get("correlation_id", ""),
                        "X-Error-Type": "unknown_error"
                    }
                )

                # Add security headers
                security_headers.add_headers(response)

                return response

        except Exception as e:
            logger.error(
                f"Error handling unknown error: {str(e)}", exc_info=True
            )
            # Return basic error response
            response = Response(
                content=json.dumps({
                    "error": "Internal server error",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }),
                status_code=500,
                media_type=CONTENT_TYPE_JSON
            )

            # Add security headers
            security_headers.add_headers(response)

            return response

    def _get_status_code(self, severity: ErrorSeverity) -> int:
        """Get HTTP status code for error severity."""
        severity_map = {
            ErrorSeverity.CRITICAL: 500,
            ErrorSeverity.HIGH: 500,
            ErrorSeverity.MEDIUM: 400,
            ErrorSeverity.LOW: 400
        }
        return severity_map.get(severity, 500)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security checks."""
    
    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware."""
        super().__init__(app)
        self._exclude_paths = {
            METRICS_ENDPOINT,
            HEALTH_ENDPOINT,
            DOCS_ENDPOINT,
            REDOC_ENDPOINT,
            OPENAPI_ENDPOINT
        }
        self._ip_lists = {
            "whitelist": set(),
            "blacklist": set()
        }
        self._rate_limits = {
            "default": {"requests": 60, "period": 60},  # 60 requests per minute
            "api": {"requests": 1000, "period": 60},
            "auth": {"requests": 10, "period": 60}
        }
        # If running in test mode, set very high rate limits
        if getattr(settings, 'TESTING', False):
            self._rate_limits = {
                "default": {"requests": 1000000, "period": 60},
                "api": {"requests": 1000000, "period": 60},
                "auth": {"requests": 1000000, "period": 60}
        }
        self._suspicious_ips = set()
        self._ip_block_duration = 3600  # 1 hour
        self._max_request_size = MAX_REQUEST_SIZE
        self._security_headers = SECURITY_HEADERS
        self._ip_lists_file = "ip_lists.json"
        self._ip_lists_lock = asyncio.Lock()
        self._load_ip_lists()

    def _load_ip_lists(self) -> None:
        """Load IP lists from file."""
        try:
            if os.path.exists(self._ip_lists_file):
                with open(self._ip_lists_file, "r") as f:
                    data = json.load(f)
                    self._ip_lists["whitelist"] = set(
                        data.get("whitelist", []))
                    self._ip_lists["blacklist"] = set(
                        data.get("blacklist", []))
        except Exception as e:
            logger.error(f"Error loading IP lists: {str(e)}", exc_info=True)

    def _save_ip_lists(self) -> None:
        """Save IP lists to file."""
        try:
            data = {
                "whitelist": list(self._ip_lists["whitelist"]),
                "blacklist": list(self._ip_lists["blacklist"])
            }
            with open(self._ip_lists_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving IP lists: {str(e)}", exc_info=True)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and perform security checks."""
        try:
            # Skip security checks for excluded paths
            if request.url.path in self._exclude_paths:
                return await call_next(request)

            # Block disallowed HTTP methods
            allowed_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}
            if request.method not in allowed_methods:
                raise ReputationError(
                    message="HTTP method not allowed",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY
                )

            # Block suspicious/malicious headers
            forbidden_headers = [
                "X-Forwarded-For", "X-Real-IP", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Forwarded-Port"
            ]
            for header in forbidden_headers:
                if header in request.headers:
                    raise ReputationError(
                        message=f"Suspicious header detected: {header}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.SECURITY
                    )

            if "cookie" in request.headers:
                cookie_val = request.headers["cookie"].lower()
                if any(domain in cookie_val for domain in ["malicious.com", "session="]):
                    raise ReputationError(
                        message="Suspicious cookie detected.",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.SECURITY
                    )

            # Block suspicious URL-encoded payloads (e.g., encoded XSS or SQLi)
            from urllib.parse import unquote
            url_query = unquote(str(request.url.query)).lower()
            if any(s in url_query for s in ["<script>", "%3cscript%3e", "%253cscript%253e", "alert("]):
                raise ReputationError(
                    message="Suspicious URL encoding detected.",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY
                )

            # Existing checks (IP, rate limit, request size, etc.)
            client_ip = request.client.host if request.client else None
            if not client_ip:
                raise ReputationError(
                    message="Could not determine client IP",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY
                )
            if client_ip in self._ip_lists["blacklist"]:
                raise ReputationError(
                    message="IP address is blacklisted",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY
                )
            if self._ip_lists["whitelist"] and client_ip not in self._ip_lists["whitelist"]:
                raise ReputationError(
                    message="IP address is not whitelisted",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SECURITY
                )
            rate_limit = self._get_rate_limit(request)
            key = f"rate_limit:{client_ip}:{request.url.path}"
            if await rate_limiter.is_rate_limited(
                key=key,
                max_requests=rate_limit["requests"],
                window=rate_limit["period"]
            ):
                raise ReputationError(
                    message="Rate limit exceeded",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.SECURITY
                )
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self._max_request_size:
                raise ReputationError(
                    message="Request body too large",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.SECURITY
                )
            response = await call_next(request)
            try:
                rate_info = await rate_limiter.get_rate_limit_info(key)
                response.headers["X-RateLimit-Limit"] = str(rate_limit["requests"])
                response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limit["requests"] - rate_info.get("local_count", 0)))
                response.headers["X-RateLimit-Reset"] = str(int(time.time() + rate_limit["period"]))
            except Exception as e:
                logger.error(f"Error adding rate limit headers: {str(e)}")
            for header, value in self._security_headers.items():
                response.headers[header] = value
            return response
        except ReputationError:
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}", exc_info=True)
            raise ReputationError(
                message=f"Security check failed: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SECURITY
            )

    async def _check_rate_limit(
            self,
            client_ip: str,
            request: Request) -> bool:
        """Check if request is within rate limit."""
        try:
            # Get rate limit config
            rate_limit = self._get_rate_limit(request)

            # Check rate limit using the rate limiter
            key = f"rate_limit:{client_ip}:{request.url.path}"
            return await rate_limiter.check_rate_limit(
                key=key,
                max_requests=rate_limit["requests"],
                window=rate_limit["period"]
            )

        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}", exc_info=True)
            return True  # Allow request if rate limiting fails

    def _get_rate_limit(self, request: Request) -> Dict[str, int]:
        """Get rate limit configuration for request."""
        path = request.url.path

        if path.startswith("/api/v1/auth"):
            return self._rate_limits["auth"]
        elif path.startswith("/api/v1"):
            return self._rate_limits["api"]
        else:
            return self._rate_limits["default"]


class CachingMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware."""
        super().__init__(app)
        self._cache = {}
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        self._max_cache_size = 1000
        self._default_ttl = 300  # 5 minutes
        self._exclude_paths = {
            "/api/v1/auth",
            "/api/v1/metrics",
            "/api/v1/health"
        }
        self._exclude_methods = {"POST", "PUT", "DELETE", "PATCH"}
        self._cache_file = "cache.json"
        self._stats_file = "cache_stats.json"
        self._cache_lock = asyncio.Lock()
        self._cleanup_interval = 60  # 1 minute
        self._max_body_size = 10 * 1024 * 1024  # 10MB
        self._last_cleanup = time.time()
        self._cache_compression = True
        self._cache_encryption = True
        self._cache_compression_level = 6
        self._cache_encryption_key = os.urandom(32)
        self._cache_encryption_nonce = os.urandom(12)

    async def initialize(self) -> None:
        """Initialize the cache."""
        try:
            await self._load_cache()
            await self._load_stats()
            logger.info("Cache initialized")
        except Exception as e:
            logger.error("Cache initialization error: %s", str(e))
            raise

    async def shutdown(self) -> None:
        """Shutdown the cache."""
        try:
            await self._save_cache()
            await self._save_stats()
            logger.info("Cache shut down")
        except Exception as e:
            logger.error("Cache shutdown error: %s", str(e))
            raise

    async def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if os.path.exists(self._cache_file):
                async with self._cache_lock:
                    import aiofiles
                    async with aiofiles.open(self._cache_file, "r") as f:
                        content = await f.read()
                        data = json.loads(content)
                        self._cache = {
                            k: {
                                "data": self._decrypt_value(v["data"]) if self._cache_encryption else v["data"],
                                "expires": datetime.fromisoformat(v["expires"])
                            }
                            for k, v in data.items()
                        }
        except Exception as e:
            logger.error("Cache load error: %s", str(e))
            self._cache = {}

    async def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            async with self._cache_lock:
                import aiofiles
                data = {
                    k: {
                        "data": self._encrypt_value(v["data"]) if self._cache_encryption else v["data"],
                        "expires": v["expires"].isoformat()
                    }
                    for k, v in self._cache.items()
                }
                async with aiofiles.open(self._cache_file, "w") as f:
                    await f.write(json.dumps(data))
        except Exception as e:
            logger.error("Cache save error: %s", str(e))

    async def _load_stats(self) -> None:
        """Load cache stats from disk."""
        try:
            if os.path.exists(self._stats_file):
                import aiofiles
                async with aiofiles.open(self._stats_file, "r") as f:
                    content = await f.read()
                    self._cache_stats = json.loads(content)
        except Exception as e:
            logger.error("Stats load error: %s", str(e))
            self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

    async def _save_stats(self) -> None:
        """Save cache stats to disk."""
        try:
            import aiofiles
            async with aiofiles.open(self._stats_file, "w") as f:
                await f.write(json.dumps(self._cache_stats))
        except Exception as e:
            logger.error("Stats save error: %s", str(e))

    async def dispatch(self, request: Request, call_next):
        """Process the request."""
        try:
            # Check if request should be cached
            if not self._should_cache(request):
                return await call_next(request)

            # Generate cache key
            cache_key = await self._generate_cache_key(request)

            # Check cache
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                self._cache_stats["hits"] += 1
                return cached_response

            self._cache_stats["misses"] += 1

            # Process request
            response = await call_next(request)

            # Cache response if cacheable
            if self._is_cacheable(response):
                await self._cache_response(cache_key, response)

            return response

        except Exception as e:
            logger.error("Cache dispatch error: %s", str(e))
            return await call_next(request)

    def _should_cache(self, request: Request) -> bool:
        """Check if request should be cached."""
        return (request.method not in self._exclude_methods and not any(
            request.url.path.startswith(path) for path in self._exclude_paths))

    async def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key."""
        try:
            # Include method, path, query params, and headers
            key_parts = [
                request.method,
                request.url.path,
                str(request.query_params),
                str(request.headers)
            ]

            # Include body hash if present
            if request.method in {"POST", "PUT", "PATCH"}:
                body = await request.body()
                if body:
                    key_parts.append(hashlib.sha256(body).hexdigest())

            return hashlib.sha256("|".join(key_parts).encode()).hexdigest()
        except Exception as e:
            logger.error("Cache key generation error: %s", str(e))
            return str(uuid.uuid4())

    async def _get_cached_response(self, key: str) -> Optional[Response]:
        """Get cached response."""
        try:
            async with self._cache_lock:
                if key not in self._cache:
                    return None

                cache_entry = self._cache[key]
                if cache_entry["expires"] <= datetime.now(timezone.utc):
                    del self._cache[key]
                    self._cache_stats["evictions"] += 1
                    return None

                return Response(
                    content=cache_entry["data"],
                    status_code=200,
                    headers={"X-Cache": "HIT"}
                )
        except Exception as e:
            logger.error("Cache get error: %s", str(e))
            return None

    async def _cache_response(self, key: str, response: Response):
        """Cache response."""
        try:
            # Check response size
            body = getattr(response, 'body', None)
            if body is not None:
                if len(body) > self._max_body_size:
                    return

                async with self._cache_lock:
                    # Check cache size
                    if len(self._cache) >= self._max_cache_size:
                        await self._evict_oldest()

                    # Add to cache
                    self._cache[key] = {
                        "data": body,
                        "expires": datetime.now(timezone.utc) +
                        timedelta(
                            seconds=self._default_ttl)}

                    # Save cache periodically
                    if time.time() - self._last_cleanup >= self._cleanup_interval:
                        await self._cleanup_cache()
                        await self._save_cache()
                        await self._save_stats()
                        self._last_cleanup = time.time()
        except Exception as e:
            logger.error("Cache set error: %s", str(e))

    async def _evict_oldest(self):
        """Evict oldest cache entry."""
        try:
            if not self._cache:
                return

            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k]["expires"]
            )
            del self._cache[oldest_key]
            self._cache_stats["evictions"] += 1
        except Exception as e:
            logger.error("Cache eviction error: %s", str(e))

    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now(timezone.utc)
            expired_keys = [
                k for k, v in self._cache.items()
                if v["expires"] <= current_time
            ]

            for key in expired_keys:
                del self._cache[key]
                self._cache_stats["evictions"] += 1

            if expired_keys:
                logger.info(
                    f"Cleaned up {len(expired_keys)} expired cache entries"
                )
        except Exception as e:
            logger.error("Cache cleanup error: %s", str(e))

    def _is_cacheable(self, response: Response) -> bool:
        """Check if response is cacheable."""
        try:
            # Check status code
            if response.status_code not in {200, 203, 204, 206}:
                return False

            # Check cache control
            cache_control = response.headers.get("cache-control", "")
            if "no-store" in cache_control or "private" in cache_control:
                return False

            # Check content type
            content_type = response.headers.get("content-type", "")
            if not any(ct in content_type for ct in {
                "application/json",
                "text/plain",
                "text/html",
                "application/xml"
            }):
                return False

            return True
        except Exception as e:
            logger.error("Cacheable check error: %s", str(e))
        return False

    async def clear_cache(self):
        """Clear the cache."""
        try:
            async with self._cache_lock:
                self._cache.clear()
                self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
                await self._save_cache()
                await self._save_stats()
                logger.info("Cache cleared")
        except Exception as e:
            logger.error("Cache clear error: %s", str(e))
            raise

    def _encrypt_value(self, value: bytes) -> bytes:
        """Encrypt cache value."""
        try:
            if not self._cache_encryption:
                return value

            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._cache_encryption_key)
            return aesgcm.encrypt(self._cache_encryption_nonce, value, None)
        except Exception as e:
            logger.error("Cache encryption error: %s", str(e))
            return value

    def _decrypt_value(self, value: bytes) -> bytes:
        """Decrypt cache value."""
        try:
            if not self._cache_encryption:
                return value

            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._cache_encryption_key)
            return aesgcm.decrypt(self._cache_encryption_nonce, value, None)
        except Exception as e:
            logger.error("Cache decryption error: %s", str(e))
            return value

    def _compress_value(self, value: bytes) -> bytes:
        """Compress cache value."""
        try:
            if not self._cache_compression:
                return value

            import zlib
            return zlib.compress(value, self._cache_compression_level)
        except Exception as e:
            logger.error("Cache compression error: %s", str(e))
            return value

    def _decompress_value(self, value: bytes) -> bytes:
        """Decompress cache value."""
        try:
            if not self._cache_compression:
                return value

            import zlib
            return zlib.decompress(value)
        except Exception as e:
            logger.error("Cache decompression error: %s", str(e))
            return value


class TransformationMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response transformation."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware."""
        super().__init__(app)
        self._max_body_size = 10 * 1024 * 1024  # 10MB
        self._exclude_paths: Set[str] = {
            "/metrics",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
        self._exclude_methods: Set[str] = {
            "GET",
            "HEAD",
            "OPTIONS"
        }
        self._content_types: Set[str] = {
            "application/json",
            "application/xml",
            "text/plain",
            "text/html"
        }
        self._transformation_stats = {
            "total_transformations": 0,
            "failed_transformations": 0,
            "memory_errors": 0,
            "timeout_errors": 0
        }
        self._max_depth = 10
        self._max_string_length = 256
        self._max_array_size = 1000
        self._max_transformation_time = 2.0  # seconds
        self._max_memory_usage = 100 * 1024 * 1024  # 100MB
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and handle transformations."""
        try:
            # Skip transformation for excluded paths/methods
            if (
                request.url.path in self._exclude_paths or
                request.method in self._exclude_methods
            ):
                return await call_next(request)
            
            # Transform request
            transformed_request = await self._transform_request(request)

            # Process request
            response = await call_next(transformed_request)

            # Transform response
            transformed_response = await self._transform_response(response)

            return transformed_response

        except Exception as e:
            logger.error(
                f"Transformation middleware error: {str(e)}", exc_info=True
            )
            self._transformation_stats["failed_transformations"] += 1
            return await call_next(request)

    async def _transform_request(self, request: Request) -> Request:
        """Transform request."""
        try:
            # Check content type
            content_type = request.headers.get(
                "content-type", "").split(";")[0].strip()
            if content_type not in self._content_types:
                return request

            # Get body
            body = await request.body()
            if not body:
                return request

            # Check body size
            if len(body) > self._max_body_size:
                raise ReputationError(
                    message="Request body too large",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.VALIDATION
                )

            # Transform based on content type
                start_time = time.time()
                memory_before = self._get_memory_usage()

                if content_type == "application/json":
                    try:
                        data = json.loads(body)
                        transformed_data = await self._transform_json_data(data)
                        transformed_body = json.dumps(
                            transformed_data).encode()
                    except json.JSONDecodeError:
                        raise ReputationError(
                            message="Invalid JSON",
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.VALIDATION
                        )

                elif content_type == "application/xml":
                    try:
                        data = xmltodict.parse(body)
                        transformed_data = await self._transform_json_data(data)
                        transformed_body = xmltodict.unparse(
                            transformed_data).encode()
                    except Exception:
                        raise ReputationError(
                            message="Invalid XML",
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.VALIDATION
                        )

                else:
                    transformed_body = body

                # Check transformation time
                if time.time() - start_time > self._max_transformation_time:
                    self._transformation_stats["timeout_errors"] += 1
                    raise ReputationError(
                        message="Transformation timeout",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.PERFORMANCE
                    )

                # Check memory usage
                memory_after = self._get_memory_usage()
                if memory_after - memory_before > self._max_memory_usage:
                    self._transformation_stats["memory_errors"] += 1
                    raise ReputationError(
                        message="Transformation memory limit exceeded",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.PERFORMANCE
                    )

                self._transformation_stats["total_transformations"] += 1

            # Cannot create a new Request with a new body; just return the original request
            # If you need to transform the body, use a custom ASGI receive function or a request wrapper
            return request

        except Exception as e:
            logger.error(
                f"Request transformation error: {str(e)}", exc_info=True
            )
            raise

    async def _transform_response(self, response: Response) -> Response:
        """Transform response."""
        try:
            # Only transform responses that have a .body attribute and are not StreamingResponse
            from starlette.responses import StreamingResponse
            if not hasattr(response, "body") or isinstance(response, StreamingResponse):
                logger.warning(
                    f"Skipping transformation for response type: {type(response).__name__}"
                )
                return response

            # Check content type
            content_type = response.headers.get(
                "content-type", "").split(";")[0].strip()
            if content_type not in self._content_types:
                return response

            # Get body
            body = response.body
            if not body:
                return response

            # Check body size
            if len(body) > self._max_body_size:
                raise ReputationError(
                    message="Response body too large",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.VALIDATION
                )

            # Transform based on content type
                start_time = time.time()
                memory_before = self._get_memory_usage()

                if content_type == "application/json":
                    try:
                        data = json.loads(body)
                        transformed_data = await self._transform_json_data(data)
                        transformed_body = json.dumps(transformed_data).encode()
                    except json.JSONDecodeError:
                        raise ReputationError(
                            message="Invalid JSON",
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.VALIDATION
                        )
                elif content_type == "application/xml":
                    try:
                        data = xmltodict.parse(body)
                        transformed_data = await self._transform_json_data(data)
                        transformed_body = xmltodict.unparse(transformed_data).encode()
                    except Exception:
                        raise ReputationError(
                            message="Invalid XML",
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.VALIDATION
                        )
                else:
                    transformed_body = body

                # Check transformation time
                if time.time() - start_time > self._max_transformation_time:
                    self._transformation_stats["timeout_errors"] += 1
                    raise ReputationError(
                        message="Transformation timeout",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.PERFORMANCE
                    )

                # Check memory usage
                memory_after = self._get_memory_usage()
                if memory_after - memory_before > self._max_memory_usage:
                    self._transformation_stats["memory_errors"] += 1
                    raise ReputationError(
                        message="Transformation memory limit exceeded",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.PERFORMANCE
                    )

                self._transformation_stats["total_transformations"] += 1

            # Create new response
            return Response(
                content=transformed_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

        except Exception as e:
            logger.error(
                f"Response transformation error: {str(e)}", exc_info=True
            )
            raise

    async def _transform_json_data(self, data: Any, depth: int = 0) -> Any:
        """Transform JSON data."""
        try:
            # Check depth
            if depth > self._max_depth:
                raise ReputationError(
                    message="Data structure too deep",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.VALIDATION
                )

            if isinstance(data, dict):
                # Transform dictionary
                transformed = {}
                for key, value in data.items():
                    if len(str(key)) > self._max_string_length:
                        raise ReputationError(
                            message="Key too long",
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.VALIDATION
                        )
                    transformed[key] = await self._transform_json_data(value, depth + 1)
                # Inject _request_timestamp at the top level if not present
                if depth == 0 and "_request_timestamp" not in transformed:
                    from datetime import datetime
                    transformed["_request_timestamp"] = datetime.now(timezone.utc).isoformat()
                return transformed

            elif isinstance(data, list):
                # Transform list
                if len(data) > self._max_array_size:
                    raise ReputationError(
                        message="Array too large",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.VALIDATION
                    )
                return [
                    await self._transform_json_data(item, depth + 1)
                    for item in data
                ]

            elif isinstance(data, str):
                # Transform string
                if len(data) > self._max_string_length:
                    raise ReputationError(
                        message="String too long",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.VALIDATION
                    )
                return data

            else:
                # Return other types as is
                return data

        except Exception as e:
            logger.error(f"JSON transformation error: {str(e)}", exc_info=True)
            raise

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except Exception as e:
            logger.error(f"Memory usage check error: {str(e)}", exc_info=True)
            return 0
