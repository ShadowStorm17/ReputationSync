import re
import time

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import logger
from app.core.monitoring import monitoring_manager
from app.core.security_manager import security_manager


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Start timing the request
        start_time = time.time()

        try:
            # Perform security checks
            await security_manager.check_request_security(request)

            # Check rate limits
            endpoint = request.url.path.split("/")[
                1
            ]  # Get the first path segment
            await security_manager.check_rate_limit(request, endpoint)

            # Process the request
            response = await call_next(request)

            # Add security headers
            response = await security_manager.add_security_headers(response)

            # Track request metrics
            processing_time = time.time() - start_time
            await monitoring_manager.track_request(
                request.url.path,
                request.method,
                response.status_code,
                processing_time,
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(processing_time)

            return response

        except HTTPException:
            # Track failed request
            process_time = time.time() - start_time
            await monitoring_manager.track_error(
                "Security check failed",
                request.url.path,
                request.method,
                process_time,
            )
            raise

        except Exception as e:
            # Track unexpected error
            process_time = time.time() - start_time
            logger.error("Security middleware error: %s", str(e))
            await monitoring_manager.track_error(
                str(e), request.url.path, request.method, process_time
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            )

    async def _add_security_headers(self, response: Response):
        """Add security headers to the response."""
        try:
            # Prevent MIME type sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"

            # Prevent clickjacking
            response.headers["X-Frame-Options"] = "DENY"

            # Enable XSS protection
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # Enable HSTS
            response.headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains; preload"

            # Content Security Policy
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://cdn.jsdelivr.net; "
                "connect-src 'self' https://api.example.com; "
                "frame-ancestors 'none'; "
                "form-action 'self'; "
                "base-uri 'self'; "
                "object-src 'none'"
            )

            # Referrer Policy
            response.headers[
                "Referrer-Policy"
            ] = "strict-origin-when-cross-origin"

            # Feature Policy
            response.headers["Permissions-Policy"] = (
                "accelerometer=(), "
                "camera=(), "
                "geolocation=(), "
                "gyroscope=(), "
                "magnetometer=(), "
                "microphone=(), "
                "payment=(), "
                "usb=()"
            )

            # Remove server information
            if "server" in response.headers:
                del response.headers["server"]

            # Add custom security headers
            response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
            response.headers["X-Download-Options"] = "noopen"
            response.headers["X-DNS-Prefetch-Control"] = "off"

            return response

        except Exception as e:
            logger.error("Error adding security headers: %s", str(e))
            return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # Validate request method
            if request.method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                await monitoring_manager.track_security_event(
                    "invalid_method", {"method": request.method}
                )
                raise HTTPException(
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                    detail="Method not allowed",
                )

            # Validate content type for POST/PUT requests
            if request.method in ["POST", "PUT"]:
                content_type = request.headers.get("content-type", "")
                if not content_type.startswith("application/json"):
                    await monitoring_manager.track_security_event(
                        "invalid_content_type", {"content_type": content_type}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="Unsupported media type",
                    )

            # Validate request size
            content_length = request.headers.get("content-length")
            if (
                content_length and int(content_length) > 1024 * 1024
            ):  # 1MB limit
                await monitoring_manager.track_security_event(
                    "request_too_large", {"size": content_length}
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request entity too large",
                )

            # Validate URL parameters
            for param, value in request.query_params.items():
                if not await self._is_safe_string(value):
                    await monitoring_manager.track_security_event(
                        "unsafe_parameter", {"param": param, "value": value}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid parameter value",
                    )

            return await call_next(request)

        except HTTPException:
            raise

        except Exception as e:
            logger.error("Request validation error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            )

    async def _is_safe_string(self, value: str) -> bool:
        """Check if a string is safe (no SQL injection, XSS, etc.)."""
        try:
            # Check for SQL injection patterns
            sql_patterns = [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
                r"(--|\b(OR|AND)\b\s+\d+\s*[=<>])",
                r"(\b(EXEC|EXECUTE|DECLARE)\b)",
            ]

            # Check for XSS patterns
            xss_patterns = [
                r"(<script|javascript:|data:text/html)",
                r"(on\w+\s*=)",
                r"(alert|confirm|prompt)\s*\(",
            ]

            # Check for path traversal
            path_patterns = [r"(\.\.\/|\.\.\\)", r"(\/etc\/|\/var\/|\/usr\/)"]

            all_patterns = sql_patterns + xss_patterns + path_patterns

            for pattern in all_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return False

            return True

        except Exception as e:
            logger.error("String safety check error: %s", str(e))
            return False
