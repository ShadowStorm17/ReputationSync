import json
import re

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.constants import CONTENT_TYPE_JSON
from app.core.logging import logger
from app.core.monitoring import monitoring_manager
from app.core.security_config import security_settings


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
                if not content_type.startswith(CONTENT_TYPE_JSON):
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
                content_length
                and int(content_length) > security_settings.MAX_REQUEST_SIZE
            ):
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

            # Validate JSON payload if content type is application/json
            if request.method in [
                "POST",
                "PUT",
            ] and CONTENT_TYPE_JSON in request.headers.get(
                "content-type", ""
            ):
                try:
                    body = await request.body()
                    if body:
                        json.loads(body)
                except json.JSONDecodeError:
                    await monitoring_manager.track_security_event(
                        "invalid_json",
                        {"content_type": request.headers.get("content-type")},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid JSON payload",
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

            # Check for command injection
            cmd_patterns = [
                r"(\b(cat|ls|rm|chmod|chown|wget|curl|nc|netcat|bash|sh)\b)",
                r"(\|\s*\w+)",
                r"(\&\s*\w+)",
                r"(\;\s*\w+)",
            ]

            all_patterns = (
                sql_patterns + xss_patterns + path_patterns + cmd_patterns
            )

            for pattern in all_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return False

            return True

        except Exception as e:
            logger.error("String safety check error: %s", str(e))
            return False
