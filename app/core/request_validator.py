"""
Request validator module.
Handles request validation and sanitization.
"""

import asyncio
import hashlib
import json
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import unquote

from fastapi import HTTPException, Request, status

from app.core.logging import logger


class RequestValidator:
    """Request validator with enhanced validation."""

    def __init__(self) -> None:
        """Initialize validator."""
        self._max_body_size = 10 * 1024 * 1024  # 10MB
        self._max_header_size = 8 * 1024  # 8KB
        self._max_path_length = 2 * 1024  # 2KB
        self._max_query_length = 1 * 1024  # 1KB
        self._max_headers = 50
        self._max_array_size = 1000
        self._max_string_length = 10000
        self._max_depth = 10
        self._validation_lock = asyncio.Lock()
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_cleanup_interval = 60  # 1 minute
        self._last_cleanup = datetime.utcnow()

        # Security patterns
        self._sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
            r"(--|\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
            r"(\b(WAITFOR|DELAY|SLEEP)\b\s*\(\d+\))",
            r"(\b(EXEC|EXECUTE|sp_)\b)",
            r"(\b(INTO|OUTFILE|DUMPFILE)\b)",
            r"(\b(LOAD_FILE|BENCHMARK)\b\s*\([^)]+\))"]

        self._xss_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(javascript:.*?\([^)]*\))",
            r"(on\w+\s*=\s*['\"][^'\"]*['\"])",
            r"(data:.*?;base64,.*?)",
            r"(<iframe[^>]*>.*?</iframe>)",
            r"(<img[^>]*onerror=.*?>)"
        ]

        self._path_traversal_patterns = [
            r"(\.\./|\.\.\\|\.\.%2f|\.\.%5c)",
            r"(%2e%2e%2f|%2e%2e%5c)",
            r"(\.\.%252f|\.\.%255c)",
            r"(\.\.%c0%af|\.\.%c1%9c)"
        ]

        self._command_injection_patterns = [
            r"(\b(cat|chmod|curl|wget|nc|netcat|bash|sh|powershell|cmd)\b)",
            r"(\|\s*\b(cat|chmod|curl|wget|nc|netcat|bash|sh|powershell|cmd)\b)",
            r"(\b(rm|del|mkdir|touch|echo)\b)",
            r"(\b(&&|\|\||;)\b)",
            r"(\b(>|<|>>|<<)\b)"]

        # Compile patterns
        self._sql_injection_regex = re.compile(
            "|".join(self._sql_injection_patterns), re.IGNORECASE)
        self._xss_regex = re.compile(
            "|".join(self._xss_patterns), re.IGNORECASE)
        self._path_traversal_regex = re.compile(
            "|".join(self._path_traversal_patterns), re.IGNORECASE)
        self._command_injection_regex = re.compile(
            "|".join(self._command_injection_patterns), re.IGNORECASE)

    async def validate_request(self, request: Request) -> None:
        """Validate the request."""
        try:
            # Check request size
            await self._validate_request_size(request)

            # Check headers
            await self._validate_headers(request)

            # Check path
            await self._validate_path(request)

            # Check query parameters
            await self._validate_query_params(request)

            # Check body
            await self._validate_body(request)

            # Clean up cache periodically
            await self._cleanup_cache()

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Request validation error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request"
            )

    async def _validate_request_size(self, request: Request) -> None:
        """Validate request size."""
        try:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self._max_body_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request entity too large"
                )
        except Exception as e:
            logger.error("Request size validation error: %s", str(e))
            raise

    async def _validate_headers(self, request: Request) -> None:
        """Validate request headers."""
        try:
            # Check number of headers
            if len(request.headers) > self._max_headers:
                raise HTTPException(
                    status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE,
                    detail="Too many headers")

            # Check header sizes
            for name, value in request.headers.items():
                if len(name) + len(value) > self._max_header_size:
                    raise HTTPException(
                        status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE,
                        detail="Header too large")

                # Check for suspicious patterns in headers
                if self._check_suspicious_patterns(value):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid header value"
                    )
        except Exception as e:
            logger.error("Header validation error: %s", str(e))
            raise

    async def _validate_path(self, request: Request) -> None:
        """Validate request path."""
        try:
            # Check path length
            if len(request.url.path) > self._max_path_length:
                raise HTTPException(
                    status_code=status.HTTP_414_URI_TOO_LONG,
                    detail="URI too long"
                )

            # Check for suspicious patterns
            decoded_path = unquote(request.url.path)
            if self._check_suspicious_patterns(decoded_path):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid path"
                )
        except Exception as e:
            logger.error("Path validation error: %s", str(e))
            raise

    async def _validate_query_params(self, request: Request) -> None:
        """Validate query parameters."""
        try:
            # Check query string length
            if len(str(request.query_params)) > self._max_query_length:
                raise HTTPException(
                    status_code=status.HTTP_414_URI_TOO_LONG,
                    detail="Query string too long"
                )

            # Check each parameter
            for param, value in request.query_params.items():
                # Check parameter name
                if self._check_suspicious_patterns(param):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid parameter name"
                    )

                # Check parameter value
                if self._check_suspicious_patterns(value):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid parameter value"
                    )
        except Exception as e:
            logger.error("Query parameter validation error: %s", str(e))
            raise

    async def _validate_body(self, request: Request) -> None:
        """Validate request body."""
        try:
            content_type = request.headers.get(
                "content-type", "").split(";")[0].strip()

            if content_type == "application/json":
                await self._validate_json_body(request)
            elif content_type == "application/xml":
                await self._validate_xml_body(request)
            elif content_type == "application/x-www-form-urlencoded":
                await self._validate_form_body(request)
            elif content_type.startswith("multipart/form-data"):
                await self._validate_multipart_body(request)
            elif content_type == "text/plain":
                await self._validate_text_body(request)
        except Exception as e:
            logger.error("Body validation error: %s", str(e))
            raise

    async def _validate_json_body(self, request: Request) -> None:
        """Validate JSON body."""
        try:
            body = await request.body()
            if not body:
                return

            try:
                data = json.loads(body)
                await self._validate_json_structure(data)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON"
                )
        except Exception as e:
            logger.error("JSON validation error: %s", str(e))
            raise

    async def _validate_xml_body(self, request: Request) -> None:
        """Validate XML body."""
        try:
            body = await request.body()
            if not body:
                return

            try:
                ET.fromstring(body)
            except ET.ParseError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid XML"
                )
        except Exception as e:
            logger.error("XML validation error: %s", str(e))
            raise

    async def _validate_form_body(self, request: Request) -> None:
        """Validate form body."""
        try:
            form = await request.form()
            for key, value in form.items():
                if isinstance(value, str):
                    if self._check_suspicious_patterns(value):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid form data"
                        )
        except Exception as e:
            logger.error("Form validation error: %s", str(e))
            raise

    async def _validate_multipart_body(self, request: Request) -> None:
        """Validate multipart body."""
        try:
            form = await request.form()
            for key, value in form.items():
                if isinstance(value, str):
                    if self._check_suspicious_patterns(value):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid multipart data"
                        )
        except Exception as e:
            logger.error("Multipart validation error: %s", str(e))
            raise

    async def _validate_text_body(self, request: Request) -> None:
        """Validate text body."""
        try:
            body = await request.body()
            if not body:
                return

            text = body.decode()
            if self._check_suspicious_patterns(text):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid text content"
                )
        except Exception as e:
            logger.error("Text validation error: %s", str(e))
            raise

    async def _validate_json_structure(
            self, data: Any, depth: int = 0) -> None:
        """Validate JSON structure."""
        try:
            if depth > self._max_depth:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON structure too deep"
                )

            if isinstance(data, dict):
                for key, value in data.items():
                    if len(str(key)) > self._max_string_length:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Key too long"
                        )
                    await self._validate_json_structure(value, depth + 1)

            elif isinstance(data, list):
                if len(data) > self._max_array_size:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Array too large"
                    )
                for item in data:
                    await self._validate_json_structure(item, depth + 1)

            elif isinstance(data, str):
                if len(data) > self._max_string_length:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="String too long"
                    )
                if self._check_suspicious_patterns(data):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid string content"
                    )
        except Exception as e:
            logger.error("JSON structure validation error: %s", str(e))
            raise

    def _check_suspicious_patterns(self, value: str) -> bool:
        """Check for suspicious patterns in string."""
        try:
            # Check SQL injection
            if self._sql_injection_regex.search(value):
                return True

            # Check XSS
            if self._xss_regex.search(value):
                return True

            # Check path traversal
            if self._path_traversal_regex.search(value):
                return True

            # Check command injection
            if self._command_injection_regex.search(value):
                return True

            return False

        except Exception as e:
            logger.error("Pattern check error: %s", str(e))
            return True

    async def _cleanup_cache(self) -> None:
        """Clean up validation cache."""
        try:
            current_time = datetime.utcnow()
            if (current_time -
                    self._last_cleanup).total_seconds() >= self._cache_cleanup_interval:
                async with self._validation_lock:
                    expired_keys = [
                        key for key, (timestamp, _) in self._cache.items()
                        if (current_time - timestamp).total_seconds() >= self._cache_ttl
                    ]

                    for key in expired_keys:
                        del self._cache[key]

                    self._last_cleanup = current_time

                    if expired_keys:
                        logger.info(
                            f"Cleaned up {
                                len(expired_keys)} expired cache entries")
        except Exception as e:
            logger.error("Cache cleanup error: %s", str(e))

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        try:
            key_parts = [
                request.method,
                request.url.path,
                str(request.query_params),
                str(request.headers)
            ]

            if request.method in {"POST", "PUT", "PATCH"}:
                body = request.body()
                if body:
                    key_parts.append(hashlib.md5(body).hexdigest())

            return hashlib.md5("|".join(key_parts).encode()).hexdigest()
        except Exception as e:
            logger.error("Cache key generation error: %s", str(e))
            return str(uuid.uuid4())


request_validator = RequestValidator()
