"""
Security manager module.
Handles security-related functionality including IP blocking, rate limiting, and request validation.
"""

import asyncio
import ipaddress
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import jwt
from fastapi import HTTPException, Request, Response, status
from passlib.context import CryptContext
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError

from app.core.cache import cache_manager
from app.core.logging import logger
from app.core.monitoring import monitoring_manager
from app.core.security_config import security_settings


class SecurityManager:
    """Security manager with enhanced functionality."""

    def __init__(self) -> None:
        """Initialize security manager."""
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self._rate_limit_store: Dict[str, List[datetime]] = {}
        self._ip_block_store: Dict[str, Dict] = {}
        self._suspicious_ips: Set[str] = set()
        self._redis: Optional[Redis] = None
        self._redis_retry_count = 0
        self._max_redis_retries = 3
        self._ip_block_expiry = 3600  # 1 hour in seconds
        self._redis_lock = asyncio.Lock()
        self._ip_block_lock = asyncio.Lock()
        self._redis_retry_delay = 1  # seconds
        self._redis_connection_timeout = 5  # seconds
        self._redis_operation_timeout = 2  # seconds
        self._ip_lists_file = "ip_lists.json"
        self._ip_lists = {
            "whitelist": set(),
            "blacklist": set()
        }
        self._load_ip_lists()

    def _load_ip_lists(self) -> None:
        """Load IP lists from file."""
        try:
            with open(self._ip_lists_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._ip_lists["whitelist"] = set(data.get("whitelist", []))
                self._ip_lists["blacklist"] = set(data.get("blacklist", []))
        except FileNotFoundError:
            # No lists file yet; keep defaults
            return
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error("Error loading IP lists: %s", e, exc_info=True)

    def _save_ip_lists(self) -> None:
        """Save IP lists to file."""
        try:
            data = {
                "whitelist": list(self._ip_lists["whitelist"]),
                "blacklist": list(self._ip_lists["blacklist"]),
            }
            with open(self._ip_lists_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Error saving IP lists: %s", str(e), exc_info=True)

    async def initialize(self) -> None:
        """Initialize the security manager."""
        try:
            async with self._redis_lock:
                if not self._redis:
                    self._redis = await cache_manager.get_redis()
                    self._redis_retry_count = 0
                    logger.info("Security manager initialized")
        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to initialize security manager: %s", str(e))
            self._redis_retry_count += 1
            if self._redis_retry_count < self._max_redis_retries:
                await asyncio.sleep(self._redis_retry_delay)
                await self.initialize()
            else:
                logger.critical(
                    "Failed to initialize Redis after maximum retries")
            raise

    async def shutdown(self) -> None:
        """Shutdown the security manager."""
        try:
            async with self._redis_lock:
                if self._redis:
                    await self._redis.close()
                    self._redis = None
            logger.info("Security manager shut down")
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Error shutting down security manager: %s", str(e))
            raise

    def verify_password(
            self,
            plain_password: str,
            hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except (ValueError) as e:
            logger.error("Password verification error: %s", str(e))
            return False

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        try:
            return self.pwd_context.hash(password)
        except (ValueError) as e:
            logger.error("Password hashing error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error hashing password"
            )

    def validate_password(self, password: str) -> bool:
        """Validate password against security policy."""
        try:
            if len(password) < security_settings.MIN_PASSWORD_LENGTH:
                return False

            if security_settings.REQUIRE_SPECIAL_CHAR and not re.search(
                    r'[!@#$%^&*(),.?":{}|<>]', password):
                return False

            if security_settings.REQUIRE_NUMBER and not re.search(
                    r'\d', password):
                return False

            if security_settings.REQUIRE_UPPERCASE and not re.search(
                    r'[A-Z]', password):
                return False

            if security_settings.REQUIRE_LOWERCASE and not re.search(
                    r'[a-z]', password):
                return False

            return True
        except Exception as e:
            logger.error("Password validation error: %s", str(e))
            return False

    def create_access_token(self, data: dict) -> str:
        """Create JWT access token."""
        try:
            to_encode = data.copy()
            expire = datetime.now(timezone.utc) + timedelta(minutes=security_settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            to_encode.update({"exp": expire})
            return jwt.encode(
                to_encode,
                security_settings.JWT_SECRET_KEY,
                algorithm=security_settings.JWT_ALGORITHM)
        except Exception as e:
            logger.error("Token creation error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating access token"
            )

    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token."""
        try:
            to_encode = data.copy()
            expire = datetime.now(timezone.utc) + timedelta(days=security_settings.REFRESH_TOKEN_EXPIRE_DAYS)
            to_encode.update({"exp": expire})
            return jwt.encode(
                to_encode,
                security_settings.JWT_SECRET_KEY,
                algorithm=security_settings.JWT_ALGORITHM)
        except Exception as e:
            logger.error("Refresh token creation error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating refresh token"
            )

    def verify_token(self, token: str) -> dict:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token, security_settings.JWT_SECRET_KEY, algorithms=[
                    security_settings.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError as e:
            logger.error("Token verification error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        try:
            if not self._redis:
                logger.warning(
                    "Redis not available, skipping rate limit check")
                return True

            key = f"rate_limit:{client_id}"
            current = await self._redis.get(key)

            if current and int(
                    current) >= security_settings.RATE_LIMIT_PER_MINUTE:
                return False

            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 60)  # 1 minute expiry
            await pipe.execute()

            return True
        except (ConnectionError, TimeoutError) as e:
            logger.error("Rate limit check error: %s", str(e))
            self._redis = None  # Reset Redis connection
            await self.initialize()  # Try to reconnect
            return True  # Allow request if rate limiting fails

    async def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked."""
        try:
            # Check Redis first
            async with self._redis_lock:
                if self._redis:
                    try:
                        key = f"blocked_ip:{ip}"
                        is_blocked = await self._redis.get(key)
                        if is_blocked:
                            return True
                    except (ConnectionError, TimeoutError):
                        self._redis = None
                        await self.initialize()

            # Check local store
            async with self._ip_block_lock:
                if ip in self._ip_block_store:
                    block_data = self._ip_block_store[ip]
                    if block_data["expires"] > datetime.now(timezone.utc):
                        return True
                    else:
                        # Clean up expired block
                        del self._ip_block_store[ip]

            return False

        except (ConnectionError, TimeoutError, KeyError) as e:
            logger.error("IP block check error: %s", str(e))
            return False

    async def block_ip(self, ip: str) -> None:
        """Block an IP address."""
        try:
            # Add to Redis
            async with self._redis_lock:
                if self._redis:
                    try:
                        key = f"blocked_ip:{ip}"
                        await self._redis.setex(key, self._ip_block_expiry, "1")
                        monitoring_manager.track_security_event(
                            "ip_blocked", {"ip": ip})
                        return
                    except (ConnectionError, TimeoutError):
                        self._redis = None
                        await self.initialize()

            # Add to local store
            async with self._ip_block_lock:
                self._ip_block_store[ip] = {
                    "timestamp": datetime.now(timezone.utc),
                    "expires": datetime.now(timezone.utc) +
                    timedelta(
                        seconds=self._ip_block_expiry)}
                monitoring_manager.track_security_event(
                    "ip_blocked", {"ip": ip})

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("IP block error: %s", str(e))
            raise

    async def cleanup_expired_blocks(self) -> None:
        """Clean up expired IP blocks."""
        try:
            async with self._redis_lock:
                if self._redis:
                    # Redis handles expiry automatically
                    return

            async with self._ip_block_lock:
                current_time = datetime.now(timezone.utc)
                expired_ips = [
                    ip for ip, data in self._ip_block_store.items()
                    if data["expires"] <= current_time
                ]

                for ip in expired_ips:
                    del self._ip_block_store[ip]

                if expired_ips:
                    logger.info(
                        f"Cleaned up {
                            len(expired_ips)} expired IP blocks")

        except (KeyError) as e:
            logger.error("IP block cleanup error: %s", str(e))
            raise

    def validate_ip_range(self, ip: str) -> bool:
        """Validate if IP is in allowed range."""
        try:
            ip_obj = ipaddress.ip_address(ip)

            # Check if IP is in private range
            if ip_obj.is_private:
                return True

            # Check if IP is in whitelist
            if ip in self._ip_lists["whitelist"]:
                return True

            # Check if IP is in blacklist
            if ip in self._ip_lists["blacklist"]:
                return False

            # Check if IP is in allowed ranges
            allowed_ranges = [
                ipaddress.ip_network("10.0.0.0/8"),
                ipaddress.ip_network("172.16.0.0/12"),
                ipaddress.ip_network("192.168.0.0/16")
            ]

            return any(ip_obj in network for network in allowed_ranges)

        except ValueError as e:
            logger.error("IP range validation error: %s", str(e))
            return False

    def mark_ip_suspicious(self, ip: str) -> None:
        """Mark an IP as suspicious."""
        try:
            self._suspicious_ips.add(ip)
            monitoring_manager.track_security_event(
                "suspicious_ip", {"ip": ip})
        except Exception as e:
            logger.error("IP marking error: %s", str(e))

    async def check_request_security(self, request: Request) -> None:
        """Perform security checks on request."""
        try:
            # Get client IP
            client_ip = request.client.host if request.client else None
            if not client_ip:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not determine client IP"
                )

            # Check if IP is blocked
            if await self.is_ip_blocked(client_ip):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="IP address is blocked"
                )

            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(
                    content_length) > 10 * 1024 * 1024:  # 10MB
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request entity too large"
                )

            # Check headers
            if not request.headers.get("User-Agent"):
                await self.block_ip(client_ip)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Missing User-Agent header"
                )

            # Check for suspicious patterns
            if any(pattern in request.url.path.lower() for pattern in [
                "wp-admin", "php", "admin", "config", ".env", "backup"
            ]):
                await self.block_ip(client_ip)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )

            # Validate IP range
            if not self.validate_ip_range(client_ip):
                self.mark_ip_suspicious(client_ip)
                monitoring_manager.track_security_event(
                    "suspicious_ip", {"ip": client_ip})

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Request security check error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error checking request security"
            )

    async def add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        try:
            if security_settings.ENABLE_HSTS:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            if security_settings.ENABLE_XSS_PROTECTION:
                response.headers["X-XSS-Protection"] = "1; mode=block"

            if security_settings.ENABLE_CONTENT_TYPE_NOSNIFF:
                response.headers["X-Content-Type-Options"] = "nosniff"

            if security_settings.ENABLE_FRAME_DENY:
                response.headers["X-Frame-Options"] = "DENY"

            if security_settings.ENABLE_NO_CACHE:
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"

            # Add Content Security Policy
            response.headers["Content-Security-Policy"] = security_settings.CONTENT_SECURITY_POLICY

            # Add Referrer-Policy
            response.headers["Referrer-Policy"] = security_settings.REFERRER_POLICY

            # Add Permissions-Policy
            response.headers["Permissions-Policy"] = security_settings.PERMISSIONS_POLICY

            # Remove Server header if present
            if "server" in response.headers:
                del response.headers["server"]

            return response
        except Exception as e:
            logger.error("Security headers error: %s", str(e))
            return response


security_manager = SecurityManager()
