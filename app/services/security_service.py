"""
Security service for enhanced protection.
Handles authentication, authorization, and security features.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
import redis.asyncio as redis
from passlib.hash import bcrypt

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AuthManager:
    """Authentication management system."""

    def __init__(self):
        """Initialize auth manager."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        self.token_ttl = 3600  # 1 hour
        self.secret_key = settings.SECRET_KEY

    async def authenticate(
        self, username: str, password: str
    ) -> Dict[str, Any]:
        """Authenticate user."""
        try:
            # Verify credentials
            user = await self._get_user(username)
            if not user or not self._verify_password(
                password, user["password_hash"]
            ):
                return {"status": "error", "message": "Invalid credentials"}

            # Generate tokens
            access_token = await self._generate_token(user, "access")
            refresh_token = await self._generate_token(user, "refresh")

            # Store refresh token
            await self._store_refresh_token(user["id"], refresh_token)

            return {
                "status": "success",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "roles": user["roles"],
                },
            }

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def verify_token(
        self, token: str, token_type: str = "access"
    ) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            # Verify token type
            if payload.get("type") != token_type:
                return {"status": "error", "message": "Invalid token type"}

            # Check if token is blacklisted
            if await self._is_token_blacklisted(token):
                return {"status": "error", "message": "Token is blacklisted"}

            return {"status": "success", "payload": payload}

        except jwt.ExpiredSignatureError:
            return {"status": "error", "message": "Token has expired"}
        except jwt.InvalidTokenError:
            return {"status": "error", "message": "Invalid token"}

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token."""
        try:
            # Verify refresh token
            verification = await self.verify_token(refresh_token, "refresh")
            if verification["status"] != "success":
                return verification

            # Get user
            user = await self._get_user(verification["payload"]["sub"])
            if not user:
                return {"status": "error", "message": "User not found"}

            # Generate new access token
            access_token = await self._generate_token(user, "access")

            return {"status": "success", "access_token": access_token}

        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _generate_token(
        self, user: Dict[str, Any], token_type: str
    ) -> str:
        """Generate JWT token."""
        now = datetime.now(timezone.utc)
        ttl = self.token_ttl if token_type == "access" else self.token_ttl * 24

        payload = {
            "sub": user["username"],
            "type": token_type,
            "roles": user["roles"],
            "iat": now,
            "exp": now + timedelta(seconds=ttl),
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    async def _store_refresh_token(self, user_id: str, token: str):
        """Store refresh token in Redis."""
        key = f"refresh_token:{user_id}"
        await self.redis.set(key, token, ex=self.token_ttl * 24)

    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        return await self.redis.exists(f"blacklist:{token}")

    async def _get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user from database."""
        # Implement user retrieval logic
        return {
            "id": "123",
            "username": username,
            "password_hash": "hash",
            "roles": ["user"],
        }

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password hash."""
        return bcrypt.verify(password, password_hash)


class RateLimiter:
    """Rate limiting system."""

    def __init__(self):
        """Initialize rate limiter."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        self.default_limit = 100  # requests per minute
        self.default_window = 60  # seconds

    async def check_rate_limit(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Check rate limit for key."""
        try:
            limit = limit or self.default_limit
            window = window or self.default_window

            # Get current count
            current = await self.redis.get(f"ratelimit:{key}")
            count = int(current) if current else 0

            if count >= limit:
                return {
                    "status": "error",
                    "message": "Rate limit exceeded",
                    "reset_in": await self.redis.ttl(f"ratelimit:{key}"),
                }

            # Increment counter
            pipe = self.redis.pipeline()
            await pipe.incr(f"ratelimit:{key}")
            await pipe.expire(f"ratelimit:{key}", window)
            await pipe.execute()

            return {"status": "success", "remaining": limit - (count + 1)}

        except Exception as e:
            logger.error(f"Rate limit error: {str(e)}")
            return {"status": "error", "message": str(e)}


class SecurityService:
    """Comprehensive security service."""

    def __init__(self):
        """Initialize security service."""
        self.auth_manager = AuthManager()
        self.rate_limiter = RateLimiter()

    async def secure_request(
        self, token: str, required_roles: List[str], rate_limit_key: str
    ) -> Dict[str, Any]:
        """Secure and validate request."""
        try:
            # Verify token
            verification = await self.auth_manager.verify_token(token)
            if verification["status"] != "success":
                return verification

            # Check roles
            user_roles = verification["payload"]["roles"]
            if not any(role in required_roles for role in user_roles):
                return {
                    "status": "error",
                    "message": "Insufficient permissions",
                }

            # Check rate limit
            rate_limit = await self.rate_limiter.check_rate_limit(
                rate_limit_key
            )
            if rate_limit["status"] != "success":
                return rate_limit

            return {"status": "success", "user": verification["payload"]}

        except Exception as e:
            logger.error(f"Security check error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def generate_api_key(
        self, user_id: str, scopes: List[str]
    ) -> Dict[str, Any]:
        """Generate API key."""
        try:
            # Generate key
            api_key = secrets.token_urlsafe(32)

            # Store key info
            key_info = {
                "user_id": user_id,
                "scopes": scopes,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await self.redis.set(f"apikey:{api_key}", json.dumps(key_info))

            return {"status": "success", "api_key": api_key, "info": key_info}

        except Exception as e:
            logger.error(f"API key generation error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def verify_api_key(
        self, api_key: str, required_scopes: List[str]
    ) -> Dict[str, Any]:
        """Verify API key and scopes."""
        try:
            # Get key info
            key_info = await self.redis.get(f"apikey:{api_key}")
            if not key_info:
                return {"status": "error", "message": "Invalid API key"}

            key_info = json.loads(key_info)

            # Check scopes
            if not all(
                scope in key_info["scopes"] for scope in required_scopes
            ):
                return {"status": "error", "message": "Insufficient scopes"}

            return {"status": "success", "key_info": key_info}

        except Exception as e:
            logger.error(f"API key verification error: {str(e)}")
            return {"status": "error", "message": str(e)}
