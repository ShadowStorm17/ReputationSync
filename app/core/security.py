from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from .config import get_settings
import secrets
import time
import hashlib
from .cache import cache

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

class APIKeyManager:
    def __init__(self):
        self.cache_prefix = "api_key:"
        self.max_keys_per_client = 5
    
    async def create_api_key(self, client_id: str) -> Dict[str, str]:
        """Create a new API key with expiration."""
        # Check if client has reached max keys
        existing_keys = await self.get_client_keys(client_id)
        if len(existing_keys) >= self.max_keys_per_client:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum of {self.max_keys_per_client} API keys allowed per client"
            )
        
        # Generate new key
        api_key = secrets.token_urlsafe(32)
        expiry = datetime.utcnow() + timedelta(days=settings.API_KEY_EXPIRY_DAYS)
        
        key_data = {
            "client_id": client_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expiry.isoformat(),
            "is_active": True
        }
        
        # Store in cache
        await cache.set(
            f"{self.cache_prefix}{api_key}",
            key_data,
            ttl=settings.API_KEY_EXPIRY_DAYS * 86400
        )
        
        return {
            "api_key": api_key,
            "expires_at": expiry.isoformat()
        }
    
    async def validate_api_key(self, api_key: str) -> bool:
        """Validate API key and check expiration."""
        key_data = await cache.get(f"{self.cache_prefix}{api_key}")
        if not key_data:
            return False
        
        if not key_data.get("is_active", False):
            return False
        
        expires_at = datetime.fromisoformat(key_data["expires_at"])
        if expires_at <= datetime.utcnow():
            return False
            
        return True
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_data = await cache.get(f"{self.cache_prefix}{api_key}")
        if not key_data:
            return False
            
        key_data["is_active"] = False
        await cache.set(
            f"{self.cache_prefix}{api_key}",
            key_data,
            ttl=86400  # Keep revoked key data for 1 day
        )
        return True
    
    async def get_client_keys(self, client_id: str) -> list:
        """Get all API keys for a client."""
        # Note: This is a simplified implementation
        # In production, you'd want to use a proper database
        return []

api_key_manager = APIKeyManager()

class RequestValidator:
    def __init__(self):
        self.timestamp_window = 300  # 5 minutes
        self.nonce_prefix = "nonce:"
    
    def _generate_signature(self, timestamp: str, nonce: str, api_key: str, body: str = "") -> str:
        """Generate request signature."""
        message = f"{timestamp}{nonce}{api_key}{body}"
        return hashlib.sha256(message.encode()).hexdigest()
    
    async def validate_request(self, request: Request, api_key: str) -> bool:
        """Validate request timestamp, nonce, and signature."""
        timestamp = request.headers.get("X-Timestamp")
        nonce = request.headers.get("X-Nonce")
        signature = request.headers.get("X-Signature")
        
        if not all([timestamp, nonce, signature]):
            raise HTTPException(
                status_code=400,
                detail="Missing required security headers"
            )
        
        # Validate timestamp
        try:
            ts = int(timestamp)
            now = int(time.time())
            if abs(now - ts) > self.timestamp_window:
                raise HTTPException(
                    status_code=400,
                    detail="Request timestamp too old"
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid timestamp format"
            )
        
        # Check nonce
        nonce_key = f"{self.nonce_prefix}{nonce}"
        if await cache.get(nonce_key):
            raise HTTPException(
                status_code=400,
                detail="Nonce already used"
            )
        await cache.set(nonce_key, "1", ttl=self.timestamp_window)
        
        # Validate signature
        body = ""
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            body = body.decode()
        
        expected_signature = self._generate_signature(timestamp, nonce, api_key, body)
        if not secrets.compare_digest(signature, expected_signature):
            raise HTTPException(
                status_code=401,
                detail="Invalid request signature"
            )
        
        return True

request_validator = RequestValidator()

async def verify_api_key(
    api_key: str = Security(api_key_header),
    request: Request = None
) -> str:
    """Verify API key and request validity."""
    if not await api_key_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key"
        )
    
    if settings.is_production and request:
        await request_validator.validate_request(request, api_key)
    
    return api_key

get_api_key = Depends(verify_api_key) 