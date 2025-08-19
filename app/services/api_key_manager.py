import os
import secrets
import logging
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class APIKeyManager:
    async def create_api_key(self, *args, **kwargs):
        """Create an API key.

        In DEBUG/TESTING, use TEST_API_KEY env var if provided; otherwise generate a secure random key.
        In non-testing environments, always generate a secure key.
        """
        if settings.DEBUG or settings.TESTING:
            test_key = os.getenv("TEST_API_KEY")
            if test_key:
                return {"api_key": test_key}
            # Generate a deterministic-length secure key for tests if env not provided
            generated = secrets.token_urlsafe(32)
            logger.warning("TEST_API_KEY not set; generated a temporary test key.")
            return {"api_key": generated}

        # Production: generate secure key
        return {"api_key": secrets.token_urlsafe(32)}

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key.

        In DEBUG/TESTING, accept TEST_API_KEY or keys prefixed with "test_" for developer convenience.
        In production, this should validate against persistent storage; raise until implemented.
        """
        if settings.DEBUG or settings.TESTING:
            allowed = os.getenv("TEST_API_KEY", "")
            return bool(api_key) and (api_key == allowed or api_key.startswith("test_"))

        # Production path should check persistence (DB/cache). Avoid insecure default.
        raise NotImplementedError("API key validation must be implemented for production use.")

    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key.

        In DEBUG/TESTING, act as a no-op.
        In production, this should remove/disable the key from persistence; raise until implemented.
        """
        if settings.DEBUG or settings.TESTING:
            return True
        raise NotImplementedError("API key revocation must be implemented for production use.")