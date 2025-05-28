from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List, Dict
import secrets
from datetime import datetime

class APIVersion:
    V1 = "v1"
    V2 = "v2"

    @staticmethod
    def all_versions() -> List[str]:
        return [APIVersion.V1, APIVersion.V2]

    @staticmethod
    def latest() -> str:
        return APIVersion.V2

    @staticmethod
    def is_deprecated(version: str) -> bool:
        return version == APIVersion.V1

class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = APIVersion.latest()
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PROJECT_NAME: str = "Instagram Stats API"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_HOSTS: List[str] = ["*"]
    API_KEY_EXPIRY_DAYS: int = 90
    
    # SSL/TLS Settings
    SSL_KEYFILE: Optional[str] = None
    SSL_CERTFILE: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # 1 hour in seconds
    
    # Instagram Graph API
    INSTAGRAM_APP_ID: str
    INSTAGRAM_APP_SECRET: str
    INSTAGRAM_ACCESS_TOKEN: str
    INSTAGRAM_API_VERSION: str = "v16.0"
    
    # Cache Settings
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 300  # 5 minutes
    REDIS_URL: str = "redis://localhost:6379"
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True
    TRACING_ENABLED: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = []
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API Documentation
    API_DOCS: Dict[str, Dict] = {
        "v1": {
            "deprecated": True,
            "sunset_date": "2024-12-31",
            "migration_guide": "/docs/v1/migration"
        },
        "v2": {
            "deprecated": False,
            "current": True,
            "release_date": "2024-01-01"
        }
    }
    
    # Environment-specific settings
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "testing"
    
    def get_cors_origins(self) -> List[str]:
        if self.is_production:
            return self.CORS_ORIGINS
        return ["*"]  # Allow all origins in development
    
    def get_redis_url(self) -> str:
        if self.is_production:
            return self.REDIS_URL
        return "redis://localhost:6379"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings() 