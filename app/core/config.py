"""
Advanced configuration management system.
Provides dynamic configuration, validation, and hot reloading.
"""

# app/core/config.py

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, Field, SecretStr, validator
from app.core.constants import CONFIG_FILE_NAME

# Configuration constants
DEFAULT_CONFIG_PATHS = [CONFIG_FILE_NAME, f"config/{CONFIG_FILE_NAME}", f"../config/{CONFIG_FILE_NAME}"]

class Settings(BaseSettings):
    """
    Unified application settings managed via environment variables.
    """

    # Environment
    ENVIRONMENT: str = Field(..., env="ENVIRONMENT")
    DEBUG: bool = False
    TESTING: bool = False

    # Application metadata
    APP_NAME: str = "Reputation Management API"
    APP_VERSION: str = "1.0.0"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 4

    # CORS
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # Security
    SECRET_KEY: SecretStr = Field(..., env="SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    RATE_LIMIT_ENABLED: bool = True
    AUTH_REQUIRED: bool = True
    SSL_ENABLED: bool = True
    ALLOWED_HOSTS: List[str] = Field(default_factory=lambda: ["*"])

    # Database
    POSTGRES_USER: str = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: SecretStr = Field(..., env="POSTGRES_PASSWORD")
    POSTGRES_SERVER: str = Field(..., env="POSTGRES_SERVER")
    POSTGRES_DB: str = Field(..., env="POSTGRES_DB")
    POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_TIMEOUT: int = 30
    ECHO_SQL: bool = False

    # Redis (caching & rate-limiting)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[SecretStr] = None
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 300

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Metrics & Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Third-party API credentials
    INSTAGRAM_ACCESS_TOKEN: Optional[SecretStr] = Field(None, env="INSTAGRAM_ACCESS_TOKEN")
    TWITTER_BEARER_TOKEN: Optional[SecretStr] = Field(None, env="TWITTER_BEARER_TOKEN")
    YOUTUBE_API_KEY: Optional[SecretStr] = Field(None, env="YOUTUBE_API_KEY")
    LINKEDIN_CLIENT_ID: Optional[str] = None
    LINKEDIN_CLIENT_SECRET: Optional[SecretStr] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

    @validator("POSTGRES_PASSWORD", pre=True)
    def ensure_secret(cls, v):
        return v if isinstance(v, SecretStr) else SecretStr(v)

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}@"
            f"{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        pwd = (
            f":{self.REDIS_PASSWORD.get_secret_value()}"
            if self.REDIS_PASSWORD
            else ""
        )
        return f"redis://{pwd}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from file with reduced complexity."""
        import yaml
        
        for config_path in DEFAULT_CONFIG_PATHS:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    continue
        return {}

    def _validate_environment(self) -> bool:
        """Validate environment configuration."""
        required_vars = ['SECRET_KEY', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_SERVER', 'POSTGRES_DB']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        return True


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()
