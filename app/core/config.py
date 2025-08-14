"""
Advanced configuration management system.
Provides dynamic configuration, validation, and hot reloading.
"""

import logging
import os
import secrets
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, root_validator, validator
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class SecurityConfig(BaseSettings):
    """Security configuration."""
    SECRET_KEY: SecretStr
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 3600  # 1 hour
    ENCRYPTION_KEY: SecretStr
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = ["*"]
    RATE_LIMIT_ENABLED: bool = True
    SSL_ENABLED: bool = True
    AUTH_REQUIRED: bool = True

    class Config:
        extra = "allow"


class CacheConfig(BaseSettings):
    """Cache configuration."""
    ENABLED: bool = True
    DEFAULT_TTL: int = 300  # 5 minutes
    MAX_SIZE_MB: int = 1024  # 1 GB
    COMPRESSION_ENABLED: bool = True
    COMPRESSION_MIN_SIZE: int = 1024  # 1 KB
    ENCRYPTION_ENABLED: bool = True
    CLEANUP_INTERVAL: int = 300  # 5 minutes

    class Config:
        extra = "allow"


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    ENABLED: bool = True
    DEFAULT_LIMIT: int = 100
    DEFAULT_WINDOW: int = 60  # 1 minute
    BURST_MULTIPLIER: float = 2.0
    MIN_LIMIT: int = 10
    MAX_LIMIT: int = 1000
    ERROR_THRESHOLD: float = 0.05

    class Config:
        extra = "allow"


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""
    ENABLED: bool = True
    COLLECTION_INTERVAL: int = 60  # 1 minute
    ANALYSIS_INTERVAL: int = 300  # 5 minutes
    ALERT_INTERVAL: int = 60  # 1 minute
    HEALTH_UPDATE_INTERVAL: int = 60  # 1 minute
    METRICS_RETENTION_DAYS: int = 7
    ALERT_RETENTION_DAYS: int = 30

    class Config:
        extra = "allow"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    LEVEL: str = "INFO"
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    FILE_ENABLED: bool = True
    FILE_PATH: str = "logs/app.log"
    MAX_SIZE_MB: int = 100
    BACKUP_COUNT: int = 5
    SYSLOG_ENABLED: bool = False
    SYSLOG_HOST: Optional[str] = None
    SYSLOG_PORT: Optional[int] = None

    class Config:
        extra = "allow"


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    URL: str
    POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_TIMEOUT: int = 30
    ECHO: bool = False
    SSL_ENABLED: bool = True
    MIGRATION_ENABLED: bool = True

    class Config:
        extra = "allow"


class APIConfig(BaseSettings):
    """API configuration."""
    TITLE: str = "Reputation Management API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Advanced reputation management system"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"
    ROOT_PATH: str = ""
    DEBUG: bool = False

    class Config:
        extra = "allow"


class PlatformConfig(BaseSettings):
    """Platform configuration."""
    INSTAGRAM_API_URL: str = "https://api.instagram.com/v1"
    INSTAGRAM_ACCESS_TOKEN: SecretStr
    TWITTER_API_URL: str = "https://api.twitter.com/2"
    TWITTER_BEARER_TOKEN: SecretStr
    YOUTUBE_API_KEY: SecretStr
    YOUTUBE_API_URL: str = "https://www.googleapis.com/youtube/v3"
    RATE_LIMIT: int = 100
    TIMEOUT: int = 30
    LINKEDIN_CLIENT_ID: str = ""
    LINKEDIN_CLIENT_SECRET: str = ""
    LINKEDIN_REDIRECT_URI: str = ""

    class Config:
        extra = "allow"


class Settings(BaseSettings):
    """Main configuration."""
    # Environment
    ENVIRONMENT: str
    DEBUG: bool = False
    TESTING: bool = False

    # Components
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    platforms: PlatformConfig = Field(default_factory=PlatformConfig)

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 4

    # Application settings
    APP_NAME: str = "Reputation Management API"
    APP_VERSION: str = "1.0.0"
    DOCS_URL: str = "/docs"
    OPENAPI_URL: str = "/openapi.json"
    METRICS_PORT: int = 9090

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    # Custom settings
    CUSTOM_SETTINGS: Dict[str, Any] = Field(default_factory=dict)

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "reputation"
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    TEST_DATABASE_URL: str = "sqlite:///./test.db"

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "localhost"

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20

    # Monitoring
    ENABLE_METRICS: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # New attribute
    API_V1_STR: str = "/api/v1"
    SLOW_QUERY_THRESHOLD: float = 1.0
    RATE_LIMIT_CALLS: int = 1000

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return (
            f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@"
            f"{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"
        )

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @root_validator(pre=True)
    def load_config_file(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config_path = os.getenv("CONFIG_FILE", "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    if isinstance(config_data, dict):
                        # Convert string secrets to SecretStr
                        if "security" in config_data:
                            sec = config_data["security"]
                            if "SECRET_KEY" in sec and isinstance(
                                    sec["SECRET_KEY"], str):
                                sec["SECRET_KEY"] = SecretStr(
                                    sec["SECRET_KEY"])
                            if "ENCRYPTION_KEY" in sec and isinstance(
                                    sec["ENCRYPTION_KEY"], str):
                                sec["ENCRYPTION_KEY"] = SecretStr(
                                    sec["ENCRYPTION_KEY"])

                        # Update values with config file data
                        values.update(config_data)
            return values
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            return values

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"


class ConfigFileHandler(FileSystemEventHandler):
    """Handler for config file changes."""

    def __init__(self, config_path: str, callback: callable):
        self.config_path = Path(config_path)
        self.callback = callback

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path) == self.config_path:
            logger.info(f"Config file modified: {event.src_path}")
            self.callback()


class ConfigManager:
    """Advanced configuration manager."""

    def __init__(self):
        """Initialize configuration manager."""
        self._lock = threading.Lock()
        self._settings = self._load_settings()
        self._observer = None
        self._setup_file_watcher()
        self._validate_config()

    def _load_settings(self) -> Settings:
        """Load settings from various sources."""
        try:
            return Settings()
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            raise

    def _setup_file_watcher(self):
        """Setup config file watcher for hot reloading."""
        try:
            config_path = os.getenv("CONFIG_FILE", "config.yaml")
            config_path = os.path.abspath(config_path)
            config_dir = os.path.dirname(config_path)

            # Create config directory if it doesn't exist
            os.makedirs(config_dir, exist_ok=True)

            # Create config file if it doesn't exist
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    yaml.safe_dump({}, f)

            event_handler = ConfigFileHandler(config_path, self._reload_config)
            self._observer = Observer()
            self._observer.schedule(event_handler, config_dir, recursive=False)
            self._observer.start()
            logger.info("Config file watcher started")

        except Exception as e:
            logger.error(f"Error setting up file watcher: {str(e)}")

    def _reload_config(self):
        """Reload configuration."""
        try:
            with self._lock:
                new_settings = self._load_settings()
                self._settings = new_settings
                self._validate_config()
                logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"Error reloading config: {str(e)}")

    def _validate_config(self):
        """Validate configuration."""
        try:
            # Validate security settings
            if self._settings.security.JWT_EXPIRATION < 300:  # 5 minutes minimum
                raise ValueError("JWT expiration too short")

            # Validate cache settings
            if self._settings.cache.MAX_SIZE_MB < 100:  # 100 MB minimum
                raise ValueError("Cache size too small")

            # Validate rate limit settings
            if self._settings.rate_limit.DEFAULT_LIMIT < self._settings.rate_limit.MIN_LIMIT:
                raise ValueError("Default rate limit below minimum")

            # Validate monitoring settings
            if self._settings.monitoring.COLLECTION_INTERVAL < 10:  # 10 seconds minimum
                raise ValueError("Monitoring collection interval too short")

            logger.info("Configuration validated successfully")

        except Exception as e:
            logger.error(f"Configuration validation error: {str(e)}")
            raise

    @property
    def settings(self) -> Settings:
        """Get current settings."""
        return self._settings

    def update_settings(self, updates: Dict[str, Any]) -> bool:
        """Update settings."""
        try:
            with self._lock:
                # Update settings
                for key, value in updates.items():
                    if hasattr(self._settings, key):
                        setattr(self._settings, key, value)

                # Validate new settings
                self._validate_config()

                # Save to file
                config_path = os.getenv("CONFIG_FILE", "config.yaml")
                with open(config_path, "w") as f:
                    yaml.safe_dump(updates, f)

                return True

        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return False

    def get_component_config(self, component: str) -> Any:
        """Get component configuration."""
        try:
            return getattr(self._settings, component)
        except AttributeError:
            logger.error(f"No configuration found for component: {component}")
            return None

    def __del__(self):
        """Cleanup on deletion."""
        if self._observer:
            self._observer.stop()
            self._observer.join()


# Global config manager instance
config_manager = ConfigManager()


@lru_cache()
def get_settings() -> Settings:
    """Get current settings."""
    return config_manager.settings

settings = get_settings()
