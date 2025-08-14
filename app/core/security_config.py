import os
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class SecuritySettings(BaseSettings):
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    RATE_LIMIT_PER_DAY: int = int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))

    # IP blocking
    MAX_FAILED_ATTEMPTS: int = int(os.getenv("MAX_FAILED_ATTEMPTS", "5"))
    BLOCK_DURATION_MINUTES: int = int(
        os.getenv("BLOCK_DURATION_MINUTES", "30")
    )
    ALLOWED_IPS: List[str] = (
        os.getenv("ALLOWED_IPS", "").split(",")
        if os.getenv("ALLOWED_IPS")
        else []
    )
    BLOCKED_IPS: List[str] = (
        os.getenv("BLOCKED_IPS", "").split(",")
        if os.getenv("BLOCKED_IPS")
        else []
    )

    # Request validation
    MAX_REQUEST_SIZE: int = int(
        os.getenv("MAX_REQUEST_SIZE", str(1024 * 1024))
    )  # 1MB
    ALLOWED_CONTENT_TYPES: List[str] = os.getenv(
        "ALLOWED_CONTENT_TYPES",
        "application/json,application/x-www-form-urlencoded,multipart/form-data",
    ).split(",")
    REQUIRED_HEADERS: List[str] = os.getenv(
        "REQUIRED_HEADERS", "User-Agent,Accept,Content-Type"
    ).split(",")

    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).split(",")
    ALLOWED_METHODS: List[str] = os.getenv(
        "ALLOWED_METHODS", "GET,POST,PUT,DELETE,PATCH"
    ).split(",")

    # Security headers
    ENABLE_HSTS: bool = os.getenv("ENABLE_HSTS", "True").lower() == "true"
    ENABLE_XSS_PROTECTION: bool = (
        os.getenv("ENABLE_XSS_PROTECTION", "True").lower() == "true"
    )
    ENABLE_CONTENT_TYPE_NOSNIFF: bool = (
        os.getenv("ENABLE_CONTENT_TYPE_NOSNIFF", "True").lower() == "true"
    )
    ENABLE_FRAME_DENY: bool = (
        os.getenv("ENABLE_FRAME_DENY", "True").lower() == "true"
    )
    ENABLE_NO_CACHE: bool = (
        os.getenv("ENABLE_NO_CACHE", "True").lower() == "true"
    )

    # JWT settings
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY", ""
    )  # Must be set in production
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(
        os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")
    )

    # Password policy
    MIN_PASSWORD_LENGTH: int = int(os.getenv("MIN_PASSWORD_LENGTH", "8"))
    REQUIRE_SPECIAL_CHAR: bool = (
        os.getenv("REQUIRE_SPECIAL_CHAR", "True").lower() == "true"
    )
    REQUIRE_NUMBER: bool = (
        os.getenv("REQUIRE_NUMBER", "True").lower() == "true"
    )
    REQUIRE_UPPERCASE: bool = (
        os.getenv("REQUIRE_UPPERCASE", "True").lower() == "true"
    )
    REQUIRE_LOWERCASE: bool = (
        os.getenv("REQUIRE_LOWERCASE", "True").lower() == "true"
    )

    # Session security
    SESSION_COOKIE_SECURE: bool = (
        os.getenv("SESSION_COOKIE_SECURE", "True").lower() == "true"
    )
    SESSION_COOKIE_HTTPONLY: bool = (
        os.getenv("SESSION_COOKIE_HTTPONLY", "True").lower() == "true"
    )
    SESSION_COOKIE_SAMESITE: str = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_MAX_AGE: int = int(
        os.getenv("SESSION_COOKIE_MAX_AGE", "3600")
    )  # 1 hour

    # Additional security settings
    ENABLE_CSRF_PROTECTION: bool = (
        os.getenv("ENABLE_CSRF_PROTECTION", "True").lower() == "true"
    )
    CSRF_TOKEN_HEADER: str = os.getenv("CSRF_TOKEN_HEADER", "X-CSRF-Token")
    CSRF_TOKEN_COOKIE: str = os.getenv("CSRF_TOKEN_COOKIE", "csrf_token")

    # API key settings
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    API_KEY_REQUIRED: bool = (
        os.getenv("API_KEY_REQUIRED", "False").lower() == "true"
    )

    # SSL/TLS settings
    SSL_CERT_FILE: str = os.getenv("SSL_CERT_FILE", "")
    SSL_KEY_FILE: str = os.getenv("SSL_KEY_FILE", "")
    SSL_VERIFY: bool = os.getenv("SSL_VERIFY", "True").lower() == "true"

    # Logging settings
    LOG_SECURITY_EVENTS: bool = (
        os.getenv("LOG_SECURITY_EVENTS", "True").lower() == "true"
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Monitoring settings
    ENABLE_SECURITY_MONITORING: bool = (
        os.getenv("ENABLE_SECURITY_MONITORING", "True").lower() == "true"
    )
    ALERT_ON_SECURITY_EVENTS: bool = (
        os.getenv("ALERT_ON_SECURITY_EVENTS", "True").lower() == "true"
    )

    # Cache settings
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes

    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = (
        os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    )
    RATE_LIMIT_STORAGE: str = os.getenv(
        "RATE_LIMIT_STORAGE", "redis"
    )  # or "memory"

    # IP blocking settings
    IP_BLOCKING_ENABLED: bool = (
        os.getenv("IP_BLOCKING_ENABLED", "True").lower() == "true"
    )
    IP_BLOCKING_STORAGE: str = os.getenv(
        "IP_BLOCKING_STORAGE", "redis"
    )  # or "memory"

    # Request validation settings
    VALIDATE_REQUEST_SIZE: bool = (
        os.getenv("VALIDATE_REQUEST_SIZE", "True").lower() == "true"
    )
    VALIDATE_CONTENT_TYPE: bool = (
        os.getenv("VALIDATE_CONTENT_TYPE", "True").lower() == "true"
    )
    VALIDATE_HEADERS: bool = (
        os.getenv("VALIDATE_HEADERS", "True").lower() == "true"
    )
    VALIDATE_PARAMETERS: bool = (
        os.getenv("VALIDATE_PARAMETERS", "True").lower() == "true"
    )

    # Security headers settings
    CONTENT_SECURITY_POLICY: str = os.getenv(
        "CONTENT_SECURITY_POLICY",
        (
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
        ),
    )

    PERMISSIONS_POLICY: str = os.getenv(
        "PERMISSIONS_POLICY",
        (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=()"
        ),
    )

    REFERRER_POLICY: str = os.getenv(
        "REFERRER_POLICY", "strict-origin-when-cross-origin"
    )

    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = True

    def validate_settings(self):
        """Validate security settings."""
        if not self.JWT_SECRET_KEY:
            raise ValueError("JWT_SECRET_KEY must be set in production")

        if self.ENABLE_CSRF_PROTECTION and not self.CSRF_TOKEN_HEADER:
            raise ValueError(
                "CSRF_TOKEN_HEADER must be set when CSRF protection is enabled"
            )

        if self.API_KEY_REQUIRED and not self.API_KEY_HEADER:
            raise ValueError(
                "API_KEY_HEADER must be set when API key is required"
            )

        if self.SSL_VERIFY and (
            not self.SSL_CERT_FILE or not self.SSL_KEY_FILE
        ):
            raise ValueError(
                "SSL_CERT_FILE and SSL_KEY_FILE must be set when SSL verification is enabled"
            )


security_settings = SecuritySettings()
security_settings.validate_settings()
