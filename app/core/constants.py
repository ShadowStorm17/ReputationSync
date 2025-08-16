"""
Application constants.
Centralized location for all string literals and configuration constants.
"""

# HTTP Content Types
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_XML = "application/xml"

# File paths and names
CONFIG_FILE_NAME = "config.yaml"
APP_LOG_FILE = "app.log"

# Database error messages
DB_MANAGER_NOT_INITIALIZED = "Database manager not initialized"

# API endpoints for exclusion
METRICS_ENDPOINT = "/metrics"
HEALTH_ENDPOINT = "/health"
DOCS_ENDPOINT = "/docs"
REDOC_ENDPOINT = "/redoc"
OPENAPI_ENDPOINT = "/openapi.json"

# Cache settings
DEFAULT_CACHE_TTL = 300  # 5 minutes
COMPRESSION_MIN_SIZE = 1024  # 1KB

# Rate limiting
DEFAULT_RATE_LIMIT_REQUESTS = 60
DEFAULT_RATE_LIMIT_PERIOD = 60  # seconds

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY", 
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

# File size limits
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BODY_SIZE = 1024 * 1024  # 1MB
MAX_LOG_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Time intervals (seconds)
LOG_ROTATION_INTERVAL = 24 * 60 * 60  # 24 hours
CACHE_CLEANUP_INTERVAL = 60  # 1 minute
METRICS_UPDATE_INTERVAL = 15  # 15 seconds
SYSTEM_METRICS_INTERVAL = 15  # 15 seconds

# Error messages
INVALID_CONTENT_TYPE_MSG = "Invalid Content-Type. Only application/json is allowed."
SUSPICIOUS_HEADER_MSG = "Suspicious header detected"
RATE_LIMIT_EXCEEDED_MSG = "Rate limit exceeded"
REQUEST_TOO_LARGE_MSG = "Request body too large"
IP_BLACKLISTED_MSG = "IP address is blacklisted"
IP_NOT_WHITELISTED_MSG = "IP address is not whitelisted"

# HTTP methods
ALLOWED_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}
CACHE_EXCLUDE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# Regex patterns
# Non-word characters
METRIC_NAME_PATTERN = r'\W'
UNDERSCORE_PATTERN = r'_+'
