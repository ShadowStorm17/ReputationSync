from enum import Enum

class ErrorCategory(str, Enum):
    SYSTEM = "system"
    BUSINESS = "business"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    SECURITY = "security"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    EXTERNAL = "external"
    UNKNOWN = "unknown" 