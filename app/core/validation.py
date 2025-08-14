"""
Validation module.
Provides input validation functions.
"""

import re
from datetime import datetime
from typing import List, Optional


# Supported platforms
SUPPORTED_PLATFORMS = [
    "linkedin",
    "twitter",
    "facebook",
    "instagram",
    "youtube",
    "github"
]

# Username patterns
USERNAME_PATTERNS = {
    "linkedin": r"^[a-zA-Z0-9\-_]{3,100}$",
    "twitter": r"^[a-zA-Z0-9_]{1,15}$",
    "facebook": r"^[a-zA-Z0-9.]{5,50}$",
    "instagram": r"^[a-zA-Z0-9._]{1,30}$",
    "youtube": r"^[a-zA-Z0-9_-]{3,30}$",
    "github": r"^[a-zA-Z0-9-]{1,39}$"
}


def validate_platform(platform: str) -> None:
    """Validate platform name."""
    if not platform or platform.lower() not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported platform. Supported platforms: {', '.join(SUPPORTED_PLATFORMS)}"
        )


def validate_username(username: str, platform: Optional[str] = None) -> None:
    """Validate username format."""
    if not username:
        raise ValueError("Username cannot be empty")

    if platform:
        pattern = USERNAME_PATTERNS.get(platform.lower())
        if pattern and not re.match(pattern, username):
            raise ValueError(
                f"Invalid username format for platform {platform}")
    else:
        # Check against all patterns
        valid = any(
            re.match(pattern, username)
            for pattern in USERNAME_PATTERNS.values()
        )
        if not valid:
            raise ValueError("Invalid username format")


def validate_date_range(start_date: str, end_date: str) -> None:
    """Validate date range."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start > end:
            raise ValueError("Start date must be before end date")

        # Check if range is not too large (e.g., max 1 year)
        if (end - start).days > 365:
            raise ValueError("Date range cannot exceed 1 year")

    except ValueError as e:
        if "time data" in str(e):
            raise ValueError("Invalid date format. Use YYYY-MM-DD")
        raise


def validate_timeframe(timeframe: str) -> None:
    """Validate timeframe format."""
    pattern = r"^(\d+)([dmy])$"
    if not re.match(pattern, timeframe):
        raise ValueError(
            "Invalid timeframe format. Use format: <number>[d|m|y] (e.g., 7d, 30d, 1y)")

    value, unit = re.match(pattern, timeframe).groups()
    value = int(value)

    if unit == "d" and value > 365:
        raise ValueError("Days cannot exceed 365")
    elif unit == "m" and value > 12:
        raise ValueError("Months cannot exceed 12")
    elif unit == "y" and value > 1:
        raise ValueError("Years cannot exceed 1")


def validate_report_type(report_type: str) -> None:
    """Validate report type."""
    valid_types = ["summary", "detailed", "trends", "comparison"]
    if report_type not in valid_types:
        raise ValueError(
            f"Invalid report type. Valid types: {', '.join(valid_types)}"
        )


def validate_comment_id(
        comment_id: str,
        platform: Optional[str] = None) -> None:
    """Validate comment ID format."""
    if not comment_id:
        raise ValueError("Comment ID cannot be empty")

    # Platform-specific validation can be added here
    if platform:
        # Add platform-specific validation if needed
        pass


def validate_metrics_request(
    platform: str,
    username: str,
    timeframe: str,
    metrics: Optional[List[str]] = None
) -> None:
    """Validate metrics request parameters."""
    validate_platform(platform)
    validate_username(username, platform)
    validate_timeframe(timeframe)

    if metrics:
        valid_metrics = [
            "sentiment",
            "engagement",
            "influence",
            "growth",
            "risk"
        ]
        invalid_metrics = [m for m in metrics if m not in valid_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {', '.join(invalid_metrics)}")
