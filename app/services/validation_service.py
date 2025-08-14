"""
Validation service for input validation and data sanitization.
Provides comprehensive data validation capabilities.
"""

import logging
import re
from typing import Any, Dict, Union

import bleach
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DataValidator:
    """Data validation system."""

    def __init__(self):
        """Initialize data validator."""
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        self.url_pattern = re.compile(
            r'^https?:\/\/'
            r'(?:www\.)?'
            r'[-a-zA-Z0-9@:%._\+~#=]{1,256}'
            r'\.[a-zA-Z0-9()]{1,6}'
            r'\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$'
        )

    def validate_input(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate input data against schema."""
        try:
            errors = []
            validated_data = {}

            for field, rules in schema.items():
                value = data.get(field)

                # Check required fields
                if rules.get('required', False) and value is None:
                    errors.append(f"Field '{field}' is required")
                    continue

                # Skip validation if field is not required and value is None
                if value is None:
                    continue

                # Validate field
                validation_result = self._validate_field(
                    field,
                    value,
                    rules
                )

                if validation_result['status'] == 'error':
                    errors.append(validation_result['message'])
                else:
                    validated_data[field] = validation_result['value']

            if errors:
                return {
                    'status': 'error',
                    'errors': errors
                }

            return {
                'status': 'success',
                'data': validated_data
            }

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _validate_field(
        self,
        field: str,
        value: Any,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate single field."""
        try:
            # Type validation
            if 'type' in rules:
                if not isinstance(value, self._get_type(rules['type'])):
                    return {
                        'status': 'error',
                        'message': f"Field '{field}' must be of type {rules['type']}"
                    }

            # Length validation
            if 'min_length' in rules and len(str(value)) < rules['min_length']:
                return {
                    'status': 'error',
                    'message': (
                        f"Field '{field}' must be at least "
                        f"{rules['min_length']} characters long"
                    )
                }

            if 'max_length' in rules and len(str(value)) > rules['max_length']:
                return {
                    'status': 'error',
                    'message': (
                        f"Field '{field}' must be at most "
                        f"{rules['max_length']} characters long"
                    )
                }

            # Range validation
            if 'min_value' in rules and value < rules['min_value']:
                return {
                    'status': 'error',
                    'message': (
                        f"Field '{field}' must be greater than or equal to "
                        f"{rules['min_value']}"
                    )
                }

            if 'max_value' in rules and value > rules['max_value']:
                return {
                    'status': 'error',
                    'message': (
                        f"Field '{field}' must be less than or equal to "
                        f"{rules['max_value']}"
                    )
                }

            # Pattern validation
            if 'pattern' in rules and not re.match(
                    rules['pattern'], str(value)):
                return {
                    'status': 'error',
                    'message': f"Field '{field}' has invalid format"
                }

            # Email validation
            if rules.get(
                'is_email',
                    False) and not self._validate_email(value):
                return {
                    'status': 'error',
                    'message': f"Field '{field}' must be a valid email address"
                }

            # URL validation
            if rules.get('is_url', False) and not self._validate_url(value):
                return {
                    'status': 'error',
                    'message': f"Field '{field}' must be a valid URL"
                }

            # Custom validation
            if 'custom_validator' in rules:
                custom_result = rules['custom_validator'](value)
                if not custom_result['valid']:
                    return {
                        'status': 'error',
                        'message': custom_result['message']
                    }

            return {
                'status': 'success',
                'value': value
            }

        except Exception as e:
            logger.error(f"Field validation error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _get_type(self, type_name: str) -> type:
        """Get Python type from type name."""
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        return type_map.get(type_name, str)

    def _validate_email(self, email: str) -> bool:
        """Validate email address."""
        return bool(self.email_pattern.match(email))

    def _validate_url(self, url: str) -> bool:
        """Validate URL."""
        return bool(self.url_pattern.match(url))


class DataSanitizer:
    """Data sanitization system."""

    def __init__(self):
        """Initialize data sanitizer."""
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li'
        ]
        self.allowed_attributes = {
            'a': ['href', 'title']
        }

    def sanitize_input(
        self,
        data: Union[Dict[str, Any], str],
        sanitize_html: bool = True
    ) -> Dict[str, Any]:
        """Sanitize input data."""
        try:
            if isinstance(data, str):
                return {
                    'status': 'success',
                    'data': self._sanitize_value(data, sanitize_html)
                }

            sanitized_data = {}
            for key, value in data.items():
                sanitized_data[key] = self._sanitize_value(
                    value,
                    sanitize_html
                )

            return {
                'status': 'success',
                'data': sanitized_data
            }

        except Exception as e:
            logger.error(f"Sanitization error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _sanitize_value(
        self,
        value: Any,
        sanitize_html: bool
    ) -> Any:
        """Sanitize single value."""
        if value is None:
            return None

        if isinstance(value, (int, float, bool)):
            return value

        if isinstance(value, str):
            # Remove control characters
            value = ''.join(char for char in value if ord(char) >= 32)

            if sanitize_html:
                value = bleach.clean(
                    value,
                    tags=self.allowed_tags,
                    attributes=self.allowed_attributes,
                    strip=True
                )

            return value.strip()

        if isinstance(value, list):
            return [
                self._sanitize_value(item, sanitize_html)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                key: self._sanitize_value(val, sanitize_html)
                for key, val in value.items()
            }

        return str(value)


class ValidationService:
    """Comprehensive validation service."""

    def __init__(self):
        """Initialize validation service."""
        self.validator = DataValidator()
        self.sanitizer = DataSanitizer()

    async def validate_and_sanitize(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        sanitize_html: bool = True
    ) -> Dict[str, Any]:
        """Validate and sanitize input data."""
        try:
            # First sanitize
            sanitized = self.sanitizer.sanitize_input(
                data,
                sanitize_html
            )
            if sanitized['status'] != 'success':
                return sanitized

            # Then validate
            validation = self.validator.validate_input(
                sanitized['data'],
                schema
            )
            if validation['status'] != 'success':
                return validation

            return {
                'status': 'success',
                'data': validation['data']
            }

        except Exception as e:
            logger.error(f"Validation and sanitization error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def create_schema(
        self,
        model: BaseModel
    ) -> Dict[str, Any]:
        """Create validation schema from Pydantic model."""
        schema = {}

        for field_name, field in model.__fields__.items():
            field_schema = {
                'type': self._get_field_type(field.type_),
                'required': field.required
            }

            # Add validators
            for field_validator in field.validators:
                if hasattr(field_validator, 'min_length'):
                    field_schema['min_length'] = field_validator.min_length
                if hasattr(field_validator, 'max_length'):
                    field_schema['max_length'] = field_validator.max_length
                if hasattr(field_validator, 'regex'):
                    field_schema['pattern'] = field_validator.regex.pattern

            schema[field_name] = field_schema

        return schema

    def _get_field_type(self, type_: type) -> str:
        """Get field type name from Python type."""
        type_map = {
            str: 'string',
            int: 'integer',
            float: 'float',
            bool: 'boolean',
            list: 'list',
            dict: 'dict'
        }
        return type_map.get(type_, 'string')
