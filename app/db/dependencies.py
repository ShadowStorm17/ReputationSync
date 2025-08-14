"""
Database dependencies.
Provides database dependencies for FastAPI.
"""

from typing import Dict, Type


from app.db.repositories import (
    MonitoringRepository,
    ReputationRepository,
    ResponseRepository,
    UserRepository,
)
from app.models.database import (
    AutoResponse,
    MonitoringConfig,
    ReputationScore,
    ResponseRule,
    ResponseTemplate,
    User,
)

# Create repository instances
user_repository = UserRepository()
reputation_repository = ReputationRepository()
monitoring_repository = MonitoringRepository()
response_template_repository = ResponseRepository.ResponseTemplateRepository()
response_rule_repository = ResponseRepository.ResponseRuleRepository()
auto_response_repository = ResponseRepository.AutoResponseRepository()

# Repository dependency mapping
REPOSITORY_MAP: Dict[Type, object] = {
    User: user_repository,
    ReputationScore: reputation_repository,
    MonitoringConfig: monitoring_repository,
    ResponseTemplate: response_template_repository,
    ResponseRule: response_rule_repository,
    AutoResponse: auto_response_repository,
}


async def get_repository(model_type: Type) -> object:
    """Get repository for model type."""
    return REPOSITORY_MAP[model_type]


async def get_user_repository() -> UserRepository:
    """Get user repository."""
    return user_repository


async def get_reputation_repository() -> ReputationRepository:
    """Get reputation repository."""
    return reputation_repository


async def get_monitoring_repository() -> MonitoringRepository:
    """Get monitoring repository."""
    return monitoring_repository


async def get_response_template_repository() -> ResponseRepository.ResponseTemplateRepository:
    """Get response template repository."""
    return response_template_repository


async def get_response_rule_repository() -> ResponseRepository.ResponseRuleRepository:
    """Get response rule repository."""
    return response_rule_repository


async def get_auto_response_repository() -> ResponseRepository.AutoResponseRepository:
    """Get auto response repository."""
    return auto_response_repository
