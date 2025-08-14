"""
API key router.
Handles API key management endpoints.
"""


from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.error_handling import ReputationError
from app.core.metrics import metrics_manager
from app.models.api_key import APIKeyCreate, APIKeyResponse, APIKeyStats
from app.services.api_key_service import APIKeyService

router = APIRouter(
    prefix="/api/v1/api-keys",
    tags=["api-keys"],
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

# Initialize services
api_key_service = APIKeyService()

# API key header
api_key_header = APIKeyHeader(name="X-API-Key")


async def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key from header."""
    try:
        return await api_key_service.validate_api_key(
            api_key=api_key, endpoint="/api/v1/api-keys", method="GET"
        )
    except ReputationError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post(
    "",
    response_model=APIKeyResponse,
    summary="Create API Key",
    description="Create a new API key for a customer based on their subscription plan.",
)
async def create_api_key(
    key_data: APIKeyCreate, api_key: str = Security(api_key_header)
):
    """Create a new API key."""
    try:
        # Validate admin API key
        await api_key_service.validate_api_key(
            api_key=api_key, endpoint="/api/v1/api-keys", method="POST"
        )

        # Create API key
        key = await api_key_service.create_api_key(key_data)

        # Record metrics
        await metrics_manager.record_api_key_metric(
            metric_type="key_creation",
            value=1,
            labels={
                "plan": key_data.subscription_plan.value,
                "status": "success",
            },
        )

        return key

    except ReputationError as e:
        # Record metrics
        await metrics_manager.record_api_key_metric(
            metric_type="key_creation",
            value=1,
            labels={
                "plan": key_data.subscription_plan.value,
                "status": "error",
            },
        )

        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Get API Key",
    description="Get details of a specific API key.",
)
async def get_api_key_details(
    key_id: str, api_key: str = Security(api_key_header)
):
    """Get API key details."""
    try:
        # Validate API key
        await get_api_key(api_key)

        # Get key details
        key = await api_key_service.get_api_key(key_id)

        return key

    except ReputationError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Revoke API Key",
    description="Revoke an existing API key.",
)
async def revoke_api_key(key_id: str, api_key: str = Security(api_key_header)):
    """Revoke an API key."""
    try:
        # Validate admin API key
        await api_key_service.validate_api_key(
            api_key=api_key, endpoint="/api/v1/api-keys", method="DELETE"
        )

        # Revoke key
        key = await api_key_service.revoke_api_key(key_id)

        # Record metrics
        await metrics_manager.record_api_key_metric(
            metric_type="key_revocation",
            value=1,
            labels={"plan": key.permissions.rate_limit, "status": "success"},
        )

        return key

    except ReputationError as e:
        # Record metrics
        await metrics_manager.record_api_key_metric(
            metric_type="key_revocation", value=1, labels={"status": "error"}
        )

        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/stats",
    response_model=APIKeyStats,
    summary="Get API Key Statistics",
    description="Get statistics about API key usage and status.",
)
async def get_api_key_stats(api_key: str = Security(api_key_header)):
    """Get API key statistics."""
    try:
        # Validate admin API key
        await api_key_service.validate_api_key(
            api_key=api_key, endpoint="/api/v1/api-keys/stats", method="GET"
        )

        # Get stats
        stats = await api_key_service.get_api_key_stats()

        return stats

    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/cleanup",
    summary="Cleanup Expired Keys",
    description="Clean up expired API keys and send notifications.",
)
async def cleanup_expired_keys(api_key: str = Security(api_key_header)):
    """Clean up expired API keys."""
    try:
        # Validate admin API key
        await api_key_service.validate_api_key(
            api_key=api_key, endpoint="/api/v1/api-keys/cleanup", method="POST"
        )

        # Clean up expired keys
        count = await api_key_service.cleanup_expired_keys()

        # Record metrics
        await metrics_manager.record_api_key_metric(
            metric_type="key_cleanup",
            value=count,
            labels={"status": "success"},
        )

        return {"message": f"Cleaned up {count} expired keys"}

    except ReputationError as e:
        # Record metrics
        await metrics_manager.record_api_key_metric(
            metric_type="key_cleanup", value=0, labels={"status": "error"}
        )

        raise HTTPException(status_code=400, detail=str(e))
