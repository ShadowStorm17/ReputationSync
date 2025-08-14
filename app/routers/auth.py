"""
Authentication router.
Handles user authentication and authorization.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.metrics import track_performance
from app.core.security import (
    Token,
    User,
    get_current_active_user,
    security_manager,
)

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)


@router.post("/token", response_model=Token)
@track_performance
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Dict[str, str]:
    """Get access token."""
    user = await security_manager.authenticate_user(
        form_data.username, form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = await security_manager.create_access_token(
        data={"sub": user.username}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=User)
@track_performance
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Get current user."""
    return current_user


@router.get("/status")
@track_performance
async def get_auth_status(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get authentication status."""
    return {
        "authenticated": True,
        "user": current_user.username,
        "roles": current_user.roles,
        "permissions": current_user.permissions,
    }
