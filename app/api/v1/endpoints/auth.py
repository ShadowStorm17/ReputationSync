"""
Authentication endpoints.
Handles user authentication and authorization.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.core.auth import (
    authenticate_user,
    create_user_token,
    get_current_active_user,
)
from app.core.security import get_password_hash
from app.db.session import get_db
from app.models.user import User

router = APIRouter()


class UserCreate(BaseModel):
    """User creation model."""

    username: str
    email: EmailStr
    password: str
    full_name: str


class Token(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str


@router.post("/login", response_model=Token)
async def login(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """Login endpoint."""
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return create_user_token(user)


@router.post("/register", response_model=Token)
async def register(
    *, db: Session = Depends(get_db), user_in: UserCreate
) -> Any:
    """Register new user."""
    # Check if user exists
    user = db.query(User).filter(User.username == user_in.username).first()
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email exists
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return create_user_token(user)


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Refresh access token."""
    return create_user_token(current_user)


@router.get("/me")
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get current user information."""
    return current_user
