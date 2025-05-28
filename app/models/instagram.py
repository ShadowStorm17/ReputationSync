from pydantic import BaseModel, Field
from typing import Optional


class InstagramUserBase(BaseModel):
    username: str = Field(..., description="Instagram username")


class InstagramUserResponse(InstagramUserBase):
    follower_count: int = Field(..., description="Number of followers")
    following_count: int = Field(..., description="Number of accounts following")
    is_private: bool = Field(default=False, description="Whether the account is private")
    post_count: Optional[int] = Field(None, description="Number of posts")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "example_user",
                "follower_count": 1000,
                "following_count": 500,
                "is_private": False,
                "post_count": 100
            }
        } 