from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import uvicorn
import redis
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Instagram Stats API Admin Dashboard")

# Security
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

# Templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
async def get_current_admin(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username != os.getenv("ADMIN_USERNAME"):
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# Routes
@app.get("/")
async def dashboard(request: Request, admin: str = Depends(get_current_admin)):
    """Main dashboard view."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "admin": admin,
            "api_stats": await get_api_stats(),
            "revenue_stats": await get_revenue_stats(),
            "user_stats": await get_user_stats()
        }
    )

@app.get("/api-keys")
async def api_keys(request: Request, admin: str = Depends(get_current_admin)):
    """API key management view."""
    return templates.TemplateResponse(
        "api_keys.html",
        {
            "request": request,
            "admin": admin,
            "api_keys": await get_api_keys_data()
        }
    )

@app.get("/analytics")
async def analytics(request: Request, admin: str = Depends(get_current_admin)):
    """Analytics and insights view."""
    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "admin": admin,
            "analytics_data": await get_analytics_data()
        }
    )

@app.get("/revenue")
async def revenue(request: Request, admin: str = Depends(get_current_admin)):
    """Revenue and billing view."""
    return templates.TemplateResponse(
        "revenue.html",
        {
            "request": request,
            "admin": admin,
            "revenue_data": await get_revenue_data()
        }
    )

@app.get("/users")
async def users(request: Request, admin: str = Depends(get_current_admin)):
    """User management view."""
    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "admin": admin,
            "users_data": await get_users_data()
        }
    )

@app.get("/system")
async def system(request: Request, admin: str = Depends(get_current_admin)):
    """System status and monitoring view."""
    return templates.TemplateResponse(
        "system.html",
        {
            "request": request,
            "admin": admin,
            "system_stats": await get_system_stats()
        }
    )

# API Routes
@app.post("/api/revoke-key/{key_id}")
async def revoke_api_key(key_id: str, admin: str = Depends(get_current_admin)):
    """Revoke an API key."""
    # Implementation
    return {"status": "success"}

@app.post("/api/create-key")
async def create_api_key(admin: str = Depends(get_current_admin)):
    """Create a new API key."""
    # Implementation
    return {"status": "success"}

@app.get("/api/stats")
async def get_api_stats():
    """Get API usage statistics."""
    # Implementation
    return {
        "total_requests": 1000000,
        "active_users": 5000,
        "error_rate": 0.1,
        "avg_response_time": 150
    }

@app.get("/api/revenue")
async def get_revenue_stats():
    """Get revenue statistics."""
    # Implementation
    return {
        "monthly_revenue": 50000,
        "active_subscriptions": 1000,
        "churn_rate": 2.5,
        "mrr_growth": 15
    }

@app.get("/api/users")
async def get_user_stats():
    """Get user statistics."""
    # Implementation
    return {
        "total_users": 10000,
        "active_today": 2000,
        "new_today": 100,
        "countries": {"US": 5000, "UK": 2000, "EU": 3000}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 