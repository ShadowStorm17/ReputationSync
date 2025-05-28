from datetime import datetime, timedelta
import jwt
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
from passlib.context import CryptContext
import sqlite3
import random

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = "your-secret-key"  # In production, use a secure secret key
ALGORITHM = "HS256"

def get_db():
    conn = sqlite3.connect('reputation_sync.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request):
    try:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        
        # For demo purposes - in production, validate against database
        if username == "admin" and password == "admin123":
            access_token = jwt.encode(
                {"sub": username, "exp": datetime.utcnow() + timedelta(days=1)},
                SECRET_KEY,
                algorithm=ALGORITHM
            )
            response = JSONResponse({"status": "success"})
            response.set_cookie(key="access_token", value=access_token, httponly=True)
            return response
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: dict = Depends(get_current_user)):
    # Generate some sample data for the dashboard
    stats = {
        "total_requests": random.randint(10000, 50000),
        "active_users": random.randint(100, 500),
        "error_rate": round(random.uniform(0.5, 5.0), 2),
        "avg_response_time": round(random.uniform(100, 500), 2),
    }
    
    # Generate sample API usage data for the last 24 hours
    hours = [(datetime.now() - timedelta(hours=x)).strftime('%H:00') for x in range(23, -1, -1)]
    hourly_usage = [random.randint(100, 1000) for _ in range(24)]
    
    stats_data = {
        "hours": hours,
        "hourly_usage": hourly_usage
    }
    
    # Sample API keys
    api_keys = [
        {"id": "1", "name": "Production API Key", "created_at": "2024-01-01", "usage_count": 15000, "last_used": "2 minutes ago"},
        {"id": "2", "name": "Development API Key", "created_at": "2024-01-15", "usage_count": 5000, "last_used": "5 hours ago"}
    ]
    
    # Sample users
    users = [
        {"name": "John Doe", "email": "john@example.com", "avatar": "https://i.pravatar.cc/150?img=1", "last_active": "2 minutes ago", "success_rate": 98.5},
        {"name": "Jane Smith", "email": "jane@example.com", "avatar": "https://i.pravatar.cc/150?img=2", "last_active": "15 minutes ago", "success_rate": 97.8}
    ]
    
    # Sample subscriptions
    subscriptions = [
        {"id": "1", "plan_name": "Enterprise", "user_email": "enterprise@example.com", "api_calls": 50000, "success_rate": 99.1, "expires_at": "2024-12-31", "days_left": 300},
        {"id": "2", "plan_name": "Professional", "user_email": "pro@example.com", "api_calls": 10000, "success_rate": 98.5, "expires_at": "2024-06-30", "days_left": 120}
    ]
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "stats_data": stats_data,
            "api_keys": api_keys,
            "users": users,
            "subscriptions": subscriptions
        }
    )

@app.get("/api/usage")
async def get_api_usage(current_user: dict = Depends(get_current_user)):
    # Generate sample API usage data for the last 24 hours
    hours = [(datetime.now() - timedelta(hours=x)).strftime('%H:00') for x in range(23, -1, -1)]
    hourly_usage = [random.randint(100, 1000) for _ in range(24)]
    
    return {
        "hours": hours,
        "hourly_usage": hourly_usage
    } 