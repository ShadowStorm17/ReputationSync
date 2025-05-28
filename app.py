from datetime import datetime, timedelta
import jwt
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
from passlib.context import CryptContext
import sqlite3
import random
import secrets
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyCreate(BaseModel):
    name: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = "your-secret-key"  # In production, use a secure secret key
ALGORITHM = "HS256"

def get_db():
    try:
        conn = sqlite3.connect('reputation_sync.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def init_db():
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Create api_keys table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            key TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            usage_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize database")

# Initialize database on startup
init_db()

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
    try:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": None}
        )
    except Exception as e:
        logger.error(f"Error rendering login page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    try:
        logger.info(f"Login attempt for username: {username}")
        
        if not username or not password:
            logger.warning("Missing username or password")
            raise HTTPException(status_code=400, detail="Username and password are required")

        # For demo purposes - in production, validate against database
        if username == "admin" and password == "admin123":
            # Create token with expiration
            token_data = {
                "sub": username,
                "exp": datetime.utcnow() + timedelta(days=1)
            }
            
            try:
                access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
                logger.info(f"Login successful for user: {username}")
                
                # Create response with cookie
                response = JSONResponse({"status": "success", "message": "Login successful"})
                response.set_cookie(
                    key="access_token",
                    value=access_token,
                    httponly=True,
                    max_age=86400,  # 1 day in seconds
                    path="/"
                )
                return response
                
            except Exception as e:
                logger.error(f"Token generation error: {str(e)}")
                raise HTTPException(status_code=500, detail="Error generating authentication token")
        else:
            logger.warning(f"Invalid credentials for username: {username}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: dict = Depends(get_current_user)):
    try:
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
        
        # Get actual API keys from database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, created_at, usage_count, 
                CASE 
                    WHEN last_used IS NULL THEN 'Never'
                    WHEN (julianday('now') - julianday(last_used)) * 24 < 1 THEN 'Less than an hour ago'
                    WHEN (julianday('now') - julianday(last_used)) * 24 < 24 THEN round((julianday('now') - julianday(last_used)) * 24) || ' hours ago'
                    ELSE round(julianday('now') - julianday(last_used)) || ' days ago'
                END as last_used
            FROM api_keys 
            WHERE is_active = 1 
            ORDER BY created_at DESC
        """)
        api_keys = [dict(row) for row in cursor.fetchall()]
        
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
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/usage")
async def get_api_usage(current_user: dict = Depends(get_current_user)):
    try:
        # Generate sample API usage data for the last 24 hours
        hours = [(datetime.now() - timedelta(hours=x)).strftime('%H:00') for x in range(23, -1, -1)]
        hourly_usage = [random.randint(100, 1000) for _ in range(24)]
        
        return {
            "hours": hours,
            "hourly_usage": hourly_usage
        }
    except Exception as e:
        logger.error(f"API usage error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_api_key")
async def create_api_key(request: Request):
    try:
        # First verify the user is authenticated
        current_user = get_current_user(request)
        
        # Get the request body
        body = await request.json()
        name = body.get('name')
        
        if not name:
            raise HTTPException(status_code=400, detail="API key name is required")
        
        # Generate a secure API key
        api_key = f"rsk_{secrets.token_urlsafe(32)}"
        
        # Store in database
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO api_keys (name, key) VALUES (?, ?)",
                (name, api_key)
            )
            conn.commit()
            logger.info(f"Created new API key with name: {name}")
            
            return {"status": "success", "key": api_key}
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error: {str(e)}")
            raise HTTPException(status_code=400, detail="API key name must be unique")
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create API key")
        finally:
            conn.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_api_key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 