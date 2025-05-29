from datetime import datetime, timedelta
import PyJWT as jwt
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
import sqlite3
import random
import secrets
from pydantic import BaseModel
import logging
import uuid
import uvicorn
import os
from pathlib import Path
from typing import List, Optional
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyCreate(BaseModel):
    name: str

class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    name: str

app = FastAPI(
    title="Admin Dashboard",
    description="Admin dashboard for monitoring API usage and statistics",
    version="1.0.0"
)

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# CORS middleware with specific origins for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.onrender.com",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_db():
    try:
        # Use DATABASE_URL from environment if available (for Render)
        db_url = os.getenv("DATABASE_URL", "dashboard.db")
        conn = sqlite3.connect(db_url)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def init_db():
    conn = None
    try:
        conn = get_db()
        c = conn.cursor()
        
        # Create tables
        tables = [
            '''CREATE TABLE IF NOT EXISTS stats
               (id INTEGER PRIMARY KEY, endpoint TEXT, requests INTEGER, 
                success_rate REAL, avg_response_time REAL)''',
            
            '''CREATE TABLE IF NOT EXISTS users
               (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT,
                email TEXT, name TEXT, avatar TEXT, last_active TEXT,
                is_admin BOOLEAN DEFAULT FALSE)''',
            
            '''CREATE TABLE IF NOT EXISTS api_keys
               (id TEXT PRIMARY KEY, name TEXT, key TEXT, user_id INTEGER,
                created_at TEXT, revoked INTEGER DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id))''',
            
            '''CREATE TABLE IF NOT EXISTS usage_stats
               (id INTEGER PRIMARY KEY, timestamp TEXT, endpoint TEXT,
                response_time INTEGER, success INTEGER, api_key TEXT,
                FOREIGN KEY(api_key) REFERENCES api_keys(id))'''
        ]
        
        for table_sql in tables:
            c.execute(table_sql)
        
        # Create default admin user if not exists
        default_admin = os.getenv("ADMIN_USERNAME", "admin")
        default_password = os.getenv("ADMIN_PASSWORD", "admin123")
        
        c.execute("SELECT * FROM users WHERE username = ?", (default_admin,))
        if not c.fetchone():
            hashed_password = get_password_hash(default_password)
            c.execute(
                "INSERT INTO users (username, password, email, name, is_admin) VALUES (?, ?, ?, ?, ?)",
                (default_admin, hashed_password, "admin@example.com", "Administrator", True)
            )
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (payload.get("sub"),))
        user = cursor.fetchone()
        conn.close()
        
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
            
        return dict(user)
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if not user or not verify_password(password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token = create_access_token({"sub": username})
        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=3600,
            path="/"
        )
        return response
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: dict = Depends(get_current_user)):
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Initialize default stats
        stats = {
            "total_requests": 0,
            "active_users": 0,
            "error_rate": 0,
            "avg_response_time": 0,
        }
        
        hours = []
        hourly_usage = []
        
        # Try to get statistics if the table exists
        try:
            cursor.execute("""
                SELECT COUNT(*) as total_requests,
                       AVG(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100 as success_rate,
                       AVG(response_time) as avg_response_time
                FROM usage_stats
                WHERE timestamp >= datetime('now', '-24 hours')
            """)
            stats_row = cursor.fetchone()
            
            if stats_row:
                stats.update({
                    "total_requests": stats_row["total_requests"] or 0,
                    "error_rate": 100 - (stats_row["success_rate"] or 0),
                    "avg_response_time": stats_row["avg_response_time"] or 0,
                })
        except sqlite3.OperationalError:
            logger.warning("usage_stats table not found or query failed")
        
        # Try to get active users if the table exists
        try:
            cursor.execute("""
                SELECT COUNT(DISTINCT api_key) as active_users
                FROM usage_stats
                WHERE timestamp >= datetime('now', '-24 hours')
            """)
            active_users_row = cursor.fetchone()
            if active_users_row:
                stats["active_users"] = active_users_row["active_users"] or 0
        except sqlite3.OperationalError:
            logger.warning("Could not fetch active users")
        
        # Try to get hourly usage data if the table exists
        try:
            cursor.execute("""
                SELECT strftime('%H:00', timestamp) as hour,
                       COUNT(*) as requests
                FROM usage_stats
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour DESC
            """)
            usage_data = cursor.fetchall()
            
            if usage_data:
                hours = [row["hour"] for row in usage_data]
                hourly_usage = [row["requests"] for row in usage_data]
            else:
                # Generate empty hourly data for the last 24 hours
                current_hour = datetime.now().hour
                hours = [f"{h:02d}:00" for h in range(current_hour, current_hour - 24, -1)]
                hourly_usage = [0] * 24
        except sqlite3.OperationalError:
            logger.warning("Could not fetch hourly usage data")
            # Generate empty hourly data
            current_hour = datetime.now().hour
            hours = [f"{h:02d}:00" for h in range(current_hour, current_hour - 24, -1)]
            hourly_usage = [0] * 24
        
        conn.close()
        
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "stats": stats,
                "hours": hours,
                "hourly_usage": hourly_usage,
                "user": current_user
            }
        )
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/create_api_key")
async def create_api_key(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    try:
        body = await request.json()
        name = body.get('name')
        
        if not name:
            raise HTTPException(status_code=400, detail="API key name is required")
        
        api_key = f"rsk_{secrets.token_urlsafe(32)}"
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO api_keys (id, name, key, user_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), name, api_key, current_user["id"], datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        
        return {"status": "success", "key": api_key}
    except sqlite3.IntegrityError as e:
        logger.error(f"API key creation error: {str(e)}")
        raise HTTPException(status_code=400, detail="API key name already exists")
    except Exception as e:
        logger.error(f"API key creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True) 