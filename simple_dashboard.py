from fastapi import FastAPI, Depends, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import sqlite3
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv
import uuid
from typing import List, Optional
import json
import secrets
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Create static and templates directories if they don't exist
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Admin Dashboard",
    description="Admin dashboard for monitoring API usage and statistics",
    version="1.0.0"
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_HOSTS", "*").split(","),
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

# Database setup
def get_db_path():
    if os.getenv("ENVIRONMENT") == "production":
        # On Render, use the /data directory which is persistent
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "dashboard.db")
    return os.path.join(os.getcwd(), "dashboard.db")

def get_db():
    try:
        db_path = get_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def init_db():
    conn = None
    try:
        db_path = get_db_path()
        print(f"Initializing database at: {db_path}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = get_db()
        c = conn.cursor()
        
        # Create tables with proper error handling
        tables = [
            '''CREATE TABLE IF NOT EXISTS stats
               (id INTEGER PRIMARY KEY, endpoint TEXT, requests INTEGER, 
                success_rate REAL, avg_response_time REAL)''',
            
            '''CREATE TABLE IF NOT EXISTS users
               (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT,
                email TEXT, name TEXT, avatar TEXT, last_active TEXT)''',
            
            '''CREATE TABLE IF NOT EXISTS api_keys
               (id TEXT PRIMARY KEY, name TEXT, key TEXT, user_id INTEGER,
                created_at TEXT, revoked INTEGER DEFAULT 0)''',
            
            '''CREATE TABLE IF NOT EXISTS subscriptions
               (id TEXT PRIMARY KEY, user_id INTEGER, plan_name TEXT,
                expires_at TEXT, status TEXT)''',
            
            '''CREATE TABLE IF NOT EXISTS usage_stats
               (id INTEGER PRIMARY KEY, timestamp TEXT, endpoint TEXT,
                response_time INTEGER, success INTEGER)'''
        ]
        
        for table_sql in tables:
            try:
                c.execute(table_sql)
                print(f"Successfully created table: {table_sql.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()}")
            except sqlite3.Error as e:
                print(f"Error creating table: {str(e)}")
                print(f"SQL: {table_sql}")
        
        conn.commit()
        print("Database initialization completed successfully")
        
        # Verify database is writable
        try:
            c.execute("INSERT INTO stats (endpoint, requests, success_rate, avg_response_time) VALUES (?, ?, ?, ?)",
                     ("/test", 0, 0.0, 0.0))
            conn.commit()
            print("Database write test successful")
        except Exception as e:
            print(f"Database write test failed: {str(e)}")
            
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        if conn:
            conn.close()

def get_real_stats():
    try:
        conn = get_db()
        c = conn.cursor()
        
        # Get total requests (last 24 hours)
        c.execute("""
            SELECT COUNT(*) FROM usage_stats 
            WHERE datetime(timestamp) > datetime('now', '-24 hours')
        """)
        total_requests = c.fetchone()[0] or 0
        
        # Get active users (last hour)
        c.execute("""
            SELECT COUNT(DISTINCT user_id) FROM usage_stats 
            WHERE datetime(timestamp) > datetime('now', '-1 hour')
        """)
        active_users = c.fetchone()[0] or 0
        
        # Calculate error rate (last hour)
        c.execute("""
            SELECT 
                ROUND(
                    (CAST(COUNT(CASE WHEN success = 0 THEN 1 END) AS FLOAT) / 
                    NULLIF(CAST(COUNT(*) AS FLOAT), 0)) * 100,
                    2
                )
            FROM usage_stats
            WHERE datetime(timestamp) > datetime('now', '-1 hour')
        """)
        error_rate = c.fetchone()[0] or 0
        
        # Calculate average response time (last hour)
        c.execute("""
            SELECT ROUND(AVG(response_time), 2)
            FROM usage_stats
            WHERE datetime(timestamp) > datetime('now', '-1 hour')
        """)
        avg_response_time = c.fetchone()[0] or 0
        
        # Get hourly usage for the last 12 hours
        c.execute("""
            SELECT 
                strftime('%H:00', datetime(timestamp, 'localtime')) as hour,
                COUNT(*) as requests
            FROM usage_stats
            WHERE datetime(timestamp) > datetime('now', '-12 hours')
            GROUP BY hour
            ORDER BY hour DESC
            LIMIT 12
        """)
        hourly_data = c.fetchall()
        
        # Format hours and usage data
        hours = []
        hourly_usage = []
        
        for hour, requests in hourly_data:
            hours.append(f"{hour}")
            hourly_usage.append(requests)
        
        # If less than 12 hours of data, pad with zeros
        while len(hours) < 12:
            hours.append("00:00")
            hourly_usage.append(0)
            
        # Reverse to show oldest first
        hours.reverse()
        hourly_usage.reverse()
        
        conn.close()
        
        return {
            "total_requests": total_requests,
            "active_users": active_users,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "hourly_usage": hourly_usage,
            "hours": hours
        }
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return {
            "total_requests": 0,
            "active_users": 0,
            "error_rate": 0,
            "avg_response_time": 0,
            "hourly_usage": [0] * 12,
            "hours": [f"{i}:00" for i in range(12)]
        }

def get_real_users():
    try:
        conn = get_db()
        c = conn.cursor()
        
        # Get most recently active users with their stats
        c.execute("""
            SELECT 
                u.name,
                u.email,
                u.avatar,
                MAX(us.timestamp) as last_active,
                COUNT(us.id) as total_requests,
                ROUND(AVG(CASE WHEN us.success = 1 THEN 1 ELSE 0 END) * 100, 2) as success_rate
            FROM users u
            LEFT JOIN usage_stats us ON us.user_id = u.id
            GROUP BY u.id
            ORDER BY last_active DESC
            LIMIT 5
        """)
        
        users = []
        for row in c.fetchall():
            name, email, avatar, last_active, total_requests, success_rate = row
            
            # Calculate relative time
            if last_active:
                try:
                    last_active_dt = datetime.fromisoformat(last_active.replace('Z', '+00:00'))
                    now = datetime.utcnow()
                    diff = now - last_active_dt
                    
                    if diff.days > 0:
                        relative_time = f"{diff.days}d ago"
                    elif diff.seconds >= 3600:
                        hours = diff.seconds // 3600
                        relative_time = f"{hours}h ago"
                    elif diff.seconds >= 60:
                        minutes = diff.seconds // 60
                        relative_time = f"{minutes}m ago"
                    else:
                        relative_time = "just now"
                except:
                    relative_time = "never"
            else:
                relative_time = "never"
                
            users.append({
                "name": name or "Anonymous",
                "email": email or "unknown@example.com",
                "avatar": avatar or f"https://ui-avatars.com/api/?name={name or 'Anonymous'}&background=random",
                "last_active": relative_time,
                "total_requests": total_requests,
                "success_rate": success_rate
            })
        
        conn.close()
        return users
    except Exception as e:
        print(f"Error getting users: {str(e)}")
        return []

def get_real_api_keys():
    try:
        conn = get_db()
        c = conn.cursor()
        
        c.execute("""
            SELECT 
                k.id,
                k.name,
                k.created_at,
                COUNT(us.id) as usage_count,
                MAX(us.timestamp) as last_used
            FROM api_keys k
            LEFT JOIN usage_stats us ON us.api_key_id = k.id
            WHERE k.revoked = 0
            GROUP BY k.id
            ORDER BY last_used DESC NULLS LAST
        """)
        
        api_keys = []
        for row in c.fetchall():
            key_id, name, created_at, usage_count, last_used = row
            
            # Format last used time
            if last_used:
                try:
                    last_used_dt = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
                    last_used_str = last_used_dt.strftime('%Y-%m-%d %H:%M')
                except:
                    last_used_str = "Never"
            else:
                last_used_str = "Never"
            
            api_keys.append({
                "id": key_id,
                "name": name,
                "created_at": datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d'),
                "usage_count": usage_count,
                "last_used": last_used_str
            })
        
        conn.close()
        return api_keys
    except Exception as e:
        print(f"Error getting API keys: {str(e)}")
        return []

def get_real_subscriptions():
    try:
        conn = get_db()
        c = conn.cursor()
        
        c.execute("""
            SELECT 
                s.id,
                s.plan_name,
                u.email as user_email,
                s.expires_at,
                s.status,
                COUNT(us.id) as api_calls,
                ROUND(AVG(CASE WHEN us.success = 1 THEN 1 ELSE 0 END) * 100, 2) as success_rate
            FROM subscriptions s
            JOIN users u ON u.id = s.user_id
            LEFT JOIN usage_stats us ON us.user_id = u.id
            WHERE s.status != 'cancelled'
            GROUP BY s.id
            ORDER BY s.expires_at ASC
        """)
        
        subscriptions = []
        for row in c.fetchall():
            sub_id, plan_name, email, expires_at, status, api_calls, success_rate = row
            
            # Calculate days until expiration
            try:
                expires_dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                now = datetime.utcnow()
                days_left = (expires_dt - now).days
                if days_left < 0:
                    status = "expired"
            except:
                days_left = 0
            
            subscriptions.append({
                "id": sub_id,
                "plan_name": plan_name,
                "user_email": email,
                "expires_at": datetime.fromisoformat(expires_at.replace('Z', '+00:00')).strftime('%Y-%m-%d'),
                "status": status,
                "days_left": days_left,
                "api_calls": api_calls,
                "success_rate": success_rate
            })
        
        conn.close()
        return subscriptions
    except Exception as e:
        print(f"Error getting subscriptions: {str(e)}")
        return []

@app.on_event("startup")
async def startup_event():
    try:
        print("Starting application...")
        print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        print(f"Current working directory: {os.getcwd()}")
        print("Initializing database...")
        init_db()
        print("Database initialized successfully")
        print("Initializing admin user...")
        init_admin()
        print("Admin user initialized successfully")
        print("Startup complete!")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Don't raise the exception - let the app continue to start
        # but log the error for debugging

# Simplified authentication for admin
def verify_admin_credentials(username: str, password: str) -> bool:
    """Simple verification for admin credentials"""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

# Authentication
def verify_password(plain_password, hashed_password):
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {str(e)}")
        return False

def get_user(username):
    conn = None
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        return user
    except Exception as e:
        print(f"Database error in get_user: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def authenticate_user(username: str, password: str):
    try:
        print(f"Attempting login with username: {username}")
        user = get_user(username)
        if not user:
            print("User not found in database")
            return False
        
        # Debug print
        print(f"Found user: {user}")
        print(f"Stored password hash: {user[2]}")
        
        if not verify_password(password, user[2]):
            print("Password verification failed")
            return False
            
        print("Authentication successful")
        return user
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return False

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Initialize admin user
def init_admin():
    conn = None
    try:
        print("Initializing admin user...")
        conn = get_db()
        c = conn.cursor()
        
        # Create users table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT,
                    email TEXT, name TEXT, avatar TEXT, last_active TEXT)''')
        
        hashed_password = pwd_context.hash(ADMIN_PASSWORD)
        print(f"Admin username: {ADMIN_USERNAME}")
        print(f"Generated password hash: {hashed_password}")
        
        try:
            c.execute("""INSERT INTO users 
                        (username, password, email, name, avatar) 
                        VALUES (?, ?, ?, ?, ?)""",
                    (ADMIN_USERNAME, hashed_password,
                    "admin@example.com", "Administrator",
                    "https://ui-avatars.com/api/?name=Administrator"))
            conn.commit()
            print("Admin user created successfully")
        except sqlite3.IntegrityError:
            c.execute("UPDATE users SET password = ? WHERE username = ?",
                    (hashed_password, ADMIN_USERNAME))
            conn.commit()
            print("Admin user password updated")
    except Exception as e:
        print(f"Error in init_admin: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        if conn:
            conn.close()

# Routes
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@app.post("/token")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        print(f"Login attempt with username: {form_data.username}")
        
        if not verify_admin_credentials(form_data.username, form_data.password):
            print("Authentication failed, redirecting to login page")
            return RedirectResponse(url='/?error=1', status_code=303)
        
        # Create access token
        access_token = create_access_token(data={"sub": form_data.username})
        response = RedirectResponse(url='/dashboard', status_code=303)
        response.set_cookie(
            key="access_token",
            value=f"Bearer {access_token}",
            httponly=True,
            samesite='lax',
            secure=os.getenv("ENVIRONMENT") == "production"
        )
        print("Login successful, redirecting to dashboard")
        return response
    except Exception as e:
        print(f"Login error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return RedirectResponse(url='/?error=1', status_code=303)

@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        # Get real data from database
        stats = get_real_stats()
        users = get_real_users()
        api_keys = get_real_api_keys()
        subscriptions = get_real_subscriptions()
        
        # Generate CSRF token
        csrf_token = generate_csrf_token()
        request.session["csrf_token"] = csrf_token
        
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": "Admin",
                "stats": stats,
                "users": users,
                "api_keys": api_keys,
                "subscriptions": subscriptions,
                "stats_json": json.dumps(stats),
                "csrf_token": csrf_token
            }
        )
    except Exception as e:
        print(f"Error in dashboard route: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# API Key Management
@app.post("/api/keys/{key_id}/revoke")
async def revoke_key(key_id: str, request: Request):
    # Add authentication check here
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,))
    conn.commit()
    conn.close()
    return JSONResponse({"status": "success"})

@app.post("/api/keys/{key_id}/regenerate")
async def regenerate_key(key_id: str, request: Request):
    # Add authentication check here
    new_key = str(uuid.uuid4())
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE api_keys SET key = ? WHERE id = ?", (new_key, key_id))
    conn.commit()
    conn.close()
    return JSONResponse({"key": new_key})

# Subscription Management
@app.post("/api/subscriptions/{sub_id}/cancel")
async def cancel_subscription(sub_id: str, request: Request):
    # Add authentication check here
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE subscriptions SET status = 'cancelled' WHERE id = ?", (sub_id,))
    conn.commit()
    conn.close()
    return JSONResponse({"status": "success"})

# Add API endpoint to record usage statistics
@app.post("/api/record-usage")
async def record_usage(
    request: Request,
    endpoint: str,
    user_id: int,
    response_time: int,
    success: bool
):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    try:
        c.execute("""
            INSERT INTO usage_stats (timestamp, endpoint, user_id, response_time, success)
            VALUES (datetime('now'), ?, ?, ?, ?)
        """, (endpoint, user_id, response_time, 1 if success else 0))
        
        conn.commit()
        return JSONResponse({"status": "success"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        conn.close()

# Add API endpoint to create new API key
@app.post("/create_api_key")
async def create_api_key(request: Request):
    try:
        # Verify CSRF token
        client_token = request.headers.get("X-CSRFToken")
        server_token = request.session.get("csrf_token")
        if not client_token or not server_token or client_token != server_token:
            raise HTTPException(status_code=403, detail="Invalid CSRF token")

        # Get request body
        body = await request.json()
        name = body.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")

        # Generate new API key
        api_key = f"rsk_{secrets.token_urlsafe(32)}"
        key_id = str(uuid.uuid4())
        
        # Store in database
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO api_keys (id, name, key, created_at) VALUES (?, ?, ?, ?)",
            (key_id, name, api_key, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        return JSONResponse({
            "success": True,
            "key": api_key,
            "id": key_id,
            "name": name
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating API key: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create API key")

# Add CSRF protection
def generate_csrf_token():
    return secrets.token_hex(32)

if __name__ == "__main__":
    init_db()
    init_admin()
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",  # Required for Render
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    ) 