from fastapi import FastAPI, Depends, HTTPException, Request
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
        return os.path.join(os.getcwd(), "dashboard.db")
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
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    # Get total requests
    c.execute("SELECT COUNT(*) FROM usage_stats")
    total_requests = c.fetchone()[0]
    
    # Get active users (users with activity in last 24 hours)
    c.execute("""
        SELECT COUNT(DISTINCT user_id) FROM usage_stats 
        WHERE datetime(timestamp) > datetime('now', '-24 hours')
    """)
    active_users = c.fetchone()[0]
    
    # Calculate error rate
    c.execute("""
        SELECT 
            ROUND(
                (CAST(COUNT(CASE WHEN success = 0 THEN 1 END) AS FLOAT) / 
                CAST(COUNT(*) AS FLOAT)) * 100,
                2
            )
        FROM usage_stats
        WHERE datetime(timestamp) > datetime('now', '-24 hours')
    """)
    error_rate = c.fetchone()[0] or 0
    
    # Calculate average response time
    c.execute("""
        SELECT ROUND(AVG(response_time), 2)
        FROM usage_stats
        WHERE datetime(timestamp) > datetime('now', '-24 hours')
    """)
    avg_response_time = c.fetchone()[0] or 0
    
    # Get hourly usage for the last 12 hours
    c.execute("""
        SELECT 
            strftime('%I %p', datetime(timestamp, 'localtime')) as hour,
            COUNT(*) as requests
        FROM usage_stats
        WHERE datetime(timestamp) > datetime('now', '-12 hours')
        GROUP BY hour
        ORDER BY datetime(timestamp)
    """)
    hourly_data = c.fetchall()
    
    hours = [row[0] for row in hourly_data] if hourly_data else []
    hourly_usage = [row[1] for row in hourly_data] if hourly_data else []
    
    conn.close()
    
    return {
        "total_requests": total_requests,
        "active_users": active_users,
        "error_rate": error_rate,
        "avg_response_time": avg_response_time,
        "hourly_usage": hourly_usage,
        "hours": hours
    }

def get_real_users():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    # Get most recently active users
    c.execute("""
        SELECT 
            u.name,
            u.email,
            u.avatar,
            MAX(us.timestamp) as last_active
        FROM users u
        LEFT JOIN usage_stats us ON us.user_id = u.id
        GROUP BY u.id
        ORDER BY last_active DESC
        LIMIT 5
    """)
    users = []
    for row in c.fetchall():
        # Calculate relative time
        last_active = row[3]
        if last_active:
            try:
                last_active_dt = datetime.fromisoformat(last_active.replace('Z', '+00:00'))
                now = datetime.utcnow()
                diff = now - last_active_dt
                
                if diff.days > 0:
                    relative_time = f"{diff.days} days ago"
                elif diff.seconds >= 3600:
                    hours = diff.seconds // 3600
                    relative_time = f"{hours} hours ago"
                elif diff.seconds >= 60:
                    minutes = diff.seconds // 60
                    relative_time = f"{minutes} minutes ago"
                else:
                    relative_time = "just now"
            except:
                relative_time = "never"
        else:
            relative_time = "never"
            
        users.append({
            "name": row[0],
            "email": row[1],
            "avatar": row[2] or f"https://ui-avatars.com/api/?name={row[0].replace(' ', '+')}",
            "last_active": relative_time
        })
    
    conn.close()
    return users

def get_real_api_keys():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    c.execute("""
        SELECT id, name, created_at
        FROM api_keys
        WHERE revoked = 0
        ORDER BY created_at DESC
    """)
    
    api_keys = []
    for row in c.fetchall():
        api_keys.append({
            "id": row[0],
            "name": row[1],
            "created_at": datetime.fromisoformat(row[2].replace('Z', '+00:00')).strftime('%Y-%m-%d')
        })
    
    conn.close()
    return api_keys

def get_real_subscriptions():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    c.execute("""
        SELECT 
            s.id,
            s.plan_name,
            u.email as user_email,
            s.expires_at,
            s.status
        FROM subscriptions s
        JOIN users u ON u.id = s.user_id
        WHERE s.status != 'cancelled'
        ORDER BY s.expires_at ASC
    """)
    
    subscriptions = []
    for row in c.fetchall():
        subscriptions.append({
            "id": row[0],
            "plan_name": row[1],
            "user_email": row[2],
            "expires_at": datetime.fromisoformat(row[3].replace('Z', '+00:00')).strftime('%Y-%m-%d'),
            "status": row[4]
        })
    
    conn.close()
    return subscriptions

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
        
        # Debug prints
        print(f"Database path: {get_db_path()}")
        print(f"Current working directory: {os.getcwd()}")
        
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            print("Authentication failed, redirecting to login page")
            return RedirectResponse(url='/?error=1', status_code=303)
        
        access_token = create_access_token(data={"sub": user[1]})
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
    token = request.cookies.get("access_token")
    if not token or not token.startswith("Bearer "):
        return RedirectResponse(url='/', status_code=303)
    
    try:
        token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username != ADMIN_USERNAME:
            raise HTTPException(status_code=401)
    except (JWTError, IndexError):
        return RedirectResponse(url='/', status_code=303)
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": username,
            "stats": get_real_stats(),
            "users": get_real_users(),
            "api_keys": get_real_api_keys(),
            "subscriptions": get_real_subscriptions()
        }
    )

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
@app.post("/api/keys/create")
async def create_api_key(request: Request, name: str):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    try:
        key_id = str(uuid.uuid4())
        api_key = str(uuid.uuid4())
        
        c.execute("""
            INSERT INTO api_keys (id, name, key, created_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (key_id, name, api_key))
        
        conn.commit()
        return JSONResponse({"status": "success", "key": api_key, "id": key_id})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    init_admin()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*"
    ) 