from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import sqlite3
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

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

@app.on_event("startup")
async def startup_event():
    print("Starting application...")
    print("Initializing database...")
    init_db()
    print("Initializing admin user...")
    init_admin()
    print("Startup complete!")

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
        return "dashboard.db"  # Store in the current directory
    return "dashboard.db"

def init_db():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stats
                 (id INTEGER PRIMARY KEY, endpoint TEXT, requests INTEGER, 
                  success_rate REAL, avg_response_time REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

# Initialize admin user
def init_admin():
    print("Initializing admin user...")  # Debug log
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    hashed_password = pwd_context.hash(ADMIN_PASSWORD)
    print(f"Admin username: {ADMIN_USERNAME}")  # Debug log
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                 (ADMIN_USERNAME, hashed_password))
        conn.commit()
        print("Admin user created successfully")  # Debug log
    except sqlite3.IntegrityError:
        # Update password if admin exists
        c.execute("UPDATE users SET password = ? WHERE username = ?",
                 (hashed_password, ADMIN_USERNAME))
        conn.commit()
        print("Admin user password updated")  # Debug log
    conn.close()

# Authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def authenticate_user(username: str, password: str):
    print(f"Attempting login with username: {username}")  # Debug log
    user = get_user(username)
    if not user:
        print("User not found in database")  # Debug log
        return False
    if not verify_password(password, user[2]):
        print("Password verification failed")  # Debug log
        return False
    print("Authentication successful")  # Debug log
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username != ADMIN_USERNAME:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# Health check endpoint for monitoring
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Routes
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user[1]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/dashboard")
async def dashboard(request: Request, current_user: str = Depends(get_current_user)):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": current_user,
            "stats": get_mock_stats()
        }
    )

def get_mock_stats():
    return {
        "total_requests": 1000000,
        "active_users": 5000,
        "error_rate": 0.1,
        "avg_response_time": 150,
        "endpoints": [
            {"name": "/api/stats", "requests": 500000},
            {"name": "/api/users", "requests": 300000},
            {"name": "/api/data", "requests": 200000}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*"
    ) 