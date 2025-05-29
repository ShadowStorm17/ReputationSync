import sqlite3
import os
from passlib.context import CryptContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_db():
    # Use DATABASE_URL from environment if available
    db_url = os.getenv("DATABASE_URL", "dashboard.db")
    
    conn = sqlite3.connect(db_url)
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
        hashed_password = pwd_context.hash(default_password)
        c.execute(
            "INSERT INTO users (username, password, email, name, is_admin) VALUES (?, ?, ?, ?, ?)",
            (default_admin, hashed_password, "admin@example.com", "Administrator", True)
        )
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db() 