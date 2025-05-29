from fastapi import FastAPI, Request, Form, HTTPException, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import bcrypt

# Create FastAPI app instance
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the templates and static directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Print directory paths for debugging
print(f"Base directory: {BASE_DIR}")
print(f"Templates directory: {TEMPLATES_DIR}")
print(f"Static directory: {STATIC_DIR}")

# Ensure the directories exist
os.makedirs(str(TEMPLATES_DIR), exist_ok=True)
os.makedirs(str(STATIC_DIR), exist_ok=True)

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Simulated user database (replace with actual database in production)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_PASSWORD_HASH = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt())

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the application is working"""
    return {"status": "ok", "message": "Application is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        return templates.TemplateResponse(
            "login.html",
            {"request": request}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": str(type(e)),
                "template_dir": str(TEMPLATES_DIR),
                "exists": os.path.exists(str(TEMPLATES_DIR)),
                "contents": os.listdir(str(TEMPLATES_DIR)) if os.path.exists(str(TEMPLATES_DIR)) else []
            }
        )

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    try:
        # Check credentials
        if username == ADMIN_USERNAME and bcrypt.checkpw(password.encode(), ADMIN_PASSWORD_HASH):
            response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
            return response
        else:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid credentials"},
                status_code=401
            )
    except Exception as e:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": str(e)},
            status_code=500
        )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Display the dashboard page"""
    try:
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/logout")
async def logout():
    """Handle logout"""
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 