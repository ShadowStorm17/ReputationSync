from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

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
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "src" / "app" / "templates"
STATIC_DIR = BASE_DIR / "src" / "app" / "static"

# Print directory paths for debugging
print(f"Base directory: {BASE_DIR}")
print(f"Templates directory: {TEMPLATES_DIR}")
print(f"Static directory: {STATIC_DIR}")

# Ensure the directories exist
os.makedirs(str(TEMPLATES_DIR), exist_ok=True)
os.makedirs(str(STATIC_DIR), exist_ok=True)

# List template directory contents
print("Template directory contents:")
try:
    print(os.listdir(str(TEMPLATES_DIR)))
except Exception as e:
    print(f"Error listing template directory: {e}")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check paths and configurations"""
    return {
        "base_dir": str(BASE_DIR),
        "templates_dir": str(TEMPLATES_DIR),
        "static_dir": str(STATIC_DIR),
        "cwd": os.getcwd(),
        "template_exists": os.path.exists(str(TEMPLATES_DIR / "login.html")),
        "template_dir_contents": os.listdir(str(TEMPLATES_DIR)) if os.path.exists(str(TEMPLATES_DIR)) else []
    }

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
