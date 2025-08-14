from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the application is working"""
    return {"status": "ok", "message": "Application is running"}
