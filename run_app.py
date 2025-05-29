import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 