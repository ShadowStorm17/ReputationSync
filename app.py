from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "src", "app", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "src", "app", "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000) 