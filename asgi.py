import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# This is for ASGI servers like uvicorn
application = app

# For compatibility with some ASGI servers
app = application 