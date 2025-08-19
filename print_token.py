from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
if token:
    masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***masked***"
    print("INSTAGRAM_ACCESS_TOKEN:", masked)
else:
    print("INSTAGRAM_ACCESS_TOKEN not set.")