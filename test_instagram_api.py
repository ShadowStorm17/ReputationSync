import os
from dotenv import load_dotenv

from app.services.instagram_service import instagram_api

load_dotenv()

access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
print("Loaded token from .env (repr):", repr(access_token))
if not access_token:
    print("Error: INSTAGRAM_ACCESS_TOKEN not set in .env")
    exit(1)

import asyncio
import httpx
import requests

async def test_instagram():
    try:
        # --- httpx test with User-Agent header ---
        url = f"https://graph.instagram.com/me?fields=id,username,account_type,media_count&access_token={access_token}"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
        print("httpx status:", response.status_code)
        print("httpx response:", response.text)

        # --- requests test ---
        r = requests.get(url, headers=headers, timeout=30)
        print("requests status:", r.status_code)
        print("requests response:", r.text)

        # --- original service test ---
        result = await instagram_api.get_user_info("me")
        print("Instagram API result:", result)
    except Exception as e:
        print("Error calling Instagram API:", e)

if __name__ == "__main__":
    asyncio.run(test_instagram()) 