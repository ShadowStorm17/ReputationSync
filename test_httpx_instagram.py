from dotenv import load_dotenv
import os
import httpx

load_dotenv()
token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
url = f"https://graph.instagram.com/me?fields=id,username,account_type,media_count&access_token={token}"

print("--- Starting Instagram Graph API test ---")
print("Requesting:", url)

try:
    response = httpx.get(url)
    print("Status code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Exception occurred:", e)

print("--- End of test ---") 