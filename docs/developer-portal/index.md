# Instagram Stats API Developer Portal

Welcome to the Instagram Stats API developer portal. This comprehensive guide will help you integrate Instagram statistics into your applications.

## Quick Links
- [Getting Started](#getting-started)
- [API Reference](../API.md)
- [SDKs & Libraries](#sdks)
- [Support](#support)
- [Status Page](https://status.example.com)

## Getting Started

### 1. Register Your Application
1. Create an account at [Developer Dashboard](https://developers.example.com)
2. Create a new application
3. Get your API credentials (client_id and client_secret)

### 2. Choose Your Plan

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Rate Limit | 60 rpm | 300 rpm | 1000 rpm |
| Support | Community | Email | 24/7 |
| SLA | None | 99.9% | 99.99% |
| Price | $0 | $49/mo | Custom |

### 3. Authentication

```python
import requests
import hmac
import hashlib
import time

def get_api_key(client_id, client_secret):
    response = requests.post(
        "https://api.example.com/api/v1/auth/token",
        json={
            "client_id": client_id,
            "client_secret": client_secret
        }
    )
    return response.json()["access_token"]

def sign_request(method, path, api_key, body=None):
    timestamp = str(int(time.time()))
    message = f"{method}\n{path}\n{timestamp}\n{body or ''}"
    signature = hmac.new(
        key=api_key.encode(),
        msg=message.encode(),
        digestmod=hashlib.sha256
    ).hexdigest()
    
    return {
        "Authorization": f"Bearer {api_key}",
        "X-Request-Timestamp": timestamp,
        "X-Request-Signature": signature
    }
```

### 4. Make Your First Request

```python
# Get profile statistics
api_key = get_api_key(CLIENT_ID, CLIENT_SECRET)
headers = sign_request("GET", "/api/v1/stats/profile", api_key)

response = requests.get(
    "https://api.example.com/api/v1/stats/profile",
    headers=headers
)
print(response.json())
```

## SDKs

Official SDKs for popular languages:
- [Python SDK](https://github.com/example/instagram-stats-python)
- [JavaScript SDK](https://github.com/example/instagram-stats-js)
- [Java SDK](https://github.com/example/instagram-stats-java)
- [PHP SDK](https://github.com/example/instagram-stats-php)
- [Ruby SDK](https://github.com/example/instagram-stats-ruby)

## Rate Limits

Each endpoint has specific rate limit costs:

| Endpoint | Cost per Request |
|----------|------------------|
| /stats/profile | 1 |
| /stats/posts | 2 |
| /stats/engagement | 2 |
| /stats/audience | 3 |

## Support

- [Documentation](https://docs.example.com)
- [API Status](https://status.example.com)
- [GitHub Issues](https://github.com/example/instagram-stats-api/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/instagram-stats-api)
- Email: api-support@example.com

## Legal
- [Terms of Service](legal/terms.md)
- [Privacy Policy](legal/privacy.md)
- [Data Processing Agreement](legal/dpa.md) 