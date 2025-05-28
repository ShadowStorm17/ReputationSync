# Instagram Stats API Reference

## API Versioning

The API uses semantic versioning (MAJOR.MINOR.PATCH) and is currently at v1. Breaking changes will only be introduced in MAJOR version updates.

- Current version: `v1`
- Base URL: `https://api.example.com/api/v1`
- Deprecation policy: 6 months notice before removing old versions

## Authentication

All API requests require authentication using an API key.

### Request Headers

```
Authorization: Bearer YOUR_API_KEY
X-Request-Timestamp: 1679395200
X-Request-Signature: HMAC_SHA256_SIGNATURE
```

### API Key Management

- Keys expire after 90 days
- Maximum 5 active keys per client
- 14-day grace period after key rotation
- Email notification 14 days before expiration

## Rate Limiting

| Tier | Requests/Minute | Burst | Monthly Limit |
|------|----------------|--------|---------------|
| Free | 60 | 10 | 10,000 |
| Pro | 300 | 50 | 100,000 |
| Enterprise | 1000 | 100 | Unlimited |

Rate limit headers in response:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1679395260
```

## Endpoints

### Authentication

#### Get Access Token

```http
POST /auth/token
Content-Type: application/json

{
    "client_id": "your_client_id",
    "client_secret": "your_client_secret"
}
```

Response:
```json
{
    "access_token": "your_api_key",
    "expires_at": "2024-06-19T10:00:00Z",
    "token_type": "Bearer"
}
```

#### Refresh Token

```http
GET /auth/refresh
Authorization: Bearer YOUR_API_KEY
```

Response:
```json
{
    "access_token": "your_new_api_key",
    "expires_at": "2024-06-19T10:00:00Z",
    "token_type": "Bearer"
}
```

### Instagram Stats

#### Get Profile Statistics

```http
GET /stats/profile
Authorization: Bearer YOUR_API_KEY
```

Response:
```json
{
    "followers_count": 10000,
    "following_count": 500,
    "media_count": 100,
    "engagement_rate": 2.5,
    "profile_views": 1500,
    "website_clicks": 200,
    "email_clicks": 50,
    "updated_at": "2024-03-21T10:00:00Z"
}
```

#### Get Post Statistics

```http
GET /stats/posts
Authorization: Bearer YOUR_API_KEY
Query Parameters:
  - start_date (YYYY-MM-DD)
  - end_date (YYYY-MM-DD)
  - limit (default: 10, max: 100)
  - offset (default: 0)
```

Response:
```json
{
    "posts": [
        {
            "id": "post_id",
            "type": "image",
            "caption": "Post caption",
            "likes": 500,
            "comments": 50,
            "saves": 25,
            "shares": 10,
            "reach": 5000,
            "impressions": 6000,
            "engagement_rate": 3.2,
            "posted_at": "2024-03-20T15:30:00Z"
        }
    ],
    "pagination": {
        "total": 100,
        "limit": 10,
        "offset": 0,
        "next": "/api/v1/stats/posts?limit=10&offset=10"
    }
}
```

#### Get Engagement Statistics

```http
GET /stats/engagement
Authorization: Bearer YOUR_API_KEY
Query Parameters:
  - period (day|week|month, default: day)
  - start_date (YYYY-MM-DD)
  - end_date (YYYY-MM-DD)
```

Response:
```json
{
    "total_engagement": 5000,
    "engagement_rate": 2.8,
    "breakdown": {
        "likes": 4000,
        "comments": 600,
        "saves": 300,
        "shares": 100
    },
    "trend": [
        {
            "date": "2024-03-20",
            "engagement_rate": 2.9,
            "total_engagement": 450
        }
    ]
}
```

#### Get Audience Statistics

```http
GET /stats/audience
Authorization: Bearer YOUR_API_KEY
```

Response:
```json
{
    "total_followers": 10000,
    "growth_rate": 1.5,
    "demographics": {
        "age_ranges": {
            "13-17": 5,
            "18-24": 25,
            "25-34": 40,
            "35-44": 20,
            "45-54": 7,
            "55+": 3
        },
        "gender": {
            "male": 48,
            "female": 51,
            "other": 1
        },
        "top_locations": [
            {
                "country": "US",
                "percentage": 35
            }
        ]
    },
    "online_times": {
        "most_active_day": "Wednesday",
        "most_active_hour": 18,
        "hourly_activity": [
            {
                "hour": 0,
                "percentage": 2.5
            }
        ]
    }
}
```

### Monitoring

#### Health Check

```http
GET /health
```

Response:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-03-21T10:00:00Z",
    "services": {
        "redis": "healthy",
        "instagram_api": "healthy"
    }
}
```

#### Metrics

```http
GET /metrics
Authorization: Bearer YOUR_API_KEY
```

Response: Prometheus format metrics

## Error Handling

### Error Response Format

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": {
            "field": "Additional error context"
        }
    }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| INVALID_CREDENTIALS | Invalid API key or signature | 401 |
| RATE_LIMIT_EXCEEDED | Too many requests | 429 |
| INVALID_PARAMETERS | Invalid request parameters | 400 |
| NOT_FOUND | Resource not found | 404 |
| INTERNAL_ERROR | Server error | 500 |
| MAINTENANCE | Service under maintenance | 503 |

### Rate Limit Exceeded Response

```json
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Rate limit exceeded. Try again in 60 seconds.",
        "details": {
            "limit": 60,
            "remaining": 0,
            "reset": "2024-03-21T10:01:00Z",
            "retry_after": 60
        }
    }
}
```

## Best Practices

1. **Caching**
   - Cache responses when possible
   - Honor Cache-Control headers
   - Use ETags for cache validation

2. **Rate Limiting**
   - Implement exponential backoff
   - Monitor rate limit headers
   - Cache responses to reduce API calls

3. **Error Handling**
   - Handle all error codes
   - Implement retry logic with backoff
   - Log and monitor errors

4. **Security**
   - Keep API keys secure
   - Rotate keys regularly
   - Use HTTPS for all requests
   - Validate request signatures

## Support

For API support:
- Email: api-support@example.com
- Status page: https://status.example.com
- Documentation: https://docs.example.com/instagram-stats-api 