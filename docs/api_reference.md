# Instagram Stats API Reference

## API Versioning

The API uses semantic versioning and supports multiple versions. Current versions:

- v2 (Current): Released 2024-01-01
- v1 (Deprecated): Sunset date 2024-12-31

### Version Migration
When a version is deprecated:
1. A sunset date will be announced at least 6 months in advance
2. Warning headers will be included in responses
3. Documentation for migration will be provided

## Authentication

The API uses API key authentication. Keys must be included in the `X-API-Key` header.

### API Key Management

- Keys expire after 90 days
- Maximum 5 active keys per client
- Keys can be revoked at any time
- New keys should be requested before expiration

### Request Signing

All requests must be signed in production:

1. Include required headers:
   - `X-Timestamp`: Current Unix timestamp
   - `X-Nonce`: Random unique string
   - `X-Signature`: Request signature
   - `X-API-Key`: Your API key

2. Generate signature:
   ```python
   message = f"{timestamp}{nonce}{api_key}{request_body}"
   signature = hashlib.sha256(message.encode()).hexdigest()
   ```

## Rate Limiting

- 100 requests per hour per IP address
- Limit headers included in responses:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

When rate limit is exceeded:
```json
{
    "detail": "Rate limit exceeded. Try again in 3600 seconds"
}
```

## Endpoints

### GET /api/v2/platforms/instagram/users/{username}

Retrieve Instagram user statistics.

**Parameters:**
- `username` (path): Instagram username (alphanumeric, dots, and underscores only)

**Headers:**
- `X-API-Key`: Your API key
- `X-Timestamp`: Current Unix timestamp
- `X-Nonce`: Random unique string
- `X-Signature`: Request signature

**Response:**
```json
{
    "username": "example_user",
    "follower_count": 1000,
    "following_count": 500,
    "is_private": false,
    "post_count": 100
}
```

**Status Codes:**
- 200: Success
- 400: Invalid username format
- 401: Invalid API key or signature
- 404: User not found
- 429: Rate limit exceeded
- 503: Instagram API unavailable

### GET /api/v2/status

Get API status information.

**Headers:**
- `X-API-Key`: Your API key

**Response:**
```json
{
    "status": "operational",
    "version": "v2",
    "environment": "production",
    "instagram_api": "connected",
    "rate_limiting": "enabled",
    "caching": "enabled"
}
```

### GET /health

Health check endpoint (no authentication required).

**Response:**
```json
{
    "status": "healthy",
    "version": "v2"
}
```

## Error Responses

All error responses follow the format:
```json
{
    "detail": "Error message"
}
```

Common error codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error
- 503: Service Unavailable

## Caching

- Successful responses are cached for 5 minutes
- Cache can be bypassed with `Cache-Control: no-cache` header
- Cache headers included in responses:
  - `Cache-Control`
  - `ETag`
  - `Last-Modified`

## Best Practices

1. **Rate Limiting:**
   - Implement exponential backoff
   - Cache responses locally
   - Monitor rate limit headers

2. **Error Handling:**
   - Handle all error codes
   - Implement retry logic for 5xx errors
   - Log warning headers

3. **Security:**
   - Rotate API keys regularly
   - Keep API keys secure
   - Validate response signatures

4. **Performance:**
   - Use compression (gzip)
   - Implement caching
   - Monitor response times

## Support

- Email: api-support@example.com
- Status Page: https://status.example.com
- Documentation: https://docs.example.com 