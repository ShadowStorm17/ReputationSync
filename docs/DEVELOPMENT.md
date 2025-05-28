# Development Guide

This guide provides instructions and best practices for developing the Instagram Stats API.

## Development Setup

### Prerequisites

1. **System Requirements**
   - Python 3.9+
   - Redis 6.0+
   - Git
   - Docker (optional)

2. **Development Tools**
   - Visual Studio Code or PyCharm
   - Python extension
   - Docker extension
   - REST Client

### Local Environment

1. **Clone Repository**
```bash
git clone https://github.com/example/instagram-stats-api.git
cd instagram-stats-api
```

2. **Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Environment Variables**
```bash
cp .env.example .env
# Edit .env with your development settings
```

5. **Start Services**
```bash
# Start Redis
docker run -d -p 6379:6379 redis:6

# Start API
uvicorn app.main:app --reload
```

## Code Structure

```
instagram_stats_api/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   ├── models/
│   │   │   └── dependencies.py
│   │   └── router.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── cache.py
│   ├── services/
│   │   ├── instagram.py
│   │   └── metrics.py
│   └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── locust/
├── scripts/
├── config/
└── docs/
```

## Development Workflow

### 1. Git Workflow

```bash
# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git add .
git commit -m "feat: add amazing feature"

# Push changes
git push origin feature/amazing-feature

# Create pull request
gh pr create
```

### 2. Code Style

1. **Formatting**
```bash
# Format code
black app tests

# Sort imports
isort app tests
```

2. **Linting**
```bash
# Check style
flake8 app tests

# Type checking
mypy app
```

3. **Style Guide**
- Follow PEP 8
- Use type hints
- Write docstrings
- Keep functions small

### 3. Testing

1. **Unit Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/unit/test_api.py -k "test_function_name"
```

2. **Integration Tests**
```bash
# Run integration tests
pytest tests/integration/

# Run with real API
REAL_API=1 pytest tests/integration/
```

3. **Load Testing**
```bash
# Run Locust tests
locust -f tests/locust/locustfile.py
```

### 4. Documentation

1. **API Documentation**
- Update OpenAPI specs
- Add endpoint descriptions
- Include request/response examples
- Document error cases

2. **Code Documentation**
- Write clear docstrings
- Add inline comments
- Update README
- Create changelogs

## API Development

### 1. Adding New Endpoints

1. **Create Route**
```python
from fastapi import APIRouter, Depends
from app.core.security import get_api_key

router = APIRouter()

@router.get("/new-endpoint")
async def new_endpoint(
    api_key: str = Depends(get_api_key)
):
    """
    New endpoint description.
    """
    return {"message": "success"}
```

2. **Add Models**
```python
from pydantic import BaseModel, Field

class NewRequest(BaseModel):
    field: str = Field(..., description="Field description")

class NewResponse(BaseModel):
    message: str
    data: dict
```

3. **Add Service Layer**
```python
from app.services.base import BaseService

class NewService(BaseService):
    async def process_request(self, data: dict) -> dict:
        # Process data
        return result
```

### 2. Error Handling

```python
from fastapi import HTTPException
from app.core.exceptions import APIError

async def endpoint():
    try:
        result = await process_data()
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": str(e)}
        )
```

### 3. Caching

```python
from app.core.cache import cache

@cache(ttl=300)  # 5 minutes
async def cached_function(key: str) -> dict:
    # Expensive operation
    return data
```

### 4. Rate Limiting

```python
from app.core.security import rate_limit

@rate_limit(limit=100, period=60)
async def rate_limited_endpoint():
    return {"message": "success"}
```

## Monitoring Development

### 1. Adding Metrics

```python
from app.core.metrics import metrics

# Counter
request_count = metrics.counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint"]
)

# Histogram
response_time = metrics.histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["endpoint"]
)
```

### 2. Logging

```python
import logging

logger = logging.getLogger(__name__)

async def function():
    try:
        result = await process()
        logger.info("Process completed", extra={
            "result": result,
            "duration": duration
        })
    except Exception as e:
        logger.error("Process failed", exc_info=e)
```

## Deployment

### 1. Local Testing

```bash
# Build Docker image
docker build -t instagram-stats-api .

# Run container
docker run -p 8000:8000 instagram-stats-api
```

### 2. Staging Deployment

```bash
# Deploy to staging
./scripts/deploy.sh staging

# Run migrations
./scripts/migrate.sh staging

# Verify deployment
./scripts/verify.sh staging
```

## Best Practices

### 1. Code Quality

- Write self-documenting code
- Follow SOLID principles
- Use dependency injection
- Keep functions pure
- Write comprehensive tests

### 2. Security

- Validate all inputs
- Sanitize outputs
- Use parameterized queries
- Follow least privilege
- Handle secrets properly

### 3. Performance

- Use async where appropriate
- Implement caching
- Optimize database queries
- Profile code regularly
- Monitor memory usage

### 4. Maintenance

- Keep dependencies updated
- Remove dead code
- Refactor regularly
- Document technical debt
- Update documentation

## Troubleshooting

### 1. Development Issues

```bash
# Clear cache
redis-cli flushall

# Reset database
./scripts/reset_db.sh

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
```

### 2. Common Problems

1. **Redis Connection**
```python
# Test connection
import redis
r = redis.Redis()
r.ping()
```

2. **API Key Issues**
```bash
# Generate new key
./scripts/generate_key.sh

# Verify key
./scripts/verify_key.sh YOUR_KEY
```

## Support

For development support:
- Email: dev-support@example.com
- Slack: #api-development
- Wiki: https://wiki.example.com/instagram-stats-api 