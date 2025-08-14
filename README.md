# Reputation Sync API

A modern, secure, and scalable API for managing reputation systems with comprehensive monitoring and observability.

## Features

- 🔒 Secure authentication and authorization
- 📊 Real-time monitoring and metrics
- 🔄 Rate limiting and IP blocking
- 🚀 High performance with async support
- 📈 Prometheus metrics integration
- 📊 Grafana dashboards
- 🔔 AlertManager integration
- 🔍 Request validation and security
- 💾 Redis caching
- 🗄️ PostgreSQL database
- 📝 Comprehensive logging
- 🧪 Test coverage

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Prometheus
- Grafana
- AlertManager

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reputation_sync.git
cd reputation_sync
```

2. Create and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services using Docker Compose:
```bash
docker-compose up -d
```

4. Access the services:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- AlertManager: http://localhost:9093

## Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest
```

4. Run linting:
```bash
flake8
black .
isort .
mypy .
```

## API Documentation

The API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Monitoring

### Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Login with default credentials (admin/admin)
3. Navigate to Dashboards
4. Import the following dashboards:
   - Reputation System Overview
   - API Performance
   - Security Metrics

### Prometheus Metrics

Prometheus metrics are available at:
- Metrics endpoint: http://localhost:8000/metrics
- Prometheus UI: http://localhost:9090

### AlertManager

AlertManager is configured to send notifications for:
- High error rates
- Slow response times
- Security events
- System resource usage

## Security

The API implements several security measures:
- JWT authentication
- Rate limiting
- IP blocking
- Request validation
- Security headers
- CORS protection
- CSRF protection
- API key authentication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 