# Admin Dashboard

A beautiful and modern admin dashboard for monitoring API usage, managing users, subscriptions, and API keys.

## Features

- 🚀 Real-time API usage monitoring
- 👥 User management
- 🔑 API key management
- 💳 Subscription management
- 📊 Beautiful statistics and charts
- 🎨 Modern UI with animations
- 🔄 Auto-refreshing data
- 🔒 Secure authentication

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/admin-dashboard.git
cd admin-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
SECRET_KEY=your-secret-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-secure-password
ENVIRONMENT=development
```

4. Initialize the database:
```bash
python simple_dashboard.py
```

### Running the Dashboard

```bash
uvicorn simple_dashboard:app --reload
```

The dashboard will be available at `http://localhost:8000`

## Using the API Client

The `APIClient` class in `api_client.py` provides an easy way to record API usage statistics.

### Basic Usage

```python
from api_client import APIClient

# Initialize client
client = APIClient('your-api-key')

# Record API usage
client.record_usage(
    endpoint='/api/data',
    user_id=123,
    response_time=150,
    success=True
)
```

### Using Context Manager

The client can automatically track response times using a context manager:

```python
with APIClient('your-api-key') as client:
    # Your API call here
    time.sleep(1)  # Simulate API call
    client.record_usage(
        endpoint='/api/users',
        user_id=456,
        response_time=client.last_response_time,
        success=True
    )
```

## Dashboard Features

### API Keys

- Create new API keys
- Revoke existing keys
- Regenerate keys
- View key creation dates

### User Management

- View active users
- Monitor user activity
- Track last active times
- User avatars and profiles

### Subscription Management

- View active subscriptions
- Cancel subscriptions
- Track expiration dates
- Monitor subscription status

### Statistics

- Total API requests
- Active users count
- Error rate monitoring
- Average response time
- Hourly usage trends
- Real-time updates

## Security

- JWT-based authentication
- Secure password hashing
- Protected API endpoints
- Cookie-based session management

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 