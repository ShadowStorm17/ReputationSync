# Admin Dashboard

A modern, secure admin dashboard built with FastAPI and Vue.js for monitoring API usage and statistics.

## Features

- 🔐 Secure authentication system
- 📊 Real-time statistics and charts
- 🌙 Modern dark theme interface
- 📱 Responsive design
- 🔄 Live data updates
- 📈 API usage monitoring
- 🔍 Detailed analytics

## Tech Stack

- Backend: FastAPI
- Frontend: Vue.js, TailwindCSS
- Database: SQLite
- Authentication: JWT
- Monitoring: Prometheus & OpenTelemetry

## Quick Start

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
# Edit .env with your settings
```

4. Run the application:
```bash
python simple_dashboard.py
```

5. Access the dashboard at: http://localhost:8000

## Production Deployment

This application is configured for deployment on Render.com. See deployment instructions in the documentation.

## Default Credentials

- Username: admin
- Password: admin123

**Note:** Change these credentials in production by setting environment variables.

## License

MIT License 