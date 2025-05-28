# Production Deployment Guide

## Prerequisites

- Linux server (Ubuntu 20.04 LTS recommended)
- Python 3.8+
- Redis 6+
- Nginx
- SSL certificate
- Domain name
- Meta/Facebook Developer account

## 1. Initial Server Setup

### System Updates
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3.8 python3.8-venv python3-pip nginx redis-server
```

### Create Service User
```bash
sudo useradd -m -s /bin/bash instagram_api
sudo usermod -aG sudo instagram_api
```

## 2. Application Setup

### Clone Repository
```bash
cd /opt
sudo git clone <repository-url> instagram_stats_api
sudo chown -R instagram_api:instagram_api instagram_stats_api
```

### Python Environment
```bash
cd instagram_stats_api
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Configuration

### Environment Variables
Create `.env` file:
```bash
# API Settings
API_VERSION=v2
ENVIRONMENT=production
DEBUG=False
PROJECT_NAME="Instagram Stats API"

# Security
SECRET_KEY=<generated-secret-key>
SSL_KEYFILE=/etc/letsencrypt/live/api.yourdomain.com/privkey.pem
SSL_CERTFILE=/etc/letsencrypt/live/api.yourdomain.com/fullchain.pem

# Instagram Graph API
INSTAGRAM_APP_ID=your-app-id
INSTAGRAM_APP_SECRET=your-app-secret
INSTAGRAM_ACCESS_TOKEN=your-access-token
INSTAGRAM_API_VERSION=v16.0

# Redis
REDIS_URL=redis://localhost:6379

# CORS
CORS_ORIGINS=["https://yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET", "POST", "OPTIONS"]
CORS_ALLOW_HEADERS=["*"]

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENABLED=true
TRACING_ENABLED=true

# Logging
LOG_LEVEL=INFO
```

### Redis Configuration
Edit `/etc/redis/redis.conf`:
```
maxmemory 256mb
maxmemory-policy allkeys-lru
supervised systemd
```

Restart Redis:
```bash
sudo systemctl restart redis
```

## 4. SSL Setup

### Install Certbot
```bash
sudo apt install certbot python3-certbot-nginx
```

### Get SSL Certificate
```bash
sudo certbot --nginx -d api.yourdomain.com
```

## 5. Nginx Configuration

Create `/etc/nginx/sites-available/instagram_api`:
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

upstream api_server {
    server unix:/run/instagram_api.sock fail_timeout=0;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'none'; frame-ancestors 'none'" always;
    
    # Logging
    access_log /var/log/nginx/api_access.log combined buffer=512k flush=1m;
    error_log /var/log/nginx/api_error.log warn;
    
    location / {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_pass http://api_server;
        proxy_redirect off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Prometheus metrics
    location /metrics {
        auth_basic "Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://api_server;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/instagram_api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 6. Systemd Service

Create `/etc/systemd/system/instagram_api.service`:
```ini
[Unit]
Description=Instagram Stats API
After=network.target

[Service]
User=instagram_api
Group=instagram_api
WorkingDirectory=/opt/instagram_stats_api
Environment="PATH=/opt/instagram_stats_api/venv/bin"
ExecStart=/opt/instagram_stats_api/venv/bin/gunicorn -c gunicorn_conf.py app.main:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl enable instagram_api
sudo systemctl start instagram_api
```

## 7. Monitoring Setup

### Prometheus
Install Prometheus:
```bash
sudo apt install prometheus
```

Edit `/etc/prometheus/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'instagram_stats_api'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana
Install Grafana:
```bash
sudo apt install grafana
```

Configure dashboards for:
- Request rates
- Response times
- Error rates
- Cache hit rates
- API key usage

## 8. Backup Setup

### Database Backups
Create backup script `/opt/instagram_stats_api/scripts/backup.sh`:
```bash
#!/bin/bash
BACKUP_DIR="/backup/redis"
DATE=$(date +%Y%m%d_%H%M%S)
redis-cli save
cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis_$DATE.rdb
find $BACKUP_DIR -type f -mtime +7 -delete
```

Add to crontab:
```bash
0 */6 * * * /opt/instagram_stats_api/scripts/backup.sh
```

## 9. Security Checklist

- [ ] Firewall configured (UFW)
- [ ] Fail2ban installed
- [ ] Regular security updates enabled
- [ ] SSL certificate auto-renewal configured
- [ ] API keys rotated regularly
- [ ] Monitoring alerts configured
- [ ] Backup verification process established
- [ ] Access logs monitored
- [ ] Rate limiting tested
- [ ] DDoS protection configured

## 10. Maintenance

### Log Rotation
Configure `/etc/logrotate.d/instagram_api`:
```
/var/log/instagram_api/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 instagram_api instagram_api
    sharedscripts
    postrotate
        systemctl reload instagram_api
    endscript
}
```

### Monitoring Alerts
Set up alerts for:
- High error rates
- High response times
- Low cache hit rates
- Certificate expiration
- Disk space usage
- Memory usage
- CPU usage

### Regular Maintenance Tasks
- Weekly: Review logs
- Monthly: Update dependencies
- Monthly: Rotate API keys
- Quarterly: SSL certificate check
- Quarterly: Security audit
- Yearly: Infrastructure review 