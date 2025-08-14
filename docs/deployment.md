# Instagram Stats API Deployment Guide

This guide provides step-by-step instructions for deploying the Instagram Stats API in a production environment.

## Prerequisites

- Ubuntu 20.04 LTS or newer
- Python 3.9+
- Redis 6.0+
- Nginx
- Certbot (for SSL)
- Prometheus
- Grafana

## System Setup

1. Create service user:
```bash
sudo useradd -r -s /bin/false instagram_api
sudo mkdir -p /opt/instagram_stats_api
sudo chown instagram_api:instagram_api /opt/instagram_stats_api
```

2. Install system dependencies:
```bash
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev redis-server nginx certbot python3-certbot-nginx
```

3. Set up Python virtual environment:
```bash
cd /opt/instagram_stats_api
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

1. Set up environment variables:
```bash
sudo mkdir /etc/instagram_api
sudo cp .env.example /etc/instagram_api/.env
sudo chown instagram_api:instagram_api /etc/instagram_api/.env
sudo chmod 600 /etc/instagram_api/.env
```

2. Configure Redis:
```bash
sudo cp config/redis/redis.conf /etc/redis/redis.conf
sudo systemctl restart redis
```

3. Set up Nginx:
```bash
sudo cp config/nginx/instagram_api.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/instagram_api.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

4. Configure SSL:
```bash
sudo certbot --nginx -d api.example.com
```

## Monitoring Setup

1. Install Prometheus:
```bash
sudo apt install -y prometheus
sudo cp config/prometheus/prometheus.yml /etc/prometheus/
sudo systemctl restart prometheus
```

2. Install Grafana:
```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

3. Import Grafana dashboard:
```bash
sudo cp config/grafana/instagram_api_dashboard.json /var/lib/grafana/dashboards/
```

## Service Setup

1. Set up systemd services:
```bash
sudo cp config/systemd/instagram_api.service /etc/systemd/system/
sudo cp config/systemd/instagram_api_monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable instagram_api instagram_api_monitor
sudo systemctl start instagram_api instagram_api_monitor
```

2. Configure log rotation:
```bash
sudo mkdir -p /var/log/instagram_api
sudo chown instagram_api:instagram_api /var/log/instagram_api
sudo cp config/logrotate.d/instagram_api /etc/logrotate.d/
```

3. Set up cron jobs:
```bash
sudo cp config/crontab /etc/cron.d/instagram_api
sudo chmod 644 /etc/cron.d/instagram_api
```

## Backup Configuration

1. Create backup directories:
```bash
sudo mkdir -p /opt/instagram_stats_api/backups
sudo chown instagram_api:instagram_api /opt/instagram_stats_api/backups
```

2. Configure remote storage (optional):
```bash
echo "REMOTE_STORAGE_URL=https://storage.example.com" >> /etc/instagram_api/.env
echo "REMOTE_STORAGE_KEY=your_secret_key" >> /etc/instagram_api/.env
```

## Security Considerations

1. Firewall configuration:
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

2. API key rotation:
- API keys are automatically rotated every 90 days
- Old keys are kept valid for 14 days after rotation
- Clients are notified via email about new keys

3. Rate limiting:
- Configured in Nginx (see `config/nginx/instagram_api.conf`)
- Default: 100 requests per minute per IP

4. SSL/TLS:
- Only TLS 1.2 and 1.3 are enabled
- Strong cipher suites only
- HSTS enabled
- Certificate auto-renewal via certbot

## Monitoring and Alerts

1. Key metrics monitored:
- Request rate and latency
- Error rate
- Cache hit rate
- System resources (CPU, memory, disk)
- SSL certificate expiration
- API key expiration

2. Alert thresholds:
- Error rate > 5%
- Response time > 1000ms
- Disk usage > 90%
- Memory usage > 90%
- CPU usage > 80%
- SSL certificate < 30 days to expiry
- Cache hit rate < 70%

3. Alert channels:
- Email notifications
- Slack webhook (if configured)
- PagerDuty (if configured)

## Maintenance Procedures

1. Updating the application:
```bash
cd /opt/instagram_stats_api
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart instagram_api
```

2. Backup verification:
```bash
python scripts/backup.py --verify
```

3. Log management:
- Logs are rotated weekly
- Compressed logs are kept for 4 weeks
- Use `journalctl -u instagram_api` for service logs

4. Database maintenance:
```bash
# Trigger Redis persistence
redis-cli save

# Compact Redis AOF file
redis-cli bgrewriteaof
```

## Troubleshooting

1. Check service status:
```bash
sudo systemctl status instagram_api
sudo systemctl status instagram_api_monitor
```

2. View logs:
```bash
tail -f /var/log/instagram_api/api.log
tail -f /var/log/instagram_api/monitor.log
```

3. Check metrics:
```bash
curl http://localhost:8000/metrics
```

4. Test API health:
```bash
curl https://api.example.com/health
```

## Support

For issues or assistance:
- Email: support@example.com
- Documentation: https://docs.example.com/instagram-stats-api
- GitHub: https://github.com/example/instagram-stats-api 