# Monitoring Guide

This document describes the monitoring setup and alerting configuration for the Instagram Stats API.

## Metrics Overview

### Application Metrics

1. **Request Metrics**
   - Request rate
   - Response time
   - Error rate
   - Status code distribution
   - Endpoint usage

2. **Cache Metrics**
   - Hit rate
   - Miss rate
   - Eviction rate
   - Memory usage
   - Key expiration

3. **API Key Metrics**
   - Active keys
   - Key expiration
   - Usage per key
   - Rate limit hits
   - Authentication failures

### System Metrics

1. **Resource Usage**
   - CPU utilization
   - Memory usage
   - Disk space
   - Network I/O
   - File descriptors

2. **Redis Metrics**
   - Connected clients
   - Commands per second
   - Memory fragmentation
   - Eviction rate
   - Persistence status

3. **Nginx Metrics**
   - Active connections
   - Request rate
   - Error rate
   - SSL handshake time
   - Upstream response time

## Prometheus Configuration

### Scrape Configuration

```yaml
scrape_configs:
  - job_name: 'instagram_stats_api'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
    
  - job_name: 'node_exporter'
    scrape_interval: 30s
    static_configs:
      - targets: ['localhost:9100']
    
  - job_name: 'redis_exporter'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9121']
    
  - job_name: 'nginx_exporter'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9113']
```

### Recording Rules

```yaml
groups:
  - name: instagram_stats_api
    rules:
      - record: api:request_rate:5m
        expr: rate(http_requests_total[5m])
      
      - record: api:error_rate:5m
        expr: rate(http_errors_total[5m]) / rate(http_requests_total[5m])
      
      - record: api:response_time:5m
        expr: rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])
      
      - record: api:cache_hit_rate:5m
        expr: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))
```

## Grafana Dashboards

### Main Dashboard

1. **Request Overview**
   - QPS by endpoint
   - Error rate
   - P95/P99 latency
   - Status codes

2. **Cache Performance**
   - Hit rate by endpoint
   - Miss rate trend
   - Memory usage
   - Eviction rate

3. **System Health**
   - CPU/Memory/Disk usage
   - Network I/O
   - Process stats
   - Error logs

### API Key Dashboard

1. **Key Usage**
   - Requests per key
   - Rate limit usage
   - Key expiration timeline
   - Authentication errors

2. **Client Stats**
   - Active clients
   - Requests by client
   - Error rate by client
   - Cache hit rate by client

## Alert Rules

### Critical Alerts

```yaml
groups:
  - name: critical_alerts
    rules:
      - alert: HighErrorRate
        expr: api:error_rate:5m > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          description: Error rate is {{ $value | humanizePercentage }} (> 5%)

      - alert: HighResponseTime
        expr: api:response_time:5m > 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High response time
          description: Average response time is {{ $value }}s (> 1s)

      - alert: LowCacheHitRate
        expr: api:cache_hit_rate:5m < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: Low cache hit rate
          description: Cache hit rate is {{ $value | humanizePercentage }} (< 70%)
```

### System Alerts

```yaml
groups:
  - name: system_alerts
    rules:
      - alert: HighCPUUsage
        expr: avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High CPU usage
          description: CPU usage is {{ $value | humanizePercentage }}

      - alert: HighMemoryUsage
        expr: node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100 < 20
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Low memory available
          description: Only {{ $value | humanizePercentage }} memory available

      - alert: DiskSpaceLow
        expr: node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: Low disk space
          description: Only {{ $value | humanizePercentage }} disk space available
```

## Alert Channels

### Email Configuration

```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'alerts@example.com'
        from: 'monitoring@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'monitoring@example.com'
        auth_password: 'app_specific_password'
```

### Slack Configuration

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX'
        channel: '#monitoring'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
```

### PagerDuty Configuration

```yaml
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'your_pagerduty_service_key'
        description: '{{ .CommonAnnotations.description }}'
        severity: '{{ .CommonLabels.severity }}'
```

## Maintenance

### Log Rotation

```nginx
/var/log/instagram_api/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 instagram_api instagram_api
    postrotate
        systemctl reload instagram_api
    endscript
}
```

### Backup Verification

```bash
# Verify Prometheus data
promtool tsdb verify /var/lib/prometheus/data

# Verify Grafana backups
grafana-cli admin verify-backup /backup/grafana
```

### Dashboard Updates

1. Export dashboards:
```bash
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:3000/api/dashboards/uid/$DASHBOARD_UID > dashboard.json
```

2. Version control:
```bash
git add config/grafana/dashboards/
git commit -m "Update monitoring dashboards"
```

## Best Practices

1. **Alert Configuration**
   - Set appropriate thresholds
   - Add clear descriptions
   - Include runbooks
   - Test alert channels

2. **Dashboard Organization**
   - Consistent naming
   - Logical grouping
   - Clear visualizations
   - Useful time ranges

3. **Metric Collection**
   - Use labels effectively
   - Follow naming conventions
   - Set retention policies
   - Monitor cardinality

4. **Performance Impact**
   - Optimize scrape intervals
   - Use recording rules
   - Monitor resource usage
   - Clean old data

## Support

For monitoring issues:
- Email: monitoring@example.com
- Slack: #monitoring-alerts
- Documentation: https://docs.example.com/monitoring 