#!/usr/bin/env python3
"""
Monitoring script for Instagram Stats API.
This script will:
1. Check system health metrics
2. Monitor API performance
3. Check SSL certificate expiration
4. Monitor disk space
5. Send alerts when thresholds are exceeded
"""

import os
import sys
import time
import logging
import asyncio
import ssl
import socket
import psutil
import aiohttp
import aiosmtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List
from prometheus_client.parser import text_string_to_metric_families

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitoring.log')
    ]
)

class AlertManager:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "monitoring@example.com")
        self.alert_emails = os.getenv("ALERT_EMAILS", "admin@example.com").split(",")
        
        # Alert thresholds
        self.thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "response_time": 1000,  # 1 second
            "disk_usage": 90,  # 90% disk usage
            "memory_usage": 90,  # 90% memory usage
            "cpu_usage": 80,  # 80% CPU usage
            "ssl_expiry_days": 30,  # 30 days before expiration
            "cache_hit_rate": 0.7,  # 70% cache hit rate
        }
    
    async def send_alert(self, subject: str, message: str, priority: str = "normal") -> None:
        """Send alert email."""
        try:
            email = EmailMessage()
            email["From"] = self.from_email
            email["To"] = ", ".join(self.alert_emails)
            email["Subject"] = f"[{priority.upper()}] {subject}"
            email.set_content(message)
            
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=True
            ) as smtp:
                await smtp.login(self.smtp_user, self.smtp_password)
                await smtp.send_message(email)
            
            logger.info("Sent alert: %s", subject)
        except Exception as e:
            logger.error("Failed to send alert: %s", str(e))

class MonitoringManager:
    def __init__(self):
        self.alert_manager = AlertManager()
        self.api_url = "http://localhost:8000"  # Update for production
        self.metrics_url = f"{self.api_url}/metrics"
        self.health_url = f"{self.api_url}/health"
        self.ssl_cert_path = settings.SSL_CERTFILE
    
    async def check_api_health(self) -> None:
        """Check API health status."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(self.health_url) as response:
                    if response.status != 200:
                        await self.alert_manager.send_alert(
                            "API Health Check Failed",
                            f"API health check returned status {response.status}",
                            "high"
                        )
        except Exception as e:
            await self.alert_manager.send_alert(
                "API Health Check Failed",
                f"Failed to connect to API: {str(e)}",
                "critical"
            )
    
    async def check_metrics(self) -> None:
        """Check Prometheus metrics."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(self.metrics_url) as response:
                    if response.status == 200:
                        text = await response.text()
                        metrics = {
                            metric.name: list(metric.samples)[0].value
                            for metric in text_string_to_metric_families(text)
                        }
                        
                        # Check error rate
                        total_requests = metrics.get("http_requests_total", 0)
                        error_requests = metrics.get("http_errors_total", 0)
                        if total_requests > 0:
                            error_rate = error_requests / total_requests
                            if error_rate > self.alert_manager.thresholds["error_rate"]:
                                await self.alert_manager.send_alert(
                                    "High Error Rate",
                                    f"Error rate is {error_rate:.2%}",
                                    "high"
                                )
                        
                        # Check response time
                        avg_response_time = metrics.get("http_request_duration_seconds", 0) * 1000
                        if avg_response_time > self.alert_manager.thresholds["response_time"]:
                            await self.alert_manager.send_alert(
                                "High Response Time",
                                f"Average response time is {avg_response_time:.2f}ms",
                                "high"
                            )
                        
                        # Check cache hit rate
                        cache_hits = metrics.get("cache_hits_total", 0)
                        cache_misses = metrics.get("cache_misses_total", 0)
                        if cache_hits + cache_misses > 0:
                            hit_rate = cache_hits / (cache_hits + cache_misses)
                            if hit_rate < self.alert_manager.thresholds["cache_hit_rate"]:
                                await self.alert_manager.send_alert(
                                    "Low Cache Hit Rate",
                                    f"Cache hit rate is {hit_rate:.2%}",
                                    "normal"
                                )
        except Exception as e:
            logger.error("Failed to check metrics: %s", str(e))
    
    def check_ssl_certificate(self) -> None:
        """Check SSL certificate expiration."""
        if not self.ssl_cert_path:
            return
        
        try:
            cert = ssl.get_server_certificate(('localhost', 443))
            x509 = ssl.PEM_cert_to_DER_cert(cert)
            expiry = ssl.cert_time_to_seconds(x509.get_notAfter())
            days_remaining = (expiry - time.time()) / (24 * 3600)
            
            if days_remaining <= self.alert_manager.thresholds["ssl_expiry_days"]:
                asyncio.create_task(self.alert_manager.send_alert(
                    "SSL Certificate Expiring Soon",
                    f"SSL certificate will expire in {int(days_remaining)} days",
                    "high"
                ))
        except Exception as e:
            logger.error("Failed to check SSL certificate: %s", str(e))
    
    def check_system_resources(self) -> None:
        """Check system resource usage."""
        try:
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            if disk_percent > self.alert_manager.thresholds["disk_usage"]:
                asyncio.create_task(self.alert_manager.send_alert(
                    "High Disk Usage",
                    f"Disk usage is at {disk_percent}%",
                    "high"
                ))
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.alert_manager.thresholds["memory_usage"]:
                asyncio.create_task(self.alert_manager.send_alert(
                    "High Memory Usage",
                    f"Memory usage is at {memory.percent}%",
                    "high"
                ))
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.alert_manager.thresholds["cpu_usage"]:
                asyncio.create_task(self.alert_manager.send_alert(
                    "High CPU Usage",
                    f"CPU usage is at {cpu_percent}%",
                    "high"
                ))
        except Exception as e:
            logger.error("Failed to check system resources: %s", str(e))

async def main():
    """Main monitoring process."""
    monitor = MonitoringManager()
    logger.info("Starting monitoring process")
    
    while True:
        try:
            # API health and metrics
            await monitor.check_api_health()
            await monitor.check_metrics()
            
            # System checks
            monitor.check_ssl_certificate()
            monitor.check_system_resources()
            
            # Wait before next check
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error("Monitoring process error: %s", str(e))
            await asyncio.sleep(60)  # Wait before retry

if __name__ == "__main__":
    asyncio.run(main()) 