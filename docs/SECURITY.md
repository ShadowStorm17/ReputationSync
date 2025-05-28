# Security Guide

This document outlines the security measures and best practices implemented in the Instagram Stats API.

## Authentication & Authorization

### API Key Management

1. **Key Generation**
   - Keys are 256-bit cryptographically secure random strings
   - Format: `ist_<random_string>`
   - Each key is associated with a specific client and tier

2. **Key Lifecycle**
   - Validity: 90 days from creation
   - Grace period: 14 days after expiration
   - Maximum active keys per client: 5
   - Automatic rotation with email notifications

3. **Key Storage**
   - Keys are hashed using Argon2id
   - Salt is unique per key
   - Original keys are never logged or stored

### Request Signing

1. **Required Headers**
```
Authorization: Bearer YOUR_API_KEY
X-Request-Timestamp: UNIX_TIMESTAMP
X-Request-Signature: HMAC_SIGNATURE
```

2. **Signature Generation**
```python
message = f"{request_method}\n{request_path}\n{timestamp}\n{request_body}"
signature = hmac.new(
    key=api_key.encode(),
    msg=message.encode(),
    digestmod=hashlib.sha256
).hexdigest()
```

3. **Timestamp Validation**
- Must be within ±300 seconds of server time
- Prevents replay attacks

## Rate Limiting

### Tier-based Limits

| Tier | RPM | Burst | Monthly |
|------|-----|-------|---------|
| Free | 60 | 10 | 10,000 |
| Pro | 300 | 50 | 100,000 |
| Enterprise | 1000 | 100 | Unlimited |

### Implementation

1. **Nginx Layer**
```nginx
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=5r/s;
limit_req_zone $binary_remote_addr zone=stats_limit:10m rate=30r/s;
```

2. **Application Layer**
- Redis-based sliding window counter
- Per-client and per-endpoint tracking
- Automatic ban for repeated abuse

## Network Security

### SSL/TLS Configuration

1. **Protocol Support**
- TLS 1.2 and 1.3 only
- Older versions disabled

2. **Cipher Suites**
```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
ssl_prefer_server_ciphers off;
```

3. **Certificate Management**
- Auto-renewal via Certbot
- OCSP stapling enabled
- Minimum 2048-bit key length

### Security Headers

```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Content-Security-Policy "default-src 'none'; frame-ancestors 'none'" always;
```

## Data Protection

### Personal Data Handling

1. **Data Minimization**
- Only essential data collected
- Automatic data expiration
- Clear data retention policies

2. **Data Encryption**
- At rest: AES-256
- In transit: TLS 1.2+
- Key rotation every 90 days

3. **Access Control**
- Role-based access control
- Audit logging
- Regular access reviews

### Logging & Monitoring

1. **Security Logs**
- Authentication attempts
- Rate limit violations
- Access patterns
- System changes

2. **Log Protection**
- Encrypted storage
- 90-day retention
- Tamper detection

3. **Alerts**
- Unusual access patterns
- Multiple failed authentications
- Rate limit breaches
- System anomalies

## Incident Response

### Security Incidents

1. **Classification**
- P0: System compromise
- P1: Data breach
- P2: DDoS attack
- P3: Minor violation

2. **Response Time**
- P0: Immediate (15 minutes)
- P1: 1 hour
- P2: 4 hours
- P3: 24 hours

3. **Notification Process**
- Internal team alert
- Client notification
- Public disclosure (if required)
- Post-mortem report

### Recovery Procedures

1. **System Compromise**
```bash
# 1. Isolate affected systems
ufw deny all

# 2. Rotate all secrets
./scripts/rotate_all_keys.sh

# 3. Verify system integrity
./scripts/verify_integrity.sh

# 4. Restore from clean backup
./scripts/restore_backup.sh

# 5. Re-enable access
ufw reset
```

2. **DDoS Mitigation**
- Rate limiting adjustment
- Traffic filtering
- CDN failover
- Client notification

## Security Testing

### Regular Tests

1. **Automated Scanning**
- Daily vulnerability scans
- Weekly dependency checks
- Monthly penetration tests

2. **Manual Reviews**
- Code security reviews
- Configuration audits
- Access control verification

3. **Compliance Checks**
- OWASP Top 10
- GDPR requirements
- Industry standards

### Vulnerability Management

1. **Severity Levels**
- Critical: Fix within 24 hours
- High: Fix within 72 hours
- Medium: Fix within 1 week
- Low: Fix within 1 month

2. **Reporting Process**
- security@example.com
- Bug bounty program
- Responsible disclosure

## Best Practices

### Development

1. **Code Security**
- Input validation
- Output encoding
- Parameterized queries
- Security testing

2. **Dependency Management**
- Regular updates
- Version pinning
- Security scanning
- Automated alerts

### Operations

1. **Access Control**
- Least privilege principle
- Regular access reviews
- Strong authentication
- Session management

2. **Monitoring**
- Real-time alerts
- Audit logging
- Performance monitoring
- Security scanning

## Compliance

### Standards

1. **Industry Standards**
- ISO 27001
- SOC 2
- GDPR
- CCPA

2. **Internal Policies**
- Security policy
- Privacy policy
- Incident response
- Business continuity

### Auditing

1. **Regular Audits**
- Quarterly internal
- Annual external
- Penetration testing
- Compliance review

2. **Documentation**
- Audit trails
- Change logs
- Access logs
- Incident reports

## Support

For security-related issues:
- Email: security@example.com
- Emergency: +1-XXX-XXX-XXXX
- Bug bounty: https://bugcrowd.com/example 