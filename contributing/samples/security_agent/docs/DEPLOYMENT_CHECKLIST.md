# 🚀 Security Agent v1.14.0 - Deployment Checklist

This comprehensive checklist ensures a successful deployment of Security Agent v1.14.0 in production environments. Follow each section carefully to ensure optimal performance, security, and reliability.

## 📋 Pre-Deployment Verification

### ✅ Environment Readiness

**Infrastructure Requirements:**
- [ ] **Python 3.8+** installed (recommended: 3.11+)
- [ ] **Node.js 16+** for Playwright testing (recommended: 18+)
- [ ] **Minimum 4GB RAM** available (8GB+ recommended for production)
- [ ] **10GB+ disk space** for logs, cache, and databases
- [ ] **Network connectivity** to GCP APIs and WebSocket ports

**GCP Prerequisites:**
- [ ] **Service Account** with appropriate permissions configured
- [ ] **Project ID** identified and accessible
- [ ] **API Credentials** (service account JSON) securely stored
- [ ] **Required GCP APIs** enabled:
  - Cloud Asset API
  - Security Center API
  - IAM API
  - Storage API
  - Monitoring API

**Security Configuration:**
- [ ] **Firewall rules** configured for ports 8000 (API) and 8765 (WebSocket)
- [ ] **SSL certificates** prepared for HTTPS deployment
- [ ] **Environment variables** securely configured (no hardcoded secrets)
- [ ] **Database encryption** enabled for sensitive data

### ✅ Code & Dependencies

**Source Code:**
- [ ] **Latest v1.14.0 code** pulled from repository
- [ ] **Git tag verified**: `git describe --tags` shows `v1.14.0`
- [ ] **No uncommitted changes**: `git status` shows clean working directory
- [ ] **Backup created** of previous version (if upgrading)

**Python Dependencies:**
- [ ] **Virtual environment** created and activated
- [ ] **Requirements installed**: `pip install -r requirements.txt`
- [ ] **Frontend requirements**: `pip install -r requirements_frontend.txt`
- [ ] **Critical dependencies verified**:
  ```bash
  pip list | grep -E "(websockets|playwright|streamlit|fastapi)"
  ```

**Frontend Dependencies:**
- [ ] **Node modules installed**: `npm install`
- [ ] **Playwright browsers**: `npx playwright install`
- [ ] **Browser binaries verified**: `npx playwright --version`

## 🔧 Configuration Checklist

### ✅ Environment Configuration

**Create/Update .env file:**
```bash
# Core Configuration
GOOGLE_CLOUD_PROJECT=your-production-project
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
ENVIRONMENT=production

# WebSocket Configuration (v1.14.0)
WEBSOCKET_ENABLED=true
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8765
SESSION_TIMEOUT=3600

# Performance & Monitoring
PERFORMANCE_MONITORING=true
CACHE_TTL=900
MAX_CONCURRENT_REQUESTS=100
DATABASE_POOL_SIZE=10

# Security Settings
CORS_ORIGINS=["https://yourdomain.com"]
SECURE_COOKIES=true
SESSION_ENCRYPTION=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/security-agent/app.log
```

**Configuration Validation:**
- [ ] **All required variables** present in .env
- [ ] **No sensitive data** committed to version control
- [ ] **File permissions** set correctly (600 for .env)
- [ ] **Service account JSON** accessible and valid

### ✅ Database Configuration

**Database Setup:**
- [ ] **Database directories** created with correct permissions:
  ```bash
  mkdir -p backend/{cache,data}
  chmod 755 backend/{cache,data}
  ```
- [ ] **Database initialization** completed:
  ```bash
  python backend/populate_sqlite.py
  ```
- [ ] **Schema migration** applied (if upgrading):
  ```bash
  python scripts/migrate_v1_14_0.py
  ```
- [ ] **Database backup** created before deployment
- [ ] **Index optimization** completed for performance

**Database Validation:**
- [ ] **Connection test** successful
- [ ] **Schema version** verified as v1.14.0
- [ ] **Performance indexes** created
- [ ] **Data integrity** checks passed

## 🌐 Network & Security Configuration

### ✅ Network Setup

**Port Configuration:**
- [ ] **Port 8000** (Backend API) accessible from load balancer
- [ ] **Port 8765** (WebSocket) configured with sticky sessions
- [ ] **Port 8501** (Frontend) accessible from users
- [ ] **Health check ports** accessible from monitoring systems

**Load Balancer Configuration:**
- [ ] **Backend load balancer** configured with health checks
- [ ] **WebSocket support** enabled with session affinity
- [ ] **SSL termination** configured at load balancer
- [ ] **Rate limiting** configured to prevent abuse

**Firewall Rules:**
```bash
# Example GCP firewall rules
gcloud compute firewall-rules create security-agent-api \
  --allow tcp:8000 --source-ranges=10.0.0.0/8

gcloud compute firewall-rules create security-agent-websocket \
  --allow tcp:8765 --source-ranges=10.0.0.0/8
```

### ✅ Security Hardening

**SSL/TLS Configuration:**
- [ ] **SSL certificates** installed and valid
- [ ] **HTTPS enforcement** enabled
- [ ] **TLS 1.2+** minimum version enforced
- [ ] **Certificate auto-renewal** configured

**Authentication & Authorization:**
- [ ] **Service account** principle of least privilege
- [ ] **API key rotation** schedule established
- [ ] **Session security** configured with secure cookies
- [ ] **CORS policies** properly configured

**Security Scanning:**
- [ ] **Vulnerability scan** completed with no critical issues
- [ ] **Dependency security** check passed
- [ ] **Container security** scan completed (if using containers)
- [ ] **Secrets scanning** completed

## 🚀 Deployment Process

### ✅ Pre-Deployment Tests

**Local Testing:**
- [ ] **Unit tests** passing: `python -m pytest tests/unit/ -v`
- [ ] **Integration tests** passing: `python -m pytest tests/integration/ -v`
- [ ] **API endpoints** responding correctly
- [ ] **WebSocket connection** established successfully

**Staging Environment:**
- [ ] **Staging deployment** completed successfully
- [ ] **End-to-end tests** passing in staging
- [ ] **Performance tests** meeting benchmarks
- [ ] **Load testing** completed with acceptable results

**Production Readiness:**
- [ ] **Monitoring** configured and alerting tested
- [ ] **Log aggregation** configured
- [ ] **Backup procedures** verified
- [ ] **Rollback plan** documented and tested

### ✅ Deployment Steps

**Backend Deployment:**
```bash
# 1. Deploy backend service
python run_backend.py --cloud -- \
  --memory=4Gi \
  --cpu=4 \
  --min-instances=2 \
  --max-instances=10 \
  --set-env-vars=ENVIRONMENT=production

# 2. Verify deployment
curl https://your-backend-url/health
curl https://your-backend-url/api/v2/health/detailed
```

**Frontend Deployment:**
```bash
# 1. Deploy frontend service
python run_frontend.py --cloud -- \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=5

# 2. Verify deployment
curl https://your-frontend-url/health
```

**WebSocket Configuration:**
```bash
# Enable WebSocket support in backend
gcloud run services update security-agent-backend \
  --set-env-vars WEBSOCKET_ENABLED=true,WEBSOCKET_PORT=8765

# Configure load balancer for WebSocket
gcloud compute backend-services update security-agent-backend \
  --session-affinity=CLIENT_IP
```

### ✅ Container Deployment (Alternative)

**Docker Deployment:**
```bash
# 1. Build production images
docker build -f Dockerfile.optimized -t security-agent:v1.14.0 .

# 2. Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# 3. Verify containers
docker-compose ps
docker-compose logs -f
```

**Kubernetes Deployment:**
```bash
# 1. Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 2. Verify deployment
kubectl get pods -n security-agent
kubectl get services -n security-agent
```

## 🔍 Post-Deployment Validation

### ✅ Service Health Checks

**Basic Health Verification:**
- [ ] **Backend API** responding: `curl https://backend-url/health`
- [ ] **Frontend UI** loading: `curl https://frontend-url/health`
- [ ] **WebSocket** connecting: Test with WebSocket client
- [ ] **Database** accessible: Check connection in logs

**Detailed Health Checks:**
```bash
# Backend detailed health
curl https://backend-url/api/v2/health/detailed

# Expected response: 200 OK with detailed system status
# {
#   "status": "healthy",
#   "database": "connected",
#   "websocket": "enabled",
#   "memory_usage": "125MB",
#   "uptime": "5m 23s"
# }
```

**WebSocket Functionality:**
```bash
# Test WebSocket connection
wscat -c wss://your-domain/ws/chat

# Send test message
{"type": "ping", "message": "health check"}

# Expected response: {"type": "pong", "status": "healthy"}
```

### ✅ Performance Validation

**API Performance Benchmarks:**
```bash
# API response time test
curl -w "%{time_total}\n" -o /dev/null -s https://backend-url/api/v1/security/analyze

# Expected: < 100ms (84% improvement target)
```

**Database Performance:**
```bash
# Database query performance test
python scripts/benchmark_queries.py

# Expected results:
# - Query time: < 50ms
# - Connection time: < 10ms
# - Index usage: > 95%
```

**Memory and CPU Usage:**
```bash
# Monitor resource usage
docker stats security-agent-backend
docker stats security-agent-frontend

# Expected usage:
# - Backend: < 200MB RAM, < 50% CPU
# - Frontend: < 100MB RAM, < 25% CPU
```

### ✅ Functional Testing

**Security Analysis Workflow:**
```bash
# Test security analysis endpoint
curl -X POST https://backend-url/api/v1/security/analyze \
  -H "Content-Type: application/json" \
  -d '{"project_id": "your-project-id", "analysis_type": "comprehensive"}'

# Expected: 200 OK with security findings
```

**WebSocket Chat Interface:**
- [ ] **Real-time messaging** working
- [ ] **Token streaming** functional
- [ ] **Session persistence** across reconnections
- [ ] **Multiple concurrent sessions** supported

**Executive Dashboard:**
- [ ] **Security metrics** displaying correctly
- [ ] **Real-time updates** functioning
- [ ] **Interactive elements** responding
- [ ] **Mobile responsiveness** verified

## 📊 Monitoring & Observability

### ✅ Monitoring Setup

**Application Monitoring:**
- [ ] **Application logs** flowing to centralized logging
- [ ] **Error tracking** configured with alerting
- [ ] **Performance metrics** collected
- [ ] **Custom business metrics** configured

**Infrastructure Monitoring:**
- [ ] **CPU/Memory metrics** collected
- [ ] **Network metrics** monitored
- [ ] **Disk usage** tracked
- [ ] **Container health** monitored

**Alert Configuration:**
```yaml
# Example alerting rules
alerts:
  - name: "API Response Time"
    condition: "avg_response_time > 500ms"
    severity: "warning"
    
  - name: "WebSocket Connection Failures"
    condition: "websocket_error_rate > 5%"
    severity: "critical"
    
  - name: "Database Connection Pool"
    condition: "db_pool_utilization > 90%"
    severity: "warning"
```

### ✅ Log Management

**Log Configuration:**
- [ ] **Structured logging** enabled (JSON format)
- [ ] **Log levels** properly configured
- [ ] **Log rotation** configured to prevent disk issues
- [ ] **Centralized logging** (ELK, Splunk, etc.) configured

**Log Verification:**
```bash
# Check log output
tail -f /var/log/security-agent/app.log

# Expected log entries:
# {"timestamp": "2025-09-08T20:00:00Z", "level": "INFO", "message": "WebSocket server started"}
# {"timestamp": "2025-09-08T20:00:01Z", "level": "INFO", "message": "Database connection established"}
```

## 🔄 Backup & Recovery

### ✅ Backup Procedures

**Database Backup:**
```bash
# Create automated backup script
cat > scripts/backup_database.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/security-agent"

mkdir -p $BACKUP_DIR

# Backup SQLite databases
sqlite3 backend/cache/gcp_data.db ".backup $BACKUP_DIR/gcp_data_$DATE.db"
sqlite3 backend/data/service_evaluations.db ".backup $BACKUP_DIR/evaluations_$DATE.db"

# Compress and encrypt backups
tar -czf $BACKUP_DIR/security-agent-backup-$DATE.tar.gz $BACKUP_DIR/*_$DATE.db
gpg --encrypt --recipient admin@company.com $BACKUP_DIR/security-agent-backup-$DATE.tar.gz

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
EOF

chmod +x scripts/backup_database.sh
```

**Configuration Backup:**
- [ ] **Environment configurations** backed up
- [ ] **SSL certificates** backed up securely
- [ ] **Deployment scripts** versioned
- [ ] **Infrastructure as code** committed to version control

### ✅ Recovery Testing

**Recovery Procedures:**
- [ ] **Database restore** procedure tested
- [ ] **Service restart** procedure documented
- [ ] **Rollback plan** tested in staging
- [ ] **Disaster recovery** runbook created

## 📋 Final Deployment Checklist

### ✅ Go-Live Verification

**Before Go-Live:**
- [ ] **All pre-deployment checks** completed ✅
- [ ] **Staging environment** mirrors production ✅
- [ ] **Performance benchmarks** met or exceeded ✅
- [ ] **Security scan** shows no critical vulnerabilities ✅
- [ ] **Team notification** sent about deployment ✅

**Go-Live Steps:**
- [ ] **Maintenance window** scheduled and communicated
- [ ] **Old version backup** created
- [ ] **New version deployed** successfully
- [ ] **Health checks** passing
- [ ] **Performance monitoring** active

**Post Go-Live:**
- [ ] **User acceptance testing** completed
- [ ] **Performance metrics** within expected ranges
- [ ] **Error rates** normal or improved
- [ ] **User feedback** positive
- [ ] **Documentation** updated with production details

### ✅ 24-Hour Post-Deployment

**Day 1 Monitoring:**
- [ ] **Error rates** monitored and within acceptable limits
- [ ] **Performance metrics** stable
- [ ] **User complaints** addressed promptly  
- [ ] **System stability** confirmed
- [ ] **Backup procedures** executed successfully

**Week 1 Review:**
- [ ] **Performance trends** analyzed
- [ ] **User adoption** metrics reviewed
- [ ] **Cost impact** assessed
- [ ] **Security posture** verified
- [ ] **Lessons learned** documented

## 🎯 Success Criteria

### Performance Targets (Must Meet)
- ✅ **API Response Time**: < 100ms (target: 84% improvement)
- ✅ **WebSocket Latency**: < 50ms
- ✅ **Database Query Time**: < 50ms  
- ✅ **Memory Usage**: < 200MB per service
- ✅ **Uptime**: > 99.9%

### Functional Requirements (Must Work)
- ✅ **Real-time WebSocket chat** functional
- ✅ **Executive dashboard** displaying live data
- ✅ **Security analysis** producing accurate results
- ✅ **Mobile responsiveness** across all devices
- ✅ **Session persistence** across reconnections

### Security Requirements (Must Pass)
- ✅ **Zero critical vulnerabilities** in security scan
- ✅ **All data encrypted** in transit and at rest
- ✅ **Authentication** properly configured
- ✅ **Input validation** preventing injection attacks
- ✅ **Audit logging** capturing all security events

---

## 📞 Emergency Contacts & Procedures

### Incident Response
- **Primary On-Call**: [Your on-call engineer]
- **Secondary On-Call**: [Backup engineer]
- **Escalation**: [Team lead/manager]

### Rollback Procedure
```bash
# Emergency rollback to v1.13.0
git checkout v1.13.0
docker-compose down
docker-compose -f docker-compose.v1.13.0.yml up -d

# Verify rollback successful
curl https://backend-url/health
```

### Support Resources
- **Documentation**: `/docs/` directory
- **Runbooks**: `/docs/runbooks/`
- **Monitoring Dashboards**: [Your monitoring URL]
- **Log Aggregation**: [Your logging URL]

---

**✅ Deployment Complete!**

**Next Steps:**
1. Monitor system health for 24-48 hours
2. Collect user feedback on new features
3. Plan optimization based on production usage
4. Schedule regular maintenance windows

**Questions?** Refer to the [Troubleshooting Guide](troubleshooting.md) or contact the development team.