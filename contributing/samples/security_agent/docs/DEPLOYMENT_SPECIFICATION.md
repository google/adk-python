# GCP Security Agent - Deployment Specification

## 1. Deployment Overview

### 1.1 Deployment Architecture
The GCP Security Agent supports two primary deployment modes:
- **Local Development**: Development and testing environment
- **Google Cloud Run**: Production deployment with auto-scaling

### 1.2 Deployment Strategy
- **Zero-downtime deployments**: Blue-green deployment strategy
- **Auto-scaling**: Based on CPU, memory, and request volume
- **Multi-region support**: Regional redundancy for high availability
- **Configuration management**: Environment-based configuration

### 1.3 Infrastructure Requirements
- **Container Runtime**: Docker or Cloud Build
- **Orchestration**: Google Cloud Run (managed)
- **Networking**: VPC connector for private resource access
- **Storage**: Cloud Storage for artifacts and logs
- **Monitoring**: Cloud Monitoring and Cloud Logging

## 2. Local Development Deployment

### 2.1 Prerequisites
```bash
# System Requirements
- Python 3.11 or higher
- pip package manager
- Git version control
- Google Cloud SDK (gcloud CLI)
- Docker (optional, for containerized development)

# Minimum Hardware
- RAM: 4GB minimum, 8GB recommended
- CPU: 2 cores minimum, 4 cores recommended
- Storage: 10GB available disk space
- Network: Broadband internet connection
```

### 2.2 Environment Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd security_agent

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project mgm-digitalconcierge

# 5. Enable required APIs
gcloud services enable cloudasset.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 2.3 Configuration Files

#### 2.3.1 Environment Variables (.env)
```bash
# GCP Configuration
GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge
VERTEX_AI_PROJECT_ID=mgm-digitalconcierge
VERTEX_AI_LOCATION=us-central1

# Application Configuration
PORT=8000
FRONTEND_PORT=8501
FRONTEND_HOST=0.0.0.0

# Development Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Optional: API Hub Configuration
APIHUB_RESOURCE_NAME=projects/mgm-digitalconcierge/locations/us-central1/apiHubs/security-hub

# Optional: Custom Service Account
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

#### 2.3.2 Development Dependencies (requirements-dev.txt)
```text
# Development tools
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0

# Testing utilities
httpx>=0.24.0
pytest-mock>=3.10.0
factory-boy>=3.2.0

# Documentation
mkdocs>=1.4.0
mkdocs-material>=9.0.0
```

### 2.4 Local Deployment Commands

#### 2.4.1 Backend Server
```bash
# Method 1: Using run script
python run_backend.py

# Method 2: Direct uvicorn
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Method 3: With custom configuration
python run_backend.py --port 8001 --project mgm-digitalconcierge
```

#### 2.4.2 Frontend Application
```bash
# Method 1: Using run script
python run_frontend.py

# Method 2: Direct streamlit
streamlit run frontend/main_app.py --server.port 8501

# Method 3: With custom configuration
python run_frontend.py --port 8502
```

#### 2.4.3 Development Workflow
```bash
# Terminal 1: Backend
python run_backend.py

# Terminal 2: Frontend
python run_frontend.py

# Terminal 3: Testing
pytest tests/ -v

# Access Points
# - Backend API: http://localhost:8000
# - API Documentation: http://localhost:8000/docs
# - Frontend UI: http://localhost:8501
# - Health Check: http://localhost:8000/health
```

### 2.5 Local Testing Configuration
```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_asset_inventory_integration.py -v

# Run with live API calls (requires GCP credentials)
pytest tests/test_real_api_transformation.py -v --gcp-project mgm-digitalconcierge
```

## 3. Google Cloud Run Deployment

### 3.1 Prerequisites
```bash
# Required Tools
- Google Cloud SDK (gcloud CLI)
- Docker (optional, for local builds)
- Git for source code management

# Required Permissions
- Cloud Run Developer
- Cloud Build Editor
- Container Registry Service Agent
- Secret Manager Admin (for credential management)
```

### 3.2 Cloud Deployment Configuration

#### 3.2.1 Environment Variables (Cloud)
```yaml
# Cloud Run Environment Variables
GOOGLE_CLOUD_PROJECT: mgm-digitalconcierge
VERTEX_AI_PROJECT_ID: mgm-digitalconcierge
VERTEX_AI_LOCATION: us-central1
PORT: 8080
ENVIRONMENT: production
LOG_LEVEL: WARNING

# Cloud-specific Settings
K_SERVICE: gcp-security-agent
K_REVISION: gcp-security-agent-001
K_CONFIGURATION: gcp-security-agent

# Resource Configuration
MEMORY_LIMIT: 2Gi
CPU_LIMIT: 2
TIMEOUT: 300
MAX_INSTANCES: 10
MIN_INSTANCES: 1
```

#### 3.2.2 Cloud Build Configuration (cloudbuild.yaml)
```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: [
      'build',
      '-t', 'gcr.io/$PROJECT_ID/gcp-security-agent:$COMMIT_SHA',
      '-t', 'gcr.io/$PROJECT_ID/gcp-security-agent:latest',
      '.'
    ]

  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/gcp-security-agent:$COMMIT_SHA']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'gcp-security-agent',
      '--image', 'gcr.io/$PROJECT_ID/gcp-security-agent:$COMMIT_SHA',
      '--region', 'us-central1',
      '--platform', 'managed',
      '--allow-unauthenticated',
      '--memory', '2Gi',
      '--cpu', '2',
      '--timeout', '300',
      '--max-instances', '10',
      '--min-instances', '1',
      '--set-env-vars', 'GOOGLE_CLOUD_PROJECT=$PROJECT_ID',
      '--set-env-vars', 'VERTEX_AI_PROJECT_ID=$PROJECT_ID',
      '--set-env-vars', 'VERTEX_AI_LOCATION=us-central1'
    ]

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'

timeout: '1200s'
```

#### 3.2.3 Dockerfile (Production)
```dockerfile
# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy application code
COPY --chown=app:app . .

# Set environment variables
ENV PYTHONPATH=/home/app
ENV PORT=8080
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 3.3 Cloud Deployment Commands

#### 3.3.1 Automated Deployment
```bash
# Deploy using run script
python run_backend.py --cloud

# Deploy with custom configuration
python run_backend.py --cloud --project mgm-digitalconcierge --region us-central1
```

#### 3.3.2 Manual Deployment Steps
```bash
# 1. Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 2. Build and deploy with Cloud Build
gcloud builds submit --config cloudbuild.yaml

# 3. Alternative: Direct deployment
gcloud run deploy gcp-security-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 1

# 4. Get service URL
gcloud run services describe gcp-security-agent \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

#### 3.3.3 Service Account Configuration
```bash
# Create service account for Cloud Run
gcloud iam service-accounts create gcp-security-agent-sa \
  --display-name "GCP Security Agent Service Account"

# Grant required permissions
gcloud projects add-iam-policy-binding mgm-digitalconcierge \
  --member "serviceAccount:gcp-security-agent-sa@mgm-digitalconcierge.iam.gserviceaccount.com" \
  --role "roles/cloudasset.viewer"

gcloud projects add-iam-policy-binding mgm-digitalconcierge \
  --member "serviceAccount:gcp-security-agent-sa@mgm-digitalconcierge.iam.gserviceaccount.com" \
  --role "roles/compute.viewer"

# Deploy with service account
gcloud run deploy gcp-security-agent \
  --service-account gcp-security-agent-sa@mgm-digitalconcierge.iam.gserviceaccount.com
```

## 4. Infrastructure Configuration

### 4.1 Network Configuration

#### 4.1.1 VPC Connector (for private resource access)
```bash
# Create VPC connector
gcloud compute networks vpc-access connectors create security-agent-connector \
  --region us-central1 \
  --subnet default \
  --subnet-project mgm-digitalconcierge \
  --min-instances 2 \
  --max-instances 10

# Deploy with VPC connector
gcloud run deploy gcp-security-agent \
  --vpc-connector security-agent-connector \
  --vpc-egress all-traffic
```

#### 4.1.2 Load Balancer Configuration
```yaml
# Global Load Balancer for multiple regions
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-agent-lb-config
data:
  backend_config: |
    regions:
      - us-central1
      - europe-west1
      - asia-east1
    health_check:
      path: /health
      interval: 30s
      timeout: 10s
```

### 4.2 Scaling Configuration

#### 4.2.1 Auto-scaling Parameters
```yaml
# Cloud Run Scaling Configuration
scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 70
  target_memory_utilization: 80
  target_concurrent_requests: 100

# Scaling triggers
triggers:
  scale_up:
    - cpu_utilization > 70%
    - memory_utilization > 80%
    - concurrent_requests > 100
  scale_down:
    - cpu_utilization < 30%
    - memory_utilization < 40%
    - concurrent_requests < 20
```

#### 4.2.2 Resource Limits
```yaml
# Resource Configuration
resources:
  limits:
    memory: 2Gi
    cpu: 2000m
  requests:
    memory: 1Gi
    cpu: 1000m

# Service Level Objectives
slo:
  response_time_p95: 2000ms
  response_time_p99: 5000ms
  availability: 99.5%
  error_rate: <1%
```

### 4.3 Security Configuration

#### 4.3.1 IAM Roles and Permissions
```yaml
# Minimum Required Roles
service_account_roles:
  - roles/cloudasset.viewer
  - roles/compute.viewer
  - roles/storage.objectViewer
  - roles/iam.securityReviewer
  - roles/recommender.viewer
  - roles/monitoring.viewer

# Custom Role for Enhanced Security
custom_role:
  name: roles/security.agent
  title: "Security Agent Role"
  description: "Custom role for GCP Security Agent"
  permissions:
    - cloudasset.assets.searchAllResources
    - compute.instances.list
    - storage.buckets.list
    - iam.roles.list
    - recommender.recommendations.list
```

#### 4.3.2 Secret Management
```bash
# Store service account key in Secret Manager
gcloud secrets create security-agent-sa-key \
  --data-file service-account-key.json

# Grant Cloud Run access to secrets
gcloud secrets add-iam-policy-binding security-agent-sa-key \
  --member "serviceAccount:gcp-security-agent-sa@mgm-digitalconcierge.iam.gserviceaccount.com" \
  --role "roles/secretmanager.secretAccessor"
```

## 5. Monitoring and Logging

### 5.1 Monitoring Configuration

#### 5.1.1 Cloud Monitoring Metrics
```yaml
# Custom Metrics
custom_metrics:
  - name: security_agent/asset_discovery_count
    description: Number of assets discovered per hour
    unit: count
  
  - name: security_agent/chat_response_time
    description: Chat response time in milliseconds
    unit: ms
  
  - name: security_agent/security_findings_count
    description: Number of security findings detected
    unit: count

# Alerting Policies
alerts:
  - name: "High Response Time"
    condition: response_time_p95 > 5000ms
    notification: security-team@company.com
  
  - name: "High Error Rate"
    condition: error_rate > 5%
    notification: dev-team@company.com
```

#### 5.1.2 Health Check Configuration
```python
# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "checks": {
            "database": check_database_connection(),
            "gcp_apis": check_gcp_api_connectivity(),
            "memory_usage": get_memory_usage(),
            "disk_space": get_disk_usage()
        }
    }
```

### 5.2 Logging Configuration

#### 5.2.1 Structured Logging
```python
import structlog

logger = structlog.get_logger()

# Log configuration
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}
```

#### 5.2.2 Log Aggregation
```yaml
# Cloud Logging Configuration
log_config:
  retention_days: 30
  filters:
    - severity >= INFO
    - resource.type = cloud_run_revision
  
  exports:
    - destination: bigquery
      dataset: security_agent_logs
      table: application_logs
    
    - destination: cloud_storage
      bucket: mgm-security-agent-logs
      path: "logs/{year}/{month}/{day}/"
```

## 6. Backup and Disaster Recovery

### 6.1 Backup Strategy

#### 6.1.1 Data Backup
```yaml
# Backup Configuration
backup_schedule:
  session_data:
    frequency: daily
    retention: 30_days
    destination: gs://mgm-security-agent-backups/sessions/
  
  configuration:
    frequency: on_change
    retention: 90_days
    destination: gs://mgm-security-agent-backups/config/
  
  logs:
    frequency: hourly
    retention: 7_days
    destination: gs://mgm-security-agent-backups/logs/
```

#### 6.1.2 Automated Backup Script
```bash
#!/bin/bash
# backup_script.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="gs://mgm-security-agent-backups"

# Backup session data
gsutil cp -r gs://mgm-security-agent-sessions/* \
  $BACKUP_BUCKET/sessions/$DATE/

# Backup configuration
gcloud run services describe gcp-security-agent \
  --format="export" > config_backup_$DATE.yaml
gsutil cp config_backup_$DATE.yaml $BACKUP_BUCKET/config/

# Cleanup old backups (retain 30 days)
gsutil -m rm -r $BACKUP_BUCKET/sessions/$(date -d '30 days ago' +%Y%m%d_*)
```

### 6.2 Disaster Recovery

#### 6.2.1 Recovery Procedures
```yaml
# Disaster Recovery Plan
recovery_procedures:
  rto: 4_hours  # Recovery Time Objective
  rpo: 1_hour   # Recovery Point Objective
  
  steps:
    1: "Assess impact and notify stakeholders"
    2: "Restore from latest backup"
    3: "Verify data integrity"
    4: "Perform functional testing"
    5: "Resume normal operations"
    6: "Post-incident review"

# Multi-region Deployment
regions:
  primary: us-central1
  secondary: us-west1
  tertiary: europe-west1
```

## 7. Performance Optimization

### 7.1 Caching Strategy

#### 7.1.1 Redis Configuration
```yaml
# Redis Cache Configuration
redis:
  instance_type: standard
  memory_size: 1GB
  region: us-central1
  auth_enabled: true
  
  cache_policies:
    asset_data:
      ttl: 300  # 5 minutes
      max_size: 100MB
    
    recommendations:
      ttl: 900  # 15 minutes
      max_size: 50MB
    
    session_data:
      ttl: 3600  # 1 hour
      max_size: 200MB
```

#### 7.1.2 CDN Configuration
```yaml
# Cloud CDN for static assets
cdn:
  origin: gcp-security-agent-static
  cache_mode: USE_ORIGIN_HEADERS
  default_ttl: 3600
  max_ttl: 86400
  
  compression: true
  http2: true
  
  cache_key_policy:
    include_host: true
    include_protocol: true
    include_query_string: false
```

### 7.2 Database Optimization

#### 7.2.1 Connection Pooling
```python
# Database connection pool configuration
database_config = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "pool_pre_ping": True
}
```

## 8. Security Hardening

### 8.1 Container Security

#### 8.1.1 Security Scanning
```yaml
# Container Security Scanning
security_scan:
  vulnerability_scanning: enabled
  binary_authorization: enabled
  
  policies:
    - require_attestation: true
    - block_high_severity: true
    - allow_only_trusted_images: true
```

#### 8.1.2 Runtime Security
```yaml
# Runtime Security Configuration
runtime_security:
  read_only_root_filesystem: true
  non_root_user: true
  capabilities_drop: ["ALL"]
  seccomp_profile: runtime/default
  apparmor_profile: runtime/default
```

### 8.2 Network Security

#### 8.2.1 Network Policies
```yaml
# Network Security Policies
network_policies:
  ingress:
    - allow_https_only: true
    - require_tls_1_3: true
    - rate_limiting: 1000_requests_per_minute
  
  egress:
    - allow_gcp_apis: true
    - block_external_http: true
    - log_all_connections: true
```

This deployment specification provides comprehensive guidance for deploying the GCP Security Agent in both local development and production cloud environments, with proper security, monitoring, and performance configurations.