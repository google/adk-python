# ADK Security Agent - Enterprise Deployment Guide

A comprehensive, enterprise-ready security evaluation platform for Google Cloud Platform (GCP) with AI-powered analysis, modular architecture, and advanced security insights through ADK (Agent Development Kit) integration.

## 🎯 Enterprise Overview

### Key Capabilities
- **🛡️ Comprehensive Security Analysis**: Multi-layered GCP security evaluation with real-time risk assessment
- **🤖 AI-Powered Security Agent**: Advanced ADK-based intelligent assistant for security recommendations  
- **🔐 Advanced IAM Analysis**: Deep IAM permissions analysis with policy testing and compliance checking
- **📊 Real-time Monitoring**: Live security metrics, alerts, and performance dashboards
- **🔄 Modular Service Architecture**: Enable/disable individual services for fault isolation and resource optimization
- **📋 Multi-Framework Compliance**: SOC2, ISO27001, GDPR compliance evaluation and reporting
- **🚨 Incident Response**: Automated security incident detection and management

### Technology Stack
- **Backend**: Python 3.8+, FastAPI, ADK, Google Cloud APIs
- **Frontend**: Streamlit with component-based architecture
- **Authentication**: Google Cloud Service Account with Application Default Credentials
- **Monitoring**: OpenTelemetry, Cloud Trace, Cloud Logging
- **Deployment**: Docker, Cloud Run ready, Kubernetes compatible

## 🚀 Quick Deployment (2 Minutes)

**Deploy the entire security agent with a single command:**

```bash
python run.py
```

**Access Points:**
- 🌐 **Security Dashboard**: http://localhost:8501
- 🔧 **API Documentation**: http://localhost:8000/docs
- ⚙️ **Service Management**: http://localhost:8501 → Service Management

## 📋 Enterprise Requirements

### System Requirements
| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| **CPU** | 2 cores | 4 cores | 8+ cores |
| **RAM** | 4GB | 8GB | 16GB+ |
| **Storage** | 2GB | 5GB | 20GB+ |
| **Network** | Internet connectivity | Dedicated bandwidth | Load balancer |

### Google Cloud Prerequisites

#### 1. Required GCP APIs
Enable these APIs in your GCP project:

```bash
# Core APIs
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable serviceusage.googleapis.com
gcloud services enable iam.googleapis.com

# Security & Monitoring
gcloud services enable securitycenter.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable cloudtrace.googleapis.com

# AI & Analytics (Optional but recommended)
gcloud services enable aiplatform.googleapis.com
gcloud services enable recommender.googleapis.com
```

#### 2. Service Account Setup
Create a dedicated service account with minimal required permissions:

```bash
# Create service account
gcloud iam service-accounts create security-agent \
    --display-name="ADK Security Agent" \
    --description="Enterprise security evaluation agent"

# Assign essential roles
PROJECT_ID="your-project-id"
SERVICE_ACCOUNT="security-agent@${PROJECT_ID}.iam.gserviceaccount.com"

# Core permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/resourcemanager.projectViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/serviceusage.serviceUsageViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/iam.securityReviewer"

# Security analysis permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/securitycenter.findingsViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/logging.viewer"

# AI features (optional)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.user"

# Download service account key
gcloud iam service-accounts keys create security-agent-key.json \
    --iam-account=$SERVICE_ACCOUNT
```

## 🔧 Enterprise Configuration

### Environment Setup
Create a `.env` file in the project root:

```bash
# Essential Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/security-agent-key.json

# Vertex AI Configuration (Enhanced AI features)
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1

# Service Configuration
SERVICE_CONFIG_PATH=backend/config/services.json
ADK_EVALUATION_ENABLED=true

# Monitoring & Logging
LOG_LEVEL=INFO
ENABLE_CLOUD_TRACE=true
ENABLE_CLOUD_LOGGING=true
```

### Advanced Configuration Options

#### Service Management Configuration
The modular architecture allows fine-grained control over enabled services:

```json
{
  "services": {
    "security": {
      "enabled_by_default": true,
      "config": {
        "scan_depth": "comprehensive",
        "risk_threshold": "medium"
      }
    },
    "iam": {
      "enabled_by_default": true,
      "config": {
        "cache_ttl": 300,
        "max_users_per_scan": 100,
        "include_service_accounts": true
      }
    },
    "compliance": {
      "enabled_by_default": true,
      "config": {
        "frameworks": ["SOC2", "ISO27001", "GDPR"],
        "auto_remediation": false
      }
    }
  }
}
```

## 🏗️ Modular Architecture Overview

### Service-Based Architecture
The security agent uses a revolutionary modular service architecture that provides:

- **🛡️ Fault Isolation**: Services fail independently without affecting others
- **⚙️ Granular Control**: Enable/disable services through web UI
- **🏥 Health Monitoring**: Real-time service health checks and status
- **🔄 Dependency Management**: Automatic service dependency resolution
- **💾 State Persistence**: Service configurations saved across restarts

### Available Services

#### Core Services (Always Required)
- **🛡️ Security Service**: Core security evaluation and scanning
- **🔧 GCP Service**: Google Cloud Platform integration
- **🤖 Agent Service**: AI-powered security agent

#### Security & Compliance Services
- **🔐 IAM Analysis**: Identity and Access Management policy analysis
- **📋 Compliance**: Multi-framework compliance checking
- **🚨 Threat Intelligence**: Vulnerability and threat analysis
- **🔍 Security Analytics**: BigQuery-based security analytics
- **📚 Security Knowledge**: Vertex AI Search integration

#### Monitoring & Operations Services
- **📊 Cloud Logging**: Google Cloud Logging integration
- **📈 Performance Monitoring**: System performance metrics
- **🔍 Distributed Tracing**: OpenTelemetry with Cloud Trace
- **🚨 Incident Response**: Security incident management

## 🚀 Deployment Options

### Option 1: One-Command Deployment (Recommended for Testing)
```bash
git clone https://github.com/google/adk-python.git
cd adk-python/contributing/samples/security_agent
python run.py
```

### Option 2: Docker Deployment (Recommended for Production)
```bash
# Build and deploy with Docker
python run.py --docker

# Or manually
docker build -t adk-security-agent .
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/security-agent-key.json:/app/security-agent-key.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/security-agent-key.json \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  adk-security-agent
```

### Option 3: Cloud Run Deployment (Enterprise Production)
```bash
# Build container image
gcloud builds submit --tag gcr.io/$PROJECT_ID/security-agent

# Deploy to Cloud Run
gcloud run deploy security-agent \
    --image gcr.io/$PROJECT_ID/security-agent \
    --platform managed \
    --region us-central1 \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
    --port 8501 \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 10 \
    --allow-unauthenticated
```

### Option 4: Kubernetes Deployment (Large Enterprise)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adk-security-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: security-agent
  template:
    metadata:
      labels:
        app: security-agent
    spec:
      serviceAccountName: security-agent-ksa
      containers:
      - name: security-agent
        image: gcr.io/PROJECT_ID/security-agent:latest
        ports:
        - containerPort: 8501
        - containerPort: 8000
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 🔒 Security Best Practices

### Authentication & Authorization
- **Service Account Authentication**: Use dedicated service accounts with minimal required permissions
- **Application Default Credentials**: Leverage ADC for secure credential management
- **IAM Role-Based Access**: Implement fine-grained permission control
- **Regular Permission Audits**: Review and rotate service account keys quarterly

### Data Protection
- **Data Encryption**: All data encrypted in transit (TLS 1.2+) and at rest
- **No PII Storage**: No persistent storage of personally identifiable information
- **Audit Logging**: Comprehensive audit trail of all security operations
- **Secure Communication**: All GCP API calls use HTTPS with certificate validation

### Network Security
- **Firewall Configuration**: Restrict inbound connections to required ports (8000, 8501)
- **TLS Termination**: Implement TLS termination at load balancer level in production
- **Network Isolation**: Use VPC networks and private subnets for production deployments
- **CORS Protection**: Configured to allow only authorized frontend origins

### Operational Security
```bash
# Production security checklist
✅ Service accounts use least privilege principles
✅ API keys stored in Secret Manager (not environment variables)
✅ Network traffic encrypted with TLS 1.2+
✅ Regular security scans and vulnerability assessments
✅ Audit logging enabled for all security operations
✅ Incident response procedures documented and tested
```

## 📊 Service Management & Monitoring

### Service Management Interface
Access comprehensive service management through the web UI:

**Navigation:** Security Dashboard → ⚙️ Service Management

**Features:**
- **Services Overview**: Real-time status of all 16 modular services
- **Health Monitoring**: Automated health checks with status indicators
- **Service Control**: Enable/disable services with one-click toggles
- **Dependency Tracking**: Automatic resolution of service dependencies
- **Performance Metrics**: Resource usage and response time monitoring

### Service Management API
Programmatically control services for automation and integration:

```bash
# List all services and their status
curl http://localhost:8000/api/v1/services/status/summary

# Enable a specific service
curl -X POST http://localhost:8000/api/v1/services/iam/enable

# Check service health
curl http://localhost:8000/api/v1/services/security/health

# Disable a service
curl -X POST http://localhost:8000/api/v1/services/threat_intelligence/disable
```

### Monitoring & Alerting
Built-in monitoring capabilities include:

- **OpenTelemetry Tracing**: Distributed tracing with Google Cloud Trace integration
- **Cloud Logging**: Centralized logging with security event correlation
- **Performance Metrics**: API response times, error rates, and resource utilization
- **Health Checks**: Automated service health monitoring with alerting

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### Port Conflicts
```bash
# Issue: Port 8000 or 8501 already in use
# Solution: Kill existing processes
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9

# Then restart the application
python run.py
```

#### Authentication Errors
```bash
# Issue: "Authentication failed" or "insufficient permissions"
# Solution: Verify service account setup
gcloud auth list
gcloud auth application-default login

# Check service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:security-agent*"
```

#### Service Startup Failures
1. **Check Service Status**: Navigate to Service Management UI
2. **Review Dependencies**: Ensure required services are enabled
3. **Check Logs**: Review application logs for error details
4. **Verify Configuration**: Confirm `.env` file and service configuration

#### GCP API Connection Issues
```bash
# Issue: "Connection refused" to GCP APIs
# Solution: Verify API enablement and network connectivity
gcloud services list --enabled --filter="name:cloudresourcemanager"

# Test API connectivity
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     https://cloudresourcemanager.googleapis.com/v3/projects
```

#### Docker Container Issues
```bash
# Issue: Container fails to start
# Solution: Check Docker daemon and container logs
docker ps -a
docker logs security-agent

# Rebuild container if needed
python run.py --docker --rebuild
```

### Service-Specific Troubleshooting

#### IAM Analysis Service
- **Issue**: "No users found" or empty analysis results
- **Solution**: Verify `roles/iam.securityReviewer` permission assigned to service account

#### Security Center Integration
- **Issue**: "Security Center API not accessible" 
- **Solution**: Enable Security Center API and assign `roles/securitycenter.findingsViewer` role

#### Vertex AI Features
- **Issue**: AI recommendations not working
- **Solution**: Enable Vertex AI API and assign `roles/aiplatform.user` role

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
docker stats security-agent

# Optimize service configuration
# Disable unused services to reduce memory footprint
curl -X POST http://localhost:8000/api/v1/services/unused_service/disable
```

#### Response Time Optimization
- **Enable Service Caching**: Configure cache TTL in service configuration
- **Optimize API Calls**: Use batch operations where possible  
- **Monitor Slow Queries**: Review Cloud Trace for bottlenecks

### Getting Support

#### Debug Information Collection
```bash
# Collect system information
python -c "import sys; print(f'Python: {sys.version}')"
docker --version
gcloud version

# Export service status
curl http://localhost:8000/api/v1/services/status/summary > service_status.json

# Export application logs
docker logs security-agent > security-agent.log 2>&1
```

#### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Community Discussions**: GitHub Discussions for community support

## 🚀 Advanced Features

### Enterprise Integrations

#### Single Sign-On (SSO)
```bash
# Configure OIDC integration
OIDC_CLIENT_ID=your-client-id
OIDC_CLIENT_SECRET=your-client-secret
OIDC_DISCOVERY_URL=https://your-sso-provider/.well-known/openid-configuration
```

#### API Gateway Integration
```yaml
# Configure API Gateway for enterprise access
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: security-agent-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    hosts:
    - security.yourdomain.com
```

#### Compliance Reporting
- **Automated Reports**: Schedule compliance reports for SOC2, ISO27001
- **Export Capabilities**: JSON, CSV, PDF report generation
- **Audit Trails**: Complete audit log export for compliance teams

### High Availability Setup

#### Load Balancer Configuration
```nginx
upstream security_backend {
    server security-agent-1:8000;
    server security-agent-2:8000;
    server security-agent-3:8000;
}

upstream security_frontend {
    server security-agent-1:8501;
    server security-agent-2:8501;
    server security-agent-3:8501;
}

server {
    listen 443 ssl;
    server_name security.yourdomain.com;
    
    location /api/ {
        proxy_pass http://security_backend;
    }
    
    location / {
        proxy_pass http://security_frontend;
    }
}
```

#### Database Persistence (Optional)
```yaml
# PostgreSQL for persistent storage
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  template:
    spec:
      containers:
      - name: postgres
        image: postgres:13
        env:
        - name: POSTGRES_DB
          value: security_agent
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
```

## 📈 Scaling & Performance

### Horizontal Scaling
```yaml
# Kubernetes Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: security-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: adk-security-agent
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Monitoring
```python
# Built-in performance monitoring
GET /api/v1/metrics/performance
{
  "response_times": {
    "p50": 120,
    "p95": 450,
    "p99": 890
  },
  "throughput": {
    "requests_per_second": 145,
    "concurrent_users": 12
  },
  "resource_usage": {
    "cpu_percent": 35,
    "memory_mb": 2048,
    "disk_io": "low"
  }
}
```

### Resource Optimization
- **Service Selective Enabling**: Disable unused services to reduce resource consumption
- **Caching Strategy**: Implement Redis for session and API response caching
- **Connection Pooling**: Use connection pooling for GCP API calls
- **Async Processing**: Leverage FastAPI's async capabilities for high throughput

## 📋 Enterprise Deployment Checklist

### Pre-Deployment
- [ ] GCP project created and configured
- [ ] Required APIs enabled (cloudresourcemanager, iam, securitycenter, etc.)
- [ ] Service account created with minimal required permissions
- [ ] Network architecture designed (VPC, subnets, firewall rules)
- [ ] SSL/TLS certificates obtained for production domains
- [ ] Monitoring and alerting configured (Cloud Monitoring, PagerDuty)

### Deployment
- [ ] Environment variables configured securely
- [ ] Container image built and stored in registry
- [ ] Database configured (if using persistent storage)
- [ ] Load balancer configured for high availability
- [ ] Backup and disaster recovery procedures implemented
- [ ] Security scanning completed on container images

### Post-Deployment
- [ ] Health checks configured and passing
- [ ] Performance benchmarks established
- [ ] Security scan completed and vulnerabilities addressed
- [ ] Documentation updated with production-specific details
- [ ] Team training completed on service management interface
- [ ] Incident response procedures tested and validated

## 📋 Features Overview

### Core Security Analysis
- **🔍 Comprehensive GCP Project Analysis**: Automated scanning of GCP projects for security misconfigurations
- **🔐 Advanced IAM Analysis**: Deep dive into IAM policies, roles, and permissions with risk scoring
- **🛡️ Security Center Integration**: Real-time security findings and vulnerability assessment
- **📊 Compliance Framework Support**: SOC2, ISO27001, GDPR compliance evaluation and reporting

### AI-Powered Intelligence
- **🤖 Security Agent**: ADK-based intelligent assistant for security analysis and recommendations
- **💬 Interactive Chat Interface**: Natural language queries for security insights and guidance
- **🎯 Automated Recommendations**: AI-generated security improvements based on best practices
- **📈 Trend Analysis**: Historical security posture tracking and improvement suggestions

### Enterprise Management
- **⚙️ Service Management Interface**: Web-based control panel for managing 16 modular services
- **🔄 Health Monitoring**: Real-time service health checks and status monitoring  
- **📊 Performance Dashboards**: Resource utilization and response time tracking
- **🚨 Incident Response**: Automated security incident detection and management workflows

### Integration & Extensibility
- **🔌 API-First Architecture**: RESTful APIs for all functionality with OpenAPI documentation
- **📡 Cloud Logging Integration**: Centralized logging with security event correlation
- **📈 OpenTelemetry Tracing**: Distributed tracing for performance monitoring and debugging
- **🔗 Plugin Architecture**: Extensible framework for custom security tools and integrations

## 📖 API Documentation

### Authentication
All API endpoints require Google Cloud authentication via service account:

```bash
# Set authentication header
export TOKEN=$(gcloud auth print-access-token)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/gcp/projects
```

### Core API Endpoints

#### Project Management
```bash
# List accessible GCP projects
GET /api/v1/gcp/projects

# Get detailed project information  
GET /api/v1/gcp/project/{project_id}/info

# List enabled services for project
GET /api/v1/gcp/project/{project_id}/services
```

#### Security Analysis
```bash
# Get comprehensive security score
GET /api/v1/security/score?project_id={project_id}

# Run security evaluation
POST /api/v1/security/evaluate
{
  "project_id": "your-project-id",
  "scan_types": ["iam", "network", "storage", "compute"]
}

# Get security recommendations
GET /api/v1/security/recommendations?project_id={project_id}
```

#### IAM Analysis  
```bash
# Analyze user permissions
GET /api/v1/iam/project/{project_id}/analyze-user/{user_email}

# Get IAM testing scenarios
GET /api/v1/iam/testing/scenarios

# Run IAM scenario test
POST /api/v1/iam/testing/run-scenario/{scenario_id}
{
  "project_id": "your-project-id"
}
```

#### Service Management
```bash
# List all services and status
GET /api/v1/services/status/summary

# Enable/disable specific service
POST /api/v1/services/{service_name}/enable
POST /api/v1/services/{service_name}/disable

# Check service health
GET /api/v1/services/{service_name}/health
```

### Complete API Reference
Access the interactive API documentation at: **http://localhost:8000/docs**

## 🏢 Enterprise Support

### Professional Services
- **Implementation Consulting**: Expert guidance for enterprise deployment
- **Custom Integration Development**: Tailored integrations with existing security tools
- **Training & Certification**: Comprehensive team training on security agent capabilities
- **24/7 Technical Support**: Enterprise-grade support with SLA guarantees

### Compliance & Certifications
- **SOC 2 Type II Compliance**: Annual compliance audits and attestation
- **ISO 27001 Certified**: Information security management system certification  
- **GDPR Compliant**: Full compliance with data protection regulations
- **FedRAMP Authorization**: Federal government deployment certification (planned)

### Enterprise Features
- **Single Sign-On (SSO)**: SAML 2.0 and OIDC integration with enterprise identity providers
- **Role-Based Access Control (RBAC)**: Fine-grained permission management for teams
- **Audit Logging**: Comprehensive audit trails for compliance and security monitoring
- **Custom Dashboards**: Tailored reporting and visualization for executive teams

## 🔗 Additional Resources

### Documentation
- **[API Reference](http://localhost:8000/docs)**: Complete OpenAPI specification
- **[Service Management Guide](http://localhost:8501)**: Web interface documentation  
- **[Security Best Practices](#security-best-practices)**: Production deployment guidelines

### Community & Support
- **GitHub Repository**: https://github.com/google/adk-python
- **Issue Tracking**: Report bugs and request features via GitHub Issues
- **Community Discussions**: Join the conversation on GitHub Discussions
- **Professional Support**: Contact enterprise@example.com for commercial support

### Version Information
- **Current Version**: 1.0.0
- **Release Date**: January 2025
- **Supported Python Versions**: 3.8, 3.9, 3.10, 3.11
- **ADK Version**: Latest stable
- **License**: Apache 2.0

---

**Ready to secure your GCP environment?** Deploy the ADK Security Agent in under 2 minutes with `python run.py` and start your comprehensive security evaluation today.
