# 🛡️ GCP Security Agent

A comprehensive, modular security evaluation platform that provides advanced security analysis capabilities for Google Cloud Platform (GCP) environments. Built with a modern, domain-driven architecture and featuring AI-powered security insights through ADK (Agent Development Kit) integration.

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit Frontend<br/>main_app.py]
        
        subgraph "Components"
            Dashboard[🏠 Dashboard]
            Security[🛡️ Security Evaluation]
            Recommendations[🎯 Recommendations]
            IAM[🔐 IAM Analysis]
            Compliance[📋 Compliance]
            Chat[💬 AI Assistant]
            MSA[📄 MSA Analysis]
            Performance[📊 Performance Monitoring]
            SRE[🔧 Day Two SRE]
            APIExplorer[🔍 API Explorer]
            OIDC[🔐 OIDC Demo]
            Incidents[🚨 Incident Response]
        end
        
        APIClient[API Client<br/>Centralized Communication]
    end
    
    subgraph "Backend Layer"
        FastAPI[FastAPI Server<br/>main.py]
        
        subgraph "Feature Modules"
            subgraph "Security Features"
                SecurityMod[security/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
                IAMMod[iam/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
                ComplianceMod[compliance/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
                RecommendationsMod[recommendations/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
            end
            
            subgraph "Platform Features"
                GCPMod[gcp/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
                MSAMod[msa/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
                DocumentationMod[documentation/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
            end
            
            subgraph "Operations Features"
                CloudLoggingMod[cloud_logging/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
                TracingMod[tracing/<br/>├── api.py<br/>├── service.py<br/>└── models.py]
            end
        end
    end
    
    subgraph "External Services"
        GCP[Google Cloud Platform<br/>APIs & Services]
        ADK[Agent Development Kit<br/>AI/ML Processing]
        VertexAI[Vertex AI<br/>Enterprise AI Features]
    end
    
    %% Frontend connections
    UI --> Dashboard
    UI --> Security
    UI --> Recommendations
    UI --> IAM
    UI --> Compliance
    UI --> Chat
    UI --> MSA
    UI --> Performance
    UI --> SRE
    UI --> APIExplorer
    UI --> OIDC
    UI --> Incidents
    
    UI --> APIClient
    APIClient --> FastAPI
    
    %% Backend connections
    FastAPI --> SecurityMod
    FastAPI --> IAMMod
    FastAPI --> ComplianceMod
    FastAPI --> RecommendationsMod
    FastAPI --> GCPMod
    FastAPI --> MSAMod
    FastAPI --> DocumentationMod
    FastAPI --> CloudLoggingMod
    FastAPI --> TracingMod
    
    %% External service connections
    SecurityMod --> GCP
    IAMMod --> GCP
    ComplianceMod --> GCP
    GCPMod --> GCP
    CloudLoggingMod --> GCP
    
    Chat --> ADK
    MSAMod --> ADK
    RecommendationsMod --> VertexAI
    
    %% Styling
    classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef backend fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef external fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef feature fill:#fff3e0,stroke:#e65100,stroke-width:1px
    
    class UI,Dashboard,Security,Recommendations,IAM,Compliance,Chat,MSA,Performance,SRE,APIExplorer,OIDC,Incidents,APIClient frontend
    class FastAPI,SecurityMod,IAMMod,ComplianceMod,RecommendationsMod,GCPMod,MSAMod,DocumentationMod,CloudLoggingMod,TracingMod backend
    class GCP,ADK,VertexAI external
```

## 🚀 Quick Start

**Deploy the entire security agent with a single command:**

```bash
python run.py
```

That's it! The script will:
- ✅ Check dependencies and setup environment
- ✅ Start modular backend with service management
- ✅ Start frontend with service control UI
- ✅ Open browser tabs automatically
- ✅ Enable/disable services as needed

**Access Points:**
- 🌐 **Frontend**: http://localhost:8501
- 🔧 **Backend API**: http://localhost:8000
- 📚 **API Documentation**: http://localhost:8000/docs
- ⚙️ **Service Management**: http://localhost:8501 → Service Management

**Deployment Options:**
```bash
python run.py                    # Modular architecture (default)
python run.py --legacy           # Legacy monolithic backend
python run.py --docker           # Docker container deployment
python run.py --backend-only     # Backend server only
python run.py --frontend-only    # Frontend only
```

## 📋 System Requirements

### Minimum Requirements
- **Docker**
- **4GB RAM** (8GB recommended)
- **2GB disk space**
- **Internet connection** for package installation

### Recommended Requirements
- **8GB RAM**
- **5GB disk space**
- **Google Cloud Project** (for Vertex AI features)

## 🏗️ Modular Service Architecture

The application features a **revolutionary modular service architecture** that allows users to enable/disable individual services independently. This prevents service failures from breaking the entire agent and provides granular control during setup.

### 🔑 Key Benefits

- **🛡️ Fault Isolation**: Services can fail independently without affecting others
- **⚙️ Enable/Disable Control**: Turn services on/off through web UI
- **🏥 Health Monitoring**: Real-time service health checks and status
- **📊 Service Management**: Complete service lifecycle management
- **🔄 Dependency Management**: Automatic service dependency resolution
- **💾 State Persistence**: Service configurations saved across restarts

### 🔧 Available Services

The system includes **16 modular services** organized by functionality:

#### **Core Services** (Always Required)
- **🛡️ Security Service**: Core security evaluation and scanning
- **🔧 GCP Service**: Google Cloud Platform integration
- **🤖 Agent Service**: AI-powered security agent

#### **Security & Compliance Services**
- **🔐 IAM Analysis**: Identity and Access Management policy analysis
- **📋 Compliance**: Multi-framework compliance checking (SOC2, ISO27001, GDPR, etc.)
- **🚨 Threat Intelligence**: Vulnerability and threat analysis
- **🔍 Security Analytics**: BigQuery-based security analytics
- **📚 Security Knowledge**: Vertex AI Search integration

#### **Monitoring & Operations Services**
- **📊 Cloud Logging**: Google Cloud Logging integration
- **📈 Performance Monitoring**: System performance metrics
- **🔍 Distributed Tracing**: OpenTelemetry with Cloud Trace
- **🚨 Incident Response**: Security incident management

#### **Integration Services**
- **📄 Documentation**: API documentation scraping
- **📋 MSA Analysis**: Microsoft Service Agreement parsing
- **🔗 API Hub**: Google API Hub integration
- **🎯 Recommendations**: AI-powered security recommendations

### Backend Structure (Modular Architecture)
```
backend/
├── core/                    # 🏗️ Service Management Core
│   ├── service_registry.py  # Central service registry
│   ├── service_config.py    # Service configuration management
│   └── base_service.py      # Base service class
├── main_modular.py          # 🚀 Modular application entry point
├── main.py                  # 📦 Legacy monolithic entry point
├── [service_name]/          # 🔧 Individual Service Modules
│   ├── api.py              # FastAPI routes
│   ├── service.py          # Business logic
│   └── models.py           # Data models
└── config/
    └── services.json       # 📋 Service configuration
```

### Frontend Structure (Component-Based)
```
frontend/
├── components/                      # 🧩 Reusable UI Components
│   ├── dashboard_view.py           # 🏠 Main dashboard
│   ├── services_management_view.py # ⚙️ Service management UI
│   ├── security_evaluation_view.py # 🛡️ Security analysis UI
│   ├── iam_analyzer_view.py        # 🔐 IAM analysis UI
│   ├── compliance_view.py          # 📋 Compliance dashboard
│   ├── chat_view.py                # 💬 AI chat interface
│   └── __init__.py                 # Component exports
├── api_client.py                   # 🌐 Backend communication
├── main_app.py                     # 🚀 Modular main application
└── enhanced_security_agent_app.py  # 📦 Legacy monolithic app
```

### 🎯 Service Management Interface

The new **Service Management** interface provides complete control:

- **📋 Services Overview**: View all services and their status
- **⚙️ Service Control**: Enable/disable services with toggle buttons  
- **🏥 Health Status**: Real-time monitoring with auto-refresh
- **🔍 Service Details**: Detailed configuration and dependencies
- **🔄 Service Actions**: Enable, disable, and restart services

### Key Improvements
- **🎯 Feature-Based Organization**: Code grouped by business domain rather than technical type
- **🔄 Reusable Components**: Frontend broken into modular, reusable components  
- **🌐 Centralized API Client**: Single point for backend communication
- **📦 Consistent Structure**: All features follow the same `api.py`, `service.py`, `models.py` pattern
- **🚀 Easy Extension**: Adding new features is now straightforward

## 🔧 Manual Installation (Alternative)

If you prefer manual installation or the automated script fails:

**📖 For detailed platform-specific instructions, see [INSTALL.md](INSTALL.md)**

### Step 1: Install Docker
Please install Docker from https://www.docker.com/get-started

### Step 2: Build and Run the Docker Container
```bash
docker build -t security-agent .
docker run -p 8000:8000 -p 8501:8501 -d --name security-agent security-agent
```

## 📋 Quick Reference

### One-Line Deployment
```bash
./run.sh
```

### Service Management
```bash
./status.sh  # Check service status
./stop.sh    # Stop all services
./run.sh     # Start all services
```

### Access URLs
- **Frontend**: http://localhost:8501
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Service Management**: http://localhost:8501 → Service Management
- **Service API**: http://localhost:8000/api/v1/services/

### Stop Services
```bash
docker stop security-agent
```

## 🛠️ Configuration

### Environment Variables
Create a `.env` file and add the following variables:
```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Service Configuration (Modular Architecture)
SERVICE_CONFIG_PATH="backend/config/services.json"

# ADK Configuration
ADK_EVALUATION_ENABLED="true"

# Vertex AI Configuration (for enterprise features)
VERTEX_AI_PROJECT_ID="your-project-id"
VERTEX_AI_LOCATION="us-central1"
```

### Service Configuration
The modular architecture uses a service configuration file to manage which services are enabled:

```json
{
  "services": {
    "iam": {
      "enabled_by_default": true,
      "config": {
        "cache_ttl": 300,
        "max_users_per_scan": 100
      }
    },
    "compliance": {
      "enabled_by_default": true,
      "config": {
        "frameworks": ["SOC2", "ISO27001", "GDPR"]
      }
    }
  },
  "runtime_status": {
    "iam": "not_configured",
    "compliance": "not_configured"
  }
}
```

### Service Management API
Control services programmatically:

```bash
# List all services
curl http://localhost:8000/api/v1/services/

# Get service details  
curl http://localhost:8000/api/v1/services/iam

# Enable a service
curl -X POST http://localhost:8000/api/v1/services/iam/enable

# Disable a service
curl -X POST http://localhost:8000/api/v1/services/threat_intelligence/disable

# Check service health
curl http://localhost:8000/api/v1/services/iam/health

# Get status summary
curl http://localhost:8000/api/v1/services/status/summary
```

## 🚀 Getting Started

Choose your preferred deployment method based on your use case:

### Option 1: One-Command Deployment (Recommended)
Perfect for quick evaluation and testing:

```bash
git clone https://github.com/google/adk-python.git
cd adk-python/contributing/samples/security_agent
python run.py
```

The script automatically:
- ✅ Detects your environment and sets up dependencies
- ✅ Starts modular backend with service management
- ✅ Launches frontend with service control UI
- ✅ Opens browser tabs to key interfaces
- ✅ Enables gradual service activation

### Option 2: Local Development
Best for development, customization, and debugging:

#### Prerequisites
- **Python 3.8+**
- **Google Cloud SDK** (optional, for GCP integration)

#### Step-by-Step Setup

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/google/adk-python.git
   cd adk-python/contributing/samples/security_agent
   ```

2. **Configure Environment**
   Create a `.env` file with your settings:
   ```bash
   # Essential Configuration
   GOOGLE_CLOUD_PROJECT="your-project-id"
   
   # Optional: Service Account (if not using default credentials)
   GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   
   # ADK Features
   ADK_EVALUATION_ENABLED="true"
   
   # Enterprise Features (optional)
   VERTEX_AI_PROJECT_ID="your-project-id"
   VERTEX_AI_LOCATION="us-central1"
   ```

3. **Launch the Application**
   ```bash
   # Create virtual environment and install dependencies
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Option A: Use the integrated launcher
   python run.py
   
   # Option B: Start services manually
   # Terminal 1: Start modular backend
   cd backend && uvicorn main_modular:app --host 0.0.0.0 --port 8000 --reload
   
   # Terminal 2: Start frontend
   cd frontend && streamlit run main_app.py --server.port 8501
   ```

4. **Access the Application**
   - 🌐 **Main Interface**: http://localhost:8501
   - ⚙️ **Service Management**: http://localhost:8501 → Service Management
   - 🔧 **API Documentation**: http://localhost:8000/docs
   - 📊 **Service Status**: http://localhost:8000/api/v1/services/status/summary

### Option 3: Docker Deployment
Ideal for production-like environments and containerized deployments:

#### Prerequisites
- **Docker Desktop** or **Docker Engine**

#### Quick Docker Setup
```bash
# Clone repository
git clone https://github.com/google/adk-python.git
cd adk-python/contributing/samples/security_agent

# Option A: Use integrated Docker launcher
python run.py --docker

# Option B: Build and run manually
docker build -t gcp-security-agent .
docker run -p 8000:8000 -p 8501:8501 gcp-security-agent
```

#### Docker Compose (Alternative)
```bash
# If docker-compose.yml exists
docker-compose up --build
```

## 🎯 First Steps After Installation

Once the application is running, follow these steps to get started:

### 1. **Verify Installation**
   - Navigate to http://localhost:8501
   - Check that the dashboard loads without errors
   - Verify backend connectivity (green status indicator)

### 2. **Configure GCP Integration** (Optional)
   ```bash
   # Authenticate with Google Cloud
   gcloud auth login
   gcloud auth application-default login
   
   # Set default project
   gcloud config set project YOUR_PROJECT_ID
   ```

### 3. **Explore Key Features**
   - 🏠 **Dashboard**: Overview of security posture and system status
   - ⚙️ **Service Management**: Enable/disable services as needed
   - 🛡️ **Security Evaluation**: Run comprehensive security scans
   - 🎯 **Recommendations**: Get AI-powered security advice
   - 🔐 **IAM Analysis**: Review user permissions and policies
   - 💬 **AI Assistant**: Chat with the security agent

### 4. **Configure Services** (New!)
   - Navigate to **Service Management** in the sidebar
   - Review available services and their status
   - Enable services gradually based on your needs
   - Monitor service health in real-time
   - Use service control toggles to manage functionality

### 5. **Test Core Functionality**
   - Select a GCP project from the sidebar
   - Run a security evaluation
   - Explore the API documentation at http://localhost:8000/docs
   - Try the service management APIs

## 🔧 Configuration Options

### Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP Project ID | ✅ | None |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service Account JSON path | ❌ | Default credentials |
| `ADK_EVALUATION_ENABLED` | Enable ADK agent features | ❌ | `false` |
| `VERTEX_AI_PROJECT_ID` | Vertex AI project | ❌ | Same as `GOOGLE_CLOUD_PROJECT` |
| `VERTEX_AI_LOCATION` | Vertex AI region | ❌ | `us-central1` |

### Application Settings
- **Backend Port**: `8000` (configurable via `--port`)
- **Frontend Port**: `8501` (configurable via `--server.port`)
- **Log Level**: `INFO` (configurable via environment)

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000/8501 in use | Kill existing processes: `lsof -ti:8000 \| xargs kill -9` |
| Service won't start | Check Service Management UI for dependencies and health status |
| Import errors | Ensure virtual environment is activated |
| GCP authentication errors | Run `gcloud auth application-default login` |
| Docker container issues | Check Docker daemon is running |
| Feature not available | Check if corresponding service is enabled in Service Management |

Need more help? See the [Troubleshooting](#troubleshooting) section below.

## 📚 Additional Documentation

- **[Modular Architecture Guide](MODULAR_ARCHITECTURE.md)**: Comprehensive guide to the service-based architecture
- **[Installation Guide](INSTALL.md)**: Detailed platform-specific setup instructions

## 🔐 GCP APIs & Service Account Permissions

This section details all Google Cloud APIs that need to be enabled and the specific IAM permissions required for the service account used by each module.

### 📋 Required GCP APIs

The following APIs must be enabled in your GCP project for full functionality:

#### Core Platform APIs
```bash
# Resource Management
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable serviceusage.googleapis.com

# Identity & Access Management
gcloud services enable iam.googleapis.com
gcloud services enable iamcredentials.googleapis.com

# Security & Compliance
gcloud services enable securitycenter.googleapis.com
gcloud services enable cloudkms.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Monitoring & Observability
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable cloudtrace.googleapis.com
gcloud services enable clouderrorreporting.googleapis.com
```

#### Feature-Specific APIs
```bash
# Compute & Infrastructure
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable appengine.googleapis.com

# Storage & Databases
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable sql.googleapis.com
gcloud services enable firestore.googleapis.com

# AI & Machine Learning
gcloud services enable aiplatform.googleapis.com
gcloud services enable ml.googleapis.com
gcloud services enable translate.googleapis.com
gcloud services enable vision.googleapis.com
gcloud services enable language.googleapis.com

# Networking
gcloud services enable dns.googleapis.com
gcloud services enable servicenetworking.googleapis.com

# DevOps & CI/CD
gcloud services enable cloudbuild.googleapis.com
gcloud services enable sourcerepo.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Recommendations
gcloud services enable recommender.googleapis.com
```

### 🛡️ Service Account IAM Roles & Permissions

Create a service account with the following roles, organized by module functionality:

#### Minimum Required Roles
```bash
# Create service account
gcloud iam service-accounts create security-agent \
    --display-name="GCP Security Agent" \
    --description="Service account for GCP Security Agent"

# Basic project access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/viewer"
```

### 📊 Module-Specific Permissions

#### **GCP Module** (`backend/gcp/`)
**APIs Used:**
- `cloudresourcemanager.googleapis.com` - Project discovery and metadata
- `serviceusage.googleapis.com` - API enablement status
- `recommender.googleapis.com` - Security recommendations

**Required Roles:**
```bash
# Resource Manager permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/resourcemanager.projectViewer"

# Service usage permissions  
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/serviceusage.serviceUsageViewer"

# Recommender access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/recommender.viewer"
```

**Granular Permissions:**
- `resourcemanager.projects.get`
- `resourcemanager.projects.list`
- `serviceusage.services.list`
- `serviceusage.services.get`
- `recommender.recommendations.get`
- `recommender.recommendations.list`

#### **Security Module** (`backend/security/`)
**APIs Used:**
- `securitycenter.googleapis.com` - Security findings and assets
- `compute.googleapis.com` - Infrastructure security scanning
- `cloudkms.googleapis.com` - Encryption key analysis

**Required Roles:**
```bash
# Security Center access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/securitycenter.findingsViewer"

# Compute security scanning
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.securityAdmin"

# KMS key analysis
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudkms.viewer"
```

**Granular Permissions:**
- `securitycenter.findings.list`
- `securitycenter.assets.list`
- `compute.instances.list`
- `compute.networks.list`
- `cloudkms.keyRings.list`
- `cloudkms.cryptoKeys.list`

#### **IAM Module** (`backend/iam/`)
**APIs Used:**
- `iam.googleapis.com` - IAM policy analysis
- `cloudresourcemanager.googleapis.com` - Project IAM policies

**Required Roles:**
```bash
# IAM policy analysis
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.securityReviewer"

# Resource Manager IAM access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/resourcemanager.projectIamAdmin"
```

**Granular Permissions:**
- `resourcemanager.projects.getIamPolicy`
- `iam.serviceAccounts.list`
- `iam.roles.list`
- `iam.roles.get`

#### **Compliance Module** (`backend/compliance/`)
**APIs Used:**
- `securitycenter.googleapis.com` - Compliance posture
- `cloudkms.googleapis.com` - Encryption compliance
- `logging.googleapis.com` - Audit log compliance

**Required Roles:**
```bash
# Security compliance monitoring
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/securitycenter.complianceViewer"

# Audit log access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.viewer"
```

**Granular Permissions:**
- `securitycenter.muteconfigs.list`
- `logging.logEntries.list`
- `cloudkms.keyRings.getIamPolicy`

#### **Cloud Logging Module** (`backend/cloud_logging/`)
**APIs Used:**
- `logging.googleapis.com` - Log querying and analysis
- `monitoring.googleapis.com` - Log-based metrics

**Required Roles:**
```bash
# Cloud Logging access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.viewer"

# Monitoring for log metrics
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/monitoring.viewer"
```

**Granular Permissions:**
- `logging.logEntries.list`
- `logging.logs.list`
- `monitoring.metricDescriptors.list`
- `monitoring.timeSeries.list`

#### **Tracing Module** (`backend/tracing/`)
**APIs Used:**
- `cloudtrace.googleapis.com` - Distributed tracing data
- `monitoring.googleapis.com` - Trace metrics

**Required Roles:**
```bash
# Cloud Trace access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudtrace.user"
```

**Granular Permissions:**
- `cloudtrace.traces.list`
- `cloudtrace.traces.get`

#### **Recommendations Module** (`backend/recommendations/`)
**APIs Used:**
- `recommender.googleapis.com` - Security recommendations
- `aiplatform.googleapis.com` - AI-powered insights

**Required Roles:**
```bash
# Recommender access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/recommender.viewer"

# Vertex AI for enhanced recommendations
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

**Granular Permissions:**
- `recommender.recommendations.list`
- `recommender.recommendations.get`
- `aiplatform.models.predict`

### 🚀 Quick Setup Script

Use this Python script to enable all required APIs and set up the service account:

```python
#!/usr/bin/env python3
"""
setup_gcp_permissions.py - GCP Security Agent Setup Script

This script automates the setup of required GCP APIs and service account permissions
for the GCP Security Agent application.

Usage:
    python setup_gcp_permissions.py --project-id YOUR_PROJECT_ID

Requirements:
    pip install google-cloud-resource-manager google-cloud-iam google-cloud-service-usage
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    from google.cloud import resourcemanager_v3
    from google.cloud import iam
    from google.cloud import service_usage_v1
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("📦 Install with: pip install google-cloud-resource-manager google-cloud-iam google-cloud-service-usage")
    sys.exit(1)


class GCPSecurityAgentSetup:
    """Setup class for GCP Security Agent permissions and APIs."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.service_account_name = "security-agent"
        self.service_account_email = f"{self.service_account_name}@{project_id}.iam.gserviceaccount.com"
        
        # Required APIs for the Security Agent
        self.required_apis = [
            "cloudresourcemanager.googleapis.com",
            "serviceusage.googleapis.com", 
            "iam.googleapis.com",
            "iamcredentials.googleapis.com",
            "securitycenter.googleapis.com",
            "cloudkms.googleapis.com",
            "secretmanager.googleapis.com",
            "monitoring.googleapis.com",
            "logging.googleapis.com",
            "cloudtrace.googleapis.com",
            "clouderrorreporting.googleapis.com",
            "compute.googleapis.com",
            "container.googleapis.com",
            "run.googleapis.com",
            "appengine.googleapis.com",
            "storage.googleapis.com",
            "bigquery.googleapis.com",
            "sql.googleapis.com",
            "firestore.googleapis.com",
            "aiplatform.googleapis.com",
            "ml.googleapis.com",
            "dns.googleapis.com",
            "servicenetworking.googleapis.com",
            "cloudbuild.googleapis.com",
            "sourcerepo.googleapis.com",
            "artifactregistry.googleapis.com",
            "recommender.googleapis.com"
        ]
        
        # Required IAM roles for the service account
        self.required_roles = [
            "roles/viewer",
            "roles/resourcemanager.projectViewer",
            "roles/serviceusage.serviceUsageViewer",
            "roles/recommender.viewer",
            "roles/securitycenter.findingsViewer",
            "roles/compute.securityAdmin",
            "roles/cloudkms.viewer",
            "roles/iam.securityReviewer",
            "roles/resourcemanager.projectIamAdmin",
            "roles/securitycenter.complianceViewer",
            "roles/logging.viewer",
            "roles/monitoring.viewer",
            "roles/cloudtrace.user",
            "roles/aiplatform.user"
        ]
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated with GCP."""
        try:
            credentials, project = default()
            print(f"✅ Authenticated with GCP (default project: {project})")
            return True
        except DefaultCredentialsError:
            print("❌ Not authenticated with GCP")
            print("🔧 Run: gcloud auth application-default login")
            return False
    
    def enable_apis(self) -> bool:
        """Enable required GCP APIs."""
        print("📡 Enabling required APIs...")
        
        try:
            # Use gcloud command for API enablement (most reliable method)
            cmd = ["gcloud", "services", "enable"] + self.required_apis + [f"--project={self.project_id}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ Successfully enabled {len(self.required_apis)} APIs")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to enable APIs: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ gcloud CLI not found. Please install Google Cloud SDK")
            return False
    
    def create_service_account(self) -> bool:
        """Create the security agent service account."""
        print("👤 Creating service account...")
        
        try:
            cmd = [
                "gcloud", "iam", "service-accounts", "create", self.service_account_name,
                "--display-name=GCP Security Agent",
                "--description=Service account for GCP Security Agent",
                f"--project={self.project_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Created service account: {self.service_account_email}")
            return True
            
        except subprocess.CalledProcessError as e:
            if "already exists" in e.stderr:
                print(f"ℹ️  Service account already exists: {self.service_account_email}")
                return True
            else:
                print(f"❌ Failed to create service account: {e.stderr}")
                return False
    
    def assign_roles(self) -> bool:
        """Assign required IAM roles to the service account."""
        print("🔐 Assigning IAM roles...")
        
        success_count = 0
        for role in self.required_roles:
            try:
                cmd = [
                    "gcloud", "projects", "add-iam-policy-binding", self.project_id,
                    f"--member=serviceAccount:{self.service_account_email}",
                    f"--role={role}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"  ✅ Assigned role: {role}")
                success_count += 1
                
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to assign role {role}: {e.stderr}")
        
        print(f"🎯 Successfully assigned {success_count}/{len(self.required_roles)} roles")
        return success_count == len(self.required_roles)
    
    def create_service_account_key(self) -> bool:
        """Create and download service account key."""
        print("🔑 Creating service account key...")
        
        key_file = "service-account-key.json"
        
        try:
            cmd = [
                "gcloud", "iam", "service-accounts", "keys", "create", key_file,
                f"--iam-account={self.service_account_email}",
                f"--project={self.project_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if Path(key_file).exists():
                print(f"✅ Service account key saved as '{key_file}'")
                return True
            else:
                print("❌ Service account key file not found after creation")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create service account key: {e.stderr}")
            return False
    
    def generate_env_config(self) -> None:
        """Generate .env configuration instructions."""
        key_path = Path("service-account-key.json").absolute()
        
        print("\n📝 Add the following to your .env file:")
        print("=" * 50)
        print(f'GOOGLE_APPLICATION_CREDENTIALS="{key_path}"')
        print(f'GOOGLE_CLOUD_PROJECT="{self.project_id}"')
        print("ADK_EVALUATION_ENABLED=true")
        print(f'VERTEX_AI_PROJECT_ID="{self.project_id}"')
        print('VERTEX_AI_LOCATION="us-central1"')
        print("=" * 50)
        
        # Optionally create .env file
        create_env = input("\n❓ Create .env file automatically? (y/N): ").lower().strip()
        if create_env == 'y':
            try:
                with open('.env', 'w') as f:
                    f.write(f'GOOGLE_APPLICATION_CREDENTIALS="{key_path}"\n')
                    f.write(f'GOOGLE_CLOUD_PROJECT="{self.project_id}"\n')
                    f.write('ADK_EVALUATION_ENABLED=true\n')
                    f.write(f'VERTEX_AI_PROJECT_ID="{self.project_id}"\n')
                    f.write('VERTEX_AI_LOCATION="us-central1"\n')
                
                print("✅ .env file created successfully!")
            except Exception as e:
                print(f"❌ Failed to create .env file: {e}")
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        print(f"🔧 Setting up GCP Security Agent permissions for project: {self.project_id}\n")
        
        # Check authentication
        if not self.check_authentication():
            return False
        
        # Enable APIs
        if not self.enable_apis():
            return False
        
        # Create service account
        if not self.create_service_account():
            return False
        
        # Assign roles
        if not self.assign_roles():
            print("⚠️  Some roles failed to assign. The application may have limited functionality.")
        
        # Create service account key
        if not self.create_service_account_key():
            return False
        
        # Generate .env configuration
        self.generate_env_config()
        
        print("\n🎉 Setup completed successfully!")
        print("🚀 You can now run the GCP Security Agent with: ./run.py")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up GCP APIs and service account permissions for the Security Agent"
    )
    parser.add_argument(
        "--project-id", 
        required=True,
        help="GCP Project ID to set up permissions for"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        setup = GCPSecurityAgentSetup(args.project_id)
        print(f"📡 Would enable {len(setup.required_apis)} APIs")
        print(f"🔐 Would assign {len(setup.required_roles)} IAM roles")
        print(f"👤 Would create service account: {setup.service_account_email}")
        return
    
    setup = GCPSecurityAgentSetup(args.project_id)
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Install dependencies
pip install google-cloud-resource-manager google-cloud-iam google-cloud-service-usage

# Run setup
python setup_gcp_permissions.py --project-id your-project-id

# Dry run to see what would be done
python setup_gcp_permissions.py --project-id your-project-id --dry-run
```

### ⚠️ Security Best Practices

1. **Principle of Least Privilege**: The permissions listed above are the minimum required. Review and adjust based on your specific needs.

2. **Service Account Key Management**: 
   - Store service account keys securely
   - Rotate keys regularly (recommended: every 90 days)
   - Never commit keys to version control

3. **API Quotas**: Monitor API usage to avoid quota limits, especially for:
   - Cloud Resource Manager API: 300 requests/minute
   - Security Center API: 100 requests/minute  
   - Recommender API: 100 requests/minute

4. **Network Security**: Consider using VPC Service Controls for additional security in production environments.

## 🌐 Networking Architecture

The GCP Security Agent uses a modern microservices architecture with clear separation between frontend and backend components. Here's how networking is implemented:

### 📊 Network Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser<br/>localhost:8501]
        CLI[API Clients<br/>curl, Postman, etc.]
    end
    
    subgraph "Application Layer - Local Machine"
        Frontend[Streamlit Frontend<br/>Port 8501<br/>0.0.0.0:8501]
        Backend[FastAPI Backend<br/>Port 8000<br/>0.0.0.0:8000]
        
        subgraph "Internal Communication"
            APIClient[API Client<br/>HTTP Requests]
        end
    end
    
    subgraph "Google Cloud Platform"
        GCPAPIs[GCP APIs<br/>*.googleapis.com<br/>HTTPS/443]
        
        subgraph "GCP Services"
            ResourceManager[Resource Manager API]
            SecurityCenter[Security Center API]
            IAMService[IAM Service API]
            CloudLogging[Cloud Logging API]
            CloudTrace[Cloud Trace API]
            VertexAI[Vertex AI API]
            CloudKMS[Cloud KMS API]
        end
    end
    
    subgraph "External Services"
        ADK[Agent Development Kit<br/>localhost:8080<br/>HTTP/WebSocket]
    end
    
    %% Client connections
    Browser --> Frontend
    CLI --> Backend
    
    %% Internal connections
    Frontend --> APIClient
    APIClient --> Backend
    
    %% External connections
    Backend --> GCPAPIs
    Backend --> ADK
    
    %% Service connections
    GCPAPIs --> ResourceManager
    GCPAPIs --> SecurityCenter
    GCPAPIs --> IAMService
    GCPAPIs --> CloudLogging
    GCPAPIs --> CloudTrace
    GCPAPIs --> VertexAI
    GCPAPIs --> CloudKMS
    
    %% Styling
    classDef client fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef app fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef gcp fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef external fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class Browser,CLI client
    class Frontend,Backend,APIClient app
    class GCPAPIs,ResourceManager,SecurityCenter,IAMService,CloudLogging,CloudTrace,VertexAI,CloudKMS gcp
    class ADK external
```

### 🔌 Network Components

#### **1. Frontend (Streamlit) - Port 8501**
- **Binding**: `0.0.0.0:8501` (accepts connections from any interface)
- **Protocol**: HTTP 
- **Purpose**: Web-based user interface
- **Access**: `http://localhost:8501`

**Key Features:**
- **Server Configuration**: 
  ```python
  streamlit run main_app.py --server.port 8501 --server.address 0.0.0.0
  ```
- **CORS**: Managed by Streamlit automatically
- **Static Assets**: Served directly by Streamlit
- **WebSocket**: Used for real-time UI updates

#### **2. Backend (FastAPI) - Port 8000**  
- **Binding**: `0.0.0.0:8000` (accepts connections from any interface)
- **Protocol**: HTTP with automatic OpenAPI documentation
- **Purpose**: REST API server and business logic
- **Access**: `http://localhost:8000`

**Key Features:**
- **CORS Configuration**:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

#### **3. Internal Communication (Frontend ↔ Backend)**
- **Protocol**: HTTP/1.1 over TCP
- **Authentication**: None (local development)
- **Request Format**: JSON REST API
- **Client**: Custom `SecurityAgentAPIClient` class

**Communication Flow:**
```python
# Frontend API Client
class SecurityAgentAPIClient:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None):
        # Makes HTTP requests to backend
        response = requests.post(f"{self.backend_url}{endpoint}", json=data)
```

### 🌍 External Network Connections

#### **1. Google Cloud Platform APIs**
- **Endpoints**: `*.googleapis.com` (HTTPS/443)
- **Authentication**: Service Account with OAuth2 
- **Protocol**: HTTPS with TLS 1.2+
- **Libraries**: Google Cloud Client Libraries

**Primary API Endpoints:**
```python
GCP_API_ENDPOINTS = {
    "Resource Manager": "https://cloudresourcemanager.googleapis.com/v3/",
    "Security Center": "https://securitycenter.googleapis.com/v1/",
    "IAM": "https://iam.googleapis.com/v1/",
    "Cloud Logging": "https://logging.googleapis.com/v2/",
    "Cloud Trace": "https://cloudtrace.googleapis.com/v2/",
    "Vertex AI": "https://aiplatform.googleapis.com/v1/",
    "Service Usage": "https://serviceusage.googleapis.com/v1/"
}
```

#### **2. Agent Development Kit (ADK)**
- **Endpoint**: `http://localhost:8080` (when running locally)
- **Protocol**: HTTP/WebSocket for streaming responses
- **Purpose**: AI/ML model inference and agent orchestration
- **Integration**: Used by chat and recommendation features

### 🔒 Security & Network Configuration

#### **Network Security Features:**

1. **CORS Protection**:
   ```python
   # Restricts frontend origins
   allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"]
   ```

2. **TLS/SSL**:
   - **Local Development**: HTTP (acceptable for localhost)
   - **Production**: HTTPS recommended with reverse proxy
   - **GCP APIs**: Always HTTPS with certificate validation

3. **Authentication Flow**:
   ```mermaid
   sequenceDiagram
       participant F as Frontend
       participant B as Backend  
       participant G as GCP APIs
       
       F->>B: API Request (localhost:8000)
       B->>B: Load Service Account Credentials
       B->>G: Authenticated API Call (HTTPS)
       G->>B: API Response
       B->>F: JSON Response
   ```

4. **Network Isolation**:
   - **Development**: All services on localhost
   - **Production**: Consider VPC, private subnets, and firewall rules

#### **Firewall & Port Configuration:**

**Required Open Ports:**
- **8501**: Streamlit frontend (inbound)
- **8000**: FastAPI backend (inbound) 
- **443**: HTTPS to GCP APIs (outbound)
- **80**: HTTP for package downloads (outbound)
- **8080**: ADK service (localhost, optional)

**Firewall Rules (Production):**
```bash
# Allow frontend access
ufw allow 8501/tcp

# Allow API access  
ufw allow 8000/tcp

# Allow HTTPS outbound (GCP APIs)
ufw allow out 443/tcp

# Block other inbound traffic
ufw deny in
```

### 📡 Request Flow Examples

#### **1. Security Evaluation Request Flow:**
```
1. User clicks "Run Security Scan" (Frontend:8501)
2. Frontend → POST /api/v1/security/evaluate (Backend:8000)
3. Backend → Security Center API (securitycenter.googleapis.com:443)
4. Backend → Service Usage API (serviceusage.googleapis.com:443)
5. Backend ← Aggregated security data
6. Frontend ← JSON response with security findings
```

#### **2. Chat Request Flow:**
```
1. User types message (Frontend:8501)
2. Frontend → POST /api/v1/agent/chat (Backend:8000)
3. Backend → ADK Service (localhost:8080)
4. Backend → GCP APIs for context data (*.googleapis.com:443)
5. Backend ← AI-generated response
6. Frontend ← Streamed chat response
```

### 🚀 Production Networking Considerations

#### **Deployment Options:**

1. **Docker Deployment**:
   ```yaml
   services:
     frontend:
       ports:
         - "8501:8501"
     backend:
       ports:
         - "8000:8000"
   ```

2. **Reverse Proxy (nginx)**:
   ```nginx
   server {
       listen 80;
       location / {
           proxy_pass http://localhost:8501;
       }
       location /api/ {
           proxy_pass http://localhost:8000;
       }
   }
   ```

3. **Cloud Deployment**:
   - **Google Cloud Run**: Automatic HTTPS and scaling
   - **Kubernetes**: Service mesh and ingress controllers
   - **Compute Engine**: Manual configuration with Load Balancer

#### **Network Performance:**
- **Local Development**: ~1-5ms latency between components
- **GCP API Calls**: ~50-200ms depending on region and API
- **Concurrent Connections**: FastAPI supports async for high throughput
- **Request Rate Limits**: Managed per GCP API quotas

#### **Monitoring & Observability:**
- **OpenTelemetry**: Traces network requests to GCP APIs
- **Cloud Trace**: Distributed tracing for request flows
- **Cloud Logging**: Network request/response logging
- **Health Checks**: `/health` endpoint for load balancer probes

### 🛠️ Network Troubleshooting

**Common Issues:**

1. **Connection Refused (localhost:8000)**:
   ```bash
   # Check if backend is running
   lsof -i :8000
   
   # Start backend manually
   cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **CORS Errors**:
   ```python
   # Update CORS origins in backend/main.py
   allow_origins=["http://localhost:8501", "https://your-domain.com"]
   ```

3. **GCP API Connectivity**:
   ```bash
   # Test API access
   curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        https://cloudresourcemanager.googleapis.com/v3/projects
   ```

4. **Network Latency**:
   ```bash
   # Test API response times
   curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/v1/gcp/projects
   ```

This networking architecture provides a robust, scalable foundation for the GCP Security Agent while maintaining security best practices and ease of development.

## 🎭 Mock vs Real Data Implementation Status

The GCP Security Agent implements a hybrid approach, combining real GCP API integrations with mock data for features that require additional setup or are demonstration-focused. Here's the complete breakdown:

### ✅ **Real GCP API Integration** (Production Ready)

#### **Core GCP Operations** (`backend/gcp/`)
- ✅ **Project Discovery**: Live GCP Resource Manager API calls
- ✅ **Service Listing**: Real Service Usage API integration  
- ✅ **Project Information**: Actual project metadata from GCP
- ✅ **API Enablement Status**: Real-time API status checking

```python
# Real API calls in gcp/service.py
def get_projects(self):
    client = ProjectsClient(credentials=self.credentials)
    request = ListProjectsRequest()
    projects = client.list_projects(request=request)
```

#### **Identity & Access Management** (`backend/iam/`)
- ✅ **IAM Policy Analysis**: Live policy retrieval from GCP
- ✅ **Role Analysis**: Real IAM role and permission checking
- ✅ **Service Account Enumeration**: Actual service account discovery
- ✅ **Permission Evaluation**: Real-time IAM policy evaluation

#### **Cloud Logging** (`backend/cloud_logging/`)
- ✅ **Log Querying**: Direct Cloud Logging API integration
- ✅ **Log Search**: Real log entry retrieval and filtering
- ✅ **Log Analytics**: Actual log-based metrics and insights

```python
# Real Cloud Logging integration
def get_recent_logs(self, project_id: str, hours: int = 24):
    entries = self.client.list_entries(
        resource_names=[f"projects/{project_id}"],
        filter_=f"timestamp >= \"{start_time.isoformat()}Z\""
    )
```

#### **Security Center Integration** (Partial)
- ✅ **Security Findings**: Basic Security Center API calls
- ✅ **Asset Discovery**: Real asset enumeration
- ⚠️ **Advanced Scanning**: Requires Security Center API setup

#### **Authentication & Authorization**
- ✅ **Service Account Authentication**: Real Google Cloud authentication
- ✅ **OAuth2 Token Management**: Live token refresh and validation
- ✅ **GCP Credentials**: Production credential handling

### 🎭 **Mock/Simulated Data** (Demo/Development)

#### **Incident Response** (`frontend/components/incident_response_view.py`)
- 🎭 **Incident Data**: Mock incident records with realistic scenarios
- 🎭 **Analytics**: Simulated incident metrics and trends
- 🎭 **Timeline**: Generated incident history and responses

```python
def get_mock_incidents():
    """Generate mock incident data."""
    return [
        {
            "id": "INC-001",
            "title": "Suspicious login attempts from unknown IP",
            "severity": "High",
            # ... realistic but simulated data
        }
    ]
```

#### **Performance Monitoring** (`frontend/components/performance_monitoring_view.py`)
- 🎭 **System Metrics**: Generated performance data
- 🎭 **SRE Dashboards**: Simulated monitoring charts
- 🎭 **Error Analytics**: Mock error distribution data

```python
# Generate mock time series data
dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
metrics = [random.randint(80, 100) for _ in range(len(dates))]
```

#### **OIDC Authentication Flow** (`frontend/components/oidc_flow_view.py`)
- 🎭 **Token Exchange**: Simulated OIDC token flow
- 🎭 **User Profile**: Mock user information display  
- 🎭 **Authorization**: Demo authentication workflow

```python
# Mock token response
mock_tokens = {
    "access_token": "mock_access_token_12345",
    "id_token": "mock_id_token_67890",
    "refresh_token": "mock_refresh_token_abcde"
}
```

#### **API Explorer Analytics** (`frontend/components/api_explorer_view.py`)
- 🎭 **Usage Statistics**: Simulated API usage metrics
- 🎭 **Response Times**: Generated performance data
- 🎭 **Error Analytics**: Mock error distribution charts

#### **Distributed Tracing** (`backend/tracing/`)
- ⚠️ **Trace Data**: Partial Cloud Trace integration (requires setup)
- 🎭 **Trace Analytics**: Simulated trace statistics
- 🎭 **Performance Metrics**: Mock tracing dashboards

### ⚠️ **Partial Integration** (Requires Additional Setup)

#### **Security Center** (`backend/security/`)
- ⚠️ **Advanced Security Scanning**: Requires Security Center API activation
- ⚠️ **Vulnerability Assessment**: Needs Security Command Center setup
- ✅ **Basic Security Evaluation**: Functional with limited data

#### **Vertex AI Integration** (`agents/agent.py`)
- ⚠️ **AI Recommendations**: Requires Vertex AI setup and billing
- ⚠️ **Enhanced Chat**: Needs Vertex AI model configuration
- ✅ **Basic Agent**: Functional with ADK local setup

#### **API Hub Integration** (`backend/services/apihub_service.py`)
- 🎭 **API Discovery**: Placeholder implementation
- ⚠️ **Real API Hub**: Requires API Hub service activation

```python
# Placeholder implementation
async def get_api_deployments(self):
    # This is a placeholder for actual API Hub API call
    return {"deployments": []}
```

#### **Documentation Service** (`backend/documentation/`)
- 🎭 **API Documentation**: Mock documentation retrieval
- ⚠️ **Real Doc Scraping**: Placeholder for actual API doc integration

### 🚀 **How to Enable Full Integration**

#### **1. Enable Missing GCP APIs**
```bash
# Enable additional APIs for full functionality
gcloud services enable securitycenter.googleapis.com
gcloud services enable cloudsecurityscanner.googleapis.com
gcloud services enable apihub.googleapis.com
```

#### **2. Configure Security Center**
```bash
# Set up Security Center (requires organization-level permissions)
gcloud alpha security-center findings list --organization=YOUR_ORG_ID
```

#### **3. Set Up Vertex AI**
```bash
# Initialize Vertex AI
gcloud ai models list --region=us-central1
```

#### **4. Replace Mock Functions with Real Implementations**

**Example: Replace Mock Incidents with Real Security Findings**
```python
# In incident_response_view.py - Replace mock function
def get_real_security_incidents(project_id: str):
    """Get real security incidents from Security Center."""
    response = api_client.get_security_findings(project_id)
    return response.get('findings', [])

# Usage in component
incidents = get_real_security_incidents(st.session_state.selected_project)
```

**Example: Enable Real Tracing Data**
```python
# In tracing/service.py - Enable real trace retrieval
def get_traces(self, project_id: str, time_range: str = "1h"):
    """Get real distributed traces from Cloud Trace."""
    trace_client = trace_v1.TraceServiceClient(credentials=self.credentials)
    traces = trace_client.list_traces(
        project_id=project_id,
        view=trace_v1.ListTracesRequest.ViewType.COMPLETE
    )
    return [trace for trace in traces]
```

### 📊 **Integration Status Summary**

| Feature | Status | Notes |
|---------|--------|-------|
| 🏗️ **GCP Projects** | ✅ Real | Full Resource Manager integration |
| 🔐 **IAM Analysis** | ✅ Real | Complete IAM policy analysis |
| 📊 **Cloud Logging** | ✅ Real | Direct Cloud Logging API |
| 🛡️ **Basic Security** | ✅ Real | Service usage and basic scanning |
| 🔍 **Advanced Security** | ⚠️ Partial | Requires Security Center setup |
| 🚨 **Incident Response** | 🎭 Mock | Demo data for incident workflows |
| 📈 **Performance Monitoring** | 🎭 Mock | Simulated metrics and dashboards |
| 🔐 **OIDC Demo** | 🎭 Mock | Educational authentication flow |
| 🎯 **AI Recommendations** | ⚠️ Partial | Requires Vertex AI configuration |
| 📡 **Distributed Tracing** | ⚠️ Partial | Basic Cloud Trace integration |
| 🔍 **API Explorer** | 🎭 Mock | Demo analytics and usage stats |

### 🎯 **Recommended Next Steps**

1. **For Production Use**: Focus on enabling Security Center and Vertex AI APIs
2. **For Development**: Mock data provides full UI/UX testing capabilities  
3. **For Demos**: Current mix provides realistic experience without complex setup
4. **For Enterprise**: Replace all mock functions with real data integrations

The current implementation provides immediate value while allowing progressive enhancement toward full production deployment.

## 🔧 How to Extend the Application

The modular architecture makes it easy to add new features. Follow these steps:

### Adding a New Backend Feature

1. **Create Feature Directory Structure**:
   ```bash
   mkdir backend/my_feature
   touch backend/my_feature/__init__.py
   touch backend/my_feature/api.py
   touch backend/my_feature/service.py
   touch backend/my_feature/models.py
   ```

2. **Define Data Models** (`backend/my_feature/models.py`):
   ```python
   from pydantic import BaseModel
   from typing import List, Optional

   class MyFeatureRequest(BaseModel):
       project_id: str
       parameter: str

   class MyFeatureResponse(BaseModel):
       success: bool
       data: dict
       error: Optional[str] = None
   ```

3. **Implement Business Logic** (`backend/my_feature/service.py`):
   ```python
   class MyFeatureService:
       def __init__(self, credentials=None, project_id=None):
           self.credentials = credentials
           self.project_id = project_id
       
       def process_request(self, request_data: dict):
           # Implement your feature logic here
           return {"result": "success"}
   ```

4. **Create API Endpoints** (`backend/my_feature/api.py`):
   ```python
   from fastapi import APIRouter, Request
   from .models import MyFeatureRequest, MyFeatureResponse
   
   router = APIRouter()
   
   @router.post("/process")
   async def process_my_feature(request: Request, req: MyFeatureRequest):
       service = request.app.state.my_feature_service
       result = service.process_request(req.dict())
       return MyFeatureResponse(success=True, data=result)
   ```

5. **Register in Main App** (`backend/main.py`):
   ```python
   # Import your feature
   from my_feature.api import router as my_feature_router
   from my_feature.service import MyFeatureService
   
   # Initialize service in lifespan function
   app.state.my_feature_service = MyFeatureService(credentials, project_id)
   
   # Include router
   app.include_router(my_feature_router, prefix="/api/v1/my-feature", tags=["My Feature"])
   ```

### Adding a New Frontend Component

1. **Create Component File** (`frontend/components/my_feature_view.py`):
   ```python
   import streamlit as st
   from api_client import api_client
   
   def render_my_feature_view():
       st.header("🆕 My New Feature")
       
       if st.button("Process My Feature"):
           response = api_client._make_request(
               "/api/v1/my-feature/process", 
               "POST", 
               {"parameter": "value"}
           )
           
           if response.get("success"):
               st.success("Feature processed successfully!")
               st.json(response.get("data"))
           else:
               st.error(f"Error: {response.get('error')}")
   ```

2. **Add API Client Methods** (`frontend/api_client.py`):
   ```python
   def process_my_feature(self, parameter: str) -> Dict[str, Any]:
       """Process my feature request."""
       return self._make_request("/api/v1/my-feature/process", "POST", {
           "parameter": parameter
       })
   ```

3. **Register Component** (`frontend/components/__init__.py`):
   ```python
   from .my_feature_view import render_my_feature_view
   
   __all__ = [
       # ... existing exports ...
       'render_my_feature_view'
   ]
   ```

4. **Add Navigation** (`frontend/main_app.py`):
   ```python
   # Add to pages dict in render_navigation()
   pages = {
       # ... existing pages ...
       "my_feature": "🆕 My Feature"
   }
   
   # Add to render_main_content()
   elif page == "my_feature":
       render_my_feature_view()
   ```

### Best Practices

- **Follow the established patterns**: Each feature should have consistent `api.py`, `service.py`, and `models.py` files
- **Use type hints**: All functions should have proper type annotations
- **Add error handling**: Always handle potential errors gracefully
- **Write tests**: Add unit tests for your services and integration tests for APIs
- **Update documentation**: Document your new feature's API endpoints and usage

## Troubleshooting

### "Connection refused" error
If you encounter a "Connection refused" error, it's likely due to a lingering backend process. To fix this:

1.  **Find the process using port 8000**:
    ```bash
    lsof -i :8000
    ```
2.  **Kill the process using its PID**:
    ```bash
    kill <PID>
    ```
3.  **Restart the agent**:
    ```bash
    ./run.sh
    ```

### "v1/models" error
If you encounter a "v1/models" error in the ADK Chat, it means the agent is trying to call the public Gemini API instead of the local server. To fix this, modify `backend/services/agent_service.py` to use `LiteLlm`:

```python
from google.adk.models.lite_llm import LiteLlm

...

llm = LiteLlm(
    model="adk",
    api_base="http://localhost:8080/run/predict",
)
agent_module.root_agent.model = llm
```

### "invalid parent name" error
If you encounter an "invalid parent name" error when fetching GCP projects, it's likely due to an issue with the Resource Manager Python client. To fix this, modify `backend/api/gcp.py` to use a direct `curl` command:

```python
import subprocess
import json

...

token_process = subprocess.run(
    ["gcloud", "auth", "application-default", "print-access-token"],
    capture_output=True, text=True, check=True
)
access_token = token_process.stdout.strip()

...

curl_command = [
    "curl", "-X", "GET",
    "https://cloudresourcemanager.googleapis.com/v3/projects",
    "--header", f"Authorization: Bearer {access_token}",
    "--header", "Content-Type: application/json"
]

response_process = subprocess.run(
    curl_command,
    capture_output=True, text=True, check=True
)

data = json.loads(response_process.stdout)
```

### Docker Daemon Not Running
If you see an error message like "Cannot connect to the Docker daemon", make sure the Docker daemon is running.

-   **macOS:** Open the Docker Desktop application.
-   **Windows:** Open the Docker Desktop application.
-   **Linux:** Run `sudo systemctl start docker`.

### Port Conflicts
If you see an error message like "Port is already allocated", it means that another application is using port 8000 or 8501. You can either stop the other application or change the ports in the `docker-compose.yml` file.

### Issues with `gcloud`
If you have issues with `gcloud` commands, make sure you have the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated:
```bash
gcloud auth login
gcloud auth application-default login
```



