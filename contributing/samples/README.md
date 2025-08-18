# ADK Security Agent for Google Cloud Platform

## 🔐 AI-Powered Security Analysis & Monitoring

The ADK Security Agent is an intelligent security management system for Google Cloud Platform (GCP) environments, built on Google's Agent Development Kit (ADK). It provides real-time security insights, vulnerability detection, and automated remediation recommendations through natural language conversations.

## 🎯 Key Features

### RADAR Methodology
- **Recognition**: Automated discovery of all GCP resources
- **Assessment**: Comprehensive security vulnerability scanning
- **Decision**: AI-driven prioritization of security issues
- **Action**: Step-by-step remediation guidance
- **Review**: Continuous monitoring and verification

### Multi-Agent Architecture
- **Coordinator Agent**: Intelligent query routing and orchestration
- **Specialized Agents**: Dedicated agents for IAM, security, compliance, and more
- **LLM-Driven Delegation**: Smart routing based on query context

### Thin Client Architecture
- **Frontend**: Lightweight Streamlit UI for user interaction
- **Backend**: Powerful FastAPI server with all intelligence and processing
- **Security**: Zero credentials in frontend, all sensitive data isolated to backend

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Cloud Project with billing enabled
- Service account with appropriate permissions

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd security_agent
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration:
# - GOOGLE_CLOUD_PROJECT=your-project-id
# - GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

5. **Enable required GCP APIs**
```bash
gcloud services enable \
  aiplatform.googleapis.com \
  cloudasset.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  recommender.googleapis.com \
  secretmanager.googleapis.com
```

### Running Locally

1. **Start the backend server**
```bash
python run_backend.py
# Backend runs on http://localhost:8000
```

2. **Start the frontend client** (in a new terminal)
```bash
python run_frontend.py
# Frontend runs on http://localhost:8501
```

3. **Access the application**
   - Open browser to http://localhost:8501
   - Start chatting with the security agent

## 📚 Documentation

### Core Documentation (BMad Method)
- **[Product Requirements (PRD)](docs/prd.md)** - Product vision, features, and roadmap
- **[Architecture](docs/architecture.md)** - System design and component architecture
- **[Tech Stack](docs/architecture/tech-stack.md)** - Technologies and frameworks used
- **[Source Tree](docs/architecture/source-tree.md)** - Project structure and organization
- **[Coding Standards](docs/architecture/coding-standards.md)** - Development guidelines

### User Guides
- **[Quick Start Guide](docs/guides/quick-start.md)** - Getting started quickly
- **[Environment Setup](docs/guides/env-setup.md)** - Detailed setup instructions
- **[Deployment Guide](docs/guides/deployment.md)** - Production deployment
- **[Testing Guide](docs/guides/testing.md)** - Testing procedures
- **[Operations Manual](docs/guides/operations.md)** - Operational procedures

### Reference Documentation
- **[API Reference](docs/reference/api-reference.md)** - Complete API documentation
- **[Agent Patterns](docs/reference/agent-patterns.md)** - ADK delegation patterns
- **[RADAR Methodology](docs/reference/radar-methodology.md)** - RADAR framework details

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  FastAPI Backend │
│  (Thin Client)  │     │  (Intelligence)  │
└─────────────────┘     └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Google Cloud APIs  │
                    │  (Assets, IAM, etc.) │
                    └─────────────────────┘
```

### Key Components

#### Frontend (Thin Client)
- Pure presentation layer
- Real-time streaming responses
- Zero business logic
- Session management

#### Backend (Intelligence Layer)
- Agent orchestration (RADAR methodology)
- GCP API integration
- Security analysis engine
- Credential management
- Context persistence

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=backend --cov=frontend

# Run specific test suite
pytest tests/test_security.py
```

## 🚢 Deployment

### Cloud Run Deployment

1. **Build and deploy backend**
```bash
cd backend
gcloud run deploy adk-security-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

2. **Build and deploy frontend**
```bash
cd frontend
gcloud run deploy adk-security-frontend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars BACKEND_URL=<backend-url>
```

### Docker Deployment

```bash
# Build containers
docker-compose build

# Run locally
docker-compose up
```

## 🔒 Security Considerations

- **Credentials**: Never commit credentials; use Secret Manager in production
- **IAM**: Follow principle of least privilege for service accounts
- **Network**: Use VPC Service Controls for additional security
- **Audit**: Enable Cloud Audit Logs for all API calls

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## 📊 Project Status

### Current Version: 4.0
- ✅ Core RADAR implementation
- ✅ Multi-agent architecture
- ✅ GCP integration
- ✅ Thin client architecture

### Roadmap
- 🔄 Enhanced vulnerability detection (Q1 2025)
- 🔄 Automated remediation (Q2 2025)
- 🔄 Multi-cloud support (Q3 2025)

## 📝 License

[LICENSE](LICENSE)

## 🆘 Support

- **Documentation**: See `/docs` directory
- **Issues**: Report bugs via GitHub Issues
- **Questions**: Contact the development team

## 🙏 Acknowledgments

Built with:
- Google Agent Development Kit (ADK)
- Google Cloud Platform
- Streamlit
- FastAPI

---

**Note**: This is an active development project. Features and APIs may change.