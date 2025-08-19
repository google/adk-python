# 🚀 GCP Security Agent - Quick Start Guide

**Get the security agent running in under 5 minutes!**

## Prerequisites

- **Python 3.8+**
- **Docker** (for containerized deployment)
- **4GB RAM** (8GB recommended)
- **Google Cloud Project** (optional for full integration)

## Installation

### Option 1: One-Command Deployment (Recommended)

```bash
# Clone repository
git clone https://github.com/google/adk-python.git
cd adk-python/contributing/samples/security_agent

# Deploy everything
python run.py
```

**That's it!** The script will automatically:
- ✅ Check dependencies and setup environment
- ✅ Start FastAPI backend server
- ✅ Launch Streamlit frontend interface
- ✅ Open browser tabs automatically

### Option 2: Docker Deployment

```bash
docker build -t gcp-security-agent .
docker run -p 8000:8000 -p 8501:8501 gcp-security-agent
```

## Access Points

After deployment, access these URLs:

- **🌐 Frontend UI**: http://localhost:8501
- **🔧 Backend API**: http://localhost:8000/docs  
- **📊 API Health**: http://localhost:8000/health

## First Steps

1. **Verify Installation**
   - Navigate to http://localhost:8501
   - Check that the dashboard loads without errors
   - Verify backend connectivity (green status indicator)

2. **Explore Key Features**
   - 🏠 **Dashboard**: Security posture overview
   - 🛡️ **Security Evaluation**: Run comprehensive scans
   - 🎯 **Recommendations**: Get AI-powered advice
   - 🔐 **IAM Analysis**: Review permissions
   - 💬 **AI Assistant**: Chat with the security agent
   - 📊 **Monitoring**: Performance and health metrics

## Optional: GCP Integration

For full GCP integration, configure authentication:

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Create .env file with your settings
echo "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID" > .env
```

## Deployment Options

```bash
python run.py                    # Full application (default)
python run.py --docker           # Docker container deployment
python run.py --backend-only     # Backend server only
python run.py --frontend-only    # Frontend only
```

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000/8501 in use | `lsof -ti:8000 | xargs kill -9` |
| Backend won't start | Check logs and dependencies |
| Import errors | Ensure virtual environment is activated |
| Docker issues | Check Docker daemon is running |

## Next Steps

- **[📖 Full Documentation](README.md)** - Complete setup and configuration guide
- **[🏗️ Architecture Guide](MODULAR_ARCHITECTURE.md)** - Service-based architecture
- **[🔧 API Reference](API_REFERENCE.md)** - Complete API documentation  
- **[🛠️ Development Guide](../../../docs/DEVELOPMENT.md)** - Extending the application

---

**Need help?** Check the [troubleshooting section](README.md#troubleshooting) or open an issue on GitHub.