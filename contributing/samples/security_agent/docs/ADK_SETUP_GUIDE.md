# Google ADK Setup Guide

## 🚨 Required for Chat Interface

The ADK Security Agent chat interface requires Google Agent Development Kit (ADK) to function. Follow these steps to set it up:

## 1. Install Google ADK

```bash
# Install Google ADK (currently in preview)
pip install google-adk

# Install Google Generative AI
pip install google-generativeai

# Install Vertex AI
pip install google-cloud-aiplatform
```

## 2. Authentication Setup

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID
```

## 3. Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Generative AI API  
gcloud services enable generativelanguage.googleapis.com

# Enable Cloud Resource Manager API
gcloud services enable cloudresourcemanager.googleapis.com
```

## 4. Environment Variables

Create a `.env` file in the project root:

```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
VERTEX_AI_LOCATION=us-central1
```

## 5. Service Account Setup (Optional)

If not using user credentials:

```bash
# Create service account
gcloud iam service-accounts create adk-security-agent

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:adk-security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create adk-key.json \
  --iam-account=adk-security-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## 6. Test Installation

```python
# Test Google ADK import
python -c "from google.adk import Agent; print('✅ ADK installed')"

# Test Vertex AI
python -c "import vertexai; print('✅ Vertex AI available')"

# Test GenAI
python -c "from google.genai import types; print('✅ GenAI available')"
```

## 7. Alternative: Mock Mode (Development Only)

If you can't install ADK immediately, you can create a mock version for development:

```python
# Create mock_adk.py for testing
class MockAgent:
    def send_message(self, query):
        return f"Mock response for: {query}"

def create_coordinator_agent(project_id):
    return MockAgent()
```

## Troubleshooting

### Common Issues:

1. **"google.adk not found"**
   - ADK is in preview - ensure you have access
   - Try: `pip install --upgrade google-adk`

2. **Authentication errors**
   - Run: `gcloud auth application-default login`
   - Check project is set: `gcloud config get-value project`

3. **API not enabled**
   - Enable required APIs in Cloud Console
   - Wait a few minutes for propagation

### Next Steps:

Once ADK is installed:
1. Restart the backend: `python run_backend.py`
2. Restart the frontend: `python run_frontend.py`  
3. The chat interface should show "Connected to ADK agents"

## Support

For ADK-specific issues:
- [Google ADK Documentation](https://cloud.google.com/agent-development-kit)
- [ADK Python Client](https://github.com/googleapis/python-adk)

For this project:
- Check the `agents/` directory for ADK integration code
- Review `coordinator_agent.py` for delegation patterns