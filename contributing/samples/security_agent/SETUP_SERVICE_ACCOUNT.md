# Service Account Setup for Vertex AI

To enable LLM-powered responses in the chat interface, you need to set up a service account with Vertex AI permissions.

## Steps:

1. **Create a Service Account** (if you don't have one):
   ```bash
   gcloud iam service-accounts create security-agent-sa \
     --display-name="Security Agent Service Account"
   ```

2. **Grant Vertex AI Permissions**:
   ```bash
   gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
     --member="serviceAccount:security-agent-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. **Create and Download Key**:
   ```bash
   gcloud iam service-accounts keys create \
     ./service-account-key.json \
     --iam-account=security-agent-sa@YOUR-PROJECT-ID.iam.gserviceaccount.com
   ```

4. **Update .env file**:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
   # OR
   SERVICE_ACCOUNT_FILENAME=service-account-key.json
   ```

5. **Place the key file** in one of these locations:
   - Project root: `/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/`
   - Keys folder: `/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/keys/`
   - Config folder: `/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/config/`

## Alternative: Use Application Default Credentials

If you have access to Vertex AI with your user account:

```bash
# Login with your Google account
gcloud auth application-default login

# Ensure Vertex AI API is enabled
gcloud services enable aiplatform.googleapis.com
```

## Verify Setup

Test your setup:
```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="YOUR-PROJECT-ID", location="us-central1")
model = GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Hello")
print(response.text)
```

## Current Status

The chat interface is configured to:
1. First look for a service account key file
2. Fall back to Application Default Credentials
3. If neither works, provide formatted responses without AI analysis

The AI analysis adds:
- Intelligent interpretation of security data
- Prioritized recommendations
- Context-aware responses based on selected tab
- Natural language explanations of technical findings