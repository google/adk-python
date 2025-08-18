# Team Setup Guide

## Quick Setup for New Team Members

### 1. Clone the Repository
```bash
git clone <repository-url>
cd security_agent
```

### 2. Set Up Your Environment

#### Create your .env file:
```bash
cp .env.example .env
```

#### Edit .env with your project details:
```bash
# Replace with YOUR Google Cloud project ID
GOOGLE_CLOUD_PROJECT=your-project-id

# Path to YOUR service account JSON file
GOOGLE_APPLICATION_CREDENTIALS=backend/config/secrets/your-service-account.json
```

### 3. Add Your Service Account Key

Place your service account JSON file in:
```
backend/config/secrets/your-service-account.json
```

**Note:** The system will automatically detect any `.json` file in the `backend/config/secrets/` directory. You don't need to use a specific filename.

### 4. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
pip install -r frontend/requirements.txt
```

### 5. Run the Application

```bash
# Start backend
python run_backend.py

# In another terminal, start frontend
python run_frontend.py
```

## Troubleshooting

### "Service account info was not in the expected format" Error

This means the system can't find or read your service account file. Check:

1. **File exists**: Ensure your service account JSON is in `backend/config/secrets/`
2. **Valid JSON**: The file should be a valid service account key from GCP
3. **Environment variable**: Check that `GOOGLE_APPLICATION_CREDENTIALS` points to the correct file
4. **Project ID**: Ensure `GOOGLE_CLOUD_PROJECT` is set to your GCP project ID

### Using Default Credentials

If you have `gcloud` CLI installed and authenticated:
```bash
gcloud auth application-default login
```

The system will fall back to these credentials if no service account file is found.

## Required GCP APIs

Ensure these APIs are enabled in your project:
```bash
gcloud services enable \
  aiplatform.googleapis.com \
  cloudasset.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  recommender.googleapis.com
```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | Your GCP project ID | `my-project-123` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON | `backend/config/secrets/sa.json` |
| `BACKEND_URL` | Backend API URL (for frontend) | `http://localhost:8000` |
| `VERTEX_AI_LOCATION` | Vertex AI region | `us-central1` |

## Notes for Different Environments

- **Development**: Use `.env` file with local paths
- **Production**: Use environment variables or Secret Manager
- **CI/CD**: Store credentials as secrets in your CI platform