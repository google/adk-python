# Enhanced GCP API Security Evaluation Agent

This project is a comprehensive security evaluation platform that demonstrates OIDC authentication flow and provides advanced security analysis capabilities for GCP APIs, now with full ADK (Agent Development Kit) integration including agent evaluation.

## 🚀 One-Command Deployment

**Deploy the entire security agent from scratch with a single command:**

```bash
python run.py
```

This script will:
- ✅ Build the Docker container
- ✅ Run the Docker container
- ✅ Start backend and frontend servers
- ✅ Open access points automatically

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

## 🛠️ Roles and Permissions

This project interacts with Google Cloud Platform (GCP) services and requires specific IAM roles and permissions for its functionality. All authentication should be handled via [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/production#adc) and **never** by hardcoding credentials.

### Required Roles

- **Vertex AI User**: For interacting with Vertex AI models (e.g., `gemini-2.0-flash-exp`).
  - Permissions: `aiplatform.user`
- **Resource Manager Viewer**: For listing and getting details of GCP projects.
  - Permissions: `resourcemanager.projects.get`, `resourcemanager.projects.list`
- **Service Usage Viewer**: For listing enabled services within a project.
  - Permissions: `serviceusage.services.list`
- **Secret Manager Secret Accessor**: If using Google Cloud Secret Manager for sensitive configuration data.
  - Permissions: `secretmanager.secrets.access`
- **Cloud Trace Agent**: For exporting OpenTelemetry traces to Google Cloud Trace.
  - Permissions: `cloudtrace.agent`
- **Cloud Logging Log Writer**: For the backend to write application logs to Google Cloud Logging.
  - Permissions: `logging.logWriter`
- **Cloud Logging Viewer**: For the frontend to read and analyze logs from Google Cloud Logging.
  - Permissions: `logging.viewer`
- **Cloud Storage Object Viewer**: For reading from Cloud Storage buckets, potentially for file uploads or data persistence.
  - Permissions: `storage.objects.get`, `storage.objects.list`
- **IAM Service Account Key Admin**: For local development setup if generating/downloading service account keys via `gcloud` for `GOOGLE_APPLICATION_CREDENTIALS`.
  - Permissions: `iam.serviceAccountKeys.create`, `iam.serviceAccountKeys.get`, `iam.serviceAccountKeys.list`, `iam.serviceAccountKeys.upload`
- **Project IAM Admin**: For broad project-level IAM management. This is generally not recommended for least privilege unless the application explicitly manages IAM policies.
  - Permissions: `resourcemanager.projects.setIamPolicy`
- **Cloud Build Editor/Admin**: If deploying via Cloud Build, the Cloud Build service account needs permissions to build and deploy.
  - Permissions: `cloudbuild.builds.editor`, `run.services.create`, `run.services.deploy`, `iam.serviceAccounts.actAs` (for Cloud Run service account)


### Service Account Configuration

It is **MANDATORY** to configure a service account with the principle of least privilege. Ensure the service account used by the application (whether running locally or in Cloud Run) has **only** the necessary roles listed above.

### Environment Variables for Authentication

Configure your environment with the following variables:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json" # Only needed for local development with a service account key file. ADC is preferred.

# Vertex AI Configuration (for enterprise features)
VERTEX_AI_PROJECT_ID="your-project-id" # Often the same as GOOGLE_CLOUD_PROJECT
VERTEX_AI_LOCATION="us-central1"
```

## 🚀 Getting Started

There are two ways to run the security agent:

- **Local Development:** Run the agent directly on your machine using Python and a virtual environment. This is the recommended method for development and experimentation.
- **Docker Deployment:** Run the agent in a Docker container. This is the recommended method for testing the deployment artifact and running the agent in a production-like environment.

### Local Development

#### Prerequisites
- **Python 3.8+**

#### 1. Clone the Repository
```bash
git clone https://github.com/google/adk-python.git
cd adk-python/contributing/samples/security_agent # Adjust if this is the root of your cloned project
```

#### 2. Configure Environment Variables
Create a file named `.env` in the project root and add the required environment variables as shown in the "Environment Variables for Authentication" section above.

#### 3. Run the Agent
```bash
python run.py
```
This script will create a virtual environment, install the dependencies, and start the agent.

#### 4. Access the Agent
- **Frontend:** [http://localhost:8501](http://localhost:8501)
- **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Docker Deployment

#### Prerequisites
- **Docker:** Make sure Docker is installed and running on your system. You can download it from [the Docker website](https://www.docker.com/get-started).

#### 1. Run the Agent
```bash
python run.py --docker
```
This script will build the Docker image and start the agent.

#### 2. Access the Agent
- **Frontend:** [http://localhost:8501](http://localhost:8501)
- **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## Troubleshooting

Refer to the `security_agent_backup/README.md` for detailed troubleshooting steps, including "Connection refused" errors, "v1/models" errors, and "invalid parent name" errors.
