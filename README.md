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

- **Vertex AI User**: For interacting with Vertex AI models (e.g., `gemini-1.5-flash-001`).
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
- **Recommender Viewer**: For accessing Active Assist recommendations.
- **Project IAM Admin**: For broad project-level IAM management. This is generally not recommended for least privilege unless the application explicitly manages IAM policies.
  - Permissions: `resourcemanager.projects.setIamPolicy`

### Service Account Configuration

It is **MANDATORY** to configure a service account with the principle of least privilege. Ensure the service account used by the application (whether running locally or in Cloud Run) has **only** the necessary roles listed above.

### Environment Variables for Authentication

For local development, create a `.env` file in the root of the project with the following content:

```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

**Note:** `GOOGLE_APPLICATION_CREDENTIALS` is only needed for local development. When deployed to Cloud Run, the attached service account will be used automatically.

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

## 📡 Backend API Interactions and Resource Usage

### External API Interactions

The backend system interacts with the following external APIs and services:

*   **Google Cloud Trace API**: Used for exporting OpenTelemetry traces, providing application performance monitoring and debugging capabilities within Google Cloud.
*   **Google Secret Manager API**: Utilized for securely storing and accessing sensitive information, such as API keys and credentials, within the Google Cloud environment.
*   **Google Cloud API Hub API**: Used for registering and managing APIs, potentially for internal or external consumption.
*   **Google Cloud Asset Inventory API**: Leveraged to analyze and retrieve metadata about Google Cloud resources and policies, enabling security configuration analysis.
*   **Google Cloud Logging API**: Employed for ingesting and managing application logs, facilitating monitoring and troubleshooting.
*   **Google Cloud DNS API**: Used for DNS-related operations, although specific use cases are not detailed in the provided snippets.
*   **Google Cloud Compute Engine API**: Used for managing Compute Engine resources, specifically to list instance information.
*   **Google Kubernetes Engine API**: Used for managing GKE clusters.
*   **Google Cloud Storage API**: Used for operations related to Cloud Storage buckets and objects.
*   **Google Cloud IAM Policy Analyzer API**: Used to analyze IAM policies to understand effective access to resources.
*   **Google Cloud Recommender API**: Used to fetch security recommendations from Active Assist.
*   **Google Cloud Security Command Center API**: Used for managing security findings and assets.
*   **Google Cloud APIs (General)**: The `GCPService` class contains methods that appear to interact with various GCP APIs based on resource types (e.g., `list_gcp_resources`).
*   **External APIs (via `requests` library)**: The `AgentService` can make arbitrary HTTP calls using the `requests` library, based on the agent's instructions, implying interaction with a variety of external services not explicitly listed here. These are likely for agents to interact with external tools or knowledge bases.

### Local and Google Cloud Resources

Running this application involves the creation and utilization of the following resources:

*   **Local Resources**:
    *   **Virtual Environments (`venv/`)**: Used for dependency management and isolation of Python packages.
    *   **Log Files**: Generated by the application for operational monitoring and debugging.
    *   **PID files (`adk_web.pid`, `backend.pid`, `frontend.pid`)**: Process ID files created to manage application instances.
    *   **Temporary files**: May be created for various processing tasks, although specific temporary file usage is not detailed in the provided code snippets.

*   **Google Cloud Resources (Utilized by the application, but not provisioned by setup scripts)**:
    *   **Cloud Trace**: Used for distributed tracing to monitor and debug the application.
    *   **Secret Manager**: Used for storing and retrieving secrets required by the application.
    *   **Cloud API Hub**: If integrated, it would be used to manage API definitions.
    *   **Cloud Asset Inventory**: Leveraged for querying GCP resource metadata.
    *   **Cloud Logging**: Used for centralizing application logs.
    *   **Google Compute Engine instances**: The application queries information about existing Compute Engine instances.
    *   **Google Kubernetes Engine clusters**: The application queries information about existing GKE clusters.
    *   **Google Cloud Storage buckets and objects**: The application interacts with existing Cloud Storage resources.
    *   **Cloud IAM Policies**: The application analyzes IAM policies.
    *   **Security Command Center findings and assets**: The application interacts with Security Command Center.

**Note**: The provided setup scripts (`run.py`, `Dockerfile`, `cloudbuild.yaml`) are responsible for setting up the Python environment, building Docker images, and running/deploying the application. They do **not** provision new Google Cloud infrastructure (e.g., creating new VMs, databases, or networks). The application interacts with *existing* GCP services and resources.
