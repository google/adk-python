# Enhanced GCP API Security Evaluation Agent

A comprehensive security evaluation platform that demonstrates OIDC authentication flow and provides advanced security analysis capabilities for GCP APIs, now with full ADK (Agent Development Kit) integration including agent evaluation.

## 🚀 One-Command Deployment

**Deploy the entire security agent from scratch with a single command:**

```bash
./run.sh
```

That's it! The script will:
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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Security Agent                  │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)  │  Backend (FastAPI)                │
│  ┌─────────────────┐   │  ┌─────────────────────────────────┐ │
│  │ Security UI     │   │  │ Core Services                   │ │
│  │ Evaluation UI   │   │  │ ├─ Security Service             │ │
│  └─────────────────┘   │  │ ├─ Agent Service                │ │
│                        │  │ ├─ Documentation Service        │ │
│                        │  │ └─ Secret Manager Service       │ │
│                        │  │                                 │ │
│                        │  │ Enhanced Services               │ │
│                        │  │ ├─ Compliance Service           │ │
│                        │  │ ├─ Threat Intelligence Service  │ │
│                        │  │ ├─ Configuration Analysis       │ │
│                        │  │ └─ Incident Response Service    │ │
│                        │  │                                 │ │
│                        │  │ 🆕 ADK Integration Services     │ │
│                        │  │ ├─ Evaluation Service           │ │
│                        │  │                                 │ │
│                        │  │ 📄 MSA Analysis Services        │ │
│                        │  │ ├─ MSA Parsing Service          │ │
│                        │  │ ├─ Google Cloud Scanner         │ │
│                        │  │ └─ Impact Analysis Service      │ │
│                        │  └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    External Integrations                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ Google ADK  │ │ Vertex AI   │ │ GCP APIs    │            │
│  │ Evaluation  │ │             │ │ Security    │            │
│  │ Framework   │ │             │ │ Command     │            │
│  └─────────────┘ └─────────────┘ │ Center      │            │
│                                  └─────────────┘            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ MSA         │ │ Google      │ │ OIDC        │            │
│  │ Documents   │ │ Cloud       │ │ Resource    │            │
│  │ & APIs      │ │ Manager     │ │ (Google,    │            │
│  │             │ │             │ │  Microsoft) │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

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

# ADK Configuration
ADK_EVALUATION_ENABLED="true"

# Vertex AI Configuration (for enterprise features)
VERTEX_AI_PROJECT_ID="your-project-id"
VERTEX_AI_LOCATION="us-central1"
```

## 🚀 Getting Started

There are two ways to run the security agent:

-   **Local Development:** Run the agent directly on your machine using Python and a virtual environment. This is the recommended method for development and experimentation.
-   **Docker Deployment:** Run the agent in a Docker container. This is the recommended method for testing the deployment artifact and running the agent in a production-like environment.

### Local Development

#### Prerequisites
-   **Python 3.8+**

#### 1. Clone the Repository
```bash
git clone https://github.com/google/adk-python.git
cd adk-python/contributing/samples/security_agent
```

#### 2. Configure Environment Variables
Create a file named `.env` and add the following, replacing the placeholder values with your own:
```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"

# ADK Configuration
ADK_EVALUATION_ENABLED="true"

# Vertex AI Configuration (for enterprise features)
VERTEX_AI_PROJECT_ID="your-project-id"
VERTEX_AI_LOCATION="us-central1"
```

#### 3. Run the Agent
```bash
./run.sh
```
This script will create a virtual environment, install the dependencies, and start the agent.

#### 4. Access the Agent
-   **Frontend:** [http://localhost:8501](http://localhost:8501)
-   **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Docker Deployment

#### Prerequisites
-   **Docker:** Make sure Docker is installed and running on your system. You can download it from [the Docker website](https://www.docker.com/get-started).

#### 1. Run the Agent
```bash
./run.sh --docker
```
This script will build the Docker image and start the agent.

#### 2. Access the Agent
-   **Frontend:** [http://localhost:8501](http://localhost:8501)
-   **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

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



