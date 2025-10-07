# Chainlit Integration Guide

This document explains how to use the Chainlit UI with your GCP Security Intelligence Platform.

## 🎯 What is Chainlit?

[Chainlit](https://github.com/Chainlit/chainlit) is a modern, open-source Python framework for building production-ready conversational AI applications. It provides:

- **Beautiful Chat UI** - Modern interface similar to ChatGPT
- **Real-time Streaming** - Stream responses as they're generated
- **File Uploads** - Upload documents for analysis
- **Multi-modal Support** - Text, images, files
- **Session Management** - Persistent conversations
- **Easy Deployment** - Deploy to any cloud platform

## 🚀 Quick Start

### Installation

```bash
# Install Chainlit
pip install chainlit==1.0.0

# Or install from requirements.txt
pip install -r requirements.txt
```

### Running the Chainlit UI

```bash
# Terminal 1: Start ADK backend (required)
cd /path/to/security_agent
adk web
# Runs on http://localhost:8000

# Terminal 2: Start Chainlit UI
chainlit run chainlit_app.py
# Runs on http://localhost:8001 (default)
```

### Custom Port

```bash
# Run on a specific port
chainlit run chainlit_app.py --port 8080

# With auto-reload for development
chainlit run chainlit_app.py --port 8080 --watch
```

## 🏗️ Architecture

```
┌─────────────┐
│   Browser   │
│  (User UI)  │
└──────┬──────┘
       │ WebSocket
┌──────▼──────────┐
│  Chainlit App   │  (port 8001)
│ chainlit_app.py │  - Chat UI
│                 │  - Session mgmt
└──────┬──────────┘  - Message routing
       │ HTTP/REST
┌──────▼──────────┐
│  ADK Backend    │  (port 8000)
│   adk web       │  - Agent logic
│                 │  - 32 tools
└──────┬──────────┘  - BigQuery access
       │
┌──────▼──────────┐
│    BigQuery     │
│  Data Platform  │
└─────────────────┘
```

## 📋 Features

### 1. **Multiple Agent Profiles (Chat Profiles)** ✨
The dropdown at the top of the UI lets users select from **4 specialized agents**:

| Agent | Icon | Specialization |
|-------|------|----------------|
| 🔒 **Security Agent** | Shield | Full access to all 32 tools across 7 categories |
| ✅ **Compliance Expert** | Certificate | PCI-DSS, HIPAA, SOC2 compliance focused |
| ☁️ **Service Discovery** | Cloud Search | GCP service onboarding and analysis |
| 📚 **Documentation Search** | Book | Confluence and knowledge base search |

Each agent has:
- Custom welcome message
- Tailored example questions
- Specialized capabilities description
- Unique icon and branding

**This is the "hack" your customer is using** - the `@cl.set_chat_profiles` decorator replaces the foundation model selector with an agent selector!

### 2. Real-time Chat
- Natural language queries to the security agent
- Streaming responses (configurable)
- Session persistence
- Error handling with user-friendly messages

### 3. Session Management
- Each user gets a unique ADK session
- Sessions maintained across page refreshes
- Clean session lifecycle (start/end handlers)

### 4. Tool Integration
All 32 tools accessible via natural language:
- **BigQuery Analysis** - Security insights, queries, exploration
- **Service Evaluation** - Compliance checking, risk assessment
- **Service Discovery** - GCP service onboarding
- **Confluence** - Documentation search
- **Feeds** - RSS feeds, release notes

### 5. Prompt Playground
- **Enabled by default** - Test and iterate on prompts
- Interactive prompt editor in the UI
- Real-time testing with your agent
- Save and share prompt variations

### 6. Environment Variables
Chainlit exposes required environment variables through the UI:
- `GOOGLE_CLOUD_PROJECT` - GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS` - Service account path
- `GOOGLE_CLOUD_LOCATION` - Region (e.g., us-central1)
- `BQ_DEFAULT_DATASET` - BigQuery dataset name
- `BQ_DEFAULT_TABLE` - BigQuery table name
- `ADK_BASE_URL` - ADK backend URL
- `CONFLUENCE_URL` - Confluence instance URL
- `CONFLUENCE_USERNAME` - Confluence user email
- `CONFLUENCE_API_TOKEN` - Confluence API token
- `CONFLUENCE_SPACES` - Comma-separated space keys

Users can provide these through the UI or via `.env` file.

## 🎨 Customization

### Adding Custom Agent Profiles (Plug & Play)

This is how your customer "hacked" the component. Edit [chainlit_app.py](../chainlit_app.py):

```python
@cl.set_chat_profiles
async def chat_profile():
    """Define multiple agent profiles for the dropdown selector."""
    return [
        cl.ChatProfile(
            name="Security Agent",
            markdown_description="🔒 **GCP Security Intelligence** - Access to 32 security tools...",
            icon="https://api.iconify.design/mdi/shield-check.svg?color=%234285f4",
        ),
        # Add your custom agents here!
        cl.ChatProfile(
            name="Cost Optimizer",
            markdown_description="💰 **Cost Analysis** - Analyze spending and optimize resources",
            icon="https://api.iconify.design/mdi/cash.svg?color=%2334a853",
        ),
        cl.ChatProfile(
            name="Incident Response",
            markdown_description="🚨 **Security Incidents** - Handle security events and alerts",
            icon="https://api.iconify.design/mdi/alert.svg?color=%23ea4335",
        ),
    ]
```

Then add custom welcome messages in the `agent_welcomes` dictionary:

```python
agent_welcomes = {
    "Security Agent": """# 🔒 GCP Security Intelligence Platform...""",
    "Cost Optimizer": """# 💰 Cost Optimization Agent...""",
    "Incident Response": """# 🚨 Incident Response Agent...""",
}
```

**Icons**: Use [Iconify](https://iconify.design/) for professional icons. Search for icons and copy the API URL.

### Branding

Edit `.chainlit` config file:

```toml
[UI]
name = "Your Company Security Platform"
description = "Custom description"
# github = "https://github.com/your-org/your-repo"
```

### Timeout Configuration

Adjust the request timeout in [chainlit_app.py](../chainlit_app.py):

```python
# In run_agent_interaction()
response = requests.post(ADK_RUN_URL, json=payload, timeout=120)  # seconds
```

### Session Timeout

Edit `.chainlit` config:

```toml
[project]
# Duration (in seconds) during which the session is saved when the connection is lost
session_timeout = 3600  # 1 hour
```

### Enable/Disable Prompt Playground

Edit `.chainlit` config:

```toml
[features]
# Show the prompt playground (enabled by default)
prompt_playground = true
```

The prompt playground allows users to:
- Test and iterate on prompts interactively
- See real-time results from the agent
- Save and share prompt variations
- Experiment with different prompt strategies

### Configure Environment Variables

Edit `.chainlit` config to specify which environment variables users must provide:

```toml
[project]
user_env = [
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "BQ_DEFAULT_DATASET",
    "ADK_BASE_URL"
]
```

If these are not in the `.env` file, Chainlit will prompt users to enter them through the UI.

## 🔧 Configuration

### Environment Variables

Create or update `.env` file:

```bash
# ADK Backend Configuration
ADK_BASE_URL=http://localhost:8000

# GCP Configuration (inherited from ADK backend)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=config/service-account.json
```

### Chainlit Configuration

The `.chainlit` file contains:
- **Telemetry** - Disabled by default
- **Session timeout** - 1 hour default
- **UI settings** - Name, description, features
- **Features** - Prompt playground, caching, etc.

## 📊 Comparison: Chainlit vs Flask

| Feature | Chainlit | Flask (app.py) |
|---------|----------|----------------|
| **UI** | Modern ChatGPT-like | Custom HTML/JS |
| **Streaming** | Native WebSocket | HTTP streaming |
| **Setup** | Zero config | Custom routes |
| **Sessions** | Built-in | Manual implementation |
| **File Upload** | Built-in | Custom implementation |
| **Multi-modal** | Native support | Custom handling |
| **Deployment** | Simple | Requires gunicorn |
| **Code Lines** | ~100 | ~700 |

## 🚀 Deployment

### Local Development

```bash
# Development mode with auto-reload
chainlit run chainlit_app.py --watch
```

### Production (Docker)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Chainlit runs on port 8000 by default
EXPOSE 8001

CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8001"]
```

### Cloud Run

```bash
# Build and deploy
gcloud run deploy security-agent-chainlit \
  --source . \
  --port 8001 \
  --set-env-vars ADK_BASE_URL=https://your-adk-backend.run.app \
  --allow-unauthenticated
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-agent-chainlit
spec:
  replicas: 2
  selector:
    matchLabels:
      app: security-agent-chainlit
  template:
    metadata:
      labels:
        app: security-agent-chainlit
    spec:
      containers:
      - name: chainlit
        image: gcr.io/PROJECT/security-agent-chainlit:latest
        ports:
        - containerPort: 8001
        env:
        - name: ADK_BASE_URL
          value: "http://adk-backend:8000"
```

## 🧪 Testing

### Manual Testing

1. Start ADK backend: `adk web`
2. Start Chainlit: `chainlit run chainlit_app.py`
3. Open browser: http://localhost:8001
4. Try example queries:
   - "Show me all critical security findings"
   - "List IAM accounts with admin privileges"
   - "Search Confluence for encryption policies"

### Automated Testing

```python
# tests/test_chainlit_integration.py
import pytest
from chainlit_app import extract_text_from_adk_response

def test_extract_text():
    mock_response = [
        {
            "content": {
                "parts": [
                    {"text": "Hello from agent"}
                ]
            }
        }
    ]
    result = extract_text_from_adk_response(mock_response)
    assert result == "Hello from agent"
```

## 🔍 Troubleshooting

### Issue: Chainlit not connecting to ADK

**Symptoms**: Error messages about connection refused

**Solution**:
```bash
# Verify ADK is running
curl http://localhost:8000/health

# Check ADK_BASE_URL in .env
echo $ADK_BASE_URL

# Restart ADK backend
adk web
```

### Issue: Timeout errors

**Symptoms**: "Request timed out" messages

**Solution**: Increase timeout in `chainlit_app.py`:
```python
response = requests.post(ADK_RUN_URL, json=payload, timeout=300)  # 5 minutes
```

### Issue: Session not persisting

**Symptoms**: Lost conversation history

**Solution**: Check `.chainlit` config:
```toml
[project]
session_timeout = 7200  # Increase to 2 hours
```

### Issue: Port already in use

**Symptoms**: "Address already in use" error

**Solution**:
```bash
# Find and kill process on port 8001
lsof -ti:8001 | xargs kill -9

# Or use a different port
chainlit run chainlit_app.py --port 8002
```

## 📚 Advanced Features

### File Upload (Future Enhancement)

```python
@cl.on_message
async def main(message: cl.Message):
    # Handle file attachments
    if message.elements:
        for element in message.elements:
            if element.type == "file":
                # Process uploaded file
                content = element.content
                # Send to ADK backend with file context
```

### Custom Actions

```python
@cl.action_callback("refresh_data")
async def on_action(action):
    # Custom button actions
    await cl.Message(content="Refreshing data...").send()
    # Trigger data refresh in backend
```

### Data Visualization

```python
import plotly.graph_objects as go

@cl.on_message
async def main(message: cl.Message):
    # Query returns data
    data = query_bigquery(message.content)

    # Create visualization
    fig = go.Figure(data=[go.Bar(x=data['x'], y=data['y'])])

    # Send as Plotly element
    await cl.Message(
        content="Here's your data visualization",
        elements=[cl.Plotly(figure=fig)]
    ).send()
```

## 🔗 Resources

- [Chainlit Documentation](https://docs.chainlit.io/)
- [Chainlit GitHub](https://github.com/Chainlit/chainlit)
- [Chainlit Examples](https://docs.chainlit.io/examples/community)
- [ADK Documentation](https://developers.google.com/adk)

## 💡 Best Practices

1. **Error Handling** - Always catch and display user-friendly errors
2. **Timeouts** - Set appropriate timeouts for long-running queries
3. **Session Management** - Clean up sessions on disconnect
4. **Logging** - Log interactions for debugging and analytics
5. **Security** - Validate inputs, sanitize outputs
6. **Performance** - Use async/await for non-blocking operations
7. **Testing** - Write unit tests for message handlers

## 🎯 Next Steps

1. **Install Chainlit**: `pip install chainlit`
2. **Start ADK Backend**: `adk web`
3. **Launch Chainlit**: `chainlit run chainlit_app.py`
4. **Test Integration**: Try example queries
5. **Customize**: Update branding and welcome message
6. **Deploy**: Follow deployment guide for your platform

---

**Need Help?** Check the [Chainlit Discord](https://discord.gg/chainlit) or file an issue on GitHub.
