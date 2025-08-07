# Agent Development Kit (ADK)

Welcome to the Agent Development Kit (ADK) - a comprehensive framework for building intelligent, AI-powered agents and applications. This repository contains core ADK components, samples, and documentation to help you build sophisticated agent-based systems.

## 🎯 What is ADK?

The Agent Development Kit (ADK) is a powerful framework that enables developers to create intelligent agents capable of:
- **🤖 Complex Reasoning**: Multi-step planning and decision making
- **🔧 Tool Integration**: Seamless integration with APIs, databases, and cloud services
- **💬 Natural Language Processing**: Advanced conversational interfaces
- **📊 Data Analysis**: Intelligent data processing and insights generation
- **🔄 Workflow Orchestration**: Automated multi-agent coordination

## 🚀 Featured Sample: GCP Security Agent

The flagship ADK sample is a comprehensive security evaluation platform for Google Cloud Platform:

**Location**: [`contributing/samples/security_agent/`](contributing/samples/security_agent/)

### Key Features
- **🛡️ Comprehensive Security Analysis**: Multi-layered GCP security evaluation with real-time risk assessment
- **🤖 AI-Powered Security Agent**: Advanced ADK-based intelligent assistant for security recommendations  
- **🔐 Advanced IAM Analysis**: Deep IAM permissions analysis with policy testing and compliance checking
- **📊 Real-time Monitoring**: Live security metrics, alerts, and performance dashboards
- **🔄 Modular Service Architecture**: Enable/disable individual services for fault isolation and resource optimization
- **📋 Multi-Framework Compliance**: SOC2, ISO27001, GDPR compliance evaluation and reporting

### Quick Start
```bash
cd contributing/samples/security_agent
python run.py
```

For detailed setup instructions, see the [Security Agent README](contributing/samples/security_agent/README.md).

## 📚 Documentation

### Core Documentation
- **[Architecture Overview](ARCHITECTURE.md)** - ADK system architecture and design principles

### Sample Applications
- **[GCP Security Agent](contributing/samples/security_agent/)** - Enterprise security evaluation platform
  - [README](contributing/samples/security_agent/README.md) - Getting started guide
  - [Architecture](contributing/samples/security_agent/MODULAR_ARCHITECTURE.md) - Modular service architecture
  - [Agent Tools](contributing/samples/security_agent/AGENT_TOOLS_ARCHITECTURE.md) - Agent and tools organization
  - [API Reference](contributing/samples/security_agent/API_REFERENCE.md) - Complete API documentation
  - [Service Documentation](contributing/samples/security_agent/SERVICE_DOCUMENTATION.md) - Detailed service specifications

## 🏗️ ADK Core Architecture

ADK follows a modular, service-based architecture that provides:

- **🔌 Pluggable Components**: Mix and match services as needed
- **🔄 Dynamic Configuration**: Enable/disable services at runtime  
- **📊 Health Monitoring**: Built-in service health checks and management
- **🚀 Scalable Deployment**: Docker, Kubernetes, and cloud-ready
- **🛡️ Security-First**: Authentication, authorization, and audit logging built-in

## 🛠️ Getting Started

### Prerequisites
- **Python 3.8+** - Core runtime requirement
- **Docker** - For containerized deployments
- **Google Cloud SDK** - For GCP integrations (optional)

### Installation

#### Option 1: Quick Start (Recommended)
```bash
git clone https://github.com/google/adk-python.git
cd adk-python

# Try the security agent sample
cd contributing/samples/security_agent
python run.py
```

#### Option 2: Development Setup
```bash
git clone https://github.com/google/adk-python.git
cd adk-python

# Install ADK core
pip install -e .

# Explore samples
ls contributing/samples/
```

### Basic Usage

ADK provides a simple API for building agents:

```python
from adk.core import Agent
from adk.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """A custom tool that processes queries."""
    return f"Processed: {query}"

# Create an agent with custom tools
agent = Agent(
    name="my_agent",
    tools=[my_custom_tool],
    description="An example ADK agent"
)

# Use the agent
result = agent.run("Hello, ADK!")
```

## 🔧 Configuration

### Environment Variables

ADK applications use environment variables for configuration:

```bash
# Core ADK Configuration
ADK_LOG_LEVEL=INFO
ADK_MODEL_PROVIDER=vertex_ai  # or 'openai', 'anthropic', etc.

# Google Cloud Integration (optional)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Vertex AI Configuration (for Google Cloud users)
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1

# OpenAI Configuration (alternative)
OPENAI_API_KEY=your-openai-key
```

### Agent Configuration

Agents can be configured through code or YAML files:

```python
from adk.core import Agent

agent = Agent(
    name="my_agent",
    model="gemini-pro",  # or any supported model
    tools=[],  # List of tools to enable
    instructions="You are a helpful assistant.",
    max_iterations=10,
    timeout=60
)
```

## 🏗️ Key Concepts

### Agents
Intelligent entities that can reason, plan, and use tools to accomplish tasks. ADK agents support:
- **Multi-step reasoning** - Break down complex tasks
- **Tool orchestration** - Use multiple tools in sequence
- **Memory management** - Remember context across conversations
- **Error handling** - Gracefully handle failures and retries

### Tools
Functions that agents can call to interact with external systems:
- **API integrations** - REST APIs, GraphQL, webhooks
- **Data processing** - File manipulation, data analysis
- **Cloud services** - Google Cloud, AWS, Azure integrations
- **Custom functions** - Your own business logic

### Services
Modular components that provide specific functionality:
- **Model services** - LLM providers (Vertex AI, OpenAI, etc.)
- **Memory services** - Conversation and context storage
- **Logging services** - Structured logging and monitoring
- **Health services** - System monitoring and diagnostics

## 🚀 Deployment

### Local Development
```bash
# Install ADK
pip install -e .

# Run the security agent sample
cd contributing/samples/security_agent
python run.py
```

### Docker Deployment
```bash
# Build from any ADK project
docker build -t my-adk-agent .
docker run -p 8000:8000 my-adk-agent
```

### Cloud Deployment
ADK applications are cloud-ready and support:
- **Google Cloud Run** - Serverless container deployment
- **Kubernetes** - Container orchestration at scale
- **Docker Compose** - Multi-service local deployment

## 🤝 Contributing

We welcome contributions to ADK! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:
- Report bugs and request features
- Submit code changes and improvements  
- Add new tools and integrations
- Improve documentation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- **Documentation**: [docs.adk.dev](https://docs.adk.dev) (coming soon)
- **GitHub**: [github.com/google/adk-python](https://github.com/google/adk-python)
- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join the community discussions

---

**Ready to build your first ADK agent?** Start with the [GCP Security Agent sample](contributing/samples/security_agent/) to see ADK in action!
