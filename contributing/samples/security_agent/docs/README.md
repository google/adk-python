# Documentation

This directory contains comprehensive documentation for the GCP Security Agent with ADK integration.

## Documentation Structure

### 📁 `/guides` - Setup & Configuration Guides
- **[QUICK_START.md](guides/QUICK_START.md)** - Get running in 5 minutes
- **[ENV_SETUP.md](guides/ENV_SETUP.md)** - Environment configuration details
- **[ADK_SETUP_GUIDE.md](ADK_SETUP_GUIDE.md)** - Complete Google ADK setup
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment instructions
- **[MANUAL_DEPLOYMENT.md](MANUAL_DEPLOYMENT.md)** - Step-by-step manual deployment

### 📁 `/architecture` - System Architecture & Design
- **[ARCHITECTURE.md](architecture/ARCHITECTURE.md)** - System design overview and patterns
- **[AGENT_TOOLS_ARCHITECTURE.md](architecture/AGENT_TOOLS_ARCHITECTURE.md)** - Agent and tools architecture
- **[CHAT_CENTRIC_ARCHITECTURE.md](architecture/CHAT_CENTRIC_ARCHITECTURE.md)** - Chat-first interface design
- **[CHAT_INTERFACE_COMPONENTS.md](architecture/CHAT_INTERFACE_COMPONENTS.md)** - Chat UI component details
- **[IMPLEMENTATION_ROADMAP.md](architecture/IMPLEMENTATION_ROADMAP.md)** - Development roadmap

### 📁 Implementation Details
- **[ADK_DELEGATION_PATTERN_IMPLEMENTATION.md](ADK_DELEGATION_PATTERN_IMPLEMENTATION.md)** - ADK agent delegation patterns
- **[CHAT_CENTRIC_IMPLEMENTATION_SUMMARY.md](CHAT_CENTRIC_IMPLEMENTATION_SUMMARY.md)** - Chat architecture implementation
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API endpoint documentation

## Quick Navigation

### Getting Started
1. Start with [QUICK_START.md](guides/QUICK_START.md) for rapid deployment
2. Configure environment using [ENV_SETUP.md](guides/ENV_SETUP.md)
3. Setup ADK with [ADK_SETUP_GUIDE.md](ADK_SETUP_GUIDE.md)

### Understanding the System
1. Review [ARCHITECTURE.md](architecture/ARCHITECTURE.md) for system overview
2. Explore [CHAT_CENTRIC_ARCHITECTURE.md](architecture/CHAT_CENTRIC_ARCHITECTURE.md) for chat design
3. Check [API_REFERENCE.md](API_REFERENCE.md) for API details

### Deployment
1. Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment
2. Use [MANUAL_DEPLOYMENT.md](MANUAL_DEPLOYMENT.md) for custom setups

## Key Concepts

### ADK Integration
- **Coordinator Agent**: LLM-driven delegation to specialized agents
- **Direct Integration**: Single version chat interface
- **Real GCP Data**: Live project data integration

### Chat-Centric Architecture
- **Primary Interface**: Chat as main user interaction
- **Agent Delegation**: Automatic routing to specialists
- **Conversation Memory**: Context persistence

### Security Features
- **IAM Analysis**: Deep permissions review
- **Vulnerability Scanning**: Real-time security assessment
- **Compliance Evaluation**: SOC2, ISO27001, GDPR frameworks

## Recent Updates

- **v2.0**: Simplified architecture with single chat version
- **Direct ADK Only**: Removed fallback modes
- **Clean Documentation**: Reorganized for clarity

---
**Last Updated**: August 2025