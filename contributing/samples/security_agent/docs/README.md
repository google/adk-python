# ADK Security Agent Documentation

This directory contains documentation for the Google ADK Security Agent with chat-centric architecture.

## 📋 Documentation Index

### Setup & Configuration
- **[ADK_SETUP_GUIDE.md](ADK_SETUP_GUIDE.md)** - Complete guide to installing and configuring Google ADK
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment instructions
- **[MANUAL_DEPLOYMENT.md](MANUAL_DEPLOYMENT.md)** - Manual deployment procedures

### Architecture & Implementation
- **[ADK_DELEGATION_PATTERN_IMPLEMENTATION.md](ADK_DELEGATION_PATTERN_IMPLEMENTATION.md)** - ADK agent delegation patterns
- **[CHAT_CENTRIC_IMPLEMENTATION_SUMMARY.md](CHAT_CENTRIC_IMPLEMENTATION_SUMMARY.md)** - Chat-first architecture overview
- **[architecture/](architecture/)** - Detailed architecture documentation

## 🚀 Quick Start

1. **Prerequisites**: Google Cloud Project with ADK access
2. **Setup**: Follow [ADK_SETUP_GUIDE.md](ADK_SETUP_GUIDE.md)
3. **Architecture**: Review [CHAT_CENTRIC_IMPLEMENTATION_SUMMARY.md](CHAT_CENTRIC_IMPLEMENTATION_SUMMARY.md)
4. **Deployment**: Use [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🎯 Key Concepts

### ADK Integration
- **Direct Integration**: Single version chat interface
- **Coordinator Agent**: LLM-driven delegation to specialized agents
- **Real GCP Data**: No mock responses - live project data only

### Chat-Centric Architecture
- **Primary Interface**: Chat is the main user interaction
- **Agent Delegation**: Automatic routing to security, IAM, storage specialists
- **Conversation Memory**: Context persistence across sessions

## 🔧 Development

### Required Dependencies
```bash
pip install google-adk google-generativeai google-cloud-aiplatform
```

### Authentication
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## 📊 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chat UI       │───▶│  Coordinator     │───▶│  Specialized    │
│   (Frontend)    │    │  Agent (ADK)     │    │  Agents (ADK)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Conversation   │    │  LLM-Driven      │    │  • Security     │
│  Memory         │    │  Routing         │    │  • IAM          │
└─────────────────┘    └──────────────────┘    │  • Storage      │
                                               │  • Compliance   │
                                               └─────────────────┘
```

## 🆕 Recent Changes

- **Simplified Architecture**: Removed enhanced/hybrid patterns
- **Single Chat Version**: Eliminated multiple chat implementations
- **Direct ADK Only**: No fallback modes or mock responses
- **Clean Documentation**: Removed obsolete implementation docs

## 📖 Additional Resources

- [Google ADK Documentation](https://cloud.google.com/agent-development-kit)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Architecture Roadmap](architecture/IMPLEMENTATION_ROADMAP.md)

---
**Last Updated**: August 2025  
**Version**: 2.0 (Simplified Architecture)