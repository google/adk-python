# Security Agent Documentation Index

This directory contains comprehensive documentation for the ADK Security Agent. Start here to navigate to the information you need.

## 📚 Core Documentation

### Getting Started
- **[Main README](../README.md)** - Project overview, features, and quick start guide with Logic Layer Architecture
- **[Deployment Guide](deployment.md)** - Complete deployment instructions for local, Docker, Cloud Run, and Kubernetes
- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions including latest troubleshooting patterns

### Architecture & Implementation
- **[Technical Implementation Checklist](TECHNICAL_IMPLEMENTATION_CHECKLIST.md)** - Development guidelines and best practices
- **[MCP Configuration](../MCP_CONFIGURATION.md)** - Model Context Protocol setup and configuration

## 🏗️ Key Features Documentation

### Context-Aware MSA Analysis
The security agent features sophisticated Monthly Service Announcement (MSA) impact analysis:

- **MSA Permission Detection** - Automatically detects permission changes (e.g., BigQuery `datasets.get` split)
- **Custom Role Impact Analysis** - Cross-references MSA changes with your project's actual IAM policies
- **Remediation Planning** - Provides specific gcloud commands for updating custom roles
- **Testing Strategies** - Includes development environment validation steps

**Key Files:**
- `/agents/gcp_security/sqlite_tool.py` - Query engine with MSA cross-referencing
- `/agents/gcp_security/vertex_sqlite_agent.py` - Agent with embedded remediation knowledge

### Knowledge Base Integration
Comprehensive knowledge management system with:

- **Coding Standards** - 7 total standards including 5 test-specific requirements
- **Enterprise Policies** - 3 security policies with CRITICAL/HIGH/MEDIUM severity levels
- **Compliance Frameworks** - SOC2, PCI-DSS compliance tracking
- **Natural Language Access** - Chat interface for policy and standard queries

**Key Files:**
- `/backend/cache/gcp_data.db` - Unified database with knowledge base tables
- [Knowledge Base Integration Report](../KNOWLEDGE_BASE_INTEGRATION.md)

### Real-Time Streaming
Token-by-token response streaming with ADK Runner integration:

- **Live Response Display** - ChatGPT-like streaming experience
- **Session Management** - Persistent conversation state
- **Error Handling** - Graceful fallback for streaming failures

## 📊 Quality Assurance

### Testing & Evaluation
- **[Evaluation Framework](../evaluation/README.md)** - Comprehensive test suite with 100% success rate
- **[Testing Complete Report](../TESTING_COMPLETE_REPORT.md)** - Production readiness validation
- **[Playwright Tests](../tests/README_PLAYWRIGHT_TESTS.md)** - UI automation testing
- **[Playwright Smoke Test Report](../PLAYWRIGHT_SMOKETEST_REPORT.md)** - Latest UI test results

### Performance & Monitoring
- **Service Health Monitoring** - Built-in health checks and metrics
- **Performance Profiling** - Load testing and bottleneck analysis
- **Security Scanning** - Automated vulnerability assessment

## 🚀 Deployment & Operations

### Environment Setup
Required environment variables and configuration:

```bash
# Core Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
DATABASE_PATH=/absolute/path/to/gcp_data.db  # Use absolute paths!

# Application URLs
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:8501

# Performance Tuning
DATA_REFRESH_INTERVAL=1800
RATE_LIMIT_CHAT=30
```

### Service Account Permissions
Your service account needs these IAM roles:
- Cloud Asset Viewer
- Security Center Admin Viewer
- Storage Admin
- IAM Security Reviewer
- Recommender Viewer
- Secret Manager Viewer
- Monitoring Viewer

## 📋 Project Reports & Status

### Achievement Summaries
- **[Achievement Summary](../ACHIEVEMENT_SUMMARY.md)** - Overall project accomplishments
- **[Service Evaluation Summary](../NEW_SERVICE_EVALUATION_SUMMARY.md)** - Latest service evaluation results

### Development Progress
- **[Stories & Implementation](stories/)** - Development user stories and todo tracking

## 🛠️ Development Guidelines

### Code Standards
The project enforces comprehensive coding standards:

- **Test Coverage**: >80% required for all new features
- **Security**: No hardcoded secrets, proper input sanitization
- **Performance**: <2 second response times for 95% of operations
- **Documentation**: All new features must include evaluation tests

### Architecture Principles
- **Single Agent Design** - No multi-agent/swarm patterns
- **Client-Server Separation** - Clean separation between frontend and backend
- **Context-Aware Intelligence** - Agent understands project state for better recommendations
- **Production Ready** - Docker deployment with monitoring and health checks

## 🔍 Quick Reference

### Essential Commands
```bash
# Start services
python run_backend.py &
python run_frontend.py

# Database operations
sqlite3 backend/cache/gcp_data.db ".tables"
sqlite3 backend/cache/gcp_data.db "SELECT COUNT(*) FROM msa_changes;"

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/status

# Evaluation
cd evaluation && python comprehensive_test_runner.py
```

### Common Issues
- **Agent not in dropdown**: Start ADK web from agent directory (`cd agents/gcp_security && adk web`)
- **Database not found**: Use absolute paths in DATABASE_PATH environment variable
- **Streaming not working**: Restart frontend after agent changes
- **MSA analysis issues**: Verify MSA and IAM data exists in database

## 📞 Support

For technical issues:
1. Check [troubleshooting.md](troubleshooting.md) for common solutions
2. Run evaluation suite: `cd evaluation && python simple_test.py`
3. Review logs in debug mode: `LOG_LEVEL=DEBUG python run_backend.py`
4. Create GitHub issue with error details and steps to reproduce

---

**Documentation Last Updated**: August 22, 2025  
**Project Status**: ✅ Production Ready  
**Test Coverage**: 🎯 100% Framework Success Rate