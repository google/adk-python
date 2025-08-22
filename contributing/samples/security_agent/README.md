# ADK Security Agent

A comprehensive Google Cloud Platform security analysis and monitoring agent built with the ADK (Agent Development Kit) framework.

## Overview

The ADK Security Agent provides intelligent security analysis, real-time monitoring, and actionable recommendations for GCP environments. It features a conversational AI interface powered by Vertex AI and comprehensive security scanning capabilities.

## Features

- **🔍 Comprehensive Security Analysis**: Automated scanning of GCP resources for security vulnerabilities
- **🤖 AI-Powered Insights**: Vertex AI integration for intelligent security recommendations  
- **📊 Real-time Monitoring**: Continuous monitoring with health checks and metrics
- **🚀 Token Streaming**: Real-time response streaming for instant feedback
- **🗄️ Smart Caching**: SQLite-based caching for optimized performance
- **🔒 Security Hardening**: Built-in protection against injection attacks
- **📈 Performance Profiling**: Load testing and bottleneck analysis tools
- **✅ Production Ready**: Docker support with health checks and monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Project with appropriate permissions
- Service account with required IAM roles
- ADK CLI installed (`pip install adk`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd security_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.template .env
# Edit .env with your GCP project details
```

4. Run setup:
```bash
python setup.py
```

### Running the Application

#### Local Development

```bash
# Start backend
python run_backend.py

# Start frontend (in another terminal)
python run_frontend.py
```

#### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

Access the application at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics
- Status: http://localhost:8000/status

## Architecture

```
security_agent/
├── agents/               # ADK agent definitions
│   └── gcp_security/    # Vertex AI security agent
├── backend/             # FastAPI backend server
│   ├── api/            # API endpoints
│   ├── middleware/     # Security middleware
│   ├── services/       # Business logic
│   └── cache/          # SQLite database
├── frontend/           # Streamlit UI
├── evaluation/         # Testing and evaluation tools
└── deploy/            # Deployment configurations
```

## Required GCP Permissions

Your service account needs these IAM roles:
- Cloud Asset Viewer
- Security Center Admin Viewer  
- Storage Admin
- IAM Security Reviewer
- Recommender Viewer
- Secret Manager Viewer
- Monitoring Viewer

## API Endpoints

### Core Security APIs
- `GET /api/v1/custom-roles/stats` - Custom role statistics
- `GET /api/v1/knowledge/stats` - Knowledge base statistics
- `GET /api/v1/iam/policies` - IAM policy analysis
- `GET /api/v1/storage/buckets` - Storage bucket analysis

### Monitoring Endpoints
- `GET /health` - Service health check
- `GET /metrics` - Prometheus-compatible metrics
- `GET /status` - Detailed service status

## Security Features

- **Input Sanitization**: Comprehensive protection against SQL, NoSQL, and command injection
- **Rate Limiting**: Configurable rate limits to prevent abuse
- **Non-root Containers**: Security-hardened Docker images
- **Environment Isolation**: Separate configurations for dev/staging/prod

## Monitoring & Observability

The application provides comprehensive monitoring through:

- **Health Checks**: Automated health monitoring with configurable thresholds
- **Metrics Collection**: System and application metrics in Prometheus format
- **Performance Profiling**: Built-in load testing and bottleneck analysis
- **Security Scanning**: Continuous vulnerability assessment

## Development

### Running Tests

```bash
# Run evaluation suite
cd evaluation
python service_evaluation_orchestrator.py --parallel

# Run specific evaluations
python service_health_monitor.py
python performance_profiler.py
python security_scanner.py
```

### Code Style

The project follows Python PEP 8 standards. Format code with:
```bash
black .
isort .
```

## Troubleshooting

### Common Issues

1. **Database not found**
   ```bash
   python populate_sqlite.py
   ```

2. **Port already in use**
   ```bash
   lsof -i :8000
   kill -9 <PID>
   ```

3. **Missing dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

See [docs/troubleshooting.md](docs/troubleshooting.md) for detailed solutions.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-org/security-agent/issues)
- Documentation: [docs/](docs/)

## Acknowledgments

Built with:
- [ADK (Agent Development Kit)](https://github.com/anthropics/adk)
- [Google Cloud Platform](https://cloud.google.com)
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)