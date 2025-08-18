# Architecture Documentation
# ADK Security Agent System Architecture

## Version 4.0 | Last Updated: 2025-01-18

## System Overview

The ADK Security Agent implements a sophisticated client-server architecture leveraging Google's Agent Development Kit (ADK) to provide intelligent security analysis for Google Cloud Platform environments. The system follows a thin-client pattern where the frontend focuses on user interaction while the backend provides all intelligence and processing capabilities.

## Architectural Principles

### Core Design Principles
1. **Separation of Concerns**: Clear boundaries between UI, business logic, and data layers
2. **Security by Design**: Zero-trust architecture with credentials isolated to backend
3. **Scalability First**: Stateless design enabling horizontal scaling
4. **Intelligence Centralization**: All AI/ML capabilities concentrated in backend services
5. **API-First Development**: All functionality exposed through well-defined APIs

### ADK Integration Principles
1. **Agent Delegation Pattern**: LLM-driven intelligent routing between specialized agents
2. **Tool-as-a-Service**: Backend APIs serve as tools for frontend agents
3. **Context Persistence**: Conversation state managed across sessions
4. **Streaming-First**: Real-time token-by-token response streaming

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Users                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│                  (Thin Client - Streamlit)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Chat Interface (streaming)                        │   │
│  │  • Session Management (lightweight)                  │   │
│  │  • API Client (httpx)                               │   │
│  │  • Real-time Response Rendering                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                     HTTPS REST API / WebSocket
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Backend Layer                            │
│                    (FastAPI Server)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           API Gateway & Request Router               │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Agent Orchestration Layer               │   │
│  │  • Coordinator Agent (delegation hub)                │   │
│  │  • RADAR Agents (Recognition, Assessment, etc.)      │   │
│  │  • Specialized Agents (IAM, Security, etc.)         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Service Layer (Tools)                   │   │
│  │  • Asset Discovery    • Security Analysis            │   │
│  │  • IAM Auditing      • Recommendations              │   │
│  │  • Monitoring        • Compliance Checking          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Data & Integration Layer                │   │
│  │  • Context Management  • Session Persistence         │   │
│  │  • Caching Layer      • Credential Management       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Services                          │
│  • Google Cloud APIs (Asset, IAM, Security Command Center)   │
│  • Secret Manager                                            │
│  • Cloud Logging & Monitoring                                │
│  • Vertex AI / Gemini Models                                 │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Frontend Components

#### 1. Thin Client (thin_client.py)
```python
Primary Responsibilities:
- User Interface rendering via Streamlit
- WebSocket connection management
- Message history display
- Real-time streaming display

Key Features:
- Zero business logic
- Stateless operation
- Pure presentation layer
- API-only backend communication
```

#### 2. Chat Interface
```python
Components:
- chat_view.py: Main chat UI component
- chat_manager.py: Local message management
- chat_streaming_base.py: Streaming response handler

Responsibilities:
- User input capture
- Response streaming
- Session state management
- Error display
```

### Backend Components

#### 1. API Gateway (main.py)
```python
Endpoints:
/api/v1/agent/* - Agent orchestration
/api/v1/gcp/* - GCP service integration
/api/v1/security/* - Security analysis
/api/v1/iam/* - IAM management
/api/v1/monitoring/* - Logs and metrics
/api/v1/assets/* - Asset inventory
/api/v1/recommendations/* - Recommender API

Middleware:
- CORS configuration
- Authentication
- Request logging
- Error handling
```

#### 2. Agent Orchestration Layer

##### Coordinator Agent Pattern
```python
class CoordinatorAgent:
    """
    Central delegation hub using ADK's TransferToAgentTool
    """
    def __init__(self):
        self.agents = {
            'recognition': RecognitionAgent(),
            'assessment': AssessmentAgent(),
            'decision': DecisionAgent(),
            'action': ActionAgent(),
            'review': ReviewAgent()
        }
    
    def delegate(self, query: str) -> Agent:
        # LLM analyzes query and selects appropriate agent
        return self.llm_select_agent(query)
```

##### RADAR Implementation
```python
RADAR Methodology Agents:
1. Recognition Agent
   - Resource discovery
   - Asset inventory
   - Dependency mapping

2. Assessment Agent
   - Vulnerability scanning
   - Risk scoring
   - Compliance checking

3. Decision Agent
   - Priority calculation
   - Impact analysis
   - Recommendation generation

4. Action Agent
   - Remediation execution
   - Playbook application
   - Change implementation

5. Review Agent
   - Verification
   - Monitoring
   - Trend analysis
```

#### 3. Service Layer (Tools)

##### Tool Registration Pattern
```python
@tool
def discover_gcp_resources(project_id: str) -> Dict:
    """
    Tool exposed to agents for resource discovery
    """
    return asset_inventory_service.discover(project_id)

@tool
def analyze_security(resources: List) -> SecurityReport:
    """
    Tool for security analysis
    """
    return security_service.analyze(resources)
```

#### 4. Data Layer

##### Context Management
```python
class ConversationManager:
    """
    Manages conversation context across sessions
    """
    def get_or_create_session(self, session_id: str) -> Session
    def add_to_history(self, session_id: str, message: str)
    def get_context(self, session_id: str) -> str
```

##### Caching Strategy
```python
Cache Layers:
1. Request Cache: 5-minute TTL for API responses
2. Resource Cache: 1-hour TTL for asset inventory
3. Analysis Cache: 30-minute TTL for security findings
4. Session Cache: 24-hour TTL for conversation context
```

## Data Flow Architecture

### Query Processing Flow
```
1. User Input → Frontend
2. Frontend → API Gateway (HTTP POST)
3. API Gateway → Coordinator Agent
4. Coordinator → LLM Analysis
5. LLM → Agent Selection
6. Selected Agent → Tool Execution
7. Tools → GCP APIs
8. Response → Agent
9. Agent → Response Synthesis
10. Synthesized Response → Frontend (Streaming)
11. Frontend → User Display
```

### Session Management Flow
```
1. Session Creation (Frontend)
2. Session ID Generation
3. Backend Session Storage
4. Context Accumulation
5. Context-Aware Processing
6. Response with Context
7. History Persistence
```

## Security Architecture

### Authentication & Authorization
```
Frontend → Backend: Session-based auth
Backend → GCP: Service Account credentials
User → System: OAuth 2.0 (future)
```

### Credential Management
```
1. Local Development:
   - Service account JSON in backend/config/secrets/
   - Never exposed to frontend

2. Production (Cloud Run):
   - Google Secret Manager
   - Workload Identity
   - Minimal IAM permissions
```

### Security Boundaries
```
Trust Zones:
1. Untrusted: User Input
2. Semi-trusted: Frontend
3. Trusted: Backend Services
4. Highly Trusted: GCP APIs
```

## Deployment Architecture

### Container Architecture
```
Frontend Container:
- Base: python:3.11-slim
- Framework: Streamlit
- Port: 8501
- Resources: 256MB RAM, 0.25 vCPU

Backend Container:
- Base: python:3.11-slim
- Framework: FastAPI + Uvicorn
- Port: 8000
- Resources: 2GB RAM, 1 vCPU
```

### Cloud Run Deployment
```yaml
Frontend Service:
  - Name: adk-security-frontend
  - Region: us-central1
  - Scaling: 0-10 instances
  - Concurrency: 100

Backend Service:
  - Name: adk-security-backend
  - Region: us-central1
  - Scaling: 1-50 instances
  - Concurrency: 1000
```

## Performance Architecture

### Optimization Strategies
1. **Connection Pooling**: Reuse GCP API connections
2. **Batch Processing**: Aggregate multiple API calls
3. **Async Operations**: Non-blocking I/O throughout
4. **Streaming Responses**: Token-by-token delivery
5. **Intelligent Caching**: Multi-tier cache strategy

### Scalability Patterns
1. **Horizontal Scaling**: Stateless services enable linear scaling
2. **Load Balancing**: Cloud Run automatic load distribution
3. **Circuit Breakers**: Prevent cascade failures
4. **Rate Limiting**: Protect against API quota exhaustion

## Integration Architecture

### GCP Service Integration
```
Primary APIs:
- Cloud Asset Inventory API
- Security Command Center API
- Cloud Resource Manager API
- IAM API
- Cloud Logging API
- Recommender API
- Secret Manager API

Integration Pattern:
- Client libraries for standardized access
- Exponential backoff for retries
- Quota management through caching
```

### ADK Framework Integration
```
Components Used:
- google.genai.adk.Agent
- google.genai.adk.Runner
- google.genai.adk.Tool
- TransferToAgentTool

Integration Points:
- Agent creation and registration
- Tool definition and binding
- Conversation management
- Context persistence
```

## Monitoring & Observability

### Logging Architecture
```
Log Levels:
- ERROR: System failures
- WARNING: Degraded functionality
- INFO: Normal operations
- DEBUG: Detailed troubleshooting

Log Aggregation:
- Cloud Logging for centralization
- Structured logging in JSON
- Correlation IDs for request tracking
```

### Metrics Collection
```
Key Metrics:
- Request latency (p50, p95, p99)
- Agent delegation decisions
- Tool execution times
- API quota usage
- Cache hit rates
- Error rates by component
```

## Error Handling Architecture

### Fallback Strategies
```python
Error Handling Chain:
1. Try primary service
2. Fallback to cache if available
3. Fallback to simplified response
4. Return graceful error message
```

### Recovery Patterns
```
1. Automatic Retry: Transient failures
2. Circuit Breaker: Prevent cascade
3. Graceful Degradation: Partial functionality
4. Manual Intervention: Critical failures
```

## Future Architecture Considerations

### Phase 2 Enhancements
- GraphQL API layer for efficient data fetching
- WebSocket support for real-time updates
- Event-driven architecture with Pub/Sub
- Federated learning for model improvements

### Phase 3 Evolution
- Microservices decomposition
- Service mesh implementation
- Multi-region deployment
- Edge computing integration

## Appendices

### A. Technology Stack
- **Frontend**: Python 3.11, Streamlit, httpx
- **Backend**: Python 3.11, FastAPI, Uvicorn
- **AI/ML**: Google ADK, Vertex AI, Gemini
- **Infrastructure**: Google Cloud Run, Secret Manager
- **Monitoring**: Cloud Logging, Cloud Monitoring

### B. API Standards
- RESTful design principles
- OpenAPI 3.0 specification
- JSON response format
- HTTP status code compliance

### C. Code Organization
```
security_agent/
├── frontend/          # Thin client
├── backend/           # Intelligence layer
│   ├── agents/       # Agent implementations
│   ├── api/          # API endpoints
│   └── config/       # Configuration
├── deploy/           # Deployment configs
├── tests/            # Test suites
└── docs/             # Documentation
```