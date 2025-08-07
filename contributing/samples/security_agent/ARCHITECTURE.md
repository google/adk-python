# ADK Security Agent - Architecture Documentation

## 📚 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Core Components](#core-components)
4. [Service Architecture](#service-architecture)
5. [Data Flow](#data-flow)
6. [API Design](#api-design)
7. [Security Model](#security-model)
8. [Deployment Architecture](#deployment-architecture)

## 🎯 System Overview

The ADK Security Agent is an enterprise-ready security evaluation platform for Google Cloud Platform (GCP) built with a revolutionary modular service architecture. It provides comprehensive security analysis, AI-powered recommendations, and real-time monitoring capabilities.

### Key Architectural Principles
- **Modular Service Architecture**: 16 independently manageable services
- **Fault Isolation**: Services fail independently without affecting others
- **Dynamic Configuration**: Enable/disable services at runtime
- **Dependency Management**: Automatic service dependency resolution
- **Asynchronous Operations**: Built on FastAPI with async/await patterns
- **Cloud-Native Design**: Ready for containerization and cloud deployment

## 🏗️ Architecture Diagrams

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit UI]
        ADK[ADK Web Interface]
    end

    subgraph "API Gateway"
        APIGW[FastAPI Backend<br/>Port 8000]
    end

    subgraph "Service Registry"
        SR[Service Registry]
        SC[Service Config]
        SM[Service Manager]
    end

    subgraph "Core Services"
        GCP[GCP Service]
        SEC[Security Service]
        AGENT[Agent Service]
    end

    subgraph "Security Services"
        IAM[IAM Analysis]
        COMP[Compliance]
        THREAT[Threat Intel]
        INC[Incident Response]
    end

    subgraph "Monitoring Services"
        LOG[Cloud Logging]
        TRACE[Cloud Trace]
        MON[Monitoring]
        PERF[Performance]
    end

    subgraph "Integration Services"
        APIH[API Hub]
        KNOW[Knowledge Base]
        ANAL[Security Analytics]
        DOC[Documentation]
    end

    subgraph "Google Cloud Platform"
        RM[Resource Manager]
        IAM_API[IAM API]
        SC_API[Security Center]
        LOG_API[Cloud Logging]
        TRACE_API[Cloud Trace]
        VA[Vertex AI]
    end

    UI --> APIGW
    ADK --> APIGW
    APIGW --> SR
    SR --> SM
    SR --> SC
    
    SM --> GCP
    SM --> SEC
    SM --> AGENT
    SM --> IAM
    SM --> COMP
    SM --> LOG
    SM --> TRACE
    SM --> APIH
    
    GCP --> RM
    IAM --> IAM_API
    SEC --> SC_API
    LOG --> LOG_API
    TRACE --> TRACE_API
    AGENT --> VA

    classDef frontend fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef gateway fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef registry fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef core fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef security fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef monitoring fill:#e0f2f1,stroke:#009688,stroke-width:2px
    classDef integration fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    classDef gcp fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px

    class UI,ADK frontend
    class APIGW gateway
    class SR,SC,SM registry
    class GCP,SEC,AGENT core
    class IAM,COMP,THREAT,INC security
    class LOG,TRACE,MON,PERF monitoring
    class APIH,KNOW,ANAL,DOC integration
    class RM,IAM_API,SC_API,LOG_API,TRACE_API,VA gcp
```

### Service Lifecycle Management

```mermaid
stateDiagram-v2
    [*] --> NOT_CONFIGURED: Service Defined
    NOT_CONFIGURED --> STARTING: Enable Service
    STARTING --> RUNNING: Initialization Success
    STARTING --> ERROR: Initialization Failed
    RUNNING --> STOPPING: Disable Service
    STOPPING --> DISABLED: Shutdown Success
    STOPPING --> ERROR: Shutdown Failed
    ERROR --> STARTING: Retry/Restart
    DISABLED --> STARTING: Re-enable Service
    RUNNING --> ERROR: Health Check Failed
    ERROR --> RUNNING: Auto-Recovery

    note right of RUNNING
        - Health checks active
        - Handling requests
        - Metrics collection
    end note

    note right of ERROR
        - Error logged
        - Dependencies notified
        - Recovery attempted
    end note
```

### Service Dependency Graph

```mermaid
graph LR
    subgraph "Core Layer (Required)"
        GCP[GCP Service]
        SEC[Security Service]
        AGENT[Agent Service]
    end

    subgraph "Security Layer"
        IAM[IAM Analysis]
        COMP[Compliance]
        THREAT[Threat Intelligence]
        INC[Incident Response]
    end

    subgraph "Monitoring Layer"
        LOG[Cloud Logging]
        TRACE[OpenTelemetry]
        MON[Monitoring]
    end

    subgraph "Integration Layer"
        APIH[API Hub]
        KNOW[Knowledge Base]
        ANAL[Analytics]
        REC[Recommendations]
    end

    IAM --> GCP
    COMP --> GCP
    LOG --> GCP
    TRACE --> GCP
    APIH --> GCP
    KNOW --> GCP
    ANAL --> GCP
    REC --> SEC
    INC --> SEC
    THREAT --> SEC

    classDef required fill:#ffcdd2,stroke:#d32f2f,stroke-width:3px
    classDef optional fill:#c8e6c9,stroke:#388e3c,stroke-width:2px

    class GCP,SEC,AGENT required
    class IAM,COMP,THREAT,INC,LOG,TRACE,MON,APIH,KNOW,ANAL,REC optional
```

## 🔧 Core Components

### Service Registry
The heart of the modular architecture, managing service lifecycle and dependencies.

```python
class ServiceRegistry:
    """Central registry for all services."""
    
    def __init__(self, config: ServiceConfig, credentials=None, project_id=None):
        self.config = config
        self.credentials = credentials
        self.project_id = project_id
        self.services: Dict[str, BaseService] = {}
        self.routers: Dict[str, Any] = {}
```

**Key Responsibilities:**
- Service instantiation and lifecycle management
- Dependency resolution and topological sorting
- Health check coordination
- Dynamic router registration
- Service status tracking

### Base Service Architecture

```mermaid
classDiagram
    class BaseService {
        <<abstract>>
        #service_name: str
        #credentials: Any
        #project_id: str
        #status: ServiceStatus
        #health_status: Dict
        +initialize()* bool
        +shutdown()* bool
        +health_check()* Dict
        +start() bool
        +stop() bool
        +restart() bool
        +check_health() Dict
        +get_status() Dict
        +is_healthy() bool
        +is_available() bool
    }

    class GCPService {
        -resource_manager_client
        -service_usage_client
        +initialize() bool
        +list_projects() List
        +get_project_info() Dict
        +list_enabled_services() List
    }

    class SecurityService {
        -security_client
        -findings_cache
        +initialize() bool
        +evaluate_security() Dict
        +get_findings() List
        +calculate_risk_score() float
    }

    class IAMPolicyAnalyzer {
        -iam_client
        -policy_analyzer
        +initialize() bool
        +analyze_permissions() Dict
        +test_scenarios() List
        +get_overprivileged_users() List
    }

    BaseService <|-- GCPService
    BaseService <|-- SecurityService
    BaseService <|-- IAMPolicyAnalyzer
```

### Service Configuration Management

```mermaid
flowchart TD
    A[Service Definition] --> B{Load Config}
    B -->|Default| C[Default Services]
    B -->|Custom| D[services.json]
    
    C --> E[Service Registry]
    D --> E
    
    E --> F{Check Dependencies}
    F -->|Met| G[Initialize Service]
    F -->|Not Met| H[Mark as Error]
    
    G --> I{Health Check}
    I -->|Healthy| J[Register Router]
    I -->|Unhealthy| K[Retry/Error]
    
    J --> L[Service Running]
    
    subgraph "Runtime Management"
        L --> M{Admin Action}
        M -->|Disable| N[Shutdown Service]
        M -->|Restart| O[Restart Service]
        M -->|Health Check| P[Check Status]
    end
```

## 📦 Service Architecture

### Service Categories

1. **Core Services (Required)**
   - Cannot be disabled
   - Foundation for other services
   - Examples: GCP, Security, Agent

2. **Security Services**
   - IAM Analysis
   - Compliance Checking
   - Threat Intelligence
   - Incident Response

3. **Monitoring Services**
   - Cloud Logging
   - OpenTelemetry Tracing
   - Performance Monitoring

4. **Integration Services**
   - API Hub
   - Security Knowledge Base
   - Analytics
   - Documentation

### Service Communication Pattern

```mermaid
sequenceDiagram
    participant Client
    participant API Gateway
    participant Service Registry
    participant Service
    participant GCP API

    Client->>API Gateway: HTTP Request
    API Gateway->>Service Registry: Get Service
    Service Registry->>Service Registry: Check Availability
    
    alt Service Available
        Service Registry->>Service: Forward Request
        Service->>GCP API: API Call (if needed)
        GCP API-->>Service: Response
        Service-->>API Gateway: Process & Return
        API Gateway-->>Client: HTTP Response
    else Service Unavailable
        Service Registry-->>API Gateway: Service Error
        API Gateway-->>Client: 503 Service Unavailable
    end
```

## 🔄 Data Flow

### Request Processing Flow

```mermaid
flowchart LR
    subgraph "Request Flow"
        A[Client Request] --> B[CORS Middleware]
        B --> C[FastAPI Router]
        C --> D{Service Check}
        D -->|Enabled| E[Service Handler]
        D -->|Disabled| F[Error Response]
        
        E --> G{Auth Required?}
        G -->|Yes| H[Validate Credentials]
        G -->|No| I[Process Request]
        H -->|Valid| I
        H -->|Invalid| J[401 Unauthorized]
        
        I --> K{External API?}
        K -->|Yes| L[GCP API Call]
        K -->|No| M[Local Processing]
        
        L --> N[Transform Response]
        M --> N
        N --> O[Return Response]
    end
```

### Security Evaluation Flow

```mermaid
flowchart TD
    A[Start Evaluation] --> B[Get Project Info]
    B --> C[Security Service]
    
    C --> D[IAM Analysis]
    C --> E[Resource Scan]
    C --> F[Compliance Check]
    C --> G[Threat Analysis]
    
    D --> H[Permission Matrix]
    E --> I[Resource Inventory]
    F --> J[Compliance Report]
    G --> K[Threat Findings]
    
    H --> L[Risk Scoring]
    I --> L
    J --> L
    K --> L
    
    L --> M[Generate Recommendations]
    M --> N[AI Enhancement]
    N --> O[Final Report]
```

## 🔌 API Design

### RESTful API Structure

```mermaid
graph TD
    subgraph "API Endpoints"
        A[/api/v1] --> B[/services]
        A --> C[/gcp]
        A --> D[/security]
        A --> E[/iam]
        A --> F[/compliance]
        A --> G[/agent]
        
        B --> B1[GET /status]
        B --> B2[POST /{name}/enable]
        B --> B3[POST /{name}/disable]
        B --> B4[GET /{name}/health]
        
        C --> C1[GET /projects]
        C --> C2[GET /project/{id}/info]
        C --> C3[GET /project/{id}/services]
        
        D --> D1[POST /evaluate]
        D --> D2[GET /score]
        D --> D3[GET /recommendations]
        
        E --> E1[GET /analyze-user/{email}]
        E --> E2[GET /testing/scenarios]
        E --> E3[POST /testing/run-scenario]
    end
```

### API Response Structure

```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "timestamp": "2025-01-08T10:30:00Z",
    "service": "security",
    "version": "1.0.0"
  },
  "error": null
}
```

## 🔒 Security Model

### Authentication & Authorization Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Google Auth
    participant GCP APIs

    User->>Frontend: Access Application
    Frontend->>Backend: Request with Credentials
    Backend->>Google Auth: Validate Credentials
    Google Auth-->>Backend: Token + Project ID
    Backend->>Backend: Check Permissions
    Backend->>GCP APIs: API Call with Token
    GCP APIs-->>Backend: Response
    Backend-->>Frontend: Processed Data
    Frontend-->>User: Display Results
```

### Security Layers

```mermaid
graph TB
    subgraph "Security Layers"
        A[Network Security<br/>- TLS 1.2+<br/>- Firewall Rules] --> B[Application Security<br/>- CORS<br/>- Input Validation]
        B --> C[Authentication<br/>- Service Account<br/>- ADC]
        C --> D[Authorization<br/>- IAM Roles<br/>- Least Privilege]
        D --> E[Data Security<br/>- Encryption at Rest<br/>- No PII Storage]
        E --> F[Audit & Monitoring<br/>- Cloud Logging<br/>- Trace]
    end
```

## 🚀 Deployment Architecture

### Container Architecture

```mermaid
graph TD
    subgraph "Docker Container"
        A[Base Image<br/>Python 3.11-slim] --> B[System Dependencies]
        B --> C[Python Dependencies]
        C --> D[Application Code]
        D --> E[Configuration]
        
        subgraph "Services"
            F[Backend API<br/>Port 8000]
            G[Frontend UI<br/>Port 8501]
            H[ADK Web<br/>Port 8080]
        end
        
        E --> F
        E --> G
        E --> H
    end
    
    subgraph "Volume Mounts"
        I[Service Account Key]
        J[Configuration Files]
        K[Logs Directory]
    end
    
    I -.-> E
    J -.-> E
    K -.-> F
    K -.-> G
```

### Cloud Run Deployment

```mermaid
flowchart LR
    subgraph "Cloud Run"
        A[Load Balancer] --> B[Service Instances]
        B --> C[Container 1]
        B --> D[Container 2]
        B --> E[Container N]
        
        subgraph "Auto-scaling"
            F[Min: 1]
            G[Max: 10]
            H[CPU: 70%]
        end
    end
    
    subgraph "Google Cloud Services"
        I[Cloud IAM]
        J[Secret Manager]
        K[Cloud Storage]
        L[Cloud Logging]
    end
    
    C --> I
    C --> J
    C --> K
    C --> L
```

### Kubernetes Architecture

```mermaid
graph TD
    subgraph "Kubernetes Cluster"
        subgraph "Namespace: security-agent"
            A[Deployment] --> B[ReplicaSet]
            B --> C[Pod 1]
            B --> D[Pod 2]
            B --> E[Pod 3]
            
            F[Service] --> C
            F --> D
            F --> E
            
            G[Ingress] --> F
            
            H[ConfigMap] -.-> C
            H -.-> D
            H -.-> E
            
            I[Secret] -.-> C
            I -.-> D
            I -.-> E
        end
        
        subgraph "Monitoring"
            J[HPA] --> B
            K[Prometheus] --> C
            K --> D
            K --> E
        end
    end
```

## 📊 Performance Architecture

### Caching Strategy

```mermaid
flowchart TD
    A[Request] --> B{Cache Check}
    B -->|Hit| C[Return Cached]
    B -->|Miss| D[Service Processing]
    
    D --> E{Cacheable?}
    E -->|Yes| F[Store in Cache]
    E -->|No| G[Return Direct]
    
    F --> H[Set TTL]
    H --> G
    
    subgraph "Cache Layers"
        I[Service-Level Cache<br/>- IAM: 5 min<br/>- Projects: 10 min]
        J[API Response Cache<br/>- Health: 30s<br/>- Status: 60s]
        K[Frontend Cache<br/>- Session Storage<br/>- Component State]
    end
```

### Monitoring & Observability

```mermaid
graph TB
    subgraph "Application"
        A[Service Metrics] --> B[OpenTelemetry]
        C[Logs] --> D[Structured Logging]
        E[Traces] --> B
    end
    
    subgraph "Google Cloud"
        B --> F[Cloud Trace]
        D --> G[Cloud Logging]
        A --> H[Cloud Monitoring]
        
        F --> I[Trace Explorer]
        G --> J[Logs Explorer]
        H --> K[Metrics Explorer]
        
        I --> L[Dashboards]
        J --> L
        K --> L
    end
```

## 🔧 Development Architecture

### Local Development Setup

```mermaid
flowchart TD
    A[Developer Machine] --> B[Virtual Environment]
    B --> C[Python Dependencies]
    
    C --> D[Backend Server<br/>uvicorn - Port 8000]
    C --> E[Frontend Server<br/>streamlit - Port 8501]
    C --> F[ADK Interface<br/>Port 8080]
    
    D --> G[Service Registry]
    G --> H[Mock Services]
    G --> I[Real GCP Services]
    
    subgraph "Development Tools"
        J[Hot Reload]
        K[Debug Mode]
        L[Test Suite]
    end
    
    J --> D
    J --> E
    K --> D
    L --> H
```

### CI/CD Pipeline

```mermaid
flowchart LR
    A[Code Push] --> B[GitHub Actions]
    
    B --> C[Lint & Format]
    B --> D[Type Check]
    B --> E[Unit Tests]
    B --> F[Integration Tests]
    
    C --> G{All Pass?}
    D --> G
    E --> G
    F --> G
    
    G -->|Yes| H[Build Container]
    G -->|No| I[Fail Build]
    
    H --> J[Push to Registry]
    J --> K[Deploy to Staging]
    K --> L[Run E2E Tests]
    L --> M{Tests Pass?}
    
    M -->|Yes| N[Deploy to Production]
    M -->|No| O[Rollback]
```

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [Service Management Guide](./SERVICE_MANAGEMENT.md)
- [Security Best Practices](./SECURITY.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

---

**Last Updated:** January 2025  
**Version:** 4.0.0  
**Maintainers:** ADK Security Team