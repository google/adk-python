"""Main FastAPI application for the security agent backend."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import os
import logging
import traceback

# Configure detailed logging  
import os
# Always use project root logs directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'backend.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Google Cloud authentication
from google.auth import default
from google.oauth2 import service_account
import json

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from api import (
    security, knowledge, agent, documentation, apihub, compliance, 
    threat_intelligence, configuration, incidents, evaluation, msa, tracing, openapi_tools, gcp, recommendations, async_security, logs, cloud_logs
)
from services.security_service import SecurityService
from services.documentation_service import DocumentationService
from services.agent_service import AgentService
from services.secret_manager_service import SecretManagerService
from services.apihub_service import APIHubService
from services.compliance_service import ComplianceService
from services.threat_intelligence_service import ThreatIntelligenceService
from services.configuration_analysis_service import ConfigurationAnalysisService
from services.incident_response_service import IncidentResponseService
from services.evaluation_service import SecurityAgentEvaluationService
from services.msa_service import MSAParsingService
from services.tracing_service import TracingService
from services.gcp_service import GCPService
from utils.openapi_converter import create_adk_compatible_openapi


def setup_service_account_credentials():
    """Set up Google Cloud service account credentials."""
    try:
        # Check for service account key file (prefer clearer variable name)
        service_account_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY_FILE') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        if service_account_path and os.path.exists(service_account_path):
            # Load service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=[
                    'https://www.googleapis.com/auth/cloud-platform',
                    'https://www.googleapis.com/auth/trace.append',
                    'https://www.googleapis.com/auth/monitoring.write',
                    'https://www.googleapis.com/auth/logging.write'
                ]
            )
            logger.info(f"✅ Service account credentials loaded from: {service_account_path}")
            logger.info(f"✅ Project ID: {project_id}")
            return credentials, project_id
        else:
            # Try service account JSON from environment variable
            service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
            if service_account_json:
                service_account_info = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=[
                        'https://www.googleapis.com/auth/cloud-platform',
                        'https://www.googleapis.com/auth/trace.append',
                        'https://www.googleapis.com/auth/monitoring.write',
                        'https://www.googleapis.com/auth/logging.write'
                    ]
                )
                project_id = service_account_info.get('project_id') or project_id
                logger.info("✅ Service account credentials loaded from environment JSON")
                logger.info(f"✅ Project ID: {project_id}")
                return credentials, project_id
            else:
                # Fall back to default credentials
                logger.warning("⚠️ No service account file found, falling back to default credentials")
                credentials, project_id = default()
                return credentials, project_id
                
    except Exception as e:
        logger.error(f"❌ Failed to set up service account credentials: {e}")
        logger.info("Falling back to default credentials")
        try:
            credentials, project_id = default()
            return credentials, project_id
        except Exception as fallback_error:
            logger.error(f"❌ Failed to get any credentials: {fallback_error}")
            return None, None


def setup_tracing(credentials=None, project_id=None):
    """Set up OpenTelemetry tracing with Cloud Trace using service account."""
    try:
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())
        
        # Set up Cloud Trace exporter with explicit credentials
        if credentials and project_id:
            cloud_trace_exporter = CloudTraceSpanExporter(
                project_id=project_id,
                credentials=credentials
            )
            logger.info(f"✅ Cloud Trace exporter configured with service account for project: {project_id}")
        else:
            cloud_trace_exporter = CloudTraceSpanExporter()
            logger.info("✅ Cloud Trace exporter configured with default credentials")
        
        # Add span processor
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(cloud_trace_exporter)
        )
        
        # Instrument requests
        RequestsInstrumentor().instrument()
        
        logger.info("✅ OpenTelemetry tracing configured with Cloud Trace")
        return True
    except Exception as e:
        logger.error(f"⚠️ Failed to configure tracing: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    print("Starting Enhanced Security Agent Backend...")
    
    # Set up service account credentials
    credentials, project_id = setup_service_account_credentials()
    app.state.gcp_credentials = credentials
    app.state.gcp_project_id = project_id
    
    # Set up tracing with service account
    tracing_enabled = setup_tracing(credentials, project_id)
    app.state.tracing_enabled = tracing_enabled
    
    # Initialize core services
    app.state.security_service = SecurityService()
    app.state.documentation_service = DocumentationService()
    app.state.agent_service = AgentService()
    
    # Initialize Secret Manager service
    app.state.secret_manager_service = SecretManagerService()
    
    # Initialize API Hub service
    app.state.apihub_service = APIHubService(app.state.secret_manager_service)
    
    # Initialize new enhanced services
    app.state.compliance_service = ComplianceService()
    app.state.threat_intelligence_service = ThreatIntelligenceService()
    app.state.configuration_analysis_service = ConfigurationAnalysisService()
    app.state.incident_response_service = IncidentResponseService()
    
    # Initialize ADK integration services
    app.state.evaluation_service = SecurityAgentEvaluationService()
    
    # Initialize MSA service
    app.state.msa_service = MSAParsingService()
    
    # Initialize tracing service
    app.state.tracing_service = TracingService()
    
    # Initialize GCP service with credentials
    app.state.gcp_service = GCPService(credentials, project_id)

    yield
    
    # Shutdown
    print("Shutting down Enhanced Security Agent Backend...")
    await app.state.agent_service.close_all_sessions()


# Create FastAPI app
app = FastAPI(
    title="Enhanced GCP API Security Evaluation Agent",
    description="Comprehensive backend API for evaluating the security stance of GCP APIs with advanced security analysis, compliance checking, threat intelligence, configuration analysis, incident response and agent evaluation. Now with OpenTelemetry tracing support and OpenAPI 3.1 to 3.0 conversion for Google ADK compatibility.",
    version="3.0.0",
    lifespan=lifespan
)

# Custom OpenAPI schema generation for ADK compatibility
def custom_openapi():
    """Generate ADK-compatible OpenAPI schema by converting 3.1 to 3.0."""
    if app.openapi_schema:
        return app.openapi_schema
    
    # Generate ADK-compatible OpenAPI 3.0.3 schema
    openapi_schema = create_adk_compatible_openapi(app)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Override FastAPI's default schema generation
app.openapi = custom_openapi

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with detailed logging."""
    error_details = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "request_url": str(request.url),
        "request_method": request.method
    }
    
    logger.error(f"Unhandled exception: {error_details}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
            "error_type": type(exc).__name__,
            "message": "An unexpected error occurred"
        }
    )


# Include API routers
app.include_router(security.router, prefix="/api/v1/security", tags=["Security"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["Knowledge Base"])
app.include_router(agent.router, prefix="/api/v1/agent", tags=["Agent"])
app.include_router(documentation.router, prefix="/api/v1/documentation", tags=["Documentation"])
app.include_router(apihub.router, prefix="/api/v1/apihub", tags=["API Hub"])

# Include new enhanced API routers
app.include_router(compliance.router, prefix="/api/v1/compliance", tags=["Compliance"])
app.include_router(threat_intelligence.router, prefix="/api/v1/threat-intelligence", tags=["Threat Intelligence"])
app.include_router(configuration.router, prefix="/api/v1/configuration", tags=["Configuration Analysis"])
app.include_router(incidents.router, prefix="/api/v1/incidents", tags=["Incident Response"])

# Include ADK integration API routers
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["Agent Evaluation"])

# Include MSA analysis API router
app.include_router(msa.router, prefix="/api/v1/msa", tags=["MSA Analysis"])

# Include tracing API router
app.include_router(tracing.router, prefix="/api/v1/tracing", tags=["Tracing"])

# Include OpenAPI tools API router
app.include_router(openapi_tools.router, prefix="/api/v1/openapi-tools", tags=["OpenAPI Tools"])
app.include_router(gcp.router, prefix="/api/v1/gcp", tags=["GCP Operations"])

# Include recommendations API router
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Security Recommendations"])

# Include async security API router
app.include_router(async_security.router, prefix="/api/v1/async-security", tags=["Async Security Operations"])

# Log Analysis endpoints
app.include_router(logs.router, tags=["Log Analysis"])

# Cloud Logging endpoints
app.include_router(cloud_logs.router, tags=["Cloud Logging"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enhanced GCP API Security Evaluation Agent Backend",
        "version": "3.0.0",
        "status": "running",
        "features": [
            "Security evaluation",
            "Knowledge base management", 
            "Documentation scraping",
            "Agent interactions",
            "API Hub toolset management",
            "Secure credential storage",
            "Compliance framework evaluation",
            "Threat intelligence analysis",
            "Security configuration analysis",
            "Incident response management",
            "Agent evaluation and testing",
            "MSA parsing and Google Cloud organization scanning"
        ],
        "new_features": [
            "SOC 2, ISO 27001, GDPR compliance checking",
            "NVD vulnerability scanning",
            "Security configuration scoring",
            "Incident response playbooks",
            "Forensics analysis capabilities",
            "OIDC authentication demonstration",
            "ADK Agent Evaluation Framework",
            "MSA document parsing and API extraction",
            "Google Cloud organization impact analysis",
            "OpenAPI 3.1 to 3.0 conversion for Google ADK compatibility",
            "Automatic optional parameter fixing for ADK toolsets",
            "API Hub registration data generation"
        ],
        "api_endpoints": {
            "security": "/api/v1/security",
            "knowledge": "/api/v1/knowledge",
            "agent": "/api/v1/agent",
            "documentation": "/api/v1/documentation",
            "apihub": "/api/v1/apihub",
            "compliance": "/api/v1/compliance",
            "threat_intelligence": "/api/v1/threat-intelligence",
            "configuration": "/api/v1/configuration",
            "incidents": "/api/v1/incidents",
            "evaluation": "/api/v1/evaluation",
            "msa": "/api/v1/msa",
            "tracing": "/api/v1/tracing",
            "openapi_tools": "/api/v1/openapi-tools"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "security_service": "available",
            "documentation_service": "available",
            "agent_service": "available",
            "secret_manager_service": "available",
            "apihub_service": "available",
            "compliance_service": "available",
            "threat_intelligence_service": "available",
            "configuration_analysis_service": "available",
            "incident_response_service": "available",
            "evaluation_service": "available",
            "msa_service": "available"
        },
        "version": "3.0.0",
        "tracing_enabled": getattr(app.state, 'tracing_enabled', False),
        "features_enabled": [
            "OIDC Authentication Demo",
            "Comprehensive Security Evaluation",
            "Compliance Framework Analysis",
            "Threat Intelligence Integration",
            "Configuration Security Analysis",
            "Incident Response Management",
            "ADK Agent Evaluation",
            "MSA Document Parsing",
            "Google Cloud Organization Scanning",
            "OpenTelemetry Tracing"
        ]
    }



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 