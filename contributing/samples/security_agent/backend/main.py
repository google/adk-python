"""Modular FastAPI application for the security agent backend."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import os
import logging
import traceback

# Configure detailed logging  
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

# Core modular imports
from core import ServiceRegistry, ServiceConfig

# Import service management API
from api.services import router as services_router

# Import utils
from utils.openapi_converter import create_adk_compatible_openapi


def setup_service_account_credentials():
    """Set up Google Cloud service account credentials using Google's standard approach."""
    try:
        # Use Google's standard default authentication flow
        credentials, project_id = default(scopes=[
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/trace.append',
            'https://www.googleapis.com/auth/monitoring.write',
            'https://www.googleapis.com/auth/logging.write'
        ])
        
        # Use project from environment if available, otherwise use detected project
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or project_id
        
        logger.info("✅ Google Cloud credentials loaded successfully")
        logger.info(f"✅ Project ID: {project_id}")
        return credentials, project_id
                
    except Exception as e:
        logger.error(f"❌ Failed to get Google Cloud credentials: {e}")
        logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set for local development")
        logger.error("or service account is attached for Cloud Run deployment")
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    print("Starting Modular Security Agent Backend...")
    
    # Set up service account credentials
    credentials, project_id = setup_service_account_credentials()
    app.state.gcp_credentials = credentials
    app.state.gcp_project_id = project_id
    
    # Initialize service configuration
    config_path = os.getenv('SERVICE_CONFIG_PATH', 'config/services.json')
    app.state.service_config = ServiceConfig(config_path)
    
    # Initialize service registry
    app.state.service_registry = ServiceRegistry(
        config=app.state.service_config,
        credentials=credentials,
        project_id=project_id
    )
    
    # Initialize all enabled services
    logger.info("Initializing services...")
    results = await app.state.service_registry.initialize_all_services()
    
    # Log initialization results
    for service_name, success in results.items():
        if success:
            logger.info(f"✅ Service initialized: {service_name}")
        else:
            logger.error(f"❌ Failed to initialize: {service_name}")
    
    # Register available routers
    for router_info in app.state.service_registry.get_available_routers():
        app.include_router(
            router_info['router'],
            prefix=router_info['prefix'],
            tags=router_info['tags']
        )
        logger.info(f"✅ Registered router: {router_info['prefix']}")

    yield
    
    # Shutdown
    print("Shutting down Modular Security Agent Backend...")
    await app.state.service_registry.shutdown_all_services()


# Create FastAPI app
app = FastAPI(
    title="Modular GCP API Security Evaluation Agent",
    description="Modular backend API for evaluating the security stance of GCP APIs with the ability to enable/disable services.",
    version="4.0.0",
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from google.auth import exceptions as auth_exceptions

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
    
    status_code = 500
    error_message = f"Internal server error: {str(exc)}"
    if isinstance(exc, auth_exceptions.DefaultCredentialsError):
        status_code = 401
        error_message = "Authentication failed. Please check your credentials."
    elif isinstance(exc, auth_exceptions.RefreshError):
        status_code = 401
        error_message = "Authentication token has expired. Please re-authenticate."
    
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": error_message,
            "error_type": type(exc).__name__,
            "message": "An unexpected error occurred"
        }
    )


# Include service management router
app.include_router(services_router, prefix="/api/v1/services", tags=["Service Management"])


@app.get("/")
async def root():
    """Root endpoint."""
    registry = app.state.service_registry
    config = app.state.service_config
    
    # Get enabled services
    enabled_services = [
        service.name for service in config.get_enabled_services()
    ]
    
    # Get service status summary
    all_statuses = registry.get_all_statuses()
    running_services = [
        name for name, status in all_statuses.items()
        if status.get('status') == 'running'
    ]
    
    return {
        "message": "Modular GCP API Security Evaluation Agent Backend",
        "version": "4.0.0",
        "status": "running",
        "services": {
            "total": len(config.get_all_services()),
            "enabled": len(enabled_services),
            "running": len(running_services),
            "list": enabled_services
        },
        "features": [
            "Modular service architecture",
            "Service enable/disable capability",
            "Service health monitoring",
            "Dynamic router registration",
            "Dependency management",
            "Service configuration persistence"
        ],
        "api_endpoints": {
            "services": "/api/v1/services",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    registry = app.state.service_registry
    config = app.state.service_config
    
    # Get all service statuses
    all_statuses = registry.get_all_statuses()
    
    # Determine overall health
    unhealthy_services = [
        name for name, status in all_statuses.items()
        if status.get('status') == 'error'
    ]
    
    overall_health = "healthy" if not unhealthy_services else "degraded"
    
    # Get service health details
    service_health = {}
    for service_name, service in registry.get_all_services().items():
        try:
            health = await service.check_health()
            service_health[service_name] = health
        except:
            service_health[service_name] = {"healthy": False, "error": "Health check failed"}
    
    return {
        "status": overall_health,
        "version": "4.0.0",
        "services": {
            "total": len(config.get_all_services()),
            "healthy": len(all_statuses) - len(unhealthy_services),
            "unhealthy": len(unhealthy_services),
            "details": service_health
        },
        "unhealthy_services": unhealthy_services
    }


if __name__ == "__main__":
    uvicorn.run(
        "main_modular:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )