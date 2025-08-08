"""
Unified FastAPI backend for ADK Security Agent
Optimized for Google Cloud ADK showcase with consistent API patterns.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
import os

# Import our unified services
from services.gcp_client_service import GCPClientService
from services.adk_evaluator_service import ADKEvaluatorService
from services.api_explorer_service import APIExplorerService
from models.api_models import APIResponse, HealthCheck, ProjectInfo
from api.v1 import gcp_router, adk_router, security_router, explorer_router
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
gcp_service: Optional[GCPClientService] = None
adk_service: Optional[ADKEvaluatorService] = None
explorer_service: Optional[APIExplorerService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global gcp_service, adk_service, explorer_service
    
    logger.info("🚀 Starting ADK Security Agent Backend...")
    
    try:
        # Initialize settings
        settings = get_settings()
        
        # Initialize core services
        gcp_service = GCPClientService(
            project_id=settings.google_cloud_project,
            credentials_file=settings.google_application_credentials
        )
        await gcp_service.initialize()
        
        adk_service = ADKEvaluatorService(gcp_client=gcp_service)
        await adk_service.initialize()
        
        explorer_service = APIExplorerService(gcp_client=gcp_service)
        await explorer_service.initialize()
        
        logger.info("✅ All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up services...")
        if gcp_service:
            await gcp_service.cleanup()
        if adk_service:
            await adk_service.cleanup()
        if explorer_service:
            await explorer_service.cleanup()

# Create FastAPI app
app = FastAPI(
    title="ADK Security Agent API",
    description="Unified API for Google Cloud ADK Security Evaluation and API Explorer",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to all responses."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(
            success=False,
            error=exc.detail,
            metadata={
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", None)
            },
            timestamp=datetime.utcnow()
        ).dict()
    )

# Health check endpoint
@app.get("/health", response_model=APIResponse[HealthCheck])
async def health_check():
    """Comprehensive health check for all services."""
    try:
        services_health = {}
        
        # Check GCP service
        if gcp_service:
            gcp_health = await gcp_service.check_health()
            services_health["gcp"] = gcp_health
        
        # Check ADK service
        if adk_service:
            adk_health = await adk_service.check_health()
            services_health["adk"] = adk_health
        
        # Check Explorer service
        if explorer_service:
            explorer_health = await explorer_service.check_health()
            services_health["explorer"] = explorer_health
        
        overall_healthy = all(
            service.get("healthy", False) 
            for service in services_health.values()
        )
        
        health_data = HealthCheck(
            status="healthy" if overall_healthy else "degraded",
            services=services_health,
            version="2.0.0",
            timestamp=datetime.utcnow()
        )
        
        return APIResponse(
            success=True,
            data=health_data,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return APIResponse(
            success=False,
            error=f"Health check failed: {str(e)}",
            timestamp=datetime.utcnow()
        )

# Root endpoint
@app.get("/", response_model=APIResponse[Dict[str, Any]])
async def root():
    """Root endpoint with API information."""
    return APIResponse(
        success=True,
        data={
            "name": "ADK Security Agent API",
            "version": "2.0.0",
            "description": "Unified API for Google Cloud ADK Security Evaluation",
            "features": [
                "GCP Project Management",
                "Security Evaluation",
                "API Discovery & Testing", 
                "ADK Feature Showcase",
                "Real-time Monitoring"
            ],
            "documentation": "/docs"
        },
        timestamp=datetime.utcnow()
    )

# Dependency injection
def get_gcp_service() -> GCPClientService:
    """Get GCP client service dependency."""
    if not gcp_service:
        raise HTTPException(status_code=503, detail="GCP service not available")
    return gcp_service

def get_adk_service() -> ADKEvaluatorService:
    """Get ADK evaluator service dependency."""
    if not adk_service:
        raise HTTPException(status_code=503, detail="ADK service not available")
    return adk_service

def get_explorer_service() -> APIExplorerService:
    """Get API explorer service dependency."""
    if not explorer_service:
        raise HTTPException(status_code=503, detail="Explorer service not available")
    return explorer_service

# Include routers
app.include_router(gcp_router, prefix="/api/v1/gcp", tags=["GCP"])
app.include_router(adk_router, prefix="/api/v1/adk", tags=["ADK"])
app.include_router(security_router, prefix="/api/v1/security", tags=["Security"])
app.include_router(explorer_router, prefix="/api/v1/explorer", tags=["API Explorer"])

# Additional utility endpoints
@app.get("/api/v1/status", response_model=APIResponse[Dict[str, Any]])
async def get_api_status():
    """Get overall API status."""
    return APIResponse(
        success=True,
        data={
            "api_version": "2.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "services": {
                "gcp": "running" if gcp_service else "stopped",
                "adk": "running" if adk_service else "stopped", 
                "explorer": "running" if explorer_service else "stopped"
            }
        },
        timestamp=datetime.utcnow()
    )

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main_unified:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )