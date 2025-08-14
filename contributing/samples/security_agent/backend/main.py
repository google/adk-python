"""FastAPI application for the security agent backend.

Clean, unified FastAPI application with all available features.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.websockets import WebSocket
from typing import Dict, Any, List, Optional
import uvicorn
import os
import logging
import asyncio
import sys
import tempfile
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Secret Manager for runtime credential loading
try:
    from google.cloud import secretmanager
    SECRETMANAGER_AVAILABLE = True
except ImportError:
    SECRETMANAGER_AVAILABLE = False
    logger.warning("Google Cloud Secret Manager not available")

# Import optional components
try:
    from backend.chat_manager import chat_manager, EnhancedChatManager
    chat_manager_available = True
except ImportError:
    chat_manager_available = False
    logger.info("Chat manager not available")

try:
    from backend.api.websocket_manager import websocket_manager, websocket_endpoint
    websocket_available = True
except ImportError:
    websocket_available = False
    logger.info("WebSocket manager not available")

try:
    from backend.api.performance_monitor import performance_monitor, PerformanceMonitor
    performance_monitor_available = True
except ImportError:
    performance_monitor_available = False
    logger.info("Performance monitor not available")

try:
    from backend.api.context_manager import context_manager, ContextAwareManager
    context_manager_available = True
except ImportError:
    context_manager_available = False
    logger.info("Context manager not available")

# Add the services directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

def setup_service_account_from_secret():
    """Setup service account credentials from Google Secret Manager."""
    if not SECRETMANAGER_AVAILABLE:
        logger.info("Secret Manager not available, using default credentials")
        return
        
    # Only fetch from Secret Manager if running in Cloud Run
    if not os.getenv('K_SERVICE'):
        logger.info("Not running in Cloud Run, using local credentials")
        return
        
    try:
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            logger.error("GOOGLE_CLOUD_PROJECT not set")
            return
            
        # Create Secret Manager client (uses Cloud Run's service account)
        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{project_id}/secrets/security-agent-sa-key/versions/latest"
        
        logger.info(f"Fetching service account key from Secret Manager: {secret_name}")
        response = client.access_secret_version(request={"name": secret_name})
        secret_data = response.payload.data.decode("UTF-8")
        
        # Create temporary file for the service account key
        temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='sa_key_')
        with os.fdopen(temp_fd, 'w') as temp_file:
            temp_file.write(secret_data)
        
        # Set the environment variable to point to the temporary file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path
        logger.info(f"✅ Service account credentials loaded from Secret Manager")
        
        return temp_path
        
    except Exception as e:
        logger.error(f"❌ Failed to load service account from Secret Manager: {e}")
        logger.info("Falling back to default Cloud Run credentials")

# Unified ADK chat service
def create_adk_chat_service(project_id):
    """Create unified ADK chat service."""
    class ADKService:
        def __init__(self, project_id):
            self.project_id = project_id
            logger.info(f"[ADK] Initialized for project: {project_id}")
        
        async def process_chat_message(self, message, context=None):
            logger.info(f"[ADK] Processing: '{message}' with context: {bool(context)}")
            
            # Simulate ADK processing
            await asyncio.sleep(0.1)
            
            return {
                "response": f"ADK processed: {message}",
                "confidence": 0.95,
                "project_id": self.project_id,
                "context_used": bool(context),
                "timestamp": datetime.now().isoformat()
            }
    
    return ADKService(project_id)

# Setup service account credentials first
setup_service_account_from_secret()

# Create FastAPI app
app = FastAPI(
    title="Security Agent Backend",
    description="Unified ADK Security Agent API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================
# API ROUTER REGISTRATION
# ===========================================
# Standardized API path structure:
# /api/v1/agent/* - Agent and chat endpoints
# /api/v1/gcp/* - Google Cloud Platform endpoints
# /api/v1/security/* - Security analysis endpoints
# /api/v1/iam/* - IAM analysis endpoints
# /api/v1/compliance/* - Compliance evaluation endpoints
# /api/v1/monitoring/* - Monitoring and logs endpoints
# /api/v1/recommendations/* - Recommendations endpoints
# /api/v1/msa/* - Master Service Agreement endpoints
# /api/v1/performance/* - Performance monitoring endpoints
# /api/v1/context/* - Context management endpoints

# Agent router (includes chat endpoints)
# Try LLM-based agent first, fallback to standard agent
agent_router_loaded = False
if chat_manager_available:
    try:
        from backend.api.agent_llm import router as agent_router
        app.include_router(agent_router, prefix="/api/v1/agent")
        logger.info("✅ LLM Agent router included at /api/v1/agent (with intelligent steering)")
        agent_router_loaded = True
    except ImportError as e:
        logger.info(f"LLM Agent router not available, trying standard router: {e}")
        
    if not agent_router_loaded:
        try:
            from backend.api.agent import router as agent_router
            app.include_router(agent_router, prefix="/api/v1/agent")
            logger.info("✅ Standard Agent router included at /api/v1/agent")
        except ImportError as e:
            logger.warning(f"No agent router available: {e}")

# GCP router
try:
    from backend.api.gcp import router as gcp_router
    app.include_router(gcp_router, prefix="/api/v1/gcp")
    logger.info("✅ GCP router included at /api/v1/gcp")
except ImportError as e:
    logger.error(f"GCP router not available: {e}")

# Security router
try:
    from backend.api.security import router as security_router
    app.include_router(security_router, prefix="/api/v1/security")
    logger.info("✅ Security router included at /api/v1/security")
except ImportError as e:
    logger.error(f"Security router not available: {e}")

# Monitoring router
try:
    from backend.api.monitoring import router as monitoring_router
    app.include_router(monitoring_router, prefix="/api/v1/monitoring")
    logger.info("✅ Monitoring router included at /api/v1/monitoring")
except ImportError as e:
    logger.warning(f"Monitoring router not available: {e}")

# Performance monitor router (if available)
if performance_monitor_available:
    try:
        from backend.api.performance_monitor import router as performance_router
        app.include_router(performance_router, prefix="/api/v1/performance")
        logger.info("✅ Performance router included at /api/v1/performance")
    except ImportError as e:
        logger.warning(f"Performance router not available: {e}")

# Context manager router (if available)
if context_manager_available:
    try:
        from backend.api.context_manager import router as context_router
        app.include_router(context_router, prefix="/api/v1/context")
        logger.info("✅ Context router included at /api/v1/context")
    except ImportError as e:
        logger.warning(f"Context router not available: {e}")

# IAM router
try:
    from backend.api.iam import router as iam_router
    app.include_router(iam_router, prefix="/api/v1/iam")
    logger.info("✅ IAM router included at /api/v1/iam")
except ImportError as e:
    logger.warning(f"IAM router not available: {e}")

# Compliance router
try:
    from backend.api.compliance import router as compliance_router
    app.include_router(compliance_router, prefix="/api/v1/compliance")
    logger.info("✅ Compliance router included at /api/v1/compliance")
except ImportError as e:
    logger.warning(f"Compliance router not available: {e}")

# Recommendations router
try:
    from backend.api.recommendations import router as recommendations_router
    app.include_router(recommendations_router, prefix="/api/v1/recommendations")
    logger.info("✅ Recommendations router included at /api/v1/recommendations")
except ImportError as e:
    logger.warning(f"Recommendations router not available: {e}")

# MSA router
try:
    from backend.api.msa import router as msa_router
    app.include_router(msa_router, prefix="/api/v1/msa")
    logger.info("✅ MSA router included at /api/v1/msa")
except ImportError as e:
    logger.warning(f"MSA router not available: {e}")

# Storage router
try:
    from backend.api.storage import router as storage_router
    app.include_router(storage_router, prefix="/api/v1/storage")
    logger.info("✅ Storage router included at /api/v1/storage")
except ImportError as e:
    logger.warning(f"Storage router not available: {e}")

# Network router
try:
    from backend.api.network import router as network_router
    app.include_router(network_router, prefix="/api/v1/network")
    logger.info("✅ Network router included at /api/v1/network")
except ImportError as e:
    logger.warning(f"Network router not available: {e}")

# Cost/FinOps router
try:
    from backend.api.cost import router as cost_router
    app.include_router(cost_router, prefix="/api/v1/cost")
    logger.info("✅ Cost router included at /api/v1/cost")
except ImportError as e:
    logger.warning(f"Cost router not available: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ADK Security Agent Backend", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "secret_manager": SECRETMANAGER_AVAILABLE,
            "chat_manager": chat_manager_available,
            "websockets": websocket_available,
            "performance_monitoring": performance_monitor_available,
            "context_awareness": context_manager_available
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "websocket": "/api/v1/agent/ws" if websocket_available else None,
            "chat": "/api/v1/agent/chat",
            "performance": "/api/v1/performance" if performance_monitor_available else None,
            "context": "/api/v1/context" if context_manager_available else None
        }
    }

# Add WebSocket support if available
if websocket_available:
    @app.websocket("/api/v1/agent/ws")
    async def websocket_endpoint_handler(websocket: WebSocket):
        await websocket_endpoint(websocket)

@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("🚀 Security Agent Backend starting up")
    logger.info(f"Features available: chat_manager={chat_manager_available}, "
               f"websockets={websocket_available}, performance={performance_monitor_available}, "
               f"context={context_manager_available}")

@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown."""
    logger.info("🛑 Security Agent Backend shutting down")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)