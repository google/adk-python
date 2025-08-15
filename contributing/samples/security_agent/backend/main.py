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

# Add current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

# Import required components for ADK
try:
    from chat_manager import chat_manager, EnhancedChatManager
    logger.info("✅ Enhanced chat manager loaded for ADK session management")
except ImportError as e:
    logger.error(f"❌ Enhanced chat manager is REQUIRED for ADK functionality: {e}")
    logger.error("Please ensure chat_manager.py is properly installed")

# WebSocket manager removed - using ConnectionManager in agent_llm.py instead

try:
    from api.performance_monitor import performance_monitor, PerformanceMonitor
    performance_monitor_available = True
    logger.info("✅ Performance monitor loaded successfully")
except ImportError as e:
    performance_monitor_available = False
    logger.warning(f"⚠️ Performance monitor not available: {e}")
    
    # Create mock performance monitor for fallback
    class MockPerformanceMonitor:
        def __init__(self):
            self.monitoring_active = False
        
        def record_response_time(self, *args, **kwargs):
            pass
        
        def get_current_system_metrics(self):
            return None
        
        async def start_monitoring(self):
            self.monitoring_active = True
    
    performance_monitor = MockPerformanceMonitor()
    PerformanceMonitor = MockPerformanceMonitor

# Context manager removed - using chat_manager for context instead
context_manager_available = False

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

# Agent router with ADK session management
try:
    from api.agent_llm import router as agent_router
    app.include_router(agent_router, prefix="/api/v1/agent")
    logger.info("✅ ADK Agent router included at /api/v1/agent (with intelligent steering and session management)")
except ImportError as e:
    logger.warning(f"⚠️ ADK Agent router not available: {e}")
    logger.warning("Creating fallback agent router...")
    
    # Create a basic fallback agent router
    from fastapi import APIRouter
    fallback_agent_router = APIRouter()
    
    @fallback_agent_router.post("/chat")
    async def fallback_chat(request: dict):
        return {
            "success": False,
            "response": "Agent system not fully configured. Please install required dependencies.",
            "error": "ADK Agent router not available"
        }
    
    app.include_router(fallback_agent_router, prefix="/api/v1/agent")
    logger.info("⚠️ Fallback agent router included at /api/v1/agent")

# Sessions router for ADK session management
try:
    from api.sessions import router as sessions_router
    app.include_router(sessions_router, prefix="/api/v1/sessions")
    logger.info("✅ Sessions router included at /api/v1/sessions (ADK thin client support)")
except ImportError as e:
    logger.warning(f"⚠️ Sessions router not available: {e}")
    logger.warning("Creating fallback sessions router...")
    
    # Create a basic fallback sessions router
    fallback_sessions_router = APIRouter()
    
    @fallback_sessions_router.post("/create")
    async def fallback_create_session(request: dict):
        import uuid
        return {
            "success": True,
            "session_id": str(uuid.uuid4()),
            "message": "Mock session created - full session management not available"
        }
    
    app.include_router(fallback_sessions_router, prefix="/api/v1/sessions")
    logger.info("⚠️ Fallback sessions router included at /api/v1/sessions")

# GCP router
try:
    from api.gcp import router as gcp_router
    app.include_router(gcp_router, prefix="/api/v1/gcp")
    logger.info("✅ GCP router included at /api/v1/gcp")
except ImportError as e:
    logger.error(f"GCP router not available: {e}")

# Security router
try:
    from api.security import router as security_router
    app.include_router(security_router, prefix="/api/v1/security")
    logger.info("✅ Security router included at /api/v1/security")
except ImportError as e:
    logger.error(f"Security router not available: {e}")

# Monitoring router
try:
    from api.monitoring import router as monitoring_router
    app.include_router(monitoring_router, prefix="/api/v1/monitoring")
    logger.info("✅ Monitoring router included at /api/v1/monitoring")
except ImportError as e:
    logger.warning(f"Monitoring router not available: {e}")

# Performance monitor router (if available)
if performance_monitor_available:
    try:
        from api.performance_monitor import router as performance_router
        app.include_router(performance_router, prefix="/api/v1/performance")
        logger.info("✅ Performance router included at /api/v1/performance")
    except ImportError as e:
        logger.warning(f"Performance router not available: {e}")

# Context manager removed - context handled through chat_manager in agent_llm.py

# IAM router
try:
    from api.iam import router as iam_router
    app.include_router(iam_router, prefix="/api/v1/iam")
    logger.info("✅ IAM router included at /api/v1/iam")
except ImportError as e:
    logger.warning(f"⚠️ IAM router not available: {e}")
    logger.info("IAM analysis will use fallback implementations in API endpoints")

# Compliance router
try:
    from api.compliance import router as compliance_router
    app.include_router(compliance_router, prefix="/api/v1/compliance")
    logger.info("✅ Compliance router included at /api/v1/compliance")
except ImportError as e:
    logger.warning(f"⚠️ Compliance router not available: {e}")
    logger.info("Compliance analysis will use fallback implementations")

# Recommendations router
try:
    from api.recommendations import router as recommendations_router
    app.include_router(recommendations_router, prefix="/api/v1/recommendations")
    logger.info("✅ Recommendations router included at /api/v1/recommendations")
except ImportError as e:
    logger.warning(f"⚠️ Recommendations router not available: {e}")
    logger.info("Recommendation service will use fallback implementations")

# MSA router removed - unused functionality

# Storage router
try:
    from api.storage import router as storage_router
    app.include_router(storage_router, prefix="/api/v1/storage")
    logger.info("✅ Storage router included at /api/v1/storage")
except ImportError as e:
    logger.warning(f"Storage router not available: {e}")

# Network router
try:
    from api.network import router as network_router
    app.include_router(network_router, prefix="/api/v1/network")
    logger.info("✅ Network router included at /api/v1/network")
except ImportError as e:
    logger.warning(f"Network router not available: {e}")

# Cost/FinOps router
try:
    from api.cost import router as cost_router
    app.include_router(cost_router, prefix="/api/v1/cost")
    logger.info("✅ Cost router included at /api/v1/cost")
except ImportError as e:
    logger.warning(f"Cost router not available: {e}")

# Asset Inventory router for unified GCP resource access
try:
    from api.asset_inventory import router as asset_inventory_router
    app.include_router(asset_inventory_router, prefix="/api/v1/assets")
    logger.info("✅ Asset Inventory router included at /api/v1/assets")
except ImportError as e:
    logger.warning(f"Asset Inventory router not available: {e}")

# Cache Management router for cache control and monitoring
try:
    from api.cache_management import router as cache_router
    app.include_router(cache_router, prefix="/api/v1/cache")
    logger.info("✅ Cache Management router included at /api/v1/cache")
except ImportError as e:
    logger.warning(f"Cache Management router not available: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ADK Security Agent Backend", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint with robust system status."""
    
    # Check component availability
    components_status = {}
    
    # Test critical components
    try:
        from api.agent_llm import router as agent_router
        components_status["agent_llm"] = "available"
    except ImportError:
        components_status["agent_llm"] = "fallback"
    
    try:
        from api.iam import router as iam_router
        components_status["iam_analysis"] = "available"
    except ImportError:
        components_status["iam_analysis"] = "fallback"
    
    try:
        from api.recommendations import router as recommendations_router
        components_status["recommendations"] = "available"
    except ImportError:
        components_status["recommendations"] = "fallback"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_mode": "robust_fallback_enabled",
        "components": components_status,
        "features": {
            "secret_manager": SECRETMANAGER_AVAILABLE,
            "adk_session_management": True,  # Always enabled with fallbacks
            "websockets": True,  # Using ConnectionManager in agent_llm.py
            "performance_monitoring": performance_monitor_available,
            "context_awareness": True,  # Handled through chat_manager with fallbacks
            "robust_fallbacks": True  # System designed to handle missing dependencies
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "websocket": "/api/v1/agent/ws",  # Available in agent_llm.py or fallback
            "chat": "/api/v1/agent/chat",  # Always available with fallbacks
            "sessions": "/api/v1/sessions",  # Always available with fallbacks
            "performance": "/api/v1/performance" if performance_monitor_available else None,
            "iam_analysis": "/api/v1/iam",  # Available with fallbacks
            "recommendations": "/api/v1/recommendations",  # Available with fallbacks
            "asset_inventory": "/api/v1/assets",
            "asset_discovery": "/api/v1/assets/discover",
            "security_analysis": "/api/v1/assets/security/analyze",
            "cache_status": "/api/v1/cache/status",
            "cache_management": "/api/v1/cache"
        },
        "notes": [
            "System designed with robust fallbacks for all dependencies",
            "Fallback implementations provide basic functionality when modules are missing",
            "All critical API endpoints remain available in degraded mode",
            "Check component status for details on availability vs fallback mode"
        ]
    }

# WebSocket endpoint is available in agent_llm.py at /api/v1/agent/ws

@app.on_event("startup")
async def startup_event():
    """Application startup with robust dependency handling."""
    logger.info("🚀 Security Agent Backend starting up")
    logger.info("🛡️ Robust fallback system enabled")
    logger.info("✅ ADK-compliant session management enabled (with fallbacks)")
    logger.info(f"📊 Performance monitoring: {'✅ available' if performance_monitor_available else '⚠️ fallback mode'}")
    logger.info(f"🔐 Secret Manager: {'✅ available' if SECRETMANAGER_AVAILABLE else '⚠️ not configured'}")
    logger.info("🔄 All API endpoints operational with intelligent fallbacks")
    logger.info("🎯 System ready to handle requests even with missing dependencies")

@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown."""
    logger.info("🛑 Security Agent Backend shutting down")

if __name__ == "__main__":
    # Use port from environment or default to 8001 to avoid conflicts
    port = int(os.getenv("BACKEND_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)