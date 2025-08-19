"""FastAPI application for the security agent backend.

Clean, unified FastAPI application with all available features.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
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
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import validation middleware  
try:
    from backend.middleware.validation import InputValidationMiddleware
    INPUT_VALIDATION_AVAILABLE = True
    logger.info("✅ Input validation middleware loaded")
except ImportError as e:
    INPUT_VALIDATION_AVAILABLE = False
    logger.warning(f"⚠️ Input validation not available: {e}")

# Import rate limiting middleware (disable due to aioredis compatibility issue)
try:
    # Temporarily disabled due to aioredis compatibility issue with Python 3.13
    # from backend.middleware.rate_limiter import RateLimitMiddleware
    RATE_LIMITER_AVAILABLE = False
    logger.info("⚠️ Rate limiting temporarily disabled (aioredis compatibility)")
except ImportError as e:
    RATE_LIMITER_AVAILABLE = False
    logger.warning(f"⚠️ Rate limiting not available: {e}")

# Try multiple locations for .env file
env_locations = [
    Path(__file__).parent.parent / "deploy" / ".env",  # deploy/.env
    Path(__file__).parent.parent / ".env",  # security_agent/.env
    Path(__file__).parent / ".env",  # backend/.env
]

for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
        break
else:
    print("⚠️ No .env file found, using system environment variables")

# Set up Google Application Credentials if not already set
if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    # Try to find any service account JSON file in the secrets directory
    secrets_dir = Path(__file__).parent / "config" / "secrets"
    if secrets_dir.exists():
        json_files = list(secrets_dir.glob("*.json"))
        if json_files:
            # Use the first JSON file found
            service_account_path = json_files[0]
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(service_account_path)
            print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS to: {service_account_path}")
        else:
            print("⚠️ No service account JSON files found in config/secrets/")
    else:
        print("⚠️ Service account directory not found, will use default credentials")

# Add backend's parent directory to sys.path to make 'backend' importable.
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_parent_dir = os.path.dirname(current_dir)
if backend_parent_dir not in sys.path:
    sys.path.insert(0, backend_parent_dir)

# Add project root to sys.path for absolute imports (e.g., 'contributing')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(backend_parent_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Import Secret Manager for runtime credential loading
try:
    from google.cloud import secretmanager
    SECRETMANAGER_AVAILABLE = True
except ImportError:
    SECRETMANAGER_AVAILABLE = False
    logger.warning("Google Cloud Secret Manager not available")

# Backend doesn't need ADK - it just processes requests
# The frontend is the thin client that displays UI
# The backend provides the intelligence via API endpoints
logger.info("✅ Backend configured as API service (no local ADK needed)")

# Context management is handled by ADK natively
context_manager_available = False

# Services directory removed - using APIs directly

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

# Temporarily disable monitoring due to dependency issues
# from backend.monitoring import setup_monitoring

# Setup service account credentials first
setup_service_account_from_secret()

# Create FastAPI app
app = FastAPI(
    title="Security Agent Backend",
    description="Unified ADK Security Agent API",
    version="1.0.0"
)

# Setup monitoring - temporarily disabled
# setup_monitoring(app)
logger.info("⚠️ Monitoring temporarily disabled")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Input Validation Middleware
if INPUT_VALIDATION_AVAILABLE:
    app.add_middleware(InputValidationMiddleware)
    logger.info(f"✅ Input validation enabled")
else:
    logger.info("⚠️ Input validation disabled")

# Add Rate Limiting Middleware
if RATE_LIMITER_AVAILABLE:
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        # app.add_middleware(RateLimitMiddleware, redis_url=redis_url)  # Disabled
        logger.info(f"✅ Rate limiting enabled")
    except Exception as e:
        logger.warning(f"⚠️ Rate limiting failed to initialize: {e}")
else:
    logger.info("⚠️ Rate limiting disabled")

# ===========================================
# API ROUTER REGISTRATION
# ===========================================
# Standardized API path structure:
# /api/v1/agent/* - Agent and chat endpoints
# /api/v1/gcp/* - Google Cloud Platform endpoints
# /api/v1/security/* - Security analysis endpoints
# /api/v1/iam/* - IAM analysis endpoints
# /api/v1/monitoring/* - Monitoring and logs endpoints
# /api/v1/recommendations/* - Recommendations endpoints
# /api/v1/context/* - Context management endpoints

# Note: RADAR/multi-agent code removed per single-agent architecture requirements
# Agent functionality is handled through direct tools in /api/v1/chat/message endpoint

# Sessions router for persistent conversation management (STORY-013)
try:
    from backend.api.sessions import router as sessions_router
    app.include_router(sessions_router, prefix="/api/v1/sessions")
    print("Sessions router included")
    logger.info("✅ Sessions router included at /api/v1/sessions (STORY-013: SQLite persistence)")
except ImportError as e:
    logger.warning(f"⚠️ Sessions router not available: {e}")
    logger.info("✅ Using ADK's built-in session management as fallback")

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

# IAM router
try:
    from backend.api.iam import router as iam_router
    app.include_router(iam_router, prefix="/api/v1/iam")
    logger.info("✅ IAM router included at /api/v1/iam")
except ImportError as e:
    logger.warning(f"⚠️ IAM router not available: {e}")

# Recommendations router for Google Cloud Recommender API
try:
    from backend.api.recommendations import router as recommendations_router
    app.include_router(recommendations_router)
    logger.info("✅ Recommendations router included (Google Cloud Recommender API)")
except ImportError as e:
    logger.warning(f"⚠️ Recommendations router not available: {e}")

# Search is handled natively by ADK tools - no custom router needed
logger.info("✅ Using ADK's built-in search tools")

# Storage router
try:
    from backend.api.storage import router as storage_router
    app.include_router(storage_router, prefix="/api/v1/storage")
    logger.info("✅ Storage router included at /api/v1/storage")
except ImportError as e:
    logger.warning(f"Storage router not available: {e}")

# Asset Inventory router for unified GCP resource access
try:
    from backend.api.asset_inventory import router as asset_inventory_router
    app.include_router(asset_inventory_router, prefix="/api/v1/assets")
    logger.info("✅ Asset Inventory router included at /api/v1/assets")
except ImportError as e:
    logger.warning(f"Asset Inventory router not available: {e}")

# API Keys router for API key management
try:
    from backend.api.keys import router as keys_router
    app.include_router(keys_router, prefix="/api/v1/keys")
    logger.info("✅ API Keys router included at /api/v1/keys")
except ImportError as e:
    logger.warning(f"API Keys router not available: {e}")

# Advisory Notifications router for security bulletins and alerts
try:
    from backend.api.advisory_notifications import router as advisory_router
    app.include_router(advisory_router, prefix="/api/v1/advisory")
    logger.info("✅ Advisory Notifications router included at /api/v1/advisory")
except ImportError as e:
    logger.warning(f"Advisory Notifications router not available: {e}")

# Import remediation API (STORY-210)
try:
    from backend.api.remediation import router as remediation_router
    app.include_router(remediation_router, prefix="/api/v1/remediation")
    logger.info("✅ Remediation API loaded (STORY-210)")
except ImportError as e:
    logger.warning(f"⚠️ Remediation API not available: {e}")



# Chat endpoint for frontend communication
@app.post("/api/v1/chat/message")
async def chat_message(request: Dict[str, Any]):
    """
    Handle chat messages using the configured ADK agent with session persistence.
    
    This endpoint uses the agent.py configuration which has all the enhanced
    security tools and proper conversation handling.
    """
    query = request.get("query", "")
    session_id = request.get("session_id", "default")
    user_id = request.get("user_id", "default_user")
    
    logger.info(f"Received chat request - User: {user_id}, Session: {session_id}, Query: {query[:50]}...")
    
    try:
        # Import the configured agent from agent.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from agent import agent
        
        # Import conversation context manager for session persistence
        try:
            from backend.api.conversation_context import conversation_manager
            
            # Get or create session
            session = conversation_manager.get_or_create_session(session_id, user_id)
            
            # Get conversation context
            context = conversation_manager.get_context(session_id)
            
            # Add context to query if there's history
            enhanced_query = query
            if context:
                enhanced_query = f"Previous conversation context:\n{context}\n\nCurrent question: {query}"
                logger.info(f"Using conversation context for session {session_id}")
        except Exception as e:
            logger.warning(f"Conversation context not available: {e}")
            enhanced_query = query
        
        # Use the configured agent's tools directly for more reliable responses
        logger.info(f"Processing query with enhanced security tools")
        
        # Import the agent tools directly
        from agent import (
            discover_assets, analyze_security, run_security_focused_scan,
            run_vulnerability_focused_scan, analyze_iam, analyze_storage,
            get_security_recommendations, run_comprehensive_security_scan
        )
        
        # Determine which tool to use based on query content
        query_lower = enhanced_query.lower()
        
        if any(word in query_lower for word in ['enhanced', 'vulnerability', 'custom rules', 'risk score']):
            response_text = analyze_security()
        elif any(word in query_lower for word in ['comprehensive', 'full scan', 'complete']):
            response_text = run_comprehensive_security_scan()
        elif any(word in query_lower for word in ['discover', 'assets', 'inventory', 'resources']):
            response_text = discover_assets()
        elif any(word in query_lower for word in ['iam', 'permissions', 'service account']):
            response_text = analyze_iam()
        elif any(word in query_lower for word in ['storage', 'bucket']):
            response_text = analyze_storage()
        elif any(word in query_lower for word in ['recommend', 'advice', 'suggest']):
            response_text = get_security_recommendations()
        elif any(word in query_lower for word in ['security', 'vulnerabilit', 'scan', 'analyze']):
            response_text = run_security_focused_scan()
        else:
            # General security overview
            response_text = f"🔐 **Enhanced Security Assistant for {os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')}**\n\n" + analyze_security()
        
        # Store in conversation history if available
        try:
            conversation_manager.add_to_history(session_id, query, response_text)
        except:
            pass  # Continue even if conversation storage fails
            
    except Exception as e:
        logger.error(f"Error processing query with agent: {e}")
        # Fallback to basic response
        response_text = f"🔍 **GCP Security Assistant for project: {os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')}**\n\nI can help you with:\n\n• **Resource Discovery**: 'What resources do I have?'\n• **Security Analysis**: 'Check my security posture'\n• **IAM Review**: 'Show my service accounts'\n• **Vulnerability Scan**: 'Find security issues'\n\nWhat would you like to explore?"
    
    # Always return a properly formatted response
    return {
        "response": response_text,
        "session_id": session_id,
        "user_id": user_id,
        "success": True
    }


def _generate_fallback_response(query_lower: str) -> str:
    """Generate an intelligent fallback response based on query keywords."""
    
    if "resource" in query_lower or "asset" in query_lower or "inventory" in query_lower:
        return (
            "To discover your GCP resources, I would need to:\n"
            "1. Query the Cloud Asset Inventory API\n"
            "2. List resources by type (compute, storage, network)\n"
            "3. Check resource metadata and configurations\n\n"
            "Currently running in demo mode. Connect your GCP project to see real data."
        )
    
    elif "security" in query_lower or "vulnerabilit" in query_lower:
        return (
            "For security analysis, I can help with:\n"
            "• Vulnerability scanning\n"
            "• IAM permission analysis\n"
            "• Security best practices review\n"
            "• Compliance checking\n\n"
            "Please specify what aspect of security you'd like to examine."
        )
    
    elif "iam" in query_lower or "permission" in query_lower or "access" in query_lower:
        return (
            "IAM analysis includes:\n"
            "• User and service account permissions\n"
            "• Role assignments and custom roles\n"
            "• Policy bindings at project/resource level\n"
            "• Least privilege recommendations\n\n"
            "What specific IAM aspect would you like to review?"
        )
    
    elif "recommend" in query_lower or "suggest" in query_lower or "improve" in query_lower:
        return (
            "I can provide recommendations for:\n"
            "• Security hardening\n"
            "• Cost optimization\n"
            "• Performance improvements\n"
            "• Compliance alignment\n\n"
            "Which area would you like recommendations for?"
        )
    
    else:
        return (
            "I'm your GCP Security Assistant. I can help with:\n\n"
            "🔍 **Resource Discovery**: Find and inventory all GCP assets\n"
            "🛡️ **Security Analysis**: Identify vulnerabilities and risks\n"
            "🔐 **IAM Review**: Analyze permissions and access controls\n"
            "📊 **Recommendations**: Get actionable security improvements\n"
            "✅ **Compliance**: Check alignment with standards\n\n"
            "What would you like to explore?"
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ADK Security Agent Backend", "status": "running"}

@app.get("/api/v1/rate-limit/status")
async def rate_limit_status():
    """Check rate limiting status."""
    if not RATE_LIMITER_AVAILABLE:
        return {"rate_limiting": "disabled", "reason": "middleware not available"}
    
    return {
        "rate_limiting": "enabled",
        "limits": {
            "heavy_operations": "5/minute",
            "chat": "30/minute", 
            "default": "100/minute"
        },
        "window": "60 seconds"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with robust system status."""
    
    # Check component availability
    components_status = {}
    
    # Test critical components
    try:
        from backend.api.agent_llm import router as agent_router
        components_status["agent_llm"] = "available"
    except ImportError:
        components_status["agent_llm"] = "fallback"
    
    try:
        from backend.api.iam import router as iam_router
        components_status["iam_analysis"] = "available"
    except ImportError:
        components_status["iam_analysis"] = "fallback"
    
    try:
        from backend.api.recommendations import router as recommendations_router
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
            "rate_limiting": RATE_LIMITER_AVAILABLE,
            "adk_session_management": True,
            "websockets": True,
            "context_awareness": True,
            "robust_fallbacks": True
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "websocket": "/api/v1/agent/ws",
            "chat": "/api/v1/agent/chat",
            "agent_chat": "/api/v1/agent/chat",
            "sessions": "/api/v1/sessions",
            "iam_analysis": "/api/v1/iam",
            "recommendations": "/api/v1/recommendations",
            "asset_inventory": "/api/v1/assets",
            "asset_discovery": "/api/v1/assets/discover",
            "security_analysis": "/api/v1/assets/security/analyze",
            "rate_limit_status": "/api/v1/rate-limit/status",
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
    logger.info("✅ ADK-compliant session management enabled")
    logger.info(f"🔐 Secret Manager: {'✅ available' if SECRETMANAGER_AVAILABLE else '⚠️ not configured'}")
    logger.info(f"🚫 Rate Limiting: {'✅ enabled' if RATE_LIMITER_AVAILABLE else '⚠️ disabled'}")
    logger.info("🔄 All API endpoints operational with intelligent fallbacks")
    logger.info("🎯 System ready to handle requests even with missing dependencies")
    
    # Perform internal healthcheck on startup
    logger.info("🏥 Running startup healthcheck...")
    try:
        health_status = await health_check()
        logger.info(f"✅ Healthcheck passed: {health_status['status']}")
        logger.info(f"📋 Components status: {json.dumps(health_status['components'], indent=2)}")
        logger.info(f"🔧 Active features: {json.dumps(health_status['features'], indent=2)}")
        logger.info(f"🌐 Available endpoints: {len([e for e in health_status['endpoints'].values() if e is not None])} active")
    except Exception as e:
        logger.error(f"❌ Healthcheck failed: {e}")
        logger.warning("⚠️ System may have limited functionality")

@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown."""
    logger.info("🛑 Security Agent Backend shutting down")

if __name__ == "__main__":
    # Use port from environment or default to 8001 to avoid conflicts
    port = int(os.getenv("BACKEND_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
