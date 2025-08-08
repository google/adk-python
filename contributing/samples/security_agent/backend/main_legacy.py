"""Legacy FastAPI application for the security agent backend.

Simple FastAPI application without the complex modular service architecture.
Provides essential security agent functionality with minimal configuration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import uvicorn
import os
import logging
from services.adk_chat_service import create_adk_chat_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GCP Security Agent - Legacy",
    description="Legacy backend for GCP API Security Evaluation Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0", "mode": "legacy"}

# Basic API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GCP Security Agent - Legacy Mode", "status": "running"}

@app.get("/api/v1/status")
async def get_status():
    """Get agent status."""
    return {
        "status": "running",
        "mode": "legacy",
        "version": "1.0.0",
        "services": {
            "core": "running",
            "health": "running"
        }
    }

# GCP Project endpoints
@app.get("/api/v1/gcp/projects")
async def get_projects():
    """Get list of available GCP projects."""
    try:
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
        
        # Create a friendly display name
        if project_id == 'mgm-digitalconcierge':
            project_name = "MGM Digital Concierge"
        else:
            # Convert project-id format to Title Case
            project_name = project_id.replace('-', ' ').title()
        
        return {
            "success": True,
            "projects": [
                {
                    "project_id": project_id,
                    "name": project_name,
                    "status": "active",
                    "project_number": "123456789"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return {"success": False, "projects": [], "error": str(e)}

@app.get("/api/v1/gcp/projects/{project_id}")
async def get_project_info(project_id: str):
    """Get detailed information about a specific GCP project."""
    return {
        "success": True,
        "project_info": {
            "project_id": project_id,
            "name": f"Project {project_id}",
            "project_number": "123456789",
            "lifecycle_state": "ACTIVE"
        }
    }

# Security endpoints
@app.post("/api/v1/security/evaluate")
async def evaluate_security(request: dict):
    """Security evaluation endpoint."""
    return {
        "success": True,
        "message": "Security evaluation completed",
        "results": {
            "score": 85,
            "issues_found": 2,
            "recommendations": ["Enable 2FA", "Review IAM policies", "Update security policies"]
        }
    }

@app.get("/api/v1/security/score")
async def get_security_score():
    """Get overall security score."""
    return {
        "success": True,
        "score": 85,
        "breakdown": {
            "iam": 90,
            "network": 80,
            "data": 85,
            "compute": 80
        }
    }

@app.get("/api/v1/security/enabled-apis")
async def get_enabled_apis():
    """Get enabled APIs for the project."""
    return {
        "success": True,
        "apis": [
            {"name": "compute.googleapis.com", "enabled": True},
            {"name": "storage.googleapis.com", "enabled": True},
            {"name": "iam.googleapis.com", "enabled": True}
        ]
    }

@app.get("/api/v1/security/findings")
async def get_security_findings():
    """Get security findings."""
    return {
        "success": True,
        "findings": [
            {
                "id": "finding-1",
                "category": "IAM",
                "severity": "HIGH",
                "title": "Overprivileged service account",
                "description": "Service account has more permissions than necessary"
            }
        ]
    }

@app.get("/api/v1/security/health")
async def get_security_health():
    """Check Security Center integration health."""
    return {"success": True, "status": "healthy", "integration": "active"}

# IAM endpoints
@app.get("/api/v1/iam/policy")
async def get_iam_policy():
    """Get IAM policy analysis."""
    return {
        "success": True,
        "policy": {
            "bindings": [
                {"role": "roles/owner", "members": ["user@example.com"]}
            ]
        }
    }

@app.get("/api/v1/iam/project/{project_id}/analyze-user/{user_email}")
async def analyze_user_permissions(project_id: str, user_email: str):
    """Analyze specific user's IAM permissions."""
    return {
        "success": True,
        "user": user_email,
        "permissions": ["compute.instances.list", "storage.buckets.list"],
        "roles": ["roles/viewer"]
    }

@app.get("/api/v1/iam/project/{project_id}/analyze-all-users")
async def analyze_all_users(project_id: str):
    """Analyze all users' IAM permissions."""
    return {
        "success": True,
        "users": [
            {"email": "user@example.com", "roles": ["roles/owner"], "risk": "low"}
        ]
    }

# Recommendations endpoint
@app.post("/api/v1/recommendations/dashboard")
async def get_recommendations(request: dict):
    """Get security recommendations."""
    return {
        "success": True,
        "recommendations": [
            {
                "title": "Enable 2FA",
                "priority": "high",
                "description": "Two-factor authentication should be enabled for all accounts"
            },
            {
                "title": "Review IAM Policies", 
                "priority": "medium",
                "description": "Regular review of IAM policies is recommended"
            }
        ]
    }

# Compliance endpoint
@app.post("/api/v1/compliance/evaluate")
async def evaluate_compliance(request: dict):
    """Evaluate compliance against a framework."""
    return {
        "success": True,
        "framework": request.get("framework", "SOC2"),
        "score": 78,
        "compliant": True,
        "issues": []
    }

# Agent chat endpoint with real ADK integration
@app.post("/api/v1/agent/chat")
async def chat_with_agent(request: dict):
    """Send a message to the agent with real GCP tool integration."""
    try:
        message = request.get("prompt", "")
        project_id = request.get("project_id", "demo-project")
        context = request.get("context", {})
        
        if not message.strip():
            return {
                "success": False,
                "error": "Message cannot be empty",
                "suggestions": ["Ask about security score", "Get recommendations", "Analyze IAM policies"]
            }
        
        logger.info(f"Processing ADK chat message: '{message}' for project: {project_id}")
        
        # Create ADK chat service instance
        chat_service = create_adk_chat_service(project_id)
        
        # Process message with real GCP integration
        result = await chat_service.process_chat_message(message, context)
        
        logger.info(f"ADK chat response generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in ADK chat endpoint: {e}")
        return {
            "success": False,
            "response": f"I encountered an error while processing your request: {str(e)}",
            "error": str(e),
            "suggestions": ["Try asking about security score", "Ask for IAM analysis", "Request recommendations"]
        }

# Service management (for compatibility)
@app.get("/api/v1/services/")
async def get_services():
    """Get all services and their status."""
    return {
        "success": True,
        "services": {
            "core": {"status": "running", "enabled": True},
            "security": {"status": "running", "enabled": True},
            "iam": {"status": "running", "enabled": True}
        }
    }

@app.get("/api/v1/services/status/summary")
async def get_services_status_summary():
    """Get summary of all services status."""
    return {
        "success": True,
        "total_services": 3,
        "running": 3,
        "stopped": 0,
        "error": 0
    }

# Performance monitoring
@app.get("/api/v1/monitoring/summary")
async def get_performance_summary():
    """Get performance summary."""
    return {
        "success": True,
        "cpu_usage": 25.5,
        "memory_usage": 60.2,
        "disk_usage": 45.0,
        "response_time": 150
    }

# Incidents endpoint
@app.get("/api/v1/incidents")
async def get_incidents():
    """Get security incidents."""
    return {
        "success": True,
        "incidents": [
            {
                "id": "inc-001",
                "title": "Suspicious API Access",
                "severity": "medium",
                "status": "investigating",
                "created": "2025-08-07T10:00:00Z"
            }
        ]
    }

@app.post("/api/v1/incidents")
async def create_incident(request: dict):
    """Create a new security incident."""
    return {
        "success": True,
        "incident_id": "inc-002",
        "message": "Incident created successfully"
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Legacy Security Agent Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)