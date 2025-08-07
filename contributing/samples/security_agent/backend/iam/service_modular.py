"""Modular IAM Policy Analyzer Service."""

import logging
from typing import Dict, List, Any, Optional
from google.auth import default
from google.cloud import resourcemanager_v3

from core.base_service import BaseService
from .service import IAMPolicyAnalyzer

logger = logging.getLogger(__name__)


class IAMService(BaseService):
    """Modular IAM service that extends BaseService."""
    
    def __init__(self, service_name: str = "iam", credentials=None, project_id=None):
        """Initialize IAM service."""
        super().__init__(service_name, credentials, project_id)
        self.analyzer = None
        self.resource_manager_client = None
        
    async def initialize(self) -> bool:
        """Initialize the IAM service."""
        try:
            # Initialize the IAM policy analyzer
            self.analyzer = IAMPolicyAnalyzer()
            
            # Initialize Resource Manager client if credentials available
            if self.credentials:
                self.resource_manager_client = resourcemanager_v3.ProjectsClient(
                    credentials=self.credentials
                )
            else:
                self.resource_manager_client = resourcemanager_v3.ProjectsClient()
            
            logger.info("IAM service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IAM service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the IAM service."""
        try:
            # Clean up resources
            self.analyzer = None
            self.resource_manager_client = None
            
            logger.info("IAM service shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown IAM service: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of the IAM service."""
        try:
            health_status = {
                "healthy": True,
                "checks": {}
            }
            
            # Check analyzer availability
            health_status["checks"]["analyzer"] = {
                "status": "healthy" if self.analyzer else "unhealthy",
                "available": self.analyzer is not None
            }
            
            # Check Resource Manager client
            health_status["checks"]["resource_manager"] = {
                "status": "healthy" if self.resource_manager_client else "unhealthy",
                "available": self.resource_manager_client is not None
            }
            
            # Test connectivity if credentials available
            if self.resource_manager_client and self.project_id:
                try:
                    # Try to get project info as a health check
                    project = self.resource_manager_client.get_project(
                        name=f"projects/{self.project_id}"
                    )
                    health_status["checks"]["gcp_connectivity"] = {
                        "status": "healthy",
                        "project_accessible": True,
                        "project_name": project.display_name
                    }
                except Exception as e:
                    health_status["checks"]["gcp_connectivity"] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "project_accessible": False
                    }
                    health_status["healthy"] = False
            
            # Overall health determination
            for check_name, check_status in health_status["checks"].items():
                if check_status.get("status") != "healthy":
                    health_status["healthy"] = False
                    break
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed for IAM service: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "checks": {}
            }
    
    # Service-specific methods
    def analyze_user_permissions(self, project_id: str, user_email: str) -> Dict[str, Any]:
        """Analyze a user's IAM permissions."""
        if not self.is_available():
            return {
                "success": False,
                "error": "IAM service is not available"
            }
        
        return self.analyzer.analyze_user_permissions(project_id, user_email)
    
    def analyze_all_users(self, project_id: str) -> Dict[str, Any]:
        """Analyze all users in a project."""
        if not self.is_available():
            return {
                "success": False,
                "error": "IAM service is not available"
            }
        
        return self.analyzer.analyze_all_users(project_id)
    
    def get_project_iam_policy(self, project_id: str) -> Dict[str, Any]:
        """Get the IAM policy for a project."""
        if not self.is_available():
            return {
                "success": False,
                "error": "IAM service is not available"
            }
        
        try:
            iam_policy = self.analyzer._get_project_iam_policy(project_id)
            return {
                "success": True,
                "project_id": project_id,
                "iam_policy": iam_policy
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id
            }