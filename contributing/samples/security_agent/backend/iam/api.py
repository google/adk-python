"""IAM related API endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from .service import IAMPolicyAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/project/{project_id}/analyze-user/{user_email}")
async def analyze_user_iam_permissions(project_id: str, user_email: str) -> Dict[str, Any]:
    """Analyze a user's IAM permissions against security best practices."""
    try:
        analyzer = IAMPolicyAnalyzer()
        result = analyzer.analyze_user_permissions(project_id, user_email)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing user IAM permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/analyze-all-users")
async def analyze_all_users_iam_permissions(project_id: str) -> Dict[str, Any]:
    """Analyze all users' IAM permissions in a project against security best practices."""
    try:
        analyzer = IAMPolicyAnalyzer()
        result = analyzer.analyze_all_users(project_id)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing all users IAM permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/policy")
async def get_project_iam_policy(project_id: str) -> Dict[str, Any]:
    """Get the full IAM policy for a project."""
    try:
        analyzer = IAMPolicyAnalyzer()
        iam_policy = analyzer._get_project_iam_policy(project_id)
        
        return {
            "success": True,
            "project_id": project_id,
            "iam_policy": iam_policy
        }
        
    except Exception as e:
        logger.error(f"Error getting IAM policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))