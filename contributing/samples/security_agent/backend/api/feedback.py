"""
Feedback API for STORY-005: Human-in-the-Loop Feedback System
============================================================

API endpoints for collecting, managing, and analyzing user feedback
with ADK evaluation integration.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
import os
import sys

# Add backend services to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.feedback_database import feedback_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/feedback", tags=["Feedback System"])

class FeedbackSubmission(BaseModel):
    """Model for feedback submission."""
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Unique message identifier")
    user_query: str = Field(..., description="Original user query")
    assistant_response: str = Field(..., description="Assistant's response")
    corrected_response: Optional[str] = Field(None, description="User's corrected response")
    rating: Optional[int] = Field(None, ge=1, le=5, description="1-5 star rating")
    thumbs_vote: Optional[str] = Field(None, pattern="^(up|down)$", description="Thumbs up/down vote")
    categories: Optional[List[str]] = Field(default_factory=list, description="Feedback categories")
    user_comments: Optional[str] = Field(None, description="User's additional comments")
    user_id: Optional[str] = Field("anonymous", description="User identifier")

    @validator('categories')
    def validate_categories(cls, v):
        """Validate feedback categories."""
        allowed_categories = [
            'accurate', 'helpful', 'incomplete', 'wrong', 'unclear',
            'too_long', 'too_short', 'irrelevant', 'outdated', 'excellent'
        ]
        if v:
            invalid_categories = [cat for cat in v if cat not in allowed_categories]
            if invalid_categories:
                raise ValueError(f"Invalid categories: {invalid_categories}")
        return v

class FeedbackResponse(BaseModel):
    """Response model for feedback operations."""
    success: bool
    feedback_id: Optional[int] = None
    message: str

class FeedbackMetrics(BaseModel):
    """Model for feedback metrics and analytics."""
    overview: Dict[str, Any]
    daily_trends: List[Dict[str, Any]]
    category_analysis: List[Dict[str, Any]]
    period_days: int

class EvalsetGeneration(BaseModel):
    """Model for evalset generation request."""
    min_feedback_count: Optional[int] = Field(10, ge=1, le=100, description="Minimum feedback items to include")
    include_corrections_only: Optional[bool] = Field(False, description="Only include feedback with corrections")
    min_rating: Optional[int] = Field(3, ge=1, le=5, description="Minimum rating for inclusion")

@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackSubmission, background_tasks: BackgroundTasks):
    """
    Submit user feedback for an assistant response.
    
    This endpoint collects feedback that will be used to improve the assistant's
    performance and generate ADK evaluation datasets.
    """
    try:
        logger.info(f"Receiving feedback for session {feedback.session_id}, message {feedback.message_id}")
        
        # Validate that we have at least one form of feedback
        if not any([
            feedback.rating,
            feedback.thumbs_vote,
            feedback.corrected_response,
            feedback.user_comments,
            feedback.categories
        ]):
            raise HTTPException(
                status_code=400,
                detail="At least one form of feedback (rating, thumbs vote, correction, comment, or category) is required"
            )
        
        # Prepare feedback data
        feedback_data = {
            'session_id': feedback.session_id,
            'message_id': feedback.message_id,
            'user_query': feedback.user_query,
            'assistant_response': feedback.assistant_response,
            'corrected_response': feedback.corrected_response,
            'rating': feedback.rating,
            'thumbs_vote': feedback.thumbs_vote,
            'categories': feedback.categories,
            'user_comments': feedback.user_comments,
            'user_id': feedback.user_id
        }
        
        # Save feedback to database
        feedback_id = feedback_db.save_feedback(feedback_data)
        
        # Schedule background tasks
        background_tasks.add_task(check_evalset_generation)
        background_tasks.add_task(update_improvement_metrics)
        
        logger.info(f"[OK] Feedback saved successfully with ID: {feedback_id}")
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Feedback submitted successfully. Thank you for helping improve the assistant!"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/metrics", response_model=FeedbackMetrics)
async def get_feedback_metrics(days: int = 30):
    """
    Get feedback analytics and metrics for the specified time period.
    
    Returns overview statistics, daily trends, and category analysis.
    """
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        metrics = feedback_db.get_feedback_metrics(days=days)
        
        return FeedbackMetrics(
            overview=metrics['overview'],
            daily_trends=metrics['daily_trends'],
            category_analysis=metrics['category_analysis'],
            period_days=metrics['period_days']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feedback metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/list")
async def list_feedback(session_id: Optional[str] = None, limit: int = 50):
    """
    List feedback entries, optionally filtered by session.
    """
    try:
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
        
        feedback_list = feedback_db.get_feedback(session_id=session_id, limit=limit)
        
        return {
            "success": True,
            "count": len(feedback_list),
            "feedback": feedback_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list feedback: {str(e)}")

@router.post("/generate-evalset")
async def generate_evalset(request: EvalsetGeneration, background_tasks: BackgroundTasks):
    """
    Generate ADK evalset from collected feedback.
    
    Creates an .evalset.json file that can be used with ADK evaluation framework
    to measure and improve assistant performance.
    """
    try:
        logger.info(f"Generating evalset with min_feedback_count={request.min_feedback_count}")
        
        # Generate evalset
        evalset = feedback_db.generate_evalset(min_feedback_count=request.min_feedback_count)
        
        if not evalset:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient feedback data to generate evalset. Need at least {request.min_feedback_count} suitable feedback items."
            )
        
        # Save to file
        file_path = feedback_db.save_evalset_to_file(evalset)
        
        # Schedule background validation
        background_tasks.add_task(validate_evalset, file_path)
        
        return {
            "success": True,
            "evalset_id": evalset["eval_set_id"],
            "eval_cases_count": len(evalset["eval_cases"]),
            "file_path": file_path,
            "message": f"Evalset generated successfully with {len(evalset['eval_cases'])} evaluation cases"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate evalset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate evalset: {str(e)}")

@router.get("/improvement-suggestions")
async def get_improvement_suggestions():
    """
    Get AI-generated suggestions for improving assistant performance based on feedback.
    """
    try:
        # Get recent feedback metrics
        metrics = feedback_db.get_feedback_metrics(days=30)
        
        suggestions = []
        
        # Analyze feedback patterns and generate suggestions
        if metrics['overview'].get('total_feedback', 0) > 0:
            avg_rating = metrics['overview'].get('avg_rating', 0)
            thumbs_down = metrics['overview'].get('thumbs_down', 0)
            thumbs_up = metrics['overview'].get('thumbs_up', 0)
            
            if avg_rating < 3.5:
                suggestions.append({
                    "priority": "high",
                    "category": "response_quality",
                    "suggestion": "Response quality is below optimal. Consider reviewing recent feedback for common issues.",
                    "metric": f"Average rating: {avg_rating:.1f}/5.0"
                })
            
            if thumbs_down > thumbs_up:
                suggestions.append({
                    "priority": "medium",
                    "category": "user_satisfaction",
                    "suggestion": "More negative than positive feedback. Review recent responses for accuracy and helpfulness.",
                    "metric": f"Thumbs down: {thumbs_down}, Thumbs up: {thumbs_up}"
                })
            
            # Analyze category patterns
            for category_data in metrics['category_analysis'][:3]:
                categories = category_data.get('categories', [])
                count = category_data.get('count', 0)
                
                if 'wrong' in categories and count > 5:
                    suggestions.append({
                        "priority": "high",
                        "category": "accuracy",
                        "suggestion": f"High number of 'wrong' categorizations ({count}). Review accuracy of recent responses.",
                        "metric": f"Wrong responses: {count}"
                    })
                
                if 'incomplete' in categories and count > 3:
                    suggestions.append({
                        "priority": "medium",
                        "category": "completeness",
                        "suggestion": f"Responses marked as incomplete ({count}). Consider providing more comprehensive answers.",
                        "metric": f"Incomplete responses: {count}"
                    })
        
        if not suggestions:
            suggestions.append({
                "priority": "low",
                "category": "general",
                "suggestion": "Feedback patterns look good! Continue monitoring for improvement opportunities.",
                "metric": "No major issues detected"
            })
        
        return {
            "success": True,
            "suggestions": suggestions,
            "analysis_period": "30 days",
            "total_feedback_analyzed": metrics['overview'].get('total_feedback', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get improvement suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

# Background task functions

async def check_evalset_generation():
    """Background task to check if we should auto-generate evalsets."""
    try:
        # Get recent feedback count
        feedback_list = feedback_db.get_feedback(limit=100)
        suitable_feedback = [
            f for f in feedback_list 
            if f.get('corrected_response') or (f.get('rating') and f['rating'] >= 4)
        ]
        
        # Auto-generate evalset if we have enough feedback
        if len(suitable_feedback) >= 25:  # Higher threshold for auto-generation
            logger.info(f"Auto-generating evalset with {len(suitable_feedback)} suitable feedback items")
            evalset = feedback_db.generate_evalset(min_feedback_count=20)
            if evalset:
                file_path = feedback_db.save_evalset_to_file(evalset)
                logger.info(f"[OK] Auto-generated evalset saved to {file_path}")
    except Exception as e:
        logger.error(f"Error in background evalset generation: {e}")

async def update_improvement_metrics():
    """Background task to update improvement tracking metrics."""
    try:
        # This would integrate with actual model performance metrics
        # For now, we'll just log that the task ran
        logger.info("[STATS] Updated improvement metrics based on recent feedback")
    except Exception as e:
        logger.error(f"Error updating improvement metrics: {e}")

async def validate_evalset(file_path: str):
    """Background task to validate generated evalset."""
    try:
        # Basic validation - check if file exists and has valid JSON
        with open(file_path, 'r') as f:
            evalset = json.load(f)
        
        if evalset.get('eval_cases') and len(evalset['eval_cases']) > 0:
            logger.info(f"[OK] Evalset validation passed: {file_path}")
            # Update database validation status
            # This would be implemented based on specific validation requirements
        else:
            logger.warning(f"[WARNING] Evalset validation failed: {file_path}")
            
    except Exception as e:
        logger.error(f"Error validating evalset {file_path}: {e}")

@router.get("/health")
async def feedback_health_check():
    """Health check for feedback system."""
    try:
        # Test database connection
        test_metrics = feedback_db.get_feedback_metrics(days=1)
        
        return {
            "status": "healthy",
            "database": "connected",
            "recent_feedback_count": test_metrics['overview'].get('total_feedback', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Feedback health check failed: {e}")
        raise HTTPException(status_code=503, detail="Feedback system unhealthy")