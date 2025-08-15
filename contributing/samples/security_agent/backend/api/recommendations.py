"""
Enhanced Security Recommendations API endpoints with Google Cloud Recommender integration
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import asyncio
import time
from datetime import datetime

# Safe imports with fallbacks for recommendation functionality
try:
    from backend.models.recommender_models import (
        RecommenderContextRequest,
        RecommendationListResponse,
        RecommendationActionRequest,
        RecommendationActionResponse,
        ChatRecommendationQuery,
        ChatRecommendationResponse,
        RecommendationProgress,
        ProgressUpdateRequest,
        ProgressStatusResponse,
        RecommenderType,
        Priority,
        RecommendationState
    )
    from backend.services.recommender_service import RecommenderService, RecommendationContext
    from backend.services.chat_recommendation_service import ChatRecommendationService
    from backend.chat_manager import chat_manager
    RECOMMENDER_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Recommender models/services not available: {e}")
    RECOMMENDER_MODELS_AVAILABLE = False
    
    # Create mock classes and enums
    from pydantic import BaseModel
    from enum import Enum
    
    class RecommenderType(Enum):
        SECURITY = "security"
        COST = "cost"
        PERFORMANCE = "performance"
    
    class Priority(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    class RecommendationState(Enum):
        ACTIVE = "active"
        DISMISSED = "dismissed"
        APPLIED = "applied"
    
    class RecommenderContextRequest(BaseModel):
        project_id: str
        location: str = "global"
        max_results: Optional[int] = None
        recommender_types: Optional[List[RecommenderType]] = None
        priority_filter: Optional[List[Priority]] = None
        state_filter: Optional[List[RecommendationState]] = None
        include_insights: bool = False
    
    class RecommendationListResponse(BaseModel):
        success: bool
        total_count: int
        filtered_count: int
        recommendations: List[Dict[str, Any]]
        analytics_summary: Optional[Dict[str, Any]] = None
        execution_time_ms: Optional[float] = None
    
    class RecommendationActionRequest(BaseModel):
        recommendation_id: str
        action: str
        dry_run: bool = True
    
    class RecommendationActionResponse(BaseModel):
        success: bool
        recommendation_id: str
        action: str
        dry_run: bool
        execution_log: List[str]
        error_message: Optional[str] = None
    
    class ChatRecommendationQuery(BaseModel):
        query: str
        context: Dict[str, Any] = {}
    
    class ChatRecommendationResponse(BaseModel):
        success: bool
        response_text: str
        recommendations: List[Dict[str, Any]] = []
        suggested_actions: List[str] = []
        follow_up_questions: List[str] = []
        context_updates: Dict[str, Any] = {}
    
    class RecommendationProgress(BaseModel):
        recommendation_id: str
        current_step: int
        total_steps: int
        completed_steps: List[int]
        overall_status: str
        user_id: str
    
    class ProgressUpdateRequest(BaseModel):
        step_number: int
    
    class ProgressStatusResponse(BaseModel):
        success: bool
        progress: RecommendationProgress
        summary: Dict[str, Any]
        next_actions: List[str]
    
    class RecommendationContext:
        def __init__(self, project_id: str, resource_name: str = "", location: str = "global", 
                     filters: Dict = None, recommender_type: RecommenderType = None):
            self.project_id = project_id
            self.resource_name = resource_name
            self.location = location
            self.filters = filters or {}
            self.recommender_type = recommender_type
    
    class MockRecommenderService:
        def __init__(self):
            self.recommendation_analytics = self
        
        async def get_all_recommendations(self, context, include_insights=False):
            return []
        
        async def get_recommendations_by_type(self, context, recommender_type):
            return []
        
        async def get_recommendations_by_priority(self, context, priority):
            return []
        
        async def apply_recommendation(self, recommendation_id, context, dry_run=True):
            return {"success": False, "message": "Mock service"}
        
        async def get_session_recommendations(self, session_id):
            return []
        
        def calculate_portfolio_metrics(self, recommendations):
            return {}
    
    class MockChatRecommendationService:
        def __init__(self, recommender_service, chat_manager=None):
            pass
        
        async def process_query(self, query):
            return ChatRecommendationResponse(
                success=True,
                response_text="Mock recommendation service is not fully configured.",
                recommendations=[],
                suggested_actions=["Configure Google Cloud Recommender API"],
                follow_up_questions=["How do I set up the recommender service?"]
            )
        
        async def get_service_metrics(self):
            return {"status": "mock", "available": False}
    
    RecommenderService = MockRecommenderService
    ChatRecommendationService = MockChatRecommendationService
    chat_manager = None

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services with enhanced configuration
recommender_service = RecommenderService()
chat_recommendation_service = ChatRecommendationService(recommender_service, chat_manager)

# LLM Agent Integration - for routing recommendation queries through the agent system
try:
    from backend.api.agent_llm import process_with_llm_agent
    LLM_AGENT_AVAILABLE = True
    logger.info("✅ LLM Agent integration available for recommendation queries")
except ImportError:
    LLM_AGENT_AVAILABLE = False
    logger.warning("⚠️ LLM Agent integration not available")
    
    # Create mock function
    async def process_with_llm_agent(query, project_id, context=None, request_id="unknown"):
        return f"Mock LLM agent response for: {query}", "MockAgent"

class RecommendationsRequest(BaseModel):
    project_id: str
    categories: Optional[List[str]] = None
    priority: Optional[str] = None

@router.post("/comprehensive", response_model=RecommendationListResponse)
async def get_comprehensive_recommendations(request: RecommenderContextRequest):
    """Get comprehensive recommendations from Google Cloud Recommender API."""
    try:
        context = RecommendationContext(
            project_id=request.project_id,
            resource_name="",
            location=request.location,
            filters={}
        )
        
        # Get all recommendations
        recommendations = await recommender_service.get_all_recommendations(context, request.include_insights)
        
        # Apply filters
        filtered_recommendations = recommendations
        
        if request.recommender_types:
            filtered_recommendations = [
                r for r in filtered_recommendations 
                if r.recommender_type in request.recommender_types
            ]
        
        if request.priority_filter:
            filtered_recommendations = [
                r for r in filtered_recommendations 
                if r.priority in request.priority_filter
            ]
        
        if request.state_filter:
            filtered_recommendations = [
                r for r in filtered_recommendations 
                if r.state in request.state_filter
            ]
        
        # Limit results
        if request.max_results:
            filtered_recommendations = filtered_recommendations[:request.max_results]
        
        # Calculate analytics
        analytics = recommender_service.recommendation_analytics.calculate_portfolio_metrics(
            filtered_recommendations
        )
        
        return RecommendationListResponse(
            success=True,
            total_count=len(recommendations),
            filtered_count=len(filtered_recommendations),
            recommendations=filtered_recommendations,
            analytics_summary=analytics,
            execution_time_ms=0.0  # Would be calculated from actual execution time
        )
        
    except Exception as e:
        logger.error(f"Error getting comprehensive recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/by-type/{recommender_type}", response_model=RecommendationListResponse)
async def get_recommendations_by_type(
    recommender_type: RecommenderType, 
    request: RecommenderContextRequest
):
    """Get recommendations for a specific recommender type."""
    try:
        context = RecommendationContext(
            project_id=request.project_id,
            resource_name="",
            location=request.location,
            recommender_type=recommender_type
        )
        
        recommendations = await recommender_service.get_recommendations_by_type(
            context, 
            recommender_type
        )
        
        analytics = recommender_service.recommendation_analytics.calculate_portfolio_metrics(
            recommendations
        )
        
        return RecommendationListResponse(
            success=True,
            total_count=len(recommendations),
            filtered_count=len(recommendations),
            recommendations=recommendations,
            analytics_summary=analytics
        )
        
    except Exception as e:
        logger.error(f"Error getting {recommender_type} recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/by-priority/{priority}", response_model=RecommendationListResponse)
async def get_recommendations_by_priority(
    priority: Priority, 
    request: RecommenderContextRequest
):
    """Get recommendations filtered by priority level."""
    try:
        context = RecommendationContext(
            project_id=request.project_id,
            resource_name="",
            location=request.location
        )
        
        recommendations = await recommender_service.get_recommendations_by_priority(
            context, 
            priority
        )
        
        analytics = recommender_service.recommendation_analytics.calculate_portfolio_metrics(
            recommendations
        )
        
        return RecommendationListResponse(
            success=True,
            total_count=len(recommendations),
            filtered_count=len(recommendations),
            recommendations=recommendations,
            analytics_summary=analytics
        )
        
    except Exception as e:
        logger.error(f"Error getting {priority} priority recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply", response_model=RecommendationActionResponse)
async def apply_recommendation(request: RecommendationActionRequest):
    """Apply a specific recommendation."""
    try:
        context = RecommendationContext(
            project_id="",  # Would be extracted from request
            resource_name="",
            location="global"
        )
        
        result = await recommender_service.apply_recommendation(
            request.recommendation_id,
            context,
            dry_run=request.dry_run
        )
        
        return RecommendationActionResponse(
            success=result["success"],
            recommendation_id=request.recommendation_id,
            action=request.action,
            dry_run=request.dry_run,
            execution_log=[result.get("message", "")],
            error_message=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error applying recommendation {request.recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/query", response_model=ChatRecommendationResponse)
async def process_chat_query(query: ChatRecommendationQuery):
    """Process a natural language query about recommendations with LLM agent routing."""
    try:
        # Enhanced query processing with LLM agent integration
        if LLM_AGENT_AVAILABLE and "recommendation" in query.query.lower():
            logger.info(f"🤖 Routing recommendation query through LLM agent system")
            
            # Route through LLM agent for intelligent processing
            agent_response, agent_used = await process_with_llm_agent(
                query.query,
                query.context.project_context.project_id,
                context={"recommendation_focused": True, "user_preferences": query.context.user_preferences}
            )
            
            # If the agent specifically handled recommendations, create enhanced response
            if "recommendation" in agent_used.lower():
                # Get actual recommendations to supplement the agent response
                context = RecommendationContext(
                    project_id=query.context.project_context.project_id,
                    resource_name="",
                    location=query.context.project_context.location
                )
                
                recommendations = await recommender_service.get_all_recommendations(context)
                
                # Create enhanced response combining agent intelligence with real data
                return ChatRecommendationResponse(
                    success=True,
                    response_text=agent_response,
                    recommendations=recommendations[:5],  # Top 5 recommendations
                    suggested_actions=[
                        "Analyze highest priority recommendation",
                        "Show detailed implementation steps",
                        "Calculate total cost savings potential"
                    ],
                    follow_up_questions=[
                        "Would you like me to prioritize these by security impact?",
                        "Should I show you the implementation timeline?",
                        "Do you want to apply any of these recommendations?"
                    ],
                    context_updates={
                        "agent_used": agent_used,
                        "llm_enhanced": True,
                        "recommendation_count": len(recommendations)
                    }
                )
        
        # Fallback to standard chat recommendation processing
        response = await chat_recommendation_service.process_query(query)
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}")
async def get_session_recommendations(session_id: str):
    """Get recommendations tracked for a specific chat session."""
    try:
        recommendations = await recommender_service.get_session_recommendations(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error getting session recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/progress/{recommendation_id}", response_model=ProgressStatusResponse)
async def update_recommendation_progress(
    recommendation_id: str, 
    request: ProgressUpdateRequest
):
    """Update progress for a recommendation implementation."""
    try:
        # This would integrate with a progress tracking system
        # For now, return a mock response
        
        progress = RecommendationProgress(
            recommendation_id=recommendation_id,
            current_step=request.step_number,
            total_steps=5,  # Would be determined from the recommendation
            completed_steps=[i for i in range(1, request.step_number + 1)],
            overall_status="in_progress" if request.step_number < 5 else "completed",
            user_id="current_user"  # Would be extracted from auth
        )
        
        return ProgressStatusResponse(
            success=True,
            progress=progress,
            summary={
                "completion_percentage": (request.step_number / 5) * 100,
                "estimated_remaining_time": max(0, (5 - request.step_number) * 30)  # 30 min per step
            },
            next_actions=["Continue to next step", "Verify current step"] if request.step_number < 5 else ["Mark as complete"]
        )
        
    except Exception as e:
        logger.error(f"Error updating progress for {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{recommendation_id}", response_model=ProgressStatusResponse)
async def get_recommendation_progress(recommendation_id: str):
    """Get progress status for a recommendation implementation."""
    try:
        # Mock progress data - would be retrieved from actual tracking system
        progress = RecommendationProgress(
            recommendation_id=recommendation_id,
            current_step=2,
            total_steps=5,
            completed_steps=[1, 2],
            overall_status="in_progress",
            user_id="current_user"
        )
        
        return ProgressStatusResponse(
            success=True,
            progress=progress,
            summary={
                "completion_percentage": 40,
                "estimated_remaining_time": 90  # minutes
            },
            next_actions=["Continue to step 3", "Review completed steps"]
        )
        
    except Exception as e:
        logger.error(f"Error getting progress for {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/{project_id}")
async def get_recommendation_analytics(project_id: str):
    """Get analytics and metrics for project recommendations with service health."""
    start_time = time.time()
    
    try:
        context = RecommendationContext(
            project_id=project_id,
            resource_name="",
            location="global"
        )
        
        # Get recommendations with performance tracking
        recommendations = await recommender_service.get_all_recommendations(context)
        analytics = recommender_service.recommendation_analytics.calculate_portfolio_metrics(recommendations)
        
        # Get service metrics
        service_metrics = await chat_recommendation_service.get_service_metrics()
        
        # Calculate response time
        response_time = time.time() - start_time
        
        return {
            "success": True,
            "project_id": project_id,
            "analytics": analytics,
            "summary": {
                "total_potential_savings": analytics.get("total_cost_savings_usd", 0),
                "high_priority_count": analytics.get("high_impact_count", 0),
                "implementation_time": analytics.get("estimated_implementation_hours", 0),
                "security_score": analytics.get("average_security_score", 0),
                "response_time_ms": round(response_time * 1000, 2)
            },
            "service_health": service_metrics,
            "recommendations_by_type": {
                rec_type.value: len([r for r in recommendations if r.recommender_type == rec_type])
                for rec_type in RecommenderType
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting analytics for {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
@router.post("/dashboard")
async def get_dashboard_recommendations(request: RecommendationsRequest):
    """Get security recommendations for dashboard display (legacy endpoint)."""
    try:
        # Convert to new format and call comprehensive endpoint
        context_request = RecommenderContextRequest(
            project_id=request.project_id,
            max_results=10
        )
        
        if request.priority:
            context_request.priority_filter = [Priority(request.priority)]
        
        response = await get_comprehensive_recommendations(context_request)
        
        # Convert to legacy format
        legacy_recommendations = []
        for rec in response.recommendations:
            legacy_recommendations.append({
                "id": rec.recommendation_id,
                "title": rec.name,
                "description": rec.description,
                "priority": rec.priority.value,
                "category": rec.recommender_type.value.split('.')[-1],
                "impact": "High" if rec.security_impact_score > 0.7 else "Medium" if rec.security_impact_score > 0.4 else "Low",
                "effort": rec.implementation_effort.value,
                "remediation": rec.executable_commands[0] if rec.executable_commands else "See detailed steps"
            })
        
        # Calculate priority distribution
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for rec in response.recommendations:
            priority_counts[rec.priority.value] += 1
        
        return {
            "project_id": request.project_id,
            "total_recommendations": response.total_count,
            **priority_counts,
            "recommendations": legacy_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard recommendations: {e}")
        return {
            "project_id": request.project_id,
            "total_recommendations": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "recommendations": [],
            "error": str(e)
        }

@router.get("/priority/{priority}")
async def get_recommendations_by_priority_legacy(
    priority: str,
    project_id: str = "default-project"
):
    """Get recommendations filtered by priority (legacy endpoint)."""
    valid_priorities = ["critical", "high", "medium", "low"]
    
    if priority not in valid_priorities:
        raise HTTPException(status_code=400, detail=f"Invalid priority. Must be one of: {valid_priorities}")
    
    try:
        context_request = RecommenderContextRequest(
            project_id=project_id,
            priority_filter=[Priority(priority)],
            max_results=50
        )
        
        response = await get_comprehensive_recommendations(context_request)
        
        # Convert to legacy format
        legacy_recommendations = []
        for rec in response.recommendations:
            legacy_recommendations.append({
                "id": rec.recommendation_id,
                "title": rec.name,
                "category": rec.recommender_type.value.split('.')[-1],
                "remediation": rec.executable_commands[0] if rec.executable_commands else "See implementation steps"
            })
        
        return {
            "project_id": project_id,
            "priority": priority,
            "count": len(legacy_recommendations),
            "recommendations": legacy_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting {priority} recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))