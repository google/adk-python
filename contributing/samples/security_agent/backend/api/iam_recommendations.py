"""
API endpoints for IAM Role Recommendations
Part of Advanced IAM Features
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json
import csv
import io
from datetime import datetime
import asyncio

from services.role_recommendation_engine import (
    RoleRecommendationEngine, 
    RoleRecommendation
)

logger = logging.getLogger(__name__)
router = APIRouter()


class RecommendationRequest(BaseModel):
    """Request model for role recommendations"""
    principal_email: str = Field(..., description="Email of the principal to analyze")
    analysis_period_days: int = Field(default=30, description="Number of days to analyze")
    force_refresh: bool = Field(default=False, description="Force refresh from audit logs")


class RecommendationResponse(BaseModel):
    """Response model for role recommendations"""
    principal: str
    principal_type: str
    current_roles: List[str]
    recommended_roles: List[str]
    confidence_score: float
    risk_reduction: str
    unused_permissions_count: int
    cost_savings: Optional[float]
    compliance_impact: List[str]
    recommendation_reason: str
    analyzed_at: str


class BulkAnalysisRequest(BaseModel):
    """Request model for bulk analysis"""
    project_id: str
    principal_filter: Optional[str] = Field(None, description="Filter principals by pattern")
    max_principals: int = Field(default=100, description="Maximum number of principals to analyze")


@router.post("/api/v1/iam/recommendations/analyze")
async def analyze_principal(request: RecommendationRequest) -> RecommendationResponse:
    """
    Analyze a single principal and generate role recommendations
    """
    logger.info(f"Analyzing IAM usage for principal: {request.principal_email}")
    
    try:
        # Initialize engine
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        engine = RoleRecommendationEngine(project_id)
        
        # Analyze usage patterns
        usage_patterns = await engine.analyze_principal_usage(
            request.principal_email,
            request.analysis_period_days
        )
        
        if not usage_patterns:
            logger.warning(f"No usage patterns found for {request.principal_email}")
            raise HTTPException(
                status_code=404,
                detail=f"No usage data found for principal: {request.principal_email}"
            )
        
        # Get current roles (mock for now)
        current_roles = ["roles/editor"]  # Would fetch from IAM API
        
        # Generate recommendations
        recommendation = engine.generate_role_recommendations(
            request.principal_email,
            usage_patterns,
            current_roles
        )
        
        # Convert to response
        response = RecommendationResponse(
            principal=recommendation.principal,
            principal_type=recommendation.principal_type,
            current_roles=recommendation.current_roles,
            recommended_roles=recommendation.recommended_roles,
            confidence_score=recommendation.confidence_score,
            risk_reduction=recommendation.risk_reduction,
            unused_permissions_count=len(recommendation.unused_permissions),
            cost_savings=recommendation.cost_impact,
            compliance_impact=recommendation.compliance_impact,
            recommendation_reason=recommendation.recommendation_reason,
            analyzed_at=recommendation.analyzed_at.isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing principal {request.principal_email}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze principal: {str(e)}"
        )


@router.get("/api/v1/iam/recommendations")
async def list_recommendations(
    limit: int = Query(default=50, le=100),
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    risk_level: Optional[str] = Query(default=None)
) -> List[RecommendationResponse]:
    """
    List all cached role recommendations
    """
    logger.info("Fetching cached role recommendations")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        engine = RoleRecommendationEngine(project_id)
        
        # Get all recommendations
        all_recommendations = engine.get_all_recommendations()
        
        # Filter by confidence
        filtered = [r for r in all_recommendations if r.confidence_score >= min_confidence]
        
        # Filter by risk level if specified
        if risk_level:
            filtered = [r for r in filtered if r.risk_reduction == risk_level]
        
        # Limit results
        filtered = filtered[:limit]
        
        # Convert to responses
        responses = []
        for rec in filtered:
            responses.append(RecommendationResponse(
                principal=rec.principal,
                principal_type=rec.principal_type,
                current_roles=rec.current_roles,
                recommended_roles=rec.recommended_roles,
                confidence_score=rec.confidence_score,
                risk_reduction=rec.risk_reduction,
                unused_permissions_count=len(rec.unused_permissions),
                cost_savings=rec.cost_impact,
                compliance_impact=rec.compliance_impact,
                recommendation_reason=rec.recommendation_reason,
                analyzed_at=rec.analyzed_at.isoformat()
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch recommendations: {str(e)}"
        )


@router.post("/api/v1/iam/recommendations/bulk-analyze")
async def bulk_analyze(
    request: BulkAnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Start bulk analysis of multiple principals
    """
    logger.info(f"Starting bulk analysis for project: {request.project_id}")
    
    try:
        # This would typically query all principals and start analysis
        # For now, return a job status
        job_id = f"bulk-analysis-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Add background task for analysis
        background_tasks.add_task(
            _run_bulk_analysis,
            request.project_id,
            request.principal_filter,
            request.max_principals,
            job_id
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "project_id": request.project_id,
            "max_principals": request.max_principals,
            "message": "Bulk analysis started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting bulk analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start bulk analysis: {str(e)}"
        )


@router.get("/api/v1/iam/recommendations/{principal_email}")
async def get_recommendation(principal_email: str) -> RecommendationResponse:
    """
    Get recommendation for a specific principal
    """
    logger.info(f"Fetching recommendation for: {principal_email}")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        engine = RoleRecommendationEngine(project_id)
        
        # Get all recommendations and find the specific one
        all_recommendations = engine.get_all_recommendations()
        recommendation = next((r for r in all_recommendations if r.principal == principal_email), None)
        
        if not recommendation:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendation found for principal: {principal_email}"
            )
        
        # Convert to response
        response = RecommendationResponse(
            principal=recommendation.principal,
            principal_type=recommendation.principal_type,
            current_roles=recommendation.current_roles,
            recommended_roles=recommendation.recommended_roles,
            confidence_score=recommendation.confidence_score,
            risk_reduction=recommendation.risk_reduction,
            unused_permissions_count=len(recommendation.unused_permissions),
            cost_savings=recommendation.cost_impact,
            compliance_impact=recommendation.compliance_impact,
            recommendation_reason=recommendation.recommendation_reason,
            analyzed_at=recommendation.analyzed_at.isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching recommendation for {principal_email}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch recommendation: {str(e)}"
        )


@router.get("/api/v1/iam/recommendations/export/csv")
async def export_recommendations_csv(
    min_confidence: float = Query(default=0.5)
) -> StreamingResponse:
    """
    Export recommendations as CSV
    """
    logger.info("Exporting recommendations to CSV")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        engine = RoleRecommendationEngine(project_id)
        
        # Get recommendations
        recommendations = engine.get_all_recommendations()
        filtered = [r for r in recommendations if r.confidence_score >= min_confidence]
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Principal", "Type", "Current Roles", "Recommended Roles",
            "Confidence", "Risk Reduction", "Unused Permissions",
            "Cost Impact", "Compliance Impact", "Reason"
        ])
        
        # Write data
        for rec in filtered:
            writer.writerow([
                rec.principal,
                rec.principal_type,
                ", ".join(rec.current_roles),
                ", ".join(rec.recommended_roles),
                f"{rec.confidence_score:.2f}",
                rec.risk_reduction,
                len(rec.unused_permissions),
                f"${rec.cost_impact:.2f}" if rec.cost_impact else "N/A",
                ", ".join(rec.compliance_impact),
                rec.recommendation_reason
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=iam_recommendations_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export recommendations: {str(e)}"
        )


@router.get("/api/v1/iam/recommendations/stats/summary")
async def get_recommendations_summary() -> Dict[str, Any]:
    """
    Get summary statistics for all recommendations
    """
    logger.info("Generating recommendations summary")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        engine = RoleRecommendationEngine(project_id)
        
        # Get all recommendations
        recommendations = engine.get_all_recommendations()
        
        if not recommendations:
            return {
                "total_principals_analyzed": 0,
                "average_confidence": 0,
                "risk_distribution": {},
                "potential_cost_savings": 0,
                "compliance_improvements": []
            }
        
        # Calculate statistics
        risk_distribution = {}
        total_cost_savings = 0
        all_compliance_impacts = set()
        
        for rec in recommendations:
            # Risk distribution
            risk_distribution[rec.risk_reduction] = risk_distribution.get(rec.risk_reduction, 0) + 1
            
            # Cost savings
            if rec.cost_impact:
                total_cost_savings += rec.cost_impact
            
            # Compliance impacts
            all_compliance_impacts.update(rec.compliance_impact)
        
        # Average confidence
        avg_confidence = sum(r.confidence_score for r in recommendations) / len(recommendations)
        
        return {
            "total_principals_analyzed": len(recommendations),
            "average_confidence": round(avg_confidence, 2),
            "risk_distribution": risk_distribution,
            "potential_cost_savings": round(total_cost_savings, 2),
            "compliance_improvements": list(all_compliance_impacts),
            "high_confidence_recommendations": len([r for r in recommendations if r.confidence_score > 0.8]),
            "custom_roles_needed": len([r for r in recommendations if r.custom_role_needed]),
            "service_accounts_analyzed": len([r for r in recommendations if r.principal_type == "serviceAccount"])
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )


async def _run_bulk_analysis(project_id: str, principal_filter: Optional[str],
                            max_principals: int, job_id: str):
    """
    Background task for bulk analysis
    """
    logger.info(f"Running bulk analysis job: {job_id}")
    
    try:
        engine = RoleRecommendationEngine(project_id)
        
        # In production, this would:
        # 1. Query all principals from IAM
        # 2. Filter based on principal_filter
        # 3. Analyze each principal
        # 4. Store results
        
        # For demo, analyze a few mock principals
        mock_principals = [
            "app-sa@project.iam.gserviceaccount.com",
            "backend-sa@project.iam.gserviceaccount.com",
            "user@example.com"
        ]
        
        for principal in mock_principals[:max_principals]:
            if principal_filter and principal_filter not in principal:
                continue
                
            try:
                # Analyze principal
                usage_patterns = await engine.analyze_principal_usage(principal, 30)
                if usage_patterns:
                    recommendation = engine.generate_role_recommendations(
                        principal,
                        usage_patterns,
                        ["roles/editor"]  # Mock current roles
                    )
                    logger.info(f"Analyzed {principal}: confidence={recommendation.confidence_score}")
            except Exception as e:
                logger.error(f"Error analyzing {principal}: {e}")
                continue
        
        logger.info(f"Bulk analysis job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Bulk analysis job {job_id} failed: {e}")