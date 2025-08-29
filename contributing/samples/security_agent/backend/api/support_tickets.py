"""
Google Cloud Support Ticket API Endpoints
=========================================

RESTful API endpoints for Google Cloud Support ticket analysis and management.
Integrates with Google Cloud Support API to analyze customer-submitted tickets.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse

from ..models.support_ticket_models import (
    SupportTicket, TicketCreationRequest, TicketUpdateRequest, TicketAnalytics,
    TicketAutomationRule, TicketComment, TicketAssignment, TicketMetadata,
    TicketPriority, TicketStatus, TicketType, IntegrationPlatform
)
from ..services.support_ticket_manager import GoogleCloudSupportManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/support-tickets", tags=["Google Cloud Support Tickets"])

# Initialize Google Cloud Support manager
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
support_manager = GoogleCloudSupportManager(project_id=project_id, database_path=database_path)


@router.get("/analyze", response_model=Dict[str, Any])
async def analyze_support_cases() -> Dict[str, Any]:
    """
    Analyze all Google Cloud Support cases for the project.
    
    Fetches support cases from Google Cloud Support API, converts them to internal format,
    analyzes patterns and trends, and provides comprehensive insights.
    """
    logger.info("Starting Google Cloud Support cases analysis")
    
    try:
        analysis_result = await support_manager.analyze_support_cases()
        
        if "error" in analysis_result:
            raise HTTPException(
                status_code=500,
                detail=f"Support case analysis failed: {analysis_result['error']}"
            )
        
        logger.info(f"Support case analysis completed: {analysis_result.get('total_cases', 0)} cases analyzed")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Support case analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Support case analysis failed: {str(e)}"
        )


@router.get("/cases", response_model=List[Dict[str, Any]])
async def list_support_cases(page_size: int = Query(50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """
    List Google Cloud Support cases for the project.
    
    Args:
        page_size: Number of cases to return (1-200)
    """
    try:
        cases = await support_manager.fetch_support_cases(page_size=page_size)
        
        # Convert cases to serializable format
        case_list = []
        for case in cases:
            case_dict = {
                "name": case.name,
                "display_name": case.display_name,
                "description": case.description,
                "priority": case.priority,
                "state": case.state.name if case.state else "UNKNOWN",
                "create_time": case.create_time.isoformat() if case.create_time else None,
                "update_time": case.update_time.isoformat() if case.update_time else None,
                "creator": getattr(case.creator, 'email', 'unknown') if case.creator else "unknown",
                "classification": case.classification.display_name if case.classification else "General"
            }
            case_list.append(case_dict)
        
        return case_list
        
    except Exception as e:
        logger.error(f"Failed to list support cases: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list support cases: {str(e)}"
        )


@router.get("/cases/{case_name}")
async def get_support_case_details(case_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific Google Cloud Support case.
    
    Args:
        case_name: Full resource name of the support case (e.g., projects/PROJECT/cases/CASE_ID)
    """
    try:
        case = await support_manager.get_case_details(case_name)
        
        if not case:
            raise HTTPException(
                status_code=404,
                detail=f"Support case not found: {case_name}"
            )
        
        # Convert to detailed dictionary
        case_details = {
            "name": case.name,
            "display_name": case.display_name,
            "description": case.description,
            "priority": case.priority,
            "state": case.state.name if case.state else "UNKNOWN",
            "create_time": case.create_time.isoformat() if case.create_time else None,
            "update_time": case.update_time.isoformat() if case.update_time else None,
            "creator": {
                "email": getattr(case.creator, 'email', 'unknown') if case.creator else "unknown",
                "display_name": getattr(case.creator, 'display_name', 'unknown') if case.creator else "unknown"
            },
            "classification": {
                "display_name": case.classification.display_name if case.classification else "General",
                "id": case.classification.id if case.classification else "general"
            },
            "time_zone": case.time_zone,
            "subscriber_email_addresses": list(case.subscriber_email_addresses) if case.subscriber_email_addresses else [],
            "language_code": case.language_code
        }
        
        return case_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get support case details for {case_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve case details: {str(e)}"
        )


@router.get("/tickets", response_model=List[SupportTicket])
async def list_tickets(
    status: Optional[TicketStatus] = Query(None, description="Filter by ticket status"),
    priority: Optional[TicketPriority] = Query(None, description="Filter by priority"),
    ticket_type: Optional[TicketType] = Query(None, description="Filter by ticket type"),
    assignee: Optional[str] = Query(None, description="Filter by assignee"),
    limit: int = Query(50, ge=1, le=200, description="Number of tickets to return"),
    offset: int = Query(0, ge=0, description="Number of tickets to skip")
) -> List[SupportTicket]:
    """
    List support tickets with filtering options.
    
    Returns tickets that have been converted from Google Cloud Support cases
    or created directly in the system.
    """
    try:
        tickets = await support_manager.list_tickets(
            status=status,
            priority=priority,
            assignee=assignee,
            platform=IntegrationPlatform.CUSTOM_API,  # GCP Support cases
            limit=limit,
            offset=offset
        )
        
        return tickets
        
    except Exception as e:
        logger.error(f"Failed to list support tickets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tickets: {str(e)}"
        )


@router.get("/tickets/{ticket_id}", response_model=SupportTicket)
async def get_ticket(ticket_id: str) -> SupportTicket:
    """
    Get a specific support ticket by ID.
    
    Args:
        ticket_id: The ticket ID to retrieve
    """
    try:
        ticket = await support_manager.get_ticket(ticket_id)
        
        if not ticket:
            raise HTTPException(
                status_code=404,
                detail=f"Ticket not found: {ticket_id}"
            )
        
        return ticket
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ticket {ticket_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve ticket: {str(e)}"
        )


@router.post("/tickets", response_model=SupportTicket)
async def create_ticket(request: TicketCreationRequest) -> SupportTicket:
    """
    Create a new support ticket.
    
    This creates an internal ticket record. For Google Cloud Support integration,
    use the analyze endpoint to sync existing cases.
    """
    try:
        ticket = await support_manager.create_ticket(request)
        logger.info(f"Created support ticket: {ticket.ticket_id}")
        return ticket
        
    except Exception as e:
        logger.error(f"Failed to create support ticket: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create ticket: {str(e)}"
        )


@router.put("/tickets/{ticket_id}", response_model=SupportTicket)
async def update_ticket(ticket_id: str, request: TicketUpdateRequest) -> SupportTicket:
    """
    Update an existing support ticket.
    
    Args:
        ticket_id: The ticket ID to update
        request: Update request with new values
    """
    try:
        # Set the ticket ID in the request
        request.ticket_id = ticket_id
        
        updated_ticket = await support_manager.update_ticket(request)
        logger.info(f"Updated support ticket: {ticket_id}")
        return updated_ticket
        
    except Exception as e:
        logger.error(f"Failed to update ticket {ticket_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update ticket: {str(e)}"
        )


@router.get("/analytics", response_model=TicketAnalytics)
async def get_ticket_analytics(
    days_back: int = Query(30, ge=1, le=365, description="Number of days to analyze")
) -> TicketAnalytics:
    """
    Get comprehensive analytics for support tickets.
    
    Provides insights into ticket volume, performance metrics, trends,
    and team performance over the specified time period.
    
    Args:
        days_back: Number of days to include in analysis (1-365)
    """
    try:
        analytics = await support_manager.get_analytics(days_back=days_back)
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to generate ticket analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analytics: {str(e)}"
        )


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_support_dashboard() -> Dict[str, Any]:
    """
    Get real-time support ticket dashboard data.
    
    Returns key metrics, trends, and insights for support ticket monitoring.
    """
    try:
        # Get recent analytics
        analytics = await support_manager.get_analytics(days_back=7)
        
        # Get ticket counts by status
        open_tickets = await support_manager.list_tickets(
            status=TicketStatus.OPEN, limit=1000
        )
        in_progress_tickets = await support_manager.list_tickets(
            status=TicketStatus.IN_PROGRESS, limit=1000
        )
        resolved_tickets = await support_manager.list_tickets(
            status=TicketStatus.RESOLVED, limit=1000
        )
        
        # Calculate key metrics
        total_open = len(open_tickets)
        total_in_progress = len(in_progress_tickets)
        total_resolved_7_days = len([t for t in resolved_tickets 
                                    if t.resolved_at and (datetime.now() - t.resolved_at).days <= 7])
        
        # Priority breakdown
        critical_tickets = await support_manager.list_tickets(
            priority=TicketPriority.CRITICAL, limit=1000
        )
        high_tickets = await support_manager.list_tickets(
            priority=TicketPriority.HIGH, limit=1000
        )
        
        dashboard_data = {
            "summary": {
                "total_open": total_open,
                "total_in_progress": total_in_progress,
                "resolved_last_7_days": total_resolved_7_days,
                "critical_priority": len([t for t in critical_tickets if t.status != TicketStatus.CLOSED]),
                "high_priority": len([t for t in high_tickets if t.status != TicketStatus.CLOSED])
            },
            "analytics": analytics.dict(),
            "trends": {
                "avg_response_time_hours": analytics.avg_response_time_hours,
                "avg_resolution_time_hours": analytics.avg_resolution_time_hours,
                "sla_compliance_percentage": analytics.sla_compliance_percentage,
                "escalation_rate": analytics.escalation_rate
            },
            "last_updated": datetime.now().isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get support dashboard data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve dashboard data: {str(e)}"
        )


@router.get("/patterns", response_model=Dict[str, Any])
async def get_ticket_patterns(
    days_back: int = Query(30, ge=1, le=365, description="Days to analyze"),
    min_occurrences: int = Query(2, ge=1, le=50, description="Minimum pattern occurrences")
) -> Dict[str, Any]:
    """
    Analyze patterns in Google Cloud Support tickets.
    
    Identifies common issues, trends, and patterns across support cases.
    
    Args:
        days_back: Number of days to analyze
        min_occurrences: Minimum occurrences for a pattern to be included
    """
    try:
        # Get tickets from the specified period
        tickets = await support_manager.list_tickets(limit=10000)  # Large limit for analysis
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_tickets = [t for t in tickets if t.created_at >= cutoff_date]
        
        # Analyze patterns
        type_distribution = {}
        priority_distribution = {}
        tag_frequency = {}
        security_domain_frequency = {}
        
        for ticket in recent_tickets:
            # Ticket type patterns
            ticket_type = ticket.ticket_type.value
            type_distribution[ticket_type] = type_distribution.get(ticket_type, 0) + 1
            
            # Priority patterns
            priority = ticket.priority.value
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            # Tag patterns
            for tag in ticket.tags:
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
            
            # Security domain patterns
            for domain in ticket.metadata.security_domains:
                security_domain_frequency[domain] = security_domain_frequency.get(domain, 0) + 1
        
        # Filter by minimum occurrences
        filtered_types = {k: v for k, v in type_distribution.items() if v >= min_occurrences}
        filtered_tags = {k: v for k, v in tag_frequency.items() if v >= min_occurrences}
        filtered_domains = {k: v for k, v in security_domain_frequency.items() if v >= min_occurrences}
        
        # Find most common patterns
        most_common_type = max(filtered_types.items(), key=lambda x: x[1]) if filtered_types else None
        most_common_tag = max(filtered_tags.items(), key=lambda x: x[1]) if filtered_tags else None
        most_common_domain = max(filtered_domains.items(), key=lambda x: x[1]) if filtered_domains else None
        
        return {
            "analysis_period": f"{days_back} days",
            "total_tickets_analyzed": len(recent_tickets),
            "patterns_found": {
                "ticket_types": len(filtered_types),
                "common_tags": len(filtered_tags),
                "security_domains": len(filtered_domains)
            },
            "distributions": {
                "ticket_types": filtered_types,
                "priorities": priority_distribution,
                "common_tags": filtered_tags,
                "security_domains": filtered_domains
            },
            "insights": {
                "most_common_issue_type": {
                    "type": most_common_type[0],
                    "occurrences": most_common_type[1],
                    "percentage": round((most_common_type[1] / len(recent_tickets)) * 100, 1)
                } if most_common_type else None,
                "most_common_tag": {
                    "tag": most_common_tag[0],
                    "occurrences": most_common_tag[1]
                } if most_common_tag else None,
                "most_affected_security_domain": {
                    "domain": most_common_domain[0],
                    "occurrences": most_common_domain[1]
                } if most_common_domain else None
            },
            "recommendations": [
                "Monitor high-frequency issue types for process improvements",
                "Consider creating knowledge base articles for common patterns",
                "Implement automated responses for frequent, low-complexity issues",
                "Review security domain patterns for potential system vulnerabilities"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze ticket patterns: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze patterns: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for Google Cloud Support integration.
    """
    try:
        # Check database connectivity
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='support_tickets'")
        table_exists = cursor.fetchone()[0] > 0
        conn.close()
        
        # Check Google Cloud Support API availability
        api_available = hasattr(support_manager, 'support_client') and support_manager.support_client is not None
        
        return {
            "status": "healthy",
            "service": "Google Cloud Support Integration",
            "version": "1.0.0",
            "database_connected": True,
            "support_tickets_table": table_exists,
            "google_cloud_support_api": api_available,
            "project_id": project_id,
            "supported_priorities": [priority.value for priority in TicketPriority],
            "supported_statuses": [status.value for status in TicketStatus],
            "supported_types": [ticket_type.value for ticket_type in TicketType],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Export router for main application
__all__ = ["router"]