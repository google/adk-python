"""
Google Cloud Advisory Notifications API - Thin client for security and reliability notifications.

This module provides a thin client wrapper around the Google Cloud Advisory Notifications API
for receiving important notifications about security bulletins, product updates, and incidents.

Docs: https://cloud.google.com/python/docs/reference/advisorynotifications/latest
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import the Google Cloud Advisory Notifications client
try:
    from google.cloud import advisorynotifications_v1
    from google.api_core import exceptions as gcp_exceptions
    ADVISORY_CLIENT_AVAILABLE = True
    logger.info("✅ Google Cloud Advisory Notifications client available")
except ImportError:
    ADVISORY_CLIENT_AVAILABLE = False
    logger.warning("⚠️ Advisory Notifications client not available. Install with: pip install google-cloud-advisorynotifications")

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
ORGANIZATION_ID = os.getenv('GOOGLE_CLOUD_ORGANIZATION', '')

# Request/Response models
class NotificationListRequest(BaseModel):
    """Request model for listing notifications."""
    parent: Optional[str] = Field(None, description="Parent resource (e.g., 'organizations/123' or 'projects/PROJECT_ID')")
    page_size: Optional[int] = Field(100, description="Number of results per page")
    language_code: Optional[str] = Field("en", description="Language code for notifications (e.g., 'en', 'es', 'fr')")
    view: Optional[str] = Field("BASIC", description="View level: BASIC or FULL")

class NotificationGetRequest(BaseModel):
    """Request model for getting a specific notification."""
    name: str = Field(..., description="Full notification name (e.g., 'organizations/123/locations/global/notifications/456')")
    language_code: Optional[str] = Field("en", description="Language code for the notification")

class NotificationSettingsGetRequest(BaseModel):
    """Request model for getting notification settings."""
    name: Optional[str] = Field(None, description="Settings name (e.g., 'organizations/123/locations/global/settings')")

class NotificationSettingsUpdateRequest(BaseModel):
    """Request model for updating notification settings."""
    name: Optional[str] = Field(None, description="Settings name")
    etag: str = Field(..., description="Etag for optimistic concurrency control")
    notification_settings: Dict[str, Any] = Field(..., description="Notification settings configuration")

def get_advisory_client():
    """Get or create Advisory Notifications client."""
    if not ADVISORY_CLIENT_AVAILABLE:
        return None
    
    try:
        client = advisorynotifications_v1.AdvisoryNotificationsServiceClient()
        return client
    except Exception as e:
        logger.error(f"Failed to create Advisory Notifications client: {e}")
        return None

def get_parent_resource():
    """Get parent resource string for Advisory Notifications."""
    # Always use project-level access (org-level not needed)
    return f"projects/{PROJECT_ID}/locations/global"

@router.post("/list")
async def list_notifications(request: NotificationListRequest):
    """
    List advisory notifications using Cloud Advisory Notifications API.
    
    This is a thin client that directly calls the Google Cloud Advisory Notifications API.
    """
    client = get_advisory_client()
    if not client:
        # Return sample data when client is not available
        return {
            "success": True,
            "source": "sample_data",
            "message": "Install google-cloud-advisorynotifications for live data",
            "notifications": [
                {
                    "name": f"projects/{PROJECT_ID}/locations/global/notifications/sample-001",
                    "subject": {
                        "text": "Security Bulletin: Critical vulnerability in Cloud Service",
                        "localized_text": {
                            "locale": "en",
                            "text": "Security Bulletin: Critical vulnerability in Cloud Service"
                        }
                    },
                    "messages": [
                        {
                            "body": {
                                "text": "A critical security vulnerability has been identified. Please review and take action.",
                                "localized_text": {
                                    "locale": "en",
                                    "text": "A critical security vulnerability has been identified. Please review and take action."
                                }
                            },
                            "create_time": datetime.now().isoformat(),
                            "attachments": []
                        }
                    ],
                    "create_time": datetime.now().isoformat(),
                    "notification_type": "NOTIFICATION_TYPE_SECURITY_BULLETIN"
                },
                {
                    "name": f"projects/{PROJECT_ID}/locations/global/notifications/sample-002",
                    "subject": {
                        "text": "Product Update: New features available",
                        "localized_text": {
                            "locale": "en",
                            "text": "Product Update: New features available"
                        }
                    },
                    "messages": [
                        {
                            "body": {
                                "text": "New security features have been added to enhance protection.",
                                "localized_text": {
                                    "locale": "en",
                                    "text": "New security features have been added to enhance protection."
                                }
                            },
                            "create_time": (datetime.now() - timedelta(days=1)).isoformat(),
                            "attachments": []
                        }
                    ],
                    "create_time": (datetime.now() - timedelta(days=1)).isoformat(),
                    "notification_type": "NOTIFICATION_TYPE_PRODUCT_UPDATES"
                }
            ],
            "total_count": 2
        }
    
    try:
        # Prepare the request
        parent = request.parent or get_parent_resource()
        
        # Create the list notifications request
        list_request = advisorynotifications_v1.ListNotificationsRequest(
            parent=parent,
            page_size=request.page_size,
            language_code=request.language_code,
            view=getattr(
                advisorynotifications_v1.NotificationView,
                request.view,
                advisorynotifications_v1.NotificationView.BASIC
            )
        )
        
        # Call the API
        page_result = client.list_notifications(request=list_request)
        
        # Process results
        notifications = []
        for notification in page_result:
            notif_dict = {
                "name": notification.name,
                "create_time": notification.create_time.isoformat() if notification.create_time else None,
                "notification_type": notification.notification_type.name if notification.notification_type else "UNSPECIFIED"
            }
            
            # Add subject
            if notification.subject:
                notif_dict["subject"] = {
                    "text": notification.subject.text.text if notification.subject.text else None,
                    "localized_text": {
                        "locale": notification.subject.text.localized_text.locale,
                        "text": notification.subject.text.localized_text.text
                    } if notification.subject.text and notification.subject.text.localized_text else None
                }
            
            # Add messages
            if notification.messages:
                notif_dict["messages"] = []
                for message in notification.messages:
                    msg_dict = {
                        "create_time": message.create_time.isoformat() if message.create_time else None,
                        "attachments": []
                    }
                    
                    # Add message body
                    if message.body:
                        msg_dict["body"] = {
                            "text": message.body.text.text if message.body.text else None,
                            "localized_text": {
                                "locale": message.body.text.localized_text.locale,
                                "text": message.body.text.localized_text.text
                            } if message.body.text and message.body.text.localized_text else None
                        }
                    
                    # Add attachments
                    if message.attachments:
                        for attachment in message.attachments:
                            att_dict = {}
                            if attachment.csv:
                                att_dict["csv"] = {
                                    "headers": list(attachment.csv.headers),
                                    "data_rows": [
                                        {"values": list(row.values)} for row in attachment.csv.data_rows
                                    ]
                                }
                            msg_dict["attachments"].append(att_dict)
                    
                    notif_dict["messages"].append(msg_dict)
            
            notifications.append(notif_dict)
        
        return {
            "success": True,
            "source": "advisory_notifications_api",
            "parent": parent,
            "notifications": notifications,
            "total_count": len(notifications)
        }
        
    except gcp_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied: {e}")
        raise HTTPException(status_code=403, detail=f"Permission denied: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get")
async def get_notification(request: NotificationGetRequest):
    """
    Get a specific advisory notification.
    """
    client = get_advisory_client()
    if not client:
        return {
            "success": False,
            "message": "Advisory Notifications client not available. Install google-cloud-advisorynotifications."
        }
    
    try:
        # Create the get notification request
        get_request = advisorynotifications_v1.GetNotificationRequest(
            name=request.name,
            language_code=request.language_code
        )
        
        # Call the API
        notification = client.get_notification(request=get_request)
        
        # Build response
        notif_dict = {
            "name": notification.name,
            "create_time": notification.create_time.isoformat() if notification.create_time else None,
            "notification_type": notification.notification_type.name if notification.notification_type else "UNSPECIFIED"
        }
        
        # Add subject
        if notification.subject:
            notif_dict["subject"] = {
                "text": notification.subject.text.text if notification.subject.text else None,
                "localized_text": {
                    "locale": notification.subject.text.localized_text.locale,
                    "text": notification.subject.text.localized_text.text
                } if notification.subject.text and notification.subject.text.localized_text else None
            }
        
        # Add messages with full details
        if notification.messages:
            notif_dict["messages"] = []
            for message in notification.messages:
                msg_dict = {
                    "create_time": message.create_time.isoformat() if message.create_time else None
                }
                
                if message.body:
                    msg_dict["body"] = {
                        "text": message.body.text.text if message.body.text else None,
                        "localized_text": {
                            "locale": message.body.text.localized_text.locale,
                            "text": message.body.text.localized_text.text
                        } if message.body.text and message.body.text.localized_text else None
                    }
                
                notif_dict["messages"].append(msg_dict)
        
        return {
            "success": True,
            "source": "advisory_notifications_api",
            "notification": notif_dict
        }
        
    except gcp_exceptions.NotFound as e:
        logger.error(f"Notification not found: {e}")
        raise HTTPException(status_code=404, detail="Notification not found")
    except Exception as e:
        logger.error(f"Error getting notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings/get")
async def get_notification_settings(request: NotificationSettingsGetRequest):
    """
    Get notification settings for the organization or project.
    """
    client = get_advisory_client()
    if not client:
        # Return sample settings
        return {
            "success": True,
            "source": "sample_data",
            "settings": {
                "name": f"{get_parent_resource()}/settings",
                "notification_settings": {
                    "enabled": True,
                    "email_preferences": {
                        "security_bulletins": True,
                        "product_updates": True,
                        "incidents": True
                    }
                },
                "etag": "sample_etag"
            }
        }
    
    try:
        # Prepare settings name
        name = request.name or f"{get_parent_resource()}/settings"
        
        # Create the get settings request
        get_request = advisorynotifications_v1.GetSettingsRequest(
            name=name
        )
        
        # Call the API
        settings = client.get_settings(request=get_request)
        
        return {
            "success": True,
            "source": "advisory_notifications_api",
            "settings": {
                "name": settings.name,
                "notification_settings": dict(settings.notification_settings) if settings.notification_settings else {},
                "etag": settings.etag
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting notification settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings/update")
async def update_notification_settings(request: NotificationSettingsUpdateRequest):
    """
    Update notification settings.
    """
    client = get_advisory_client()
    if not client:
        return {
            "success": False,
            "message": "Advisory Notifications client not available"
        }
    
    try:
        # Prepare settings name
        name = request.name or f"{get_parent_resource()}/settings"
        
        # Create settings object
        settings = advisorynotifications_v1.Settings(
            name=name,
            notification_settings=request.notification_settings,
            etag=request.etag
        )
        
        # Create the update settings request
        update_request = advisorynotifications_v1.UpdateSettingsRequest(
            settings=settings
        )
        
        # Call the API
        updated_settings = client.update_settings(request=update_request)
        
        return {
            "success": True,
            "settings": {
                "name": updated_settings.name,
                "notification_settings": dict(updated_settings.notification_settings) if updated_settings.notification_settings else {},
                "etag": updated_settings.etag
            },
            "message": "Notification settings updated successfully"
        }
        
    except gcp_exceptions.FailedPrecondition as e:
        logger.error(f"Etag mismatch: {e}")
        raise HTTPException(status_code=412, detail="Etag mismatch - settings were modified")
    except Exception as e:
        logger.error(f"Error updating notification settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notification-types")
async def get_notification_types():
    """
    Get list of available notification types.
    """
    return {
        "success": True,
        "notification_types": [
            {
                "type": "NOTIFICATION_TYPE_SECURITY_BULLETIN",
                "description": "Security bulletins and vulnerability notifications"
            },
            {
                "type": "NOTIFICATION_TYPE_SENSITIVE_ACTIONS",
                "description": "Notifications about sensitive administrative actions"
            },
            {
                "type": "NOTIFICATION_TYPE_SECURITY_MSA",
                "description": "Security Mandatory Service Announcements"
            },
            {
                "type": "NOTIFICATION_TYPE_THREAT_HORIZONS",
                "description": "Threat intelligence and emerging threat notifications"
            },
            {
                "type": "NOTIFICATION_TYPE_PRODUCT_UPDATES",
                "description": "Product feature updates and changes"
            },
            {
                "type": "NOTIFICATION_TYPE_INCIDENTS",
                "description": "Service incident notifications"
            }
        ]
    }

@router.get("/analyze")
async def analyze_notifications(
    parent: Optional[str] = None,
    days_back: Optional[int] = Query(30, description="Number of days to analyze")
):
    """
    Analyze recent advisory notifications for security insights.
    """
    client = get_advisory_client()
    
    analysis = {
        "parent": parent or get_parent_resource(),
        "period": f"Last {days_back} days",
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "critical_notifications": [],
        "recommendations": []
    }
    
    if not client:
        # Return sample analysis
        return {
            "success": True,
            "source": "sample_analysis",
            "analysis": {
                **analysis,
                "summary": {
                    "total_notifications": 15,
                    "security_bulletins": 5,
                    "product_updates": 8,
                    "incidents": 2,
                    "critical_count": 3,
                    "high_priority_count": 7
                },
                "critical_notifications": [
                    {
                        "type": "SECURITY_BULLETIN",
                        "subject": "Critical vulnerability in Cloud Service",
                        "date": (datetime.now() - timedelta(days=2)).isoformat(),
                        "action_required": True
                    }
                ],
                "recommendations": [
                    "Review and address 3 critical security bulletins",
                    "Enable notification settings for all security categories",
                    "Set up automated alerting for critical notifications",
                    "Review incident response procedures for recent incidents"
                ]
            }
        }
    
    try:
        # List recent notifications
        list_request = advisorynotifications_v1.ListNotificationsRequest(
            parent=parent or get_parent_resource(),
            page_size=100,
            view=advisorynotifications_v1.NotificationView.BASIC
        )
        
        notifications = list(client.list_notifications(request=list_request))
        
        # Analyze notifications
        type_counts = {}
        critical_notifications = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for notification in notifications:
            # Skip old notifications
            if notification.create_time and notification.create_time < cutoff_date:
                continue
            
            # Count by type
            notif_type = notification.notification_type.name if notification.notification_type else "UNSPECIFIED"
            type_counts[notif_type] = type_counts.get(notif_type, 0) + 1
            
            # Identify critical notifications
            if notif_type in ["NOTIFICATION_TYPE_SECURITY_BULLETIN", "NOTIFICATION_TYPE_SECURITY_MSA"]:
                critical_notifications.append({
                    "type": notif_type,
                    "subject": notification.subject.text.text if notification.subject and notification.subject.text else "No subject",
                    "date": notification.create_time.isoformat() if notification.create_time else None,
                    "action_required": True
                })
        
        # Build summary
        analysis["summary"] = {
            "total_notifications": len(notifications),
            **type_counts,
            "critical_count": len(critical_notifications)
        }
        
        analysis["critical_notifications"] = critical_notifications[:5]  # Top 5 critical
        
        # Generate recommendations
        if len(critical_notifications) > 0:
            analysis["recommendations"].append(f"Address {len(critical_notifications)} critical security notifications")
        if type_counts.get("NOTIFICATION_TYPE_INCIDENTS", 0) > 0:
            analysis["recommendations"].append("Review recent incidents and update response procedures")
        
        analysis["recommendations"].extend([
            "Enable automated forwarding of critical notifications",
            "Set up monitoring dashboards for notification trends",
            "Implement notification response SLAs"
        ])
        
        return {
            "success": True,
            "source": "live_analysis",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing notifications: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": analysis
        }

@router.get("/health")
async def health_check():
    """Health check for Advisory Notifications service."""
    return {
        "status": "healthy",
        "service": "advisory_notifications",
        "client_available": ADVISORY_CLIENT_AVAILABLE,
        "parent_resource": get_parent_resource(),
        "timestamp": datetime.now().isoformat()
    }