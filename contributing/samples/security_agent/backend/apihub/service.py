import os
import logging
from google.auth import default
from google.cloud import apihub_v1
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from opentelemetry import trace

logger = logging.getLogger(__name__)

class APIHubService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.credentials = None
        self.project_id = None
        self.apihub_client = None
        
        # Initialize credentials and API Hub client
        try:
            self.credentials, self.project_id = default()
            self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', self.project_id)
            
            # Initialize API Hub client
            self.apihub_client = apihub_v1.ApiHubClient(credentials=self.credentials)
            logger.info(f"✅ API Hub client initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize API Hub client: {e}")
            self.apihub_client = None
    
    def discover_apis(self, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """
        Discover APIs from Google Cloud API Hub.
        
        Args:
            project_id: GCP project ID (uses default if not provided)
            location: Location for API Hub (default: global)
            
        Returns:
            Dict containing discovered APIs or error information
        """
        if not self.apihub_client:
            return {
                "success": False,
                "error": "API Hub client not initialized. Ensure API Hub API is enabled.",
                "apis": []
            }
        
        with self.tracer.start_as_current_span("APIHubService.discover_apis") as span:
            project_id = project_id or self.project_id
            span.set_attribute("project_id", project_id)
            span.set_attribute("location", location)
            
            try:
                parent = f"projects/{project_id}/locations/{location}"
                
                # List APIs from API Hub
                request = apihub_v1.ListApisRequest(parent=parent)
                response = self.apihub_client.list_apis(request=request)
                
                apis = []
                for api in response:
                    api_data = {
                        "name": api.name,
                        "display_name": api.display_name,
                        "description": api.description,
                        "create_time": api.create_time,
                        "update_time": api.update_time,
                        "labels": dict(api.labels) if api.labels else {},
                        "api_id": api.name.split('/')[-1] if api.name else None
                    }
                    apis.append(api_data)
                
                # If no APIs found, provide helpful information
                if not apis:
                    return {
                        "success": True,
                        "apis": [],
                        "message": "No APIs found in API Hub. APIs need to be registered in API Hub to appear here.",
                        "setup_help": {
                            "register_apis": "Use 'gcloud apihub apis create' to register your APIs",
                            "api_discovery": "API Hub discovers APIs from various sources including API Gateway",
                            "documentation": "https://cloud.google.com/api-hub/docs"
                        }
                    }
                
                span.set_attribute("apis_found", len(apis))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "apis": apis,
                    "total_count": len(apis),
                    "project_id": project_id,
                    "location": location
                }
                
            except Exception as e:
                error_msg = f"Failed to discover APIs: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "apis": [],
                    "help": "Ensure API Hub API is enabled and you have proper permissions"
                }
    
    def get_api_versions(self, api_name: str, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """
        Get versions for a specific API.
        
        Args:
            api_name: Name of the API
            project_id: GCP project ID
            location: Location for API Hub
            
        Returns:
            Dict containing API versions or error information
        """
        if not self.apihub_client:
            return {
                "success": False,
                "error": "API Hub client not initialized",
                "versions": []
            }
        
        with self.tracer.start_as_current_span("APIHubService.get_api_versions") as span:
            project_id = project_id or self.project_id
            span.set_attribute("project_id", project_id)
            span.set_attribute("api_name", api_name)
            
            try:
                parent = f"projects/{project_id}/locations/{location}/apis/{api_name}"
                
                # List versions for the API
                request = apihub_v1.ListVersionsRequest(parent=parent)
                response = self.apihub_client.list_versions(request=request)
                
                versions = []
                for version in response:
                    version_data = {
                        "name": version.name,
                        "display_name": version.display_name,
                        "description": version.description,
                        "state": version.state.name if version.state else "UNKNOWN",
                        "create_time": version.create_time,
                        "update_time": version.update_time,
                        "labels": dict(version.labels) if version.labels else {},
                        "version_id": version.name.split('/')[-1] if version.name else None
                    }
                    versions.append(version_data)
                
                span.set_attribute("versions_found", len(versions))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "versions": versions,
                    "api_name": api_name,
                    "total_count": len(versions)
                }
                
            except Exception as e:
                error_msg = f"Failed to get API versions for {api_name}: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "versions": []
                }
    
    def get_api_specs(self, version_name: str, project_id: str = None) -> Dict[str, Any]:
        """
        Get specifications for a specific API version.
        
        Args:
            version_name: Full name of the API version
            project_id: GCP project ID
            
        Returns:
            Dict containing API specifications or error information
        """
        if not self.apihub_client:
            return {
                "success": False,
                "error": "API Hub client not initialized",
                "specs": []
            }
        
        with self.tracer.start_as_current_span("APIHubService.get_api_specs") as span:
            span.set_attribute("version_name", version_name)
            
            try:
                # List specs for the version
                request = apihub_v1.ListSpecsRequest(parent=version_name)
                response = self.apihub_client.list_specs(request=request)
                
                specs = []
                for spec in response:
                    spec_data = {
                        "name": spec.name,
                        "display_name": spec.display_name,
                        "spec_type": spec.spec_type,
                        "source_uri": spec.source_uri,
                        "create_time": spec.create_time,
                        "update_time": spec.update_time,
                        "labels": dict(spec.labels) if spec.labels else {},
                        "spec_id": spec.name.split('/')[-1] if spec.name else None
                    }
                    specs.append(spec_data)
                
                span.set_attribute("specs_found", len(specs))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "specs": specs,
                    "version_name": version_name,
                    "total_count": len(specs)
                }
                
            except Exception as e:
                error_msg = f"Failed to get API specs for {version_name}: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "specs": []
                }
    
    def search_apis(self, query: str, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """
        Search for APIs in API Hub.
        
        Args:
            query: Search query
            project_id: GCP project ID
            location: Location for API Hub
            
        Returns:
            Dict containing search results or error information
        """
        if not self.apihub_client:
            return {
                "success": False,
                "error": "API Hub client not initialized",
                "results": []
            }
        
        with self.tracer.start_as_current_span("APIHubService.search_apis") as span:
            project_id = project_id or self.project_id
            span.set_attribute("project_id", project_id)
            span.set_attribute("query", query)
            
            try:
                # Get all APIs and filter by query (API Hub may not have full-text search)
                all_apis_result = self.discover_apis(project_id, location)
                
                if not all_apis_result.get("success"):
                    return all_apis_result
                
                all_apis = all_apis_result.get("apis", [])
                
                # Simple search filtering
                query_lower = query.lower()
                filtered_apis = []
                
                for api in all_apis:
                    if (query_lower in api.get("display_name", "").lower() or
                        query_lower in api.get("description", "").lower() or
                        query_lower in api.get("name", "").lower()):
                        filtered_apis.append(api)
                
                span.set_attribute("results_found", len(filtered_apis))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "results": filtered_apis,
                    "query": query,
                    "total_results": len(filtered_apis),
                    "total_apis_searched": len(all_apis)
                }
                
            except Exception as e:
                error_msg = f"Failed to search APIs: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "results": []
                }
    
    def get_api_analytics(self, project_id: str = None, location: str = "global", days: int = 30) -> Dict[str, Any]:
        """
        Get API analytics and usage statistics.
        
        Args:
            project_id: GCP project ID
            location: Location for API Hub
            days: Number of days to look back
            
        Returns:
            Dict containing API analytics or error information
        """
        if not self.apihub_client:
            return {
                "success": False,
                "error": "API Hub client not initialized",
                "analytics": {}
            }
        
        with self.tracer.start_as_current_span("APIHubService.get_api_analytics") as span:
            try:
                # Get API discovery data to create analytics
                apis_result = self.discover_apis(project_id, location)
                
                if not apis_result.get("success"):
                    return apis_result
                
                apis = apis_result.get("apis", [])
                
                # Create analytics from available data
                analytics = {
                    "total_apis": len(apis),
                    "api_breakdown": {
                        "by_labels": {},
                        "by_creation_time": {}
                    },
                    "recent_activity": [],
                    "top_apis": []
                }
                
                # Analyze APIs
                for api in apis:
                    # Breakdown by labels
                    for label_key, label_value in api.get("labels", {}).items():
                        if label_key not in analytics["api_breakdown"]["by_labels"]:
                            analytics["api_breakdown"]["by_labels"][label_key] = {}
                        if label_value not in analytics["api_breakdown"]["by_labels"][label_key]:
                            analytics["api_breakdown"]["by_labels"][label_key][label_value] = 0
                        analytics["api_breakdown"]["by_labels"][label_key][label_value] += 1
                    
                    # Add to top APIs (based on display name for now)
                    analytics["top_apis"].append({
                        "name": api.get("display_name", "Unknown"),
                        "id": api.get("api_id"),
                        "description": api.get("description", "")[:100] + "..." if len(api.get("description", "")) > 100 else api.get("description", "")
                    })
                
                # Limit top APIs to 10
                analytics["top_apis"] = analytics["top_apis"][:10]
                
                span.set_attribute("total_apis_analyzed", len(apis))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "analytics": analytics,
                    "project_id": project_id,
                    "time_range_days": days
                }
                
            except Exception as e:
                error_msg = f"Failed to get API analytics: {str(e)}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "analytics": {}
                }