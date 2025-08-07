"""
Google API Tools

Tools for making generic Google Cloud API calls and API Hub integration.
"""

from typing import Dict, Any, Optional, List
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.apihub_tool.apihub_toolset import APIHubToolset
from google.adk.auth import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.tools.openapi_tool.auth.auth_helpers import dict_to_auth_scheme
import requests
import json


def call_google_api(
    service: str,
    version: str,
    resource_path: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Constructs and executes a REST call to a specified Google Cloud API endpoint
    by calling the backend's generic API endpoint.

    Args:
        service: The Google Cloud service name (e.g., 'storage', 'cloudresourcemanager').
        version: The API version (e.g., 'v1', 'v3').
        resource_path: The resource path for the API call (e.g., 'b/my-bucket/o').
        method: The HTTP method to use (GET, POST, PUT, DELETE).
        body: The JSON body for POST or PUT requests.
        tool_context: The context for the tool execution.

    Returns:
        The JSON response from the API as a string, or an error message.
    """
    try:
        request_data = {
            "service": service,
            "version": version,
            "resource_path": resource_path,
            "method": method,
            "body": body
        }
        
        response = requests.post("http://localhost:8000/api/v1/gcp/call-api", json=request_data)
        response.raise_for_status()
        
        response_data = response.json()
        if response_data.get("success"):
            return json.dumps(response_data.get("response", {}), indent=2)
        else:
            return f"Error from backend API: {response_data.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error calling backend for Google API: {e}"


def create_apihub_toolset(
    toolset_name: str,
    apihub_resource_name: str,
    auth_type: str = "service_account",
    auth_config: Optional[Dict[str, Any]] = None,
    tool_filter: Optional[List[str]] = None,
    description: str = ""
) -> APIHubToolset:
    """Create an API Hub toolset for dynamic tool access.
    
    Args:
        toolset_name: Name for the toolset.
        apihub_resource_name: API Hub resource name.
        auth_type: Authentication type ('service_account', 'access_token', 'oauth2').
        auth_config: Authentication configuration.
        tool_filter: List of tool names to include.
        description: Toolset description.
        
    Returns:
        Configured APIHubToolset instance.
        
    Raises:
        Exception: If toolset creation fails.
    """
    try:
        toolset_kwargs = {
            "name": toolset_name,
            "description": description,
            "apihub_resource_name": apihub_resource_name,
            "tool_filter": tool_filter
        }
        
        if auth_type == "service_account":
            if auth_config and "service_account_json" in auth_config:
                toolset_kwargs["service_account_json"] = auth_config["service_account_json"]
            # Otherwise use default credentials
            
        elif auth_type == "access_token":
            if auth_config and "access_token" in auth_config:
                toolset_kwargs["access_token"] = auth_config["access_token"]
            else:
                raise Exception("Access token not provided in auth config")
                
        elif auth_type == "oauth2":
            if auth_config and "oauth2_config" in auth_config:
                oauth2_config = auth_config["oauth2_config"]
                oauth_scheme = dict_to_auth_scheme(oauth2_config)
                
                auth_credential = AuthCredential(
                    auth_type=AuthCredentialTypes.OAUTH2,
                    oauth2=OAuth2Auth(
                        client_id=oauth2_config.get("client_id"),
                        client_secret=oauth2_config.get("client_secret")
                    )
                )
                
                toolset_kwargs["auth_scheme"] = oauth_scheme
                toolset_kwargs["auth_credential"] = auth_credential
            else:
                raise Exception("OAuth2 configuration not provided")
        
        return APIHubToolset(**toolset_kwargs)
        
    except Exception as e:
        raise Exception(f"Failed to create API Hub toolset: {str(e)}")


def get_available_toolsets() -> List[Dict[str, Any]]:
    """Get list of available API Hub toolsets from configuration.
    
    Returns:
        List of toolset configurations.
    """
    # This would typically read from a configuration file or database
    # For now, return an empty list - toolsets should be configured via the backend
    return []