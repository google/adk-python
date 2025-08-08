"""
Direct ADK Agent - Pure ADK Implementation
Uses RestApiTool to call GCP APIs directly, no backend middleware needed.
"""

import os
from typing import List, Dict, Any
from google.adk import Agent
from google.adk.tools.openapi_tool import RestApiTool
from google.genai import types
from .base_agent import initialize_vertex_ai

def create_direct_security_agent(project_id: str) -> Agent:
    """Create ADK agent that calls GCP APIs directly using RestApiTool."""
    
    initialize_vertex_ai()
    
    # Define all your backend services as direct REST API tools
    security_tools = [
        # Security Center API
        RestApiTool(
            name="get_security_findings",
            description="Get security findings from GCP Security Center",
            base_url="https://securitycenter.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "paths": {
                    f"/v1/organizations/{{org_id}}/sources/-/findings": {
                        "get": {
                            "summary": "List security findings",
                            "parameters": [
                                {
                                    "name": "org_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"}
                                },
                                {
                                    "name": "filter",
                                    "in": "query", 
                                    "schema": {"type": "string"}
                                }
                            ]
                        }
                    }
                }
            }
        ),
        
        # IAM API
        RestApiTool(
            name="get_iam_policy",
            description="Get IAM policy for GCP project",
            base_url="https://cloudresourcemanager.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "paths": {
                    f"/v1/projects/{project_id}:getIamPolicy": {
                        "post": {
                            "summary": "Get IAM policy",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "options": {"type": "object"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ),
        
        # Compute Engine API
        RestApiTool(
            name="list_compute_instances",
            description="List compute instances in project",
            base_url="https://compute.googleapis.com",
            spec={
                "openapi": "3.0.0", 
                "paths": {
                    f"/compute/v1/projects/{project_id}/aggregated/instances": {
                        "get": {
                            "summary": "List all instances",
                            "parameters": [
                                {
                                    "name": "filter",
                                    "in": "query",
                                    "schema": {"type": "string"}
                                }
                            ]
                        }
                    }
                }
            }
        ),
        
        # Asset Inventory API
        RestApiTool(
            name="list_assets",
            description="List all assets in GCP project",
            base_url="https://cloudasset.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "paths": {
                    f"/v1/projects/{project_id}/assets": {
                        "get": {
                            "summary": "List project assets",
                            "parameters": [
                                {
                                    "name": "assetTypes",
                                    "in": "query",
                                    "schema": {"type": "array", "items": {"type": "string"}}
                                },
                                {
                                    "name": "contentType", 
                                    "in": "query",
                                    "schema": {"type": "string"}
                                }
                            ]
                        }
                    }
                }
            }
        ),
        
        # Cloud Storage API
        RestApiTool(
            name="list_storage_buckets",
            description="List Cloud Storage buckets",
            base_url="https://storage.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "paths": {
                    "/storage/v1/b": {
                        "get": {
                            "summary": "List buckets",
                            "parameters": [
                                {
                                    "name": "project",
                                    "in": "query",
                                    "required": True,
                                    "schema": {"type": "string"}
                                }
                            ]
                        }
                    }
                }
            }
        ),
        
        # Service Usage API
        RestApiTool(
            name="list_enabled_services",
            description="List enabled services in project",
            base_url="https://serviceusage.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "paths": {
                    f"/v1/projects/{project_id}/services": {
                        "get": {
                            "summary": "List services",
                            "parameters": [
                                {
                                    "name": "filter",
                                    "in": "query",
                                    "schema": {"type": "string"}
                                }
                            ]
                        }
                    }
                }
            }
        )
    ]
    
    # Create the direct agent
    agent = Agent(
        model='gemini-2.0-flash-exp',
        name='direct_security_agent',
        description='Direct GCP security agent using RestApiTool',
        instruction=f"""
        You are a GCP security agent with DIRECT access to Google Cloud APIs.
        
        You have these capabilities through RestApiTool:
        - get_security_findings: Query Security Center for vulnerabilities
        - get_iam_policy: Analyze IAM permissions and roles
        - list_compute_instances: Inventory compute resources  
        - list_assets: Get comprehensive asset inventory
        - list_storage_buckets: Analyze Cloud Storage security
        - list_enabled_services: Check enabled APIs
        
        For project: {project_id}
        
        When asked about security:
        1. Use get_security_findings to check for active threats
        2. Use get_iam_policy to analyze access controls
        3. Use list_assets for comprehensive inventory
        4. Synthesize findings into actionable recommendations
        
        You call GCP APIs DIRECTLY - no backend middleware needed.
        Always provide specific, actionable security advice.
        """,
        tools=security_tools,
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                )
            ]
        )
    )
    
    print(f"🚀 Direct ADK agent created with {len(security_tools)} RestApiTools")
    print(f"📡 Calling GCP APIs directly for project: {project_id}")
    
    return agent


def create_rest_api_tool_from_service_config(service_name: str, config: Dict[str, Any], project_id: str) -> RestApiTool:
    """Convert backend service config to RestApiTool for direct API access."""
    
    # Map service names to GCP API endpoints
    api_mappings = {
        'security': {
            'base_url': 'https://securitycenter.googleapis.com',
            'paths': {
                '/v1/organizations/{org_id}/sources/-/findings': 'get'
            }
        },
        'iam': {
            'base_url': 'https://cloudresourcemanager.googleapis.com', 
            'paths': {
                f'/v1/projects/{project_id}:getIamPolicy': 'post',
                f'/v1/projects/{project_id}:testIamPermissions': 'post'
            }
        },
        'compliance': {
            'base_url': 'https://cloudresourcemanager.googleapis.com',
            'paths': {
                f'/v1/projects/{project_id}': 'get'
            }
        },
        'gcp': {
            'base_url': 'https://cloudresourcemanager.googleapis.com',
            'paths': {
                f'/v1/projects/{project_id}': 'get',
                '/v1/projects': 'get'
            }
        }
    }
    
    if service_name not in api_mappings:
        return None
        
    mapping = api_mappings[service_name]
    
    # Build OpenAPI spec
    spec = {
        "openapi": "3.0.0",
        "info": {"title": f"{service_name} API", "version": "1.0.0"},
        "paths": {}
    }
    
    for path, method in mapping['paths'].items():
        spec['paths'][path] = {
            method: {
                "summary": f"{service_name} operation",
                "description": config.get('description', f"{service_name} API operation")
            }
        }
    
    return RestApiTool(
        name=f"direct_{service_name}",
        description=config.get('description', f"Direct {service_name} API access"),
        base_url=mapping['base_url'],
        spec=spec
    )


# Factory function
def create_direct_adk_chat_service(project_id: str) -> Agent:
    """Create direct ADK agent - replaces your entire backend layer."""
    return create_direct_security_agent(project_id)