"""
Hybrid ADK Agent - Smart Tool Selection
Direct GCP API calls for simple operations + Backend services for value-add operations
"""

import os
from typing import List, Dict, Any
from google.adk import Agent
from google.adk.tools.openapi_tool import RestApiTool
from google.genai import types
from .base_agent import initialize_vertex_ai

def create_hybrid_security_agent(project_id: str) -> Agent:
    """
    Create optimized ADK agent using:
    - RestApiTool for direct GCP API calls (no backend proxy needed)
    - Backend services only for value-add operations (KB, custom logic, etc.)
    """
    
    initialize_vertex_ai()
    
    # DIRECT GCP API CALLS (No backend needed)
    direct_gcp_tools = [
        # Security Center - Direct access
        RestApiTool(
            name="get_security_findings",
            description="Get security findings directly from GCP Security Center",
            base_url="https://securitycenter.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "Security Center", "version": "v1"},
                "paths": {
                    "/v1/organizations/{org_id}/sources/-/findings": {
                        "get": {
                            "summary": "List security findings",
                            "parameters": [
                                {"name": "org_id", "in": "path", "required": True, "schema": {"type": "string"}},
                                {"name": "filter", "in": "query", "schema": {"type": "string"}},
                                {"name": "pageSize", "in": "query", "schema": {"type": "integer"}}
                            ]
                        }
                    }
                }
            }
        ),
        
        # IAM - Direct access
        RestApiTool(
            name="get_iam_policy", 
            description="Get IAM policy directly from GCP",
            base_url="https://cloudresourcemanager.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "Resource Manager", "version": "v1"},
                "paths": {
                    f"/v1/projects/{project_id}:getIamPolicy": {
                        "post": {
                            "summary": "Get project IAM policy",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "options": {
                                                    "type": "object",
                                                    "properties": {
                                                        "requestedPolicyVersion": {"type": "integer"}
                                                    }
                                                }
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
        
        # Compute Engine - Direct access
        RestApiTool(
            name="list_compute_instances",
            description="List compute instances directly from GCP",
            base_url="https://compute.googleapis.com",
            spec={
                "openapi": "3.0.0", 
                "info": {"title": "Compute Engine", "version": "v1"},
                "paths": {
                    f"/compute/v1/projects/{project_id}/aggregated/instances": {
                        "get": {
                            "summary": "List all VM instances",
                            "parameters": [
                                {"name": "filter", "in": "query", "schema": {"type": "string"}},
                                {"name": "maxResults", "in": "query", "schema": {"type": "integer"}}
                            ]
                        }
                    }
                }
            }
        ),
        
        # Cloud Storage - Direct access
        RestApiTool(
            name="list_storage_buckets",
            description="List Cloud Storage buckets directly",
            base_url="https://storage.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "Cloud Storage", "version": "v1"},
                "paths": {
                    "/storage/v1/b": {
                        "get": {
                            "summary": "List storage buckets",
                            "parameters": [
                                {"name": "project", "in": "query", "required": True, "schema": {"type": "string"}},
                                {"name": "maxResults", "in": "query", "schema": {"type": "integer"}}
                            ]
                        }
                    }
                }
            }
        ),
        
        # Asset Inventory - Direct access
        RestApiTool(
            name="search_all_resources",
            description="Search all resources using Cloud Asset Inventory",
            base_url="https://cloudasset.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "Cloud Asset", "version": "v1"},
                "paths": {
                    "/v1/assets:searchAll": {
                        "get": {
                            "summary": "Search all accessible resources",
                            "parameters": [
                                {"name": "scope", "in": "query", "required": True, "schema": {"type": "string"}},
                                {"name": "query", "in": "query", "schema": {"type": "string"}},
                                {"name": "assetTypes", "in": "query", "schema": {"type": "array", "items": {"type": "string"}}},
                                {"name": "pageSize", "in": "query", "schema": {"type": "integer"}}
                            ]
                        }
                    }
                }
            }
        ),
        
        # Service Usage - Direct access  
        RestApiTool(
            name="list_enabled_services",
            description="List enabled APIs directly from Service Usage API",
            base_url="https://serviceusage.googleapis.com",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "Service Usage", "version": "v1"},
                "paths": {
                    f"/v1/projects/{project_id}/services": {
                        "get": {
                            "summary": "List project services",
                            "parameters": [
                                {"name": "filter", "in": "query", "schema": {"type": "string"}},
                                {"name": "pageSize", "in": "query", "schema": {"type": "integer"}}
                            ]
                        }
                    }
                }
            }
        )
    ]
    
    # VALUE-ADD BACKEND SERVICES (Keep these!)
    backend_services = [
        # Knowledge Base Articles - Customer specific data
        RestApiTool(
            name="search_knowledge_base",
            description="Search customer-specific security knowledge base articles",
            base_url="http://localhost:8000",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "ADK Backend", "version": "v1"},
                "paths": {
                    "/api/v1/knowledge/search": {
                        "post": {
                            "summary": "Search knowledge base",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "query": {"type": "string"},
                                                "category": {"type": "string"},
                                                "project_id": {"type": "string"}
                                            },
                                            "required": ["query"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ),
        
        # Custom Security Recommendations - Business logic
        RestApiTool(
            name="get_custom_recommendations",
            description="Get AI-powered security recommendations based on customer context",
            base_url="http://localhost:8000",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "ADK Backend", "version": "v1"}, 
                "paths": {
                    "/api/v1/recommendations/custom": {
                        "post": {
                            "summary": "Generate custom recommendations",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "project_id": {"type": "string"},
                                                "findings": {"type": "array"},
                                                "context": {"type": "object"},
                                                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                                            },
                                            "required": ["project_id"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ),
        
        # Custom Compliance Framework - Multi-standard evaluation
        RestApiTool(
            name="evaluate_custom_compliance",
            description="Evaluate compliance against multiple frameworks with customer-specific rules",
            base_url="http://localhost:8000",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "ADK Backend", "version": "v1"},
                "paths": {
                    "/api/v1/compliance/evaluate-custom": {
                        "post": {
                            "summary": "Custom compliance evaluation",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object", 
                                            "properties": {
                                                "project_id": {"type": "string"},
                                                "frameworks": {"type": "array", "items": {"type": "string"}},
                                                "custom_rules": {"type": "array"},
                                                "baseline": {"type": "object"}
                                            },
                                            "required": ["project_id", "frameworks"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ),
        
        # Store Customer Analysis - Persistent data
        RestApiTool(
            name="store_analysis_result",
            description="Store analysis results and customer insights for future reference",
            base_url="http://localhost:8000",
            spec={
                "openapi": "3.0.0",
                "info": {"title": "ADK Backend", "version": "v1"},
                "paths": {
                    "/api/v1/analysis/store": {
                        "post": {
                            "summary": "Store analysis results",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "project_id": {"type": "string"},
                                                "analysis_type": {"type": "string"},
                                                "results": {"type": "object"},
                                                "timestamp": {"type": "string"},
                                                "metadata": {"type": "object"}
                                            },
                                            "required": ["project_id", "analysis_type", "results"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )
    ]
    
    # Combine all tools
    all_tools = direct_gcp_tools + backend_services
    
    # Create the hybrid agent
    agent = Agent(
        model='gemini-2.0-flash-exp',
        name='hybrid_security_agent',
        description='Hybrid security agent with direct GCP access + value-add backend services',
        instruction=f"""
        You are an optimized GCP security agent with HYBRID capabilities.
        
        🔥 DIRECT GCP ACCESS (No backend proxy):
        - get_security_findings: Query Security Center directly
        - get_iam_policy: Get IAM policies directly  
        - list_compute_instances: List VMs directly
        - list_storage_buckets: Check Cloud Storage directly
        - search_all_resources: Use Asset Inventory directly
        - list_enabled_services: Check enabled APIs directly
        
        💎 VALUE-ADD BACKEND SERVICES (Use these for customer-specific data):
        - search_knowledge_base: Find customer KB articles and documentation
        - get_custom_recommendations: Get AI-powered, context-aware recommendations
        - evaluate_custom_compliance: Multi-framework compliance with custom rules
        - store_analysis_result: Save findings for customer's historical analysis
        
        📡 WORKFLOW INTELLIGENCE:
        1. Use DIRECT GCP calls for raw data gathering (faster, no proxy overhead)
        2. Use BACKEND SERVICES for business logic, customer context, and persistent data
        3. Always store important findings using store_analysis_result
        4. Search KB for customer-specific guidance before giving generic advice
        
        For project: {project_id}
        
        EXAMPLE WORKFLOW:
        User: "What's my security posture?"
        1. get_security_findings (direct GCP) → Get raw security data
        2. get_iam_policy (direct GCP) → Get permissions data  
        3. search_knowledge_base (backend) → Find customer's security policies
        4. get_custom_recommendations (backend) → Generate contextual advice
        5. store_analysis_result (backend) → Save for future reference
        
        This gives you SPEED (direct APIs) + INTELLIGENCE (custom backend logic).
        """,
        tools=all_tools,
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                )
            ]
        )
    )
    
    print(f"🚀 Hybrid ADK agent created:")
    print(f"  📡 {len(direct_gcp_tools)} direct GCP API tools (no backend proxy)")
    print(f"  💎 {len(backend_services)} value-add backend services")  
    print(f"  🎯 Project: {project_id}")
    
    return agent


# Factory function
def create_hybrid_adk_chat_service(project_id: str) -> Agent:
    """Create optimized hybrid ADK agent - best of both worlds."""
    return create_hybrid_security_agent(project_id)