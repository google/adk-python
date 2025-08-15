"""
Security Agent

The main security analysis agent for GCP environments.
This agent provides comprehensive security evaluation capabilities.
"""

import os
from typing import List
from google.adk import Agent
from google.genai import types

# Import base agent functionality
from .base_agent import initialize_vertex_ai, collect_tools_from_modules

# Import tool modules
from ..tools.gcp_tools import project_tools, storage_tools, asset_inventory_tools
from ..tools.api_tools import google_api_tools
from ..tools.security_tools import knowledge_base_tools
from ..tools.analysis_tools import dependency_analysis

# Import API Hub toolset creation
from ..tools.api_tools.google_api_tools import create_apihub_toolset


def create_security_agent() -> Agent:
    """Create and configure the security agent with all necessary tools.
    
    Returns:
        Configured Agent instance ready for security analysis tasks.
    """
    # Initialize Vertex AI
    initialize_vertex_ai()
    
    # Collect base security tools including comprehensive Asset Inventory
    base_tools = [
        project_tools.get_gcp_projects,
        project_tools.get_project_info,
        project_tools.get_project_services,
        storage_tools.analyze_gcs_bucket_security,
        google_api_tools.call_google_api,
        knowledge_base_tools.evaluate_api_security,
        knowledge_base_tools.scrape_api_documentation,
        dependency_analysis.get_api_dependency_graph,
        dependency_analysis.propagate_risk,
        # Comprehensive Asset Inventory tools
        asset_inventory_tools.discover_gcp_resources,
        asset_inventory_tools.get_compute_instances,
        asset_inventory_tools.get_storage_buckets,
        asset_inventory_tools.get_cloud_functions,
        asset_inventory_tools.get_databases,
        asset_inventory_tools.get_kubernetes_clusters,
        asset_inventory_tools.analyze_security_assets,
        asset_inventory_tools.search_assets_by_name,
        asset_inventory_tools.get_asset_inventory_summary
    ]
    
    # Dynamically load API Hub toolset if configured
    apihub_resource_name = os.environ.get('APIHUB_RESOURCE_NAME')
    if apihub_resource_name:
        try:
            print(f"Attempting to load API Hub Toolset for resource: {apihub_resource_name}")
            apihub_toolset = create_apihub_toolset(
                toolset_name="security_apihub_tools",
                apihub_resource_name=apihub_resource_name,
                description="Tools dynamically loaded from API Hub for security evaluations"
            )
            
            # Add tools from API Hub toolset
            tools = apihub_toolset.get_tools()
            base_tools.extend(tools)
            print(f"✅ Added {len(tools)} tools from API Hub toolset")
        except Exception as e:
            print(f"❌ Failed to load API Hub Toolset: {e}")
    
    # Create the security agent
    agent = Agent(
        model='gemini-2.5-flash',
        name='security_agent',
        description=(
            'Security evaluation agent for evaluating the security stance of '
            'onboarding new GCP APIs using public documentation and JSON knowledge base,'
            'and dynamic API Hub tools. This agent also provides dependency analysis '
            'and risk propagation.'
        ),
        instruction="""
            You are a comprehensive security evaluation agent for GCP APIs and projects with unified 
            Asset Inventory access to ALL GCP services and resources.
            
            Your enhanced capabilities include:
            
            UNIFIED RESOURCE DISCOVERY:
            - Use discover_gcp_resources for natural language queries like "show me my compute instances"
            - Use get_compute_instances for all VM instances and their security analysis
            - Use get_storage_buckets for all storage buckets with security recommendations
            - Use get_cloud_functions for all serverless functions 
            - Use get_databases for all databases (Cloud SQL, Spanner, BigQuery, etc.)
            - Use get_kubernetes_clusters for all GKE clusters
            - Use analyze_security_assets for comprehensive security posture analysis
            - Use search_assets_by_name to find resources by name patterns
            - Use get_asset_inventory_summary for complete project overview
            
            TRADITIONAL PROJECT TOOLS:
            - Use get_gcp_projects to list accessible GCP projects
            - Use get_project_info for detailed project information
            - Use get_project_services for enabled services
            
            SECURITY ANALYSIS:
            - Use evaluate_api_security for knowledge base security assessments
            - Use scrape_api_documentation for documentation analysis
            - Use get_api_dependency_graph for dependency visualization
            - Use propagate_risk for risk propagation analysis
            
            INTELLIGENT QUERY PROCESSING:
            When users ask questions like:
            - "What compute instances do I have?" → Use get_compute_instances
            - "Show me my databases" → Use get_databases  
            - "Tell me about my cloud functions" → Use get_cloud_functions
            - "Analyze my security" → Use analyze_security_assets
            - "What resources are in my project?" → Use get_asset_inventory_summary
            - Any natural language query → Use discover_gcp_resources
            
            You have REAL-TIME access to the user's complete GCP infrastructure through the 
            Asset Inventory API. Always use these tools to provide accurate, current information
            about their actual resources rather than generic responses.
            
            RESPONSE PATTERN:
            1. Use the appropriate asset discovery tool based on the user's query
            2. Analyze the real data returned from the Asset Inventory API
            3. Provide specific security recommendations based on actual findings
            4. Reference the API calls made (logged to cloudasset.googleapis.com)
            5. Always prioritize actionable insights over generic advice
        """,
        tools=base_tools,
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ]
        )
    )
    
    print(f"🔑 Using Vertex AI with Application Default Credentials")
    print(f"🤖 Security agent created with {len(base_tools)} tools including comprehensive Asset Inventory")
    print(f"📊 Asset Inventory tools: 9 specialized functions for unified GCP resource discovery")
    
    return agent


# Create the root agent instance
root_agent = create_security_agent()