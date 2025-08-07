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
from ..tools.gcp_tools import project_tools, storage_tools
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
    
    # Collect base security tools
    base_tools = [
        project_tools.get_gcp_projects,
        project_tools.get_project_info,
        project_tools.get_project_services,
        storage_tools.analyze_gcs_bucket_security,
        google_api_tools.call_google_api,
        knowledge_base_tools.evaluate_api_security,
        knowledge_base_tools.scrape_api_documentation,
        dependency_analysis.get_api_dependency_graph,
        dependency_analysis.propagate_risk
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
        instruction=\"\"\"
            You are a comprehensive security evaluation agent for GCP APIs and projects. 
            
            Your primary functions:
            - Use get_gcp_projects to list the user's accessible GCP projects
            - Use get_project_info to get detailed information about specific projects
            - Use get_project_services to list enabled services in a project
            - Use evaluate_api_security to assess GCP API security using the knowledge base
            - Use scrape_api_documentation to extract security information from documentation URLs
            - Use get_api_dependency_graph to visualize API dependencies
            - Use propagate_risk to identify at-risk services due to dependencies
            - When asked about GCS buckets, use the `analyze_gcs_bucket_security` tool to provide actionable recommendations.
            - For any other Google Cloud API interactions, use the generic `call_google_api` tool.
            - Always provide actionable recommendations and reference official documentation
            
            When a user asks about their GCP environment:
            1. First use get_gcp_projects to see what projects they have access to
            2. Use get_project_services to see what services are enabled
            3. Use get_project_info for detailed project information when needed
            4. Use evaluate_api_security for knowledge base security assessments
            5. Provide comprehensive security recommendations based on actual project data
            
            You have access to LIVE GCP data through proper authentication, so you can analyze
            the user's actual projects, services, and configurations. Always use real data
            when available rather than generic responses.
            
            If an API is not found in your knowledge base, inform the user and suggest updating it.
            Always prioritize security best practices and compliance requirements.
            \"\"\",
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
    print(f"🤖 Security agent created with {len(base_tools)} tools")
    
    return agent


# Create the root agent instance
root_agent = create_security_agent()