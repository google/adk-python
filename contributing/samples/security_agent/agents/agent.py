"""Security agent with API Hub integration.

This is the legacy agent file - functionality has been refactored into
the new organized structure. This file now imports from the new modules
and maintains backward compatibility.
"""

# Import the organized agent structure
from .security_agent import root_agent, create_security_agent

# Legacy imports for backward compatibility
from ..tools.gcp_tools.project_tools import get_gcp_projects, get_project_info, get_project_services
from ..tools.gcp_tools.storage_tools import analyze_gcs_bucket_security
from ..tools.api_tools.google_api_tools import call_google_api, create_apihub_toolset, get_available_toolsets
from ..tools.security_tools.knowledge_base_tools import evaluate_api_security, scrape_api_documentation, load_security_kb
from ..tools.analysis_tools.dependency_analysis import get_api_dependency_graph, propagate_risk

# Export the root agent from the new organized structure
# This maintains backward compatibility while using the new architecture
__all__ = [
    'root_agent',
    'create_security_agent',
    'get_gcp_projects',
    'get_project_info', 
    'get_project_services',
    'analyze_gcs_bucket_security',
    'call_google_api',
    'create_apihub_toolset',
    'get_available_toolsets',
    'evaluate_api_security',
    'scrape_api_documentation',
    'load_security_kb',
    'get_api_dependency_graph',
    'propagate_risk'
]