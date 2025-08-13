"""Frontend components package for the security agent - reorganized by feature area."""

# Dashboard components
from .dashboard.dashboard_view import render_dashboard_view
from .dashboard.multi_agent_graph_view import render_multi_agent_graph_view

# Security components
from .security.security_evaluation_view import render_security_evaluation_view, render_security_summary_card
from .security.iam_analyzer_view import render_iam_analyzer_view, render_iam_summary_card
from .security.incident_response_view import render_incident_response_view, render_incident_summary_card

# Compliance components
from .compliance.compliance_view import render_compliance_view, render_compliance_summary_card

# Chat components
from .chat.chat_view import render_chat_view

# Roadmap components
from .roadmap.roadmap_view import render_roadmap_view

# Monitoring components
from .monitoring.performance_monitoring_view import (
    render_performance_monitoring_view, 
    render_day_two_sre_view, 
    render_performance_summary_card
)
# Services management removed - using ADK agent routing instead

# Shared components
from .shared.recommendations_view import render_recommendations_view, render_recommendations_summary_card
from .shared.msa_analysis_view import render_msa_analysis_view, render_msa_summary_card
from .shared.api_explorer_view import render_api_explorer_view, render_api_explorer_summary_card
from .shared.gcp_api_explorer_view import render_gcp_api_explorer_view, render_gcp_api_explorer_summary_card

# ADK demos now integrated into main chat interface

__all__ = [
    # Dashboard
    'render_dashboard_view',
    'render_multi_agent_graph_view',
    
    # Security
    'render_security_evaluation_view',
    'render_security_summary_card',
    'render_iam_analyzer_view',
    'render_iam_summary_card',
    'render_incident_response_view',
    'render_incident_summary_card',
    
    # Compliance
    'render_compliance_view',
    'render_compliance_summary_card',
    
    # Chat
    'render_chat_view',
    
    # Roadmap
    'render_roadmap_view',
    
    # Monitoring
    'render_performance_monitoring_view',
    'render_day_two_sre_view',
    'render_performance_summary_card',
# 'render_services_management_enhanced_view',  # Removed - ADK agent routing
    
    # Shared
    'render_recommendations_view',
    'render_recommendations_summary_card',
    'render_msa_analysis_view',
    'render_msa_summary_card',
    'render_api_explorer_view',
    'render_api_explorer_summary_card',
    'render_gcp_api_explorer_view',
    'render_gcp_api_explorer_summary_card',
]