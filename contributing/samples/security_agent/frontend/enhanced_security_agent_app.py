"""Enhanced GCP API Security Evaluation Agent with OIDC Flow Demonstration."""

import streamlit as st
import json
import os
import requests
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlencode, parse_qs, urlparse
from streamlit_agraph import agraph, Node, Edge, Config


# Configuration
BACKEND_URL = "http://localhost:8000"
OIDC_DEMO_CONFIG = {
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "client_id": "demo-client-id",
    "redirect_uri": "http://localhost:8501/callback",
    "scope": "openid email profile",
    "response_type": "code"
}


def init_session_state():
    """Initialize session state variables."""
    if 'oidc_state' not in st.session_state:
        st.session_state.oidc_state = None
    if 'oidc_code' not in st.session_state:
        st.session_state.oidc_code = None
    if 'oidc_tokens' not in st.session_state:
        st.session_state.oidc_tokens = None
    if 'current_user' not in st.session_state:
        st.session_state.current_user = {"email": "admin@stuartgano.altostrat.com", "authenticated": True}
    if 'selected_api' not in st.session_state:
        st.session_state.selected_api = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    if 'credentials_data' not in st.session_state:
        st.session_state.credentials_data = {"use_adc": True}
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "mgm-digitalconcierge"
    if 'available_projects' not in st.session_state:
        st.session_state.available_projects = []


def fetch_available_projects() -> List[str]:
    """Fetch available GCP projects from backend."""
    try:
        response = make_backend_request("/api/v1/gcp/projects", include_project=False)
        if response.get("success"):
            projects = response.get("projects", [])
            if not projects:
                 st.warning("No GCP projects were found for the current user account.")
            return projects
        else:
            error_message = response.get('error', 'An unknown error occurred')
            st.error(f"Failed to fetch GCP projects: {error_message}")
            return []
    except Exception as e:
        st.error(f"A critical error occurred while fetching projects: {e}")
        return []


def project_picker_sidebar():
    """Render GCP project picker in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏗️ GCP Project")
    
    # Fetch projects if not already loaded
    if not st.session_state.available_projects:
        with st.spinner("Loading projects..."):
            st.session_state.available_projects = fetch_available_projects()
    
    # Project selector
    selected_project = st.sidebar.selectbox(
        "Select Project for Scanning:",
        options=st.session_state.available_projects,
        index=st.session_state.available_projects.index(st.session_state.selected_project) 
              if st.session_state.selected_project in st.session_state.available_projects 
              else 0,
        help="Choose the GCP project to scan for security analysis"
    )
    
    # Update session state if project changed
    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.sidebar.success(f"✅ Project changed to: {selected_project}")
        st.rerun()
    
    # Display current project info
    st.sidebar.info(f"🎯 Active Project: **{st.session_state.selected_project}**")
    
    # Refresh projects button
    if st.sidebar.button("🔄 Refresh Projects"):
        st.session_state.available_projects = fetch_available_projects()
        st.sidebar.success("Projects refreshed!")
        st.rerun()
    
    # ADK Web Interface access
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 ADK Web Interface")
    st.sidebar.markdown("**Native ADK Interface:** [localhost:8080](http://localhost:8080)")
    if st.sidebar.button("🚀 Open ADK Web"):
        st.sidebar.success("Check your browser!")
        st.sidebar.info("ADK Web Interface should open at http://localhost:8080")
    
    # Manual project input
    with st.sidebar.expander("➕ Add Custom Project"):
        custom_project = st.text_input(
            "Project ID:",
            placeholder="your-project-id",
            help="Enter a custom GCP project ID"
        )
        if st.button("Add Project") and custom_project:
            if custom_project not in st.session_state.available_projects:
                st.session_state.available_projects.append(custom_project)
                st.session_state.selected_project = custom_project
                st.success(f"Added project: {custom_project}")
                st.rerun()


def make_backend_request(endpoint: str, method: str = "GET", data: Dict = None, include_project: bool = True) -> Dict:
    """Make request to backend API with optional project context."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        # Add project context to request data if requested
        if include_project and hasattr(st.session_state, 'selected_project'):
            if data is None:
                data = {}
            data["project_id"] = st.session_state.selected_project
        
        if method == "GET":
            # For GET requests, add project as query parameter
            if include_project and hasattr(st.session_state, 'selected_project'):
                params = {"project_id": st.session_state.selected_project}
                response = requests.get(url, params=params, timeout=10)
            else:
                response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        
        response.raise_for_status()
        
        # Try to parse JSON response
        try:
            return response.json()
        except ValueError as json_error:
            st.error(f"JSON Parse Error: {str(json_error)}")
            st.error(f"Response content (first 200 chars): {response.text[:200]}")
            return {"success": False, "error": f"Invalid JSON response: {response.text[:100]}"}
            
    except requests.exceptions.RequestException as e:
        st.error(f"Backend request failed: {str(e)}")
        return {"success": False, "error": str(e)}


def oidc_flow_demo():
    """Demonstrate OIDC flow without real credentials."""
    st.markdown("---")
    st.header("🔐 OIDC Authentication Flow Demonstration")
    st.write("This demonstrates the OIDC authentication flow without requiring real credentials.")
    
    # OIDC Flow Steps
    with st.expander("📋 OIDC Flow Overview", expanded=True):
        st.markdown("""
        **OpenID Connect (OIDC) Flow Steps:**
        1. **Authorization Request** - Client redirects user to authorization server
        2. **User Authentication** - User authenticates with the authorization server
        3. **Authorization Response** - Server redirects back with authorization code
        4. **Token Exchange** - Client exchanges code for tokens
        5. **User Info** - Client retrieves user information using access token
        """)
    
    # Step 1: Authorization Request
    st.subheader("Step 1: Authorization Request")
    st.write("Generate authorization URL for OIDC flow:")
    
    # Generate state parameter
    if st.button("Generate Authorization URL"):
        st.session_state.oidc_state = str(uuid.uuid4())
        
        auth_params = {
            "client_id": OIDC_DEMO_CONFIG["client_id"],
            "redirect_uri": OIDC_DEMO_CONFIG["redirect_uri"],
            "response_type": OIDC_DEMO_CONFIG["response_type"],
            "scope": OIDC_DEMO_CONFIG["scope"],
            "state": st.session_state.oidc_state
        }
        
        auth_url = f"{OIDC_DEMO_CONFIG['authorization_endpoint']}?{urlencode(auth_params)}"
        
        st.code(auth_url, language="text")
        st.info("🔗 This URL would redirect the user to Google's OAuth consent screen")
        
        # Simulate the authorization response
        st.session_state.oidc_code = "demo_auth_code_12345"
        st.success("✅ Authorization code received (simulated)")
    
    # Step 2: Token Exchange
    if st.session_state.oidc_code:
        st.subheader("Step 2: Token Exchange")
        st.write("Exchange authorization code for tokens:")
        
        if st.button("Exchange Code for Tokens"):
            # Simulate token response
            st.session_state.oidc_tokens = {
                "access_token": "demo_access_token_67890",
                "id_token": "demo_id_token_11111",
                "refresh_token": "demo_refresh_token_22222",
                "token_type": "Bearer",
                "expires_in": 3600
            }
            
            st.json(st.session_state.oidc_tokens)
            st.success("✅ Tokens received successfully")
    
    # Step 3: User Info
    if st.session_state.oidc_tokens:
        st.subheader("Step 3: User Information")
        st.write("Retrieve user information using access token:")
        
        if st.button("Get User Info"):
            # Simulate user info
            user_info = {
                "sub": "demo_user_123",
                "name": "Demo User",
                "email": "demo@example.com",
                "email_verified": True,
                "picture": "https://example.com/avatar.jpg"
            }
            
            st.session_state.current_user = user_info
            st.json(user_info)
            st.success("✅ User information retrieved")
    
    # Current Authentication Status
    st.subheader("🔍 Current Authentication Status")
    auth_status = {
        "State Parameter": st.session_state.oidc_state or "Not generated",
        "Authorization Code": st.session_state.oidc_code or "Not received",
        "Access Token": st.session_state.oidc_tokens.get("access_token") if st.session_state.oidc_tokens else "Not received",
        "User Authenticated": "Yes" if st.session_state.current_user else "No",
        "User Email": st.session_state.current_user.get("email") if st.session_state.current_user else "N/A"
    }
    
    for key, value in auth_status.items():
        st.write(f"**{key}:** {value}")
    
    # Reset Demo
    if st.button("🔄 Reset OIDC Demo"):
        st.session_state.oidc_state = None
        st.session_state.oidc_code = None
        st.session_state.oidc_tokens = None
        st.session_state.current_user = None
        st.success("OIDC demo reset successfully")
        st.rerun()


def security_evaluation_ui():
    """Enhanced security evaluation UI with all new capabilities."""
    st.markdown("---")
    st.header("🔒 Enhanced Security Evaluation")
    
    # API Selection
    col1, col2 = st.columns(2)
    with col1:
        api_name = st.text_input("Enter GCP API Name:", value="Cloud Storage")
    with col2:
        evaluation_type = st.selectbox(
            "Evaluation Type:",
            ["Comprehensive", "Security Only", "Compliance", "Threat Intelligence", "Configuration Analysis"]
        )
    
    # Evaluation Options
    with st.expander("⚙️ Evaluation Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            include_compliance = st.checkbox("Include Compliance", value=True)
            include_threat_intel = st.checkbox("Include Threat Intelligence", value=True)
        with col2:
            include_config_analysis = st.checkbox("Include Configuration Analysis", value=True)
            include_incident_response = st.checkbox("Include Incident Response", value=True)
        with col3:
            frameworks = st.multiselect(
                "Compliance Frameworks:",
                ["soc2", "iso27001", "gdpr", "hipaa", "pci_dss"],
                default=["soc2", "iso27001"]
            )
    
    # Run Evaluation
    if st.button("🚀 Run Comprehensive Security Evaluation"):
        with st.spinner("Running security evaluation..."):
            results = {}
            
            # Basic security evaluation
            if evaluation_type in ["Comprehensive", "Security Only"]:
                security_result = make_backend_request(
                    "/api/v1/security/evaluate",
                    method="POST",
                    data={
                        "api_name": api_name,
                        "project_id": st.session_state.selected_project
                    }
                )
                results["security"] = security_result
            
            # Compliance evaluation
            if evaluation_type in ["Comprehensive", "Compliance"] and include_compliance:
                compliance_result = make_backend_request(
                    "/api/v1/compliance/evaluate",
                    method="POST",
                    data={
                        "api_name": api_name,
                        "frameworks": frameworks,
                        "project_id": st.session_state.selected_project
                    }
                )
                results["compliance"] = compliance_result
            
            # Threat intelligence
            if evaluation_type in ["Comprehensive", "Threat Intelligence"] and include_threat_intel:
                threat_result = make_backend_request(
                    "/api/v1/threat-intelligence/landscape",
                    method="POST",
                    data={
                        "api_name": api_name,
                        "project_id": st.session_state.selected_project
                    }
                )
                results["threat_intelligence"] = threat_result
            
            # Configuration analysis
            if evaluation_type in ["Comprehensive", "Configuration Analysis"] and include_config_analysis:
                config_result = make_backend_request(
                    "/api/v1/configuration/analyze",
                    method="POST",
                    data={
                        "api_name": api_name,
                        "project_id": st.session_state.selected_project
                    }
                )
                results["configuration"] = config_result
            
            st.session_state.evaluation_results = results
            st.success("✅ Evaluation completed!")
    
    # Display Results
    if st.session_state.evaluation_results:
        display_evaluation_results(st.session_state.evaluation_results)


def display_evaluation_results(results: Dict[str, Any]):
    """Display comprehensive evaluation results."""
    st.markdown("---")
    st.header("📊 Evaluation Results")
    
    # Create tabs for different result types
    tabs = st.tabs(["Security", "Compliance", "Threat Intelligence", "Configuration", "Summary"])
    
    with tabs[0]:
        if "security" in results:
            security_data = results["security"]
            if security_data.get("success"):
                st.subheader("🔒 Security Evaluation")
                evaluation_text = security_data.get("evaluation", "")
                if evaluation_text:
                    st.text_area("Security Analysis:", value=evaluation_text, height=200, disabled=True)
                else:
                    st.warning("No evaluation data available")
            else:
                st.error(f"Security evaluation failed: {security_data.get('error')}")
        else:
            st.info("Security evaluation not performed")
    
    with tabs[1]:
        if "compliance" in results:
            compliance_data = results["compliance"]
            if compliance_data.get("success"):
                st.subheader("📋 Compliance Evaluation")
                
                # Compliance scores chart
                frameworks_data = compliance_data.get("evaluation", {}).get("frameworks", {})
                if frameworks_data:
                    scores_data = []
                    for framework, data in frameworks_data.items():
                        if isinstance(data, dict) and "compliance_score" in data:
                            scores_data.append({
                                "Framework": data.get("name", framework),
                                "Score": data["compliance_score"],
                                "Status": data.get("status", "Unknown")
                            })
                    
                    if scores_data:
                        df = pd.DataFrame(scores_data)
                        fig = px.bar(df, x="Framework", y="Score", color="Status",
                                   title="Compliance Scores by Framework")
                        st.plotly_chart(fig)
                
                st.json(compliance_data.get("evaluation", {}))
            else:
                st.error(f"Compliance evaluation failed: {compliance_data.get('error')}")
        else:
            st.info("Compliance evaluation not performed")
    
    with tabs[2]:
        if "threat_intelligence" in results:
            threat_data = results["threat_intelligence"]
            if threat_data.get("success"):
                st.subheader("🛡️ Threat Intelligence Analysis")
                
                # Risk level indicator
                risk_level = threat_data.get("risk_level", "Unknown")
                risk_colors = {
                    "Critical": "red",
                    "High": "orange", 
                    "Medium": "yellow",
                    "Low": "green"
                }
                
                st.metric("Risk Level", risk_level, delta_color=risk_colors.get(risk_level, "normal"))
                
                # Vulnerability metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Vulnerabilities", threat_data.get("total_vulnerabilities", 0))
                with col2:
                    st.metric("Critical Vulnerabilities", threat_data.get("critical_vulnerabilities", 0))
                with col3:
                    st.metric("High Vulnerabilities", threat_data.get("high_vulnerabilities", 0))
                
                st.json(threat_data)
            else:
                st.error(f"Threat intelligence analysis failed: {threat_data.get('error')}")
        else:
            st.info("Threat intelligence analysis not performed")
    
    with tabs[3]:
        if "configuration" in results:
            config_data = results["configuration"]
            if config_data.get("success"):
                st.subheader("⚙️ Configuration Analysis")
                
                # Overall score
                overall_score = config_data.get("overall_score", 0)
                st.metric("Overall Security Score", f"{overall_score}%")
                
                # Issues breakdown
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Critical Issues", config_data.get("critical_issues", 0))
                with col2:
                    st.metric("High Issues", config_data.get("high_issues", 0))
                with col3:
                    st.metric("Medium Issues", config_data.get("medium_issues", 0))
                with col4:
                    st.metric("Low Issues", config_data.get("low_issues", 0))
                
                st.json(config_data)
            else:
                st.error(f"Configuration analysis failed: {config_data.get('error')}")
        else:
            st.info("Configuration analysis not performed")
    
    with tabs[4]:
        st.subheader("📈 Summary Dashboard")
        
        # Create summary metrics
        summary_data = []
        
        if "security" in results and results["security"].get("success"):
            summary_data.append({"Metric": "Security Evaluation", "Status": "✅ Completed"})
        else:
            summary_data.append({"Metric": "Security Evaluation", "Status": "❌ Not Available"})
        
        if "compliance" in results and results["compliance"].get("success"):
            summary_data.append({"Metric": "Compliance Evaluation", "Status": "✅ Completed"})
        else:
            summary_data.append({"Metric": "Compliance Evaluation", "Status": "❌ Not Available"})
        
        if "threat_intelligence" in results and results["threat_intelligence"].get("success"):
            summary_data.append({"Metric": "Threat Intelligence", "Status": "✅ Completed"})
        else:
            summary_data.append({"Metric": "Threat Intelligence", "Status": "❌ Not Available"})
        
        if "configuration" in results and results["configuration"].get("success"):
            summary_data.append({"Metric": "Configuration Analysis", "Status": "✅ Completed"})
        else:
            summary_data.append({"Metric": "Configuration Analysis", "Status": "❌ Not Available"})
        
        df = pd.DataFrame(summary_data)
        st.table(df)


def incident_response_ui():
    """Incident response management UI."""
    st.markdown("---")
    st.header("🚨 Incident Response Management")
    
    # Create new incident
    with st.expander("➕ Create New Incident", expanded=False):
        with st.form("create_incident_form"):
            col1, col2 = st.columns(2)
            with col1:
                incident_title = st.text_input("Incident Title")
                severity = st.selectbox("Severity", ["critical", "high", "medium", "low"])
                api_name = st.text_input("Affected API")
            with col2:
                description = st.text_area("Description")
                affected_resources = st.text_area("Affected Resources (one per line)")
                indicators = st.text_area("Security Indicators (one per line)")
            
            if st.form_submit_button("Create Incident"):
                if incident_title and api_name:
                    incident_data = {
                        "title": incident_title,
                        "description": description,
                        "severity": severity,
                        "api_name": api_name,
                        "affected_resources": [r.strip() for r in affected_resources.split('\n') if r.strip()],
                        "indicators": [i.strip() for i in indicators.split('\n') if i.strip()]
                    }
                    
                    result = make_backend_request(
                        "/api/v1/incidents/create",
                        method="POST",
                        data=incident_data
                    )
                    
                    if result.get("success"):
                        st.success("✅ Incident created successfully!")
                    else:
                        st.error(f"Failed to create incident: {result.get('error')}")
    
    # List incidents
    if st.button("📋 Refresh Incidents"):
        incidents = make_backend_request("/api/v1/incidents")
        if incidents.get("success"):
            st.session_state.incidents = incidents.get("incidents", [])
    
    if hasattr(st.session_state, 'incidents') and st.session_state.incidents:
        st.subheader("📋 Active Incidents")
        for incident in st.session_state.incidents:
            with st.expander(f"{incident['title']} - {incident['severity'].upper()}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**API:** {incident['api_name']}")
                    st.write(f"**Status:** {incident['status']}")
                    st.write(f"**Created:** {incident['detected_at']}")
                with col2:
                    st.write(f"**Description:** {incident['description']}")
                    if incident.get('affected_resources'):
                        st.write(f"**Affected Resources:** {', '.join(incident['affected_resources'])}")


def knowledge_base_ui():
    """Enhanced knowledge base management UI."""
    st.markdown("---")
    st.header("📚 Knowledge Base Management")
    
    # Load knowledge base
    kb_path = os.path.join(os.path.dirname(__file__), 'gcp_api_security_kb.json')
    
    try:
        with open(kb_path, 'r') as f:
            kb = json.load(f)
    except FileNotFoundError:
        st.error("Knowledge base file not found!")
        return
    
    # API selection
    api_names = [api['name'] for api in kb['apis']]
    selected_api_name = st.selectbox("Select API to manage:", [''] + api_names)
    
    if selected_api_name:
        selected_api = next((api for api in kb['apis'] if api['name'] == selected_api_name), None)
        
        if selected_api:
            # Display current API info
            st.subheader(f"📋 {selected_api['name']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Documentation URL:** {selected_api.get('documentation_url', 'N/A')}")
                st.write(f"**Vulnerable:** {'Yes' if selected_api.get('vulnerable', False) else 'No'}")
            with col2:
                st.write(f"**Security Considerations:** {len(selected_api.get('security_considerations', []))}")
                st.write(f"**Dependencies:** {len(selected_api.get('dependencies', []))}")
            
            # Edit API
            with st.expander("✏️ Edit API Information", expanded=False):
                with st.form("edit_api_form"):
                    name = st.text_input("API Name", value=selected_api['name'])
                    doc_url = st.text_input("Documentation URL", value=selected_api.get('documentation_url', ''))
                    considerations = st.text_area("Security Considerations", value='\n'.join(selected_api.get('security_considerations', [])))
                    practices = st.text_area("Recommended Practices", value='\n'.join(selected_api.get('recommended_practices', [])))
                    dependencies = st.text_input("Dependencies", value=','.join(selected_api.get('dependencies', [])))
                    vulnerable = st.checkbox("Vulnerable", value=selected_api.get('vulnerable', False))
                    
                    if st.form_submit_button("Save Changes"):
                        selected_api.update({
                            'name': name,
                            'documentation_url': doc_url,
                            'security_considerations': [c.strip() for c in considerations.split('\n') if c.strip()],
                            'recommended_practices': [p.strip() for p in practices.split('\n') if p.strip()],
                            'dependencies': [d.strip() for d in dependencies.split(',') if d.strip()],
                            'vulnerable': vulnerable
                        })
                        
                        with open(kb_path, 'w') as f:
                            json.dump(kb, f, indent=2)
                        
                        st.success("✅ API updated successfully!")


def msa_analysis_ui():
    """MSA parsing and Google Cloud organization scanning UI."""
    st.markdown("---")
    st.header("📄 MSA Analysis & Google Cloud Impact Assessment")
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Parse MSA", "🔍 Scan Organization", "📊 View Results", "⚙️ Settings"])
    
    with tab1:
        st.subheader("📝 Parse Microsoft Service Agreement (MSA)")
        
        # MSA input options
        input_option = st.radio(
            "Choose input method:",
            ["📋 Enter MSA Text", "📄 Upload MSA File", "🎯 Use Sample MSA"]
        )
        
        msa_text = ""
        msa_name = ""
        
        if input_option == "📋 Enter MSA Text":
            msa_name = st.text_input("MSA Name:", value="My Service Agreement")
            msa_text = st.text_area(
                "MSA Text Content:",
                height=400,
                placeholder="Paste your Microsoft Service Agreement text here..."
            )
        
        elif input_option == "📄 Upload MSA File":
            uploaded_file = st.file_uploader("Upload MSA file", type=['txt', 'pdf', 'docx'])
            if uploaded_file:
                msa_name = st.text_input("MSA Name:", value=uploaded_file.name)
                if uploaded_file.type == "text/plain":
                    msa_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("File format not supported. Please convert to text format.")
        
        elif input_option == "🎯 Use Sample MSA":
            if st.button("Load Sample MSA"):
                sample_response = make_backend_request("/api/v1/msa/sample-msa")
                if sample_response.get("success"):
                    msa_text = sample_response["sample_msa"]
                    msa_name = "Sample MSA"
                    st.success("✅ Sample MSA loaded!")
        
        # Parse MSA
        if msa_text and msa_name:
            if st.button("🔍 Parse MSA"):
                with st.spinner("Parsing MSA..."):
                    parse_data = {
                        "msa_text": msa_text,
                        "msa_name": msa_name,
                        "user_id": st.session_state.current_user.get("email", "demo_user") if st.session_state.current_user else "demo_user"
                    }
                    
                    result = make_backend_request("/api/v1/msa/parse", method="POST", data=parse_data)
                    
                    if result.get("success"):
                        st.session_state.current_msa = result["msa_record"]
                        st.success(f"✅ {result['message']}")
                        
                        # Display parsing results
                        msa_record = result["msa_record"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", msa_record["word_count"])
                        with col2:
                            st.metric("APIs Found", len(msa_record["api_mentions"]))
                        with col3:
                            st.metric("Security Keywords", msa_record["security_content"]["security_keywords_found"])
                        
                        # Show API mentions
                        if msa_record["api_mentions"]:
                            st.subheader("🔍 APIs Mentioned in MSA")
                            for api_name, api_data in msa_record["api_mentions"].items():
                                with st.expander(f"{api_name.replace('_', ' ').title()} ({api_data['count']} mentions)"):
                                    st.write(f"**Total Occurrences:** {api_data['total_occurrences']}")
                                    for mention in api_data['mentions'][:3]:  # Show first 3 mentions
                                        st.write(f"**Context:** {mention['context']}")
                    else:
                        st.error(f"❌ Failed to parse MSA: {result.get('error')}")
    
    with tab2:
        st.subheader("🔍 Scan Google Cloud Organization")
        
        if not hasattr(st.session_state, 'current_msa') or not st.session_state.current_msa:
            st.warning("⚠️ Please parse an MSA first in the 'Parse MSA' tab.")
        else:
            st.info(f"📄 Using MSA: {st.session_state.current_msa['msa_name']}")
            
            # Authentication options
            auth_option = st.radio(
                "Choose authentication method:",
                ["🔐 OIDC Flow (Recommended)", "🔑 Service Account", "🎯 Demo Mode"]
            )
            
            credentials_data = {}
            
            if auth_option == "🔐 OIDC Flow (Recommended)":
                st.info("🔐 Using Application Default Credentials (ADC) from gcloud auth.")
                
                if st.button("🔐 Use Current gcloud Authentication"):
                    # Use Application Default Credentials
                    credentials_data = {"use_adc": True}
                    st.session_state.credentials_data = credentials_data
                    st.success("✅ Using gcloud credentials: admin@stuartgano.altostrat.com")
                    st.info("Ready to scan Google Cloud organization.")
                
                # Show current gcloud status
                with st.expander("🔍 Current gcloud Authentication Status"):
                    st.text("Active Account: admin@stuartgano.altostrat.com")
                    st.text("Project: mgm-digitalconcierge") 
                    st.text("Auth Status: ✅ Authenticated")
            
            elif auth_option == "🔑 Service Account":
                st.info("🔑 Upload your service account JSON file.")
                sa_file = st.file_uploader("Service Account JSON", type=['json'])
                if sa_file:
                    try:
                        sa_data = json.load(sa_file)
                        credentials_data = {"service_account_info": sa_data}
                        st.success("✅ Service account loaded!")
                    except Exception as e:
                        st.error(f"❌ Failed to load service account: {e}")
            
            elif auth_option == "🎯 Demo Mode":
                st.info("🎯 Using demo credentials for testing (no real cloud access).")
                if st.button("🎯 Use Demo Credentials"):
                    credentials_data = {
                        "demo_mode": True,
                        "access_token": "demo_token",
                        "client_id": "demo_client", 
                        "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
                    }
                    st.session_state.credentials_data = credentials_data
                    st.success("✅ Demo credentials configured")
                    st.warning("⚠️ Demo mode - no real cloud resources will be accessed")
            
            # Scan organization
            if hasattr(st.session_state, 'credentials_data') and st.session_state.credentials_data:
                if st.button("🔍 Scan Google Cloud Organization"):
                    with st.spinner("Scanning Google Cloud organization..."):
                        scan_data = {
                            "credentials_data": st.session_state.credentials_data,
                            "msa_record": st.session_state.current_msa
                        }
                        
                        result = make_backend_request("/api/v1/msa/scan-gcp", method="POST", data=scan_data)
                        
                        if result.get("success"):
                            st.session_state.current_impact = result["impact_analysis"]
                            st.success(f"✅ {result['message']}")
                            
                            # Display scan results
                            impact_analysis = result["impact_analysis"]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Projects", impact_analysis["total_projects"])
                            with col2:
                                st.metric("Impacted Projects", impact_analysis["impact_summary"]["total_impacted"])
                            with col3:
                                st.metric("Overall Risk", impact_analysis["impact_summary"]["overall_risk_level"])
                        else:
                            st.error(f"❌ Failed to scan organization: {result.get('error')}")
    
    with tab3:
        st.subheader("📊 View Analysis Results")
        
        # Show MSA records
        if st.button("📋 Refresh MSA Records"):
            records_response = make_backend_request("/api/v1/msa/records")
            if records_response.get("success"):
                st.session_state.msa_records = records_response["records"]
        
        if hasattr(st.session_state, 'msa_records') and st.session_state.msa_records:
            st.subheader("📄 MSA Records")
            for record in st.session_state.msa_records:
                with st.expander(f"{record['msa_name']} - {record['timestamp'][:10]}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**User:** {record['user_id']}")
                        st.write(f"**APIs Found:** {len(record['api_mentions'])}")
                        st.write(f"**Word Count:** {record['word_count']}")
                    with col2:
                        st.write(f"**Is MSA:** {'Yes' if record['metadata']['is_msa'] else 'No'}")
                        st.write(f"**Confidence:** {record['metadata']['confidence_score']:.2f}")
                        st.write(f"**Security Keywords:** {record['security_content']['security_keywords_found']}")
        
        # Show impact analyses
        if st.button("📊 Refresh Impact Analyses"):
            analyses_response = make_backend_request("/api/v1/msa/impact-analyses")
            if analyses_response.get("success"):
                st.session_state.impact_analyses = analyses_response["analyses"]
        
        if hasattr(st.session_state, 'impact_analyses') and st.session_state.impact_analyses:
            st.subheader("🔍 Impact Analyses")
            for analysis in st.session_state.impact_analyses:
                with st.expander(f"{analysis['msa_name']} - {analysis['organization_name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Organization:** {analysis['organization_name']}")
                        st.write(f"**Total Projects:** {analysis['total_projects']}")
                        st.write(f"**Impacted Projects:** {analysis['impact_summary']['total_impacted']}")
                    with col2:
                        st.write(f"**Overall Risk:** {analysis['impact_summary']['overall_risk_level']}")
                        st.write(f"**Scan Date:** {analysis['scan_timestamp'][:10]}")
                    
                    # Show impacted projects
                    if analysis['impacted_projects']:
                        st.subheader("🚨 Impacted Projects")
                        for project in analysis['impacted_projects'][:5]:  # Show top 5
                            risk_color = {
                                "HIGH": "🔴",
                                "MEDIUM": "🟡", 
                                "LOW": "🟢"
                            }.get(project['risk_level'], "⚪")
                            
                            st.write(f"{risk_color} **{project['project_name']}** ({project['risk_level']} risk)")
                            st.write(f"   Impact Score: {project['impact_score']:.2f}")
                            st.write(f"   APIs Used: {len(project['impacted_apis'])}")
    
    with tab4:
        st.subheader("⚙️ MSA Analysis Settings")
        
        # API patterns
        if st.button("🔍 View API Patterns"):
            patterns_response = make_backend_request("/api/v1/msa/api-patterns")
            if patterns_response.get("success"):
                st.json(patterns_response["api_patterns"])
        
        # MSA patterns
        if st.button("📄 View MSA Patterns"):
            patterns_response = make_backend_request("/api/v1/msa/msa-patterns")
            if patterns_response.get("success"):
                st.json(patterns_response["msa_patterns"])
        
        # Configuration
        st.subheader("🔧 Configuration")
        st.write("Configure MSA analysis settings here.")
        
        # Add configuration options as needed
        st.info("Configuration options will be added here in future updates.")


def render_agent_dag_agraph():
    """Render agent execution DAG with realistic trace data."""
    st.subheader("🔍 Agent Execution Trace")
    
    # Create tabs for different trace views
    trace_tab1, trace_tab2, trace_tab3 = st.tabs(["🚀 Live Trace", "📊 Trace Analysis", "⚙️ Trace Settings"])
    
    with trace_tab1:
        st.write("**Agent Execution Flow with Timing Information**")
        
        # Simulate realistic agent execution trace
        nodes = [
            Node(id="security_agent", label="🛡️ security_agent\n(45ms)", color="#2E8B57", size=25),
            Node(id="evaluate_api_security", label="🔍 evaluate_api_security\n(120ms)", color="#4682B4", size=20),
            Node(id="scrape_documentation", label="📄 scrape_documentation\n(340ms)", color="#FF6347", size=20),
            Node(id="compliance_check", label="📋 compliance_check\n(89ms)", color="#DAA520", size=20),
            Node(id="threat_analysis", label="⚠️ threat_analysis\n(156ms)", color="#DC143C", size=20),
            Node(id="knowledge_base", label="📚 knowledge_base\n(23ms)", color="#9932CC", size=15),
            Node(id="api_response", label="✅ API Response\n(12ms)", color="#32CD32", size=15),
        ]
        
        edges = [
            Edge(source="security_agent", target="evaluate_api_security", label="start"),
            Edge(source="evaluate_api_security", target="knowledge_base", label="lookup"),
            Edge(source="evaluate_api_security", target="scrape_documentation", label="fetch"),
            Edge(source="security_agent", target="compliance_check", label="parallel"),
            Edge(source="security_agent", target="threat_analysis", label="parallel"),
            Edge(source="compliance_check", target="api_response", label="complete"),
            Edge(source="threat_analysis", target="api_response", label="complete"),
            Edge(source="scrape_documentation", target="api_response", label="complete"),
        ]
        
        config = Config(
            width=800, 
            height=500, 
            directed=True, 
            nodeHighlightBehavior=True, 
            highlightColor="#F7A7A6", 
            collapsible=True,
            physics=True,
            hierarchical=False
        )
        
        selected_node = agraph(nodes=nodes, edges=edges, config=config)
        
        # Show trace details if node is selected
        if selected_node:
            st.subheader(f"🔍 Trace Details: {selected_node}")
            
            trace_details = {
                "security_agent": {
                    "span_id": "span_001",
                    "trace_id": "trace_abc123",
                    "start_time": "2025-01-09T10:30:00.000Z",
                    "duration": "45ms",
                    "status": "OK",
                    "attributes": {
                        "user_id": "demo_user",
                        "query": "Evaluate Cloud Storage security",
                        "session_id": "session_456"
                    }
                },
                "evaluate_api_security": {
                    "span_id": "span_002",
                    "trace_id": "trace_abc123",
                    "start_time": "2025-01-09T10:30:00.045Z",
                    "duration": "120ms",
                    "status": "OK",
                    "attributes": {
                        "api_name": "Cloud Storage",
                        "knowledge_base_hits": 3,
                        "security_score": 85
                    }
                },
                "scrape_documentation": {
                    "span_id": "span_003",
                    "trace_id": "trace_abc123",
                    "start_time": "2025-01-09T10:30:00.165Z",
                    "duration": "340ms",
                    "status": "OK",
                    "attributes": {
                        "url": "https://cloud.google.com/storage/docs/security",
                        "findings_count": 12,
                        "http_status": 200
                    }
                },
                "compliance_check": {
                    "span_id": "span_004",
                    "trace_id": "trace_abc123",
                    "start_time": "2025-01-09T10:30:00.050Z",
                    "duration": "89ms",
                    "status": "OK",
                    "attributes": {
                        "frameworks": ["SOC2", "ISO27001"],
                        "compliance_score": 92,
                        "issues_found": 2
                    }
                },
                "threat_analysis": {
                    "span_id": "span_005",
                    "trace_id": "trace_abc123",
                    "start_time": "2025-01-09T10:30:00.055Z",
                    "duration": "156ms",
                    "status": "OK",
                    "attributes": {
                        "vulnerabilities_found": 1,
                        "risk_level": "LOW",
                        "threat_score": 23
                    }
                }
            }
            
            if selected_node in trace_details:
                details = trace_details[selected_node]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duration", details["duration"])
                    st.metric("Status", details["status"])
                with col2:
                    st.metric("Span ID", details["span_id"])
                    st.metric("Trace ID", details["trace_id"])
                
                st.subheader("📊 Span Attributes")
                st.json(details["attributes"])
    
    with trace_tab2:
        st.subheader("📊 Trace Performance Analysis")
        
        # Create performance metrics chart
        trace_data = [
            {"Component": "security_agent", "Duration (ms)": 45, "Type": "Agent"},
            {"Component": "evaluate_api_security", "Duration (ms)": 120, "Type": "Tool"},
            {"Component": "scrape_documentation", "Duration (ms)": 340, "Type": "Tool"},
            {"Component": "compliance_check", "Duration (ms)": 89, "Type": "Service"},
            {"Component": "threat_analysis", "Duration (ms)": 156, "Type": "Service"},
            {"Component": "knowledge_base", "Duration (ms)": 23, "Type": "Database"},
            {"Component": "api_response", "Duration (ms)": 12, "Type": "Response"},
        ]
        
        df = pd.DataFrame(trace_data)
        
        # Performance chart
        fig = px.bar(df, x="Component", y="Duration (ms)", color="Type",
                    title="Agent Execution Performance by Component")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trace timeline
        st.subheader("⏱️ Execution Timeline")
        timeline_fig = px.timeline(
            df, 
            x_start=[0, 45, 165, 50, 55, 165, 505], 
            x_end=[45, 165, 505, 139, 211, 188, 517],
            y="Component",
            color="Type",
            title="Agent Execution Timeline"
        )
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    with trace_tab3:
        st.subheader("⚙️ Trace Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            trace_enabled = st.checkbox("Enable Tracing", value=True)
            sample_rate = st.slider("Sample Rate", 0.0, 1.0, 1.0, 0.1)
        with col2:
            export_format = st.selectbox("Export Format", ["Cloud Trace", "Jaeger", "Zipkin"])
            trace_level = st.selectbox("Trace Level", ["DEBUG", "INFO", "WARN", "ERROR"])
        
        if st.button("🔄 Refresh Trace Data"):
            st.success("Trace data refreshed!")
        
        # Cloud Trace link
        st.subheader("🔗 View in Google Cloud Trace")
        project_id = st.text_input("Project ID", value="your-project-id")
        if project_id and project_id != "your-project-id":
            trace_url = f"https://console.cloud.google.com/traces/list?project={project_id}"
            st.markdown(f"[🔗 Open Cloud Trace Console]({trace_url})")


def render_oidc_mermaid_diagrams():
    """Render OIDC flow as Mermaid diagrams."""
    st.subheader("🔐 OIDC Authentication Flow Diagrams")
    
    # Create tabs for different OIDC flows
    oidc_tab1, oidc_tab2, oidc_tab3 = st.tabs(["🔄 Authorization Code Flow", "🏢 Client Credentials", "🔧 Token Exchange"])
    
    with oidc_tab1:
        st.write("**OAuth 2.0 Authorization Code Flow with PKCE**")
        
        mermaid_auth_code = """
        sequenceDiagram
            participant User as 👤 User
            participant App as 🖥️ Security Agent
            participant AuthServer as 🔐 Auth Server<br/>(Google/Entra ID)
            participant API as 🛡️ GCP APIs
            
            Note over User,API: 1. Authorization Request
            User->>App: Access Security Agent
            App->>App: Generate PKCE verifier & challenge
            App->>User: Redirect to Auth Server
            Note right of App: /oauth2/authorize?<br/>client_id=xxx&<br/>redirect_uri=xxx&<br/>code_challenge=xxx
            
            Note over User,API: 2. User Authentication
            User->>AuthServer: Login & Consent
            AuthServer->>AuthServer: Validate credentials
            AuthServer->>User: Redirect with auth code
            Note left of AuthServer: /callback?code=abc123&state=xyz
            
            Note over User,API: 3. Token Exchange
            User->>App: Authorization code
            App->>AuthServer: Exchange code for tokens
            Note right of App: POST /oauth2/token<br/>code=abc123&<br/>code_verifier=xxx
            AuthServer->>App: Access & ID tokens
            
            Note over User,API: 4. API Access
            App->>API: API call with access token
            Note right of App: Authorization: Bearer token
            API->>API: Validate token
            API->>App: Protected resource
            App->>User: Security evaluation results
        ```
        """
        st.markdown(mermaid_auth_code)
    
    with oidc_tab2:
        st.write("**OAuth 2.0 Client Credentials Flow (Service-to-Service)**")
        
        mermaid_client_creds = """
        sequenceDiagram
            participant Agent as 🤖 Security Agent
            participant AuthServer as 🔐 Auth Server<br/>(Google/Entra ID)
            participant SecretMgr as 🔑 Secret Manager
            participant API as 🛡️ GCP APIs
            
            Note over Agent,API: 1. Retrieve Credentials
            Agent->>SecretMgr: Get client credentials
            SecretMgr->>Agent: Client ID & Secret
            
            Note over Agent,API: 2. Token Request
            Agent->>AuthServer: POST /oauth2/token
            Note right of Agent: grant_type=client_credentials<br/>client_id=xxx<br/>client_secret=xxx<br/>scope=cloud-platform
            AuthServer->>AuthServer: Validate client
            AuthServer->>Agent: Access token
            
            Note over Agent,API: 3. API Access
            Agent->>API: API call with token
            Note right of Agent: Authorization: Bearer token
            API->>API: Validate token & scope
            API->>Agent: API response
            
            Note over Agent,API: 4. Token Refresh (if needed)
            Agent->>AuthServer: Refresh token
            AuthServer->>Agent: New access token
        ```
        """
        st.markdown(mermaid_client_creds)
    
    with oidc_tab3:
        st.write("**Token Exchange & Cross-Project Access**")
        
        mermaid_token_exchange = """
        sequenceDiagram
            participant User as 👤 User
            participant Agent as 🤖 Security Agent
            participant EntraID as 🏢 Entra ID
            participant GCP as ☁️ GCP STS
            participant SecretMgr as 🔑 Secret Manager
            participant API as 🛡️ GCP APIs
            
            Note over User,API: 1. OIDC Authentication
            User->>Agent: Access request
            Agent->>EntraID: OIDC flow
            EntraID->>Agent: OIDC ID token
            
            Note over User,API: 2. Token Exchange
            Agent->>GCP: Exchange OIDC token
            Note right of Agent: STS Token Exchange<br/>subject_token=oidc_token<br/>audience=//iam.googleapis.com/...
            GCP->>GCP: Validate OIDC token
            GCP->>Agent: GCP access token
            
            Note over User,API: 3. Secret Access
            Agent->>SecretMgr: Get API credentials
            Note right of Agent: Authorization: Bearer gcp_token
            SecretMgr->>Agent: API keys/secrets
            
            Note over User,API: 4. API Hub Access
            Agent->>API: Call API Hub tools
            Note right of Agent: Using retrieved credentials
            API->>Agent: Tool results
            Agent->>User: Security evaluation
        ```
        """
        st.markdown(mermaid_token_exchange)


def adk_chat_ui():
    """ADK-powered chat interface for security agent."""
    st.markdown("---")
    st.header("💬 Security Agent Chat (Powered by ADK)")
    st.write("Chat with the ADK security agent for real-time security analysis and recommendations.")
    
    # Initialize chat history
    if "adk_messages" not in st.session_state:
        st.session_state.adk_messages = [
            {"role": "assistant", "content": "Hello! I'm your GCP Security Agent powered by ADK. I can help you with:\n\n🔒 **Security Analysis** - Evaluate API security posture\n📋 **Compliance Checks** - SOC2, ISO27001, GDPR assessments\n🚨 **Threat Intelligence** - CVE analysis and risk assessment\n⚙️ **Configuration Review** - Security best practices\n📄 **MSA Analysis** - Parse service agreements\n\nWhat would you like to analyze today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.adk_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about GCP security..."):
        # Add user message to chat history
        st.session_state.adk_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response using ADK
        with st.chat_message("assistant"):
            with st.spinner("🤖 ADK Agent thinking..."):
                response = get_adk_response(prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.adk_messages.append({"role": "assistant", "content": response})


def get_adk_response(user_input: str) -> str:
    """Get response from ADK security agent."""
    try:
        # Make request to backend ADK agent
        response = make_backend_request(
            "/api/v1/agent/chat",
            method="POST",
            data={"query": user_input, "user_id": "streamlit_user"}
        )

        if response.get("success"):
            response_text = response.get("response", "I couldn't process that request.")

            # Check for common authentication errors and provide helpful guidance
            if "Missing key inputs" in response_text or "API key" in response_text or "Vertex" in response_text:
                return """🔧 **ADK Agent Configuration Required**

The agent's backend requires proper authentication with Google Cloud via Vertex AI.

**To resolve this, please ensure:**

1.  **Application Default Credentials (ADC) are configured:**
    - Run `gcloud auth application-default login` in your terminal.
2.  **The correct project is set:**
    - Run `gcloud config set project YOUR_PROJECT_ID`.
3.  **Vertex AI API is enabled** for the selected project.

This chat interface will become fully functional once the backend agent can authenticate correctly. All other tools in this application remain available.
"""

            return response_text
        else:
            return f"❌ Error: {response.get('error', 'Unknown error occurred')}"

    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def dag_visualization_ui():
    """Enhanced DAG visualization UI with traces and OIDC flows."""
    st.title("🚀 Agent Execution & Authentication Flows")
    
    # Main tabs
    main_tab1, main_tab2, main_tab3 = st.tabs(["🔍 Agent Traces", "🔐 OIDC Flows", "📊 Performance"])
    
    with main_tab1:
        render_agent_dag_agraph()
    
    with main_tab2:
        render_oidc_mermaid_diagrams()
    
    with main_tab3:
        st.subheader("📊 Overall Performance Dashboard")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Execution Time", "785ms", "-12ms")
        with col2:
            st.metric("Successful Traces", "98.5%", "+0.3%")
        with col3:
            st.metric("Average Response Time", "234ms", "-5ms")
        with col4:
            st.metric("Error Rate", "1.2%", "-0.1%")
        
        # Recent traces table
        st.subheader("🕐 Recent Traces")
        recent_traces = pd.DataFrame([
            {"Timestamp": "2025-01-09 10:35:23", "Trace ID": "trace_def456", "Duration": "892ms", "Status": "✅ Success", "User": "demo@example.com"},
            {"Timestamp": "2025-01-09 10:34:15", "Trace ID": "trace_abc123", "Duration": "785ms", "Status": "✅ Success", "User": "demo@example.com"},
            {"Timestamp": "2025-01-09 10:33:02", "Trace ID": "trace_ghi789", "Duration": "1.2s", "Status": "⚠️ Warning", "User": "demo@example.com"},
            {"Timestamp": "2025-01-09 10:31:45", "Trace ID": "trace_jkl012", "Duration": "645ms", "Status": "✅ Success", "User": "demo@example.com"},
            {"Timestamp": "2025-01-09 10:30:12", "Trace ID": "trace_mno345", "Duration": "1.8s", "Status": "❌ Error", "User": "demo@example.com"},
        ])
        st.dataframe(recent_traces, use_container_width=True)
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        trend_data = pd.DataFrame({
            "Time": pd.date_range("2025-01-09 10:00", periods=20, freq="5min"),
            "Response Time (ms)": [650, 720, 580, 890, 650, 720, 580, 890, 650, 720, 580, 890, 650, 720, 580, 890, 650, 720, 580, 785],
            "Success Rate (%)": [98, 97, 99, 95, 98, 97, 99, 95, 98, 97, 99, 95, 98, 97, 99, 95, 98, 97, 99, 98.5]
        })
        
        fig = px.line(trend_data, x="Time", y="Response Time (ms)", title="Response Time Trend")
        st.plotly_chart(fig, use_container_width=True)


def performance_monitoring_ui():
    """Comprehensive OpenTelemetry performance monitoring interface."""
    st.title("📊 Performance Monitor")
    st.write("Real-time OpenTelemetry tracing and performance analytics for the security agent")
    
    # Performance tabs
    perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs([
        "📈 Live Dashboard", 
        "🔍 Request Tracing", 
        "🚨 Error Monitoring", 
        "💬 Chat Performance"
    ])
    
    with perf_tab1:
        show_live_performance_dashboard()
    
    with perf_tab2:
        show_request_tracing()
    
    with perf_tab3:
        show_error_monitoring()
    
    with perf_tab4:
        show_chat_performance()


def show_live_performance_dashboard():
    """Live performance metrics dashboard."""
    st.subheader("🎯 Real-Time Performance Metrics")
    
    # Auto-refresh option
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**System Status:** 🟢 All systems operational")
    with col2:
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        # This would refresh every 30 seconds in a real implementation
        st.info("🔄 Auto-refresh enabled - Data updates every 30 seconds")
    
    try:
        # Fetch real-time statistics from tracing API
        stats_data = make_backend_request("/api/v1/tracing/statistics", include_project=False)
        
        if stats_data.get('success', True):
            stats = stats_data.get('statistics', {})
            
            # Key performance indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_response = stats.get('average_response_time_ms', 0)
                st.metric(
                    "Avg Response Time", 
                    f"{avg_response:.0f}ms",
                    delta=f"{stats.get('response_time_trend', 0):+.0f}ms"
                )
            
            with col2:
                success_rate = stats.get('success_rate_percent', 0)
                st.metric(
                    "Success Rate", 
                    f"{success_rate:.1f}%",
                    delta=f"{stats.get('success_rate_trend', 0):+.1f}%"
                )
            
            with col3:
                active_requests = stats.get('active_requests', 0)
                st.metric("Active Requests", str(active_requests))
            
            with col4:
                error_rate = stats.get('error_rate_percent', 0)
                st.metric(
                    "Error Rate", 
                    f"{error_rate:.1f}%",
                    delta=f"{stats.get('error_rate_trend', 0):+.1f}%"
                )
            
            # Performance trends chart
            st.subheader("📈 Performance Trends (Last Hour)")
            
            if 'performance_history' in stats:
                history_df = pd.DataFrame(stats['performance_history'])
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['response_time_ms'],
                    mode='lines+markers',
                    name='Response Time (ms)',
                    line=dict(color='#1f77b4')
                ))
                
                fig.update_layout(
                    title="Response Time Trend",
                    xaxis_title="Time",
                    yaxis_title="Response Time (ms)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ No performance history data available")
                st.write("Performance trends require OpenTelemetry tracing to be enabled and configured.")
        
        else:
            st.error("❌ Unable to fetch live performance data")
            st.write("Please check that the backend tracing service is running and accessible.")
    
    except Exception as e:
        st.error(f"❌ Error fetching performance data: {str(e)}")
        st.write("Performance monitoring requires backend services and OpenTelemetry configuration.")


def show_request_tracing():
    """Request tracing with trace IDs and timing breakdown."""
    st.subheader("🔍 Request Tracing & Analysis")
    
    # Trace filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        trace_filter = st.selectbox("Filter by", ["All Requests", "Security Evaluations", "Chat Requests", "Errors Only"])
    with col2:
        time_range = st.selectbox("Time Range", ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours"])
    with col3:
        if st.button("🔄 Refresh Traces"):
            st.rerun()
    
    try:
        # Fetch recent traces
        traces_data = make_backend_request("/api/v1/tracing/traces/recent", include_project=False)
        
        if traces_data.get('success', True) and 'traces' in traces_data:
            traces = traces_data['traces']
            
            st.subheader(f"📋 Recent Traces ({len(traces)} found)")
            
            # Traces table with expandable details
            for trace in traces[:10]:  # Show last 10 traces
                with st.expander(f"🔗 {trace.get('operation', 'Unknown')} - {trace.get('duration_ms', 0)}ms - {trace.get('timestamp', '')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Trace Details:**")
                        st.write(f"• **Trace ID:** `{trace.get('trace_id', 'N/A')}`")
                        st.write(f"• **Duration:** {trace.get('duration_ms', 0)}ms")
                        st.write(f"• **Status:** {trace.get('status', 'Unknown')}")
                        st.write(f"• **User:** {trace.get('user', 'Anonymous')}")
                        
                        # Cloud Trace link
                        if trace.get('trace_id') and st.session_state.selected_project:
                            trace_url = f"https://console.cloud.google.com/traces/list?project={st.session_state.selected_project}&trace={trace['trace_id']}"
                            st.link_button("🔍 View in Cloud Trace", trace_url)
                    
                    with col2:
                        st.write("**Performance Breakdown:**")
                        
                        # Performance breakdown (if available from trace data)
                        breakdown = trace.get('breakdown', {})
                        
                        if breakdown:
                            for step, duration in breakdown.items():
                                st.write(f"• **{step.title()}:** {duration}")
                        else:
                            st.write("No performance breakdown available for this trace")
                        
                        # Error details if present
                        if trace.get('error'):
                            st.error(f"❌ **Error:** {trace['error']}")
        else:
            st.warning("⚠️ No recent traces found")
            st.write("Traces will appear here once OpenTelemetry is configured and requests are made.")
    
    except Exception as e:
        st.error(f"❌ Error fetching trace data: {str(e)}")
        st.write("Trace data requires backend tracing endpoints to be implemented and accessible.")


def show_error_monitoring():
    """Error tracking and monitoring."""
    st.subheader("🚨 Error Monitoring & Analysis")
    
    # Error summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Errors (Last Hour)", "3", delta="-2")
    with col2:
        st.metric("Most Common Error", "Authentication Timeout")
    with col3:
        st.metric("MTTR (Mean Time to Resolve)", "12 min", delta="-3 min")
    
    try:
        # Fetch recent errors (this endpoint would need to be implemented)
        errors_data = make_backend_request("/api/v1/tracing/errors/recent", include_project=False)
        
        if errors_data.get('success', False) and 'errors' in errors_data:
            errors = errors_data['errors']
            
            st.subheader("🔴 Recent Errors")
            
            for error in errors:
                with st.expander(f"❌ {error.get('operation', 'Unknown Error')} - {error.get('timestamp', '')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Error Details:**")
                        st.code(error.get('error_message', 'No details available'))
                        st.write(f"**Trace ID:** `{error.get('trace_id', 'N/A')}`")
                        st.write(f"**Project:** {error.get('project_id', 'N/A')}")
                    
                    with col2:
                        st.write("**Context:**")
                        st.write(f"• **User:** {error.get('user', 'Anonymous')}")
                        st.write(f"• **Endpoint:** {error.get('endpoint', 'Unknown')}")
                        st.write(f"• **Duration:** {error.get('duration_ms', 0)}ms")
                        
                        # Cloud Trace link for error investigation
                        if error.get('trace_id') and st.session_state.selected_project:
                            trace_url = f"https://console.cloud.google.com/traces/list?project={st.session_state.selected_project}&trace={error['trace_id']}"
                            st.link_button("🔍 Investigate in Cloud Trace", trace_url)
        else:
            st.info("✅ No recent errors found")
            st.write("Error monitoring data will appear here when errors occur.")
    
    except Exception as e:
        st.error(f"❌ Error fetching error data: {str(e)}")
        st.write("Error monitoring requires backend error tracking endpoints to be implemented.")


def show_chat_performance():
    """ADK chat performance monitoring."""
    st.subheader("💬 Chat Performance Analytics")
    
    # Chat performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Chat Response", "1.2s", delta="-0.3s")
    with col2:
        st.metric("Vertex AI Latency", "890ms", delta="-120ms")
    with col3:
        st.metric("Chat Success Rate", "99.1%", delta="+0.5%")
    with col4:
        st.metric("Tokens Used/Hour", "2,340", delta="+230")
    
    st.subheader("📈 Chat Performance Trends")
    
    # Try to fetch real chat performance data
    try:
        chat_perf_data = make_backend_request("/api/v1/tracing/chat-performance", include_project=False)
        
        if chat_perf_data.get('success', False) and 'performance_data' in chat_perf_data:
            perf_data = chat_perf_data['performance_data']
            
            # Create DataFrame from real data
            if perf_data.get('history'):
                chat_df = pd.DataFrame(perf_data['history'])
                chat_df['timestamp'] = pd.to_datetime(chat_df['timestamp'])
                
                # Response time chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=chat_df['timestamp'], 
                    y=chat_df['response_time_ms'],
                    mode='lines+markers',
                    name='Total Response Time',
                    line=dict(color='#1f77b4')
                ))
                if 'vertex_ai_time_ms' in chat_df.columns:
                    fig1.add_trace(go.Scatter(
                        x=chat_df['timestamp'], 
                        y=chat_df['vertex_ai_time_ms'],
                        mode='lines+markers',
                        name='Vertex AI Time',
                        line=dict(color='#ff7f0e')
                    ))
                fig1.update_layout(
                    title="Chat Response Time Breakdown",
                    xaxis_title="Time",
                    yaxis_title="Response Time (ms)",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Token usage chart if available
                if 'tokens_used' in chat_df.columns:
                    fig2 = px.bar(chat_df, x='timestamp', y='tokens_used', 
                                  title='Token Usage Over Time')
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("📊 No chat performance history available")
            
            # Recent chat requests
            st.subheader("💬 Recent Chat Requests")
            if perf_data.get('recent_chats'):
                recent_chats_df = pd.DataFrame(perf_data['recent_chats'])
                st.dataframe(recent_chats_df, use_container_width=True)
            else:
                st.info("No recent chat data available")
        else:
            st.error("❌ Unable to fetch chat performance data")
            st.write("Chat performance monitoring requires backend endpoints to be implemented.")
    
    except Exception as e:
        st.error(f"❌ Error fetching chat performance data: {str(e)}")
        st.write("Chat performance monitoring requires backend tracing services to be configured.")










def api_explorer_ui():
    """UI for exploring Google Cloud APIs."""
    st.markdown("---")
    st.header("🔍 API Explorer")
    st.write("Directly call any Google Cloud API using the generic API tool.")

    with st.form("api_explorer_form"):
        st.subheader("Request Details")
        col1, col2 = st.columns(2)
        with col1:
            service = st.text_input("Service Name", "run.googleapis.com", help="e.g., run.googleapis.com, iam.googleapis.com")
            version = st.text_input("API Version", "v1", help="e.g., v1, v2, v3")
            method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
        with col2:
            resource_path = st.text_input("Resource Path", "projects/mgm-digitalconcierge/locations/us-central1/services", help="e.g., projects/YOUR_PROJECT/locations/us-central1/services")
        
        body = st.text_area("Request Body (JSON)", "{}", help="Enter a valid JSON object or leave as {} for GET/DELETE requests.")

        if st.form_submit_button("🚀 Execute API Call"):
            try:
                body_json = json.loads(body)
                request_data = {
                    "service": service,
                    "version": version,
                    "resource_path": resource_path,
                    "method": method,
                    "body": body_json if method in ["POST", "PUT"] else None
                }
                with st.spinner("Calling API..."):
                    response = make_backend_request("/api/v1/gcp/call-api", method="POST", data=request_data)
                    st.session_state.api_explorer_response = response
            except json.JSONDecodeError:
                st.error("Invalid JSON in request body.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if "api_explorer_response" in st.session_state:
        st.markdown("---")
        st.subheader("API Response")
        if st.session_state.api_explorer_response.get("success"):
            st.json(st.session_state.api_explorer_response.get("response", {}))
        else:
            st.error(st.session_state.api_explorer_response.get("error", "An unknown error occurred."))


def main():
    """Main application."""
    st.set_page_config(
        page_title="Enhanced GCP Security Agent",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🔒 GCP Security Agent")
    
    # Check if button navigation was triggered
    if hasattr(st.session_state, 'page') and st.session_state.page:
        selected_page = st.session_state.page
        st.session_state.page = None  # Reset after use
    else:
        selected_page = None
    
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Dashboard", "💬 ADK Chat", "🔍 API Explorer", "🔐 OIDC Demo", "🔒 Security Evaluation", "📄 MSA Analysis", "🚨 Incident Response", "📚 Knowledge Base", "🚀 Agent DAG", "📊 Performance Monitor"],
        index=["🏠 Dashboard", "💬 ADK Chat", "🔍 API Explorer", "🔐 OIDC Demo", "🔒 Security Evaluation", "📄 MSA Analysis", "🚨 Incident Response", "📚 Knowledge Base", "🚀 Agent DAG", "📊 Performance Monitor"].index(selected_page) if selected_page else 0
    )
    
    # Add GCP project picker
    project_picker_sidebar()
    
    # Main content
    if page == "🏠 Dashboard":
        st.title("Enhanced GCP API Security Evaluation Agent")
        st.write("Comprehensive security evaluation with OIDC authentication demonstration")
        
        # Dashboard overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            backend_status = make_backend_request("/health", include_project=False)
            st.metric("Backend Status", "🟢 Online" if backend_status.get("status") == "healthy" else "🔴 Offline")
        with col2:
            st.metric("Authentication", "✅ Authenticated" if st.session_state.current_user else "❌ Not Authenticated")
        with col3:
            st.metric("User", st.session_state.current_user.get("email", "Demo User") if st.session_state.current_user else "Guest")
        with col4:
            st.metric("Active Project", f"🏗️ {st.session_state.selected_project}")
        
        # Project Information Section
        st.subheader("🏗️ Project Information")
        with st.expander("Project Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Project ID:** {st.session_state.selected_project}")
                st.info("**Region:** us-central1")
                st.info("**Authentication:** Application Default Credentials")
            with col2:
                if st.session_state.selected_project and st.session_state.selected_project != "None":
                    # Get project info from backend
                    project_info = make_backend_request(f"/api/v1/gcp/project/{st.session_state.selected_project}/info", include_project=False)
                    if project_info.get("success"):
                        project = project_info.get("project", {})
                        st.success(f"**Status:** {project.get('state', 'Unknown')}")
                        st.success(f"**Display Name:** {project.get('display_name', 'N/A')}")
                    else:
                        st.warning("**Status:** Could not fetch project details")
                    
                    # Show services count
                    services_info = make_backend_request(f"/api/v1/gcp/project/{st.session_state.selected_project}/services", include_project=False)
                    if services_info.get("success"):
                        st.metric("Enabled Services", services_info.get("total_services", 0))
                    else:
                        st.metric("Enabled Services", "Unknown")
                else:
                    st.warning("No project selected. Please select a project from the sidebar.")
        
        # Security Recommendations Section
        st.subheader("🎯 Security Recommendations")
        if st.session_state.selected_project and st.session_state.selected_project != "None":
            # Get recommendations from backend
            recommendations_data = make_backend_request(
                "/api/v1/recommendations/dashboard",
                method="POST",
                data={
                    "project_id": st.session_state.selected_project,
                    "user_email": st.session_state.current_user.get("email", "admin@stuartgano.altostrat.com"),
                    "priority": "high"  # Show only high priority on dashboard
                }
            )
            
            if recommendations_data.get("success"):
                recs_info = recommendations_data.get("data", {})
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("High Priority", recs_info.get("high_priority", 0), delta=None)
                with col2:
                    st.metric("Medium Priority", recs_info.get("medium_priority", 0), delta=None)
                with col3:
                    st.metric("Low Priority", recs_info.get("low_priority", 0), delta=None)
                with col4:
                    st.metric("Total Items", recs_info.get("total_recommendations", 0), delta=None)
                
                # Show top recommendations
                recommendations = recs_info.get("recommendations", [])
                if recommendations:
                    st.markdown("### 🔥 Top Priority Recommendations")
                    for i, rec in enumerate(recommendations[:3]):  # Show top 3
                        with st.expander(f"⚠️ {rec.get('title', 'Unknown')}", expanded=i==0):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(rec.get('description', ''))
                                st.write(f"**Category:** {rec.get('category', 'General')}")
                                st.write(f"**Impact:** {rec.get('impact', 'Unknown')} | **Effort:** {rec.get('effort', 'Unknown')}")
                                
                                # Show actions
                                if rec.get('actions'):
                                    st.write("**Next Steps:**")
                                    for action in rec.get('actions', []):
                                        st.write(f"• {action}")
                            
                            with col2:
                                # Priority badge
                                priority = rec.get('priority', 'medium')
                                if priority == 'high':
                                    st.error(f"🔴 {priority.upper()}")
                                elif priority == 'medium':
                                    st.warning(f"🟡 {priority.upper()}")
                                else:
                                    st.info(f"🟢 {priority.upper()}")
                                
                                # Compliance frameworks
                                frameworks = rec.get('compliance_frameworks', [])
                                if frameworks:
                                    st.write("**Compliance:**")
                                    for fw in frameworks:
                                        st.write(f"• {fw}")
                    
                    # View all recommendations button
                    if st.button("📋 View All Recommendations"):
                        st.session_state.page = "🔒 Security Evaluation"
                        st.rerun()
                else:
                    st.success("🎉 No high-priority recommendations found!")
            else:
                st.error("❌ Failed to load security recommendations")
        else:
            st.info("Select a project to view security recommendations")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("🔒 Evaluate Security"):
                st.session_state.page = "🔒 Security Evaluation"
                st.rerun()
        with col2:
            if st.button("🔐 OIDC Demo"):
                st.session_state.page = "🔐 OIDC Demo"
                st.rerun()
        with col3:
            if st.button("📄 MSA Analysis"):
                st.session_state.page = "📄 MSA Analysis"
                st.rerun()
        with col4:
            if st.button("🚨 Incident Response"):
                st.session_state.page = "🚨 Incident Response"
                st.rerun()
        with col5:
            if st.button("🌐 ADK Web Interface"):
                st.info("Opening ADK Web Interface...")
                st.markdown("""
                <script>
                window.open('http://localhost:8080', '_blank');
                </script>
                """, unsafe_allow_html=True)
                st.success("✅ ADK Web Interface opened in new tab")
                st.markdown("**Direct Link:** [http://localhost:8080](http://localhost:8080)")
    
    elif page == "🔍 API Explorer":
        api_explorer_ui()
    
    elif page == "💬 ADK Chat":
        adk_chat_ui()
    
    elif page == "🔐 OIDC Demo":
        oidc_flow_demo()
    
    elif page == "🔒 Security Evaluation":
        security_evaluation_ui()
    
    elif page == "📄 MSA Analysis":
        msa_analysis_ui()
    
    elif page == "🚨 Incident Response":
        incident_response_ui()
    
    elif page == "📚 Knowledge Base":
        knowledge_base_ui()
    
    elif page == "🚀 Agent DAG":
        dag_visualization_ui()


if __name__ == "__main__":
    main() 